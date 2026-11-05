"""
Appointment scheduling API endpoints for tenant dashboard.
Appointments live in PostgreSQL per tenant (protected by RLS) and are mirrored to
Google Calendar when the tenant has connected their Google account.
"""

import logging
import uuid
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.models import AgentConfig
from app.appointments.models import Appointment
from app.appointments.services import BOGOTA, AppointmentService, BookingOutcome
from app.auth.dependencies import get_current_user
from app.auth.models import UserContext
from app.core.database import bind_tenant, get_db_session
from app.integrations.google.calendar import CALENDAR_NOT_CONNECTED, GoogleCalendarService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/appointments", tags=["Appointments"])

VALID_STATUSES = {"confirmed", "completed", "cancelled"}
SLOT_TAKEN_DETAIL = "Ese horario ya está ocupado. Elige otra fecha u hora."


class AppointmentItemSchema(BaseModel):
    id: str
    patientName: str
    phone: str
    service: str
    date: str
    time: str
    status: str = "confirmed"
    googleEventId: str
    htmlLink: Optional[str] = None
    createdAt: Optional[str] = None


class AppointmentListResponse(BaseModel):
    items: list[AppointmentItemSchema]
    total: int


class AppointmentCreate(BaseModel):
    """Schema for manual appointment booking from dashboard."""

    customer_name: str
    customer_phone: str
    start_time: str
    end_time: str
    service: Optional[str] = None
    date_label: Optional[str] = None
    time_label: Optional[str] = None
    notes: Optional[str] = None


class AppointmentResponse(BaseModel):
    """Result of booking an appointment."""

    status: str
    message: str
    synced_with_google: bool = False
    event_id: Optional[str] = None
    html_link: Optional[str] = None
    appointment: Optional[AppointmentItemSchema] = None


class AvailabilityResponse(BaseModel):
    """Availability query result."""

    date: str
    availability: str
    google_connected: bool = False


class AppointmentStatusUpdate(BaseModel):
    status: str


def _require_org_id(user: UserContext) -> uuid.UUID:
    """Extract organization_id or raise 403."""
    if not user.organization_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="An organization context is required.",
        )
    return user.organization_id


async def _service(db: AsyncSession, org_id: uuid.UUID) -> AppointmentService:
    await bind_tenant(db, org_id)
    return AppointmentService(db, org_id, GoogleCalendarService(db, str(org_id)))


def _to_item(appointment: Appointment) -> AppointmentItemSchema:
    start = appointment.starts_at.astimezone(BOGOTA)
    end = appointment.ends_at.astimezone(BOGOTA)
    return AppointmentItemSchema(
        id=str(appointment.id),
        patientName=appointment.customer_name,
        phone=appointment.customer_phone,
        service=appointment.service,
        date=start.strftime("%Y-%m-%d"),
        time=f"{start:%H:%M} - {end:%H:%M}",
        status=appointment.status,
        googleEventId=appointment.google_event_id or "",
        htmlLink=appointment.html_link,
        createdAt=appointment.created_at.isoformat() if appointment.created_at else None,
    )


async def _find_or_404(service: AppointmentService, appointment_id: uuid.UUID) -> Appointment:
    appointment = await service.get(appointment_id)
    if appointment is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Cita no encontrada.")
    return appointment


def _parse_window(payload: AppointmentCreate) -> tuple[datetime, datetime]:
    try:
        start = datetime.fromisoformat(payload.start_time)
        end = datetime.fromisoformat(payload.end_time)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Fecha u hora inválida.",
        ) from exc

    start = start if start.tzinfo else start.replace(tzinfo=BOGOTA)
    end = end if end.tzinfo else end.replace(tzinfo=BOGOTA)
    if end <= start:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="La hora final debe ser posterior a la inicial.",
        )
    return start, end


@router.get(
    "",
    response_model=AppointmentListResponse,
    summary="List appointments for the current tenant",
)
async def list_appointments(
    user: UserContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
) -> AppointmentListResponse:
    """Return every channel's appointments for the authenticated tenant."""
    service = await _service(db, _require_org_id(user))
    items = [_to_item(appointment) for appointment in await service.list_all()]
    return AppointmentListResponse(items=items, total=len(items))


@router.get(
    "/availability",
    response_model=AvailabilityResponse,
    summary="Check calendar availability for a date",
)
async def check_availability(
    date: str = Query(..., description="Date in YYYY-MM-DD format"),
    user: UserContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
) -> AvailabilityResponse:
    """
    Query Google Calendar availability for the tenant's primary calendar.
    Without a connected calendar, reports the tenant's own agenda for that date.
    """
    org_id = _require_org_id(user)
    service = await _service(db, org_id)

    availability = await service.calendar.get_availability(date)
    if availability != CALENDAR_NOT_CONNECTED:
        return AvailabilityResponse(date=date, availability=availability, google_connected=True)

    config = (await db.execute(select(AgentConfig))).scalar_one_or_none()
    answers = (config.onboarding_answers or {}) if config else {}
    schedule = answers.get("schedule") or "Lunes a Viernes 08:00 - 18:00"
    booked = [
        f"{item.time} ({item.patientName})"
        for item in map(_to_item, await service.list_all())
        if item.date == date and item.status == "confirmed"
    ]
    message = (
        f"Turnos ocupados el {date}: {', '.join(booked)}. Resto del horario ({schedule}) disponible."
        if booked
        else f"Agenda disponible el {date} según horario ({schedule})."
    )
    return AvailabilityResponse(date=date, availability=message, google_connected=False)


@router.post(
    "",
    response_model=AppointmentResponse,
    summary="Create appointment and mirror it to Google Calendar",
)
async def create_appointment(
    payload: AppointmentCreate,
    user: UserContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
) -> AppointmentResponse:
    """Book a manual appointment. Occupied slots are rejected with 409."""
    start, end = _parse_window(payload)
    service = await _service(db, _require_org_id(user))

    result = await service.book(
        payload.customer_name.strip(),
        payload.customer_phone.strip(),
        start,
        end,
        payload.service or payload.notes or "Consulta / Asesoría",
        "dashboard",
        require_calendar=False,
    )
    if result.outcome == BookingOutcome.SLOT_TAKEN:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=SLOT_TAKEN_DETAIL)
    if result.outcome != BookingOutcome.BOOKED:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="No se pudo crear la cita en Google Calendar.",
        )

    appointment = result.appointment
    return AppointmentResponse(
        status="success",
        message=(
            "¡Cita guardada y sincronizada en Google Calendar!"
            if result.synced
            else "Cita guardada. Conecta Google Calendar para sincronizarla."
        ),
        synced_with_google=result.synced,
        event_id=appointment.google_event_id,
        html_link=appointment.html_link,
        appointment=_to_item(appointment),
    )


@router.patch(
    "/{appointment_id}/status",
    response_model=AppointmentItemSchema,
    summary="Update appointment status (confirmed, completed, cancelled)",
)
async def update_appointment_status(
    appointment_id: uuid.UUID,
    payload: AppointmentStatusUpdate,
    user: UserContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
) -> AppointmentItemSchema:
    if payload.status not in VALID_STATUSES:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Estado de cita inválido.",
        )

    service = await _service(db, _require_org_id(user))
    appointment = await _find_or_404(service, appointment_id)

    if payload.status == "cancelled":
        await service.cancel(appointment, require_calendar=False)
    elif not await service.set_status(appointment, payload.status):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=SLOT_TAKEN_DETAIL)

    return _to_item(appointment)


@router.delete(
    "/{appointment_id}",
    status_code=status.HTTP_200_OK,
    summary="Delete an appointment and remove it from Google Calendar",
)
async def delete_appointment(
    appointment_id: uuid.UUID,
    user: UserContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
) -> dict:
    service = await _service(db, _require_org_id(user))
    appointment = await _find_or_404(service, appointment_id)

    await service.cancel(appointment, require_calendar=False)
    await db.delete(appointment)
    await db.commit()

    return {"status": "success", "message": "Cita eliminada y desincronizada de Google Calendar."}
