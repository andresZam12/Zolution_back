"""
Appointment scheduling API endpoints for tenant dashboard.
Persists appointments in PostgreSQL per tenant (protected by RLS) and synchronizes
with Google Calendar when the tenant has connected their Google account.
"""

import logging
import uuid
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm.attributes import flag_modified

from app.agents.models import AgentConfig
from app.auth.dependencies import get_current_user
from app.auth.models import UserContext
from app.core.database import get_db_session
from app.integrations.google.calendar import GoogleCalendarService
from app.tenants.models import Organization

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/appointments", tags=["Appointments"])


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


def _require_org_id(user: UserContext) -> uuid.UUID:
    """Extract organization_id or raise 403."""
    if not user.organization_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="An organization context is required.",
        )
    return user.organization_id


async def _get_or_create_agent_config(db: AsyncSession, org_id: uuid.UUID) -> AgentConfig:
    try:
        await db.execute(
            text("SELECT set_config('app.current_organization_id', :org_id, true)"),
            {"org_id": str(org_id)},
        )
    except Exception:
        pass

    stmt = select(AgentConfig).where(AgentConfig.organization_id == org_id)
    cfg = (await db.execute(stmt)).scalar_one_or_none()
    if cfg is None:
        org_res = await db.execute(select(Organization).where(Organization.id == org_id))
        org = org_res.scalar_one_or_none()
        cfg = AgentConfig(
            organization_id=org_id,
            status="active",
            llm_provider="openai",
            llm_model="gpt-4o-mini",
            onboarding_answers={
                "companyName": org.name if org else "Mi Empresa",
                "business_name": org.name if org else "Mi Empresa",
                "appointments": [],
            },
        )
        db.add(cfg)
        await db.flush()
    return cfg


@router.get(
    "",
    response_model=AppointmentListResponse,
    summary="List persisted appointments for the current tenant",
)
async def list_appointments(
    user: UserContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
) -> AppointmentListResponse:
    """Return all appointments stored in PostgreSQL for the authenticated tenant."""
    org_id = _require_org_id(user)
    cfg = await _get_or_create_agent_config(db, org_id)
    ans = dict(cfg.onboarding_answers or {})
    raw_list = ans.get("appointments")
    if not isinstance(raw_list, list):
        raw_list = []

    items: list[AppointmentItemSchema] = []
    for item in raw_list:
        if isinstance(item, dict) and item.get("id"):
            items.append(
                AppointmentItemSchema(
                    id=str(item.get("id")),
                    patientName=str(item.get("patientName", "Cliente")),
                    phone=str(item.get("phone", "")),
                    service=str(item.get("service", "Consulta General")),
                    date=str(item.get("date", "")),
                    time=str(item.get("time", "")),
                    status=str(item.get("status", "confirmed")),
                    googleEventId=str(item.get("googleEventId", "")),
                    htmlLink=item.get("htmlLink"),
                    createdAt=item.get("createdAt"),
                )
            )

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
    If Google Calendar is not yet connected, checks availability against the
    tenant's PostgreSQL appointments for that date.
    """
    org_id = _require_org_id(user)
    cfg = await _get_or_create_agent_config(db, org_id)
    service = GoogleCalendarService(db, str(org_id))

    try:
        result = await service.get_availability(date)
        return AvailabilityResponse(date=date, availability=result, google_connected=True)
    except ValueError:
        ans = dict(cfg.onboarding_answers or {})
        raw_list = ans.get("appointments") if isinstance(ans.get("appointments"), list) else []
        booked_for_date = [
            f"{a.get('time')} ({a.get('patientName')})"
            for a in raw_list
            if isinstance(a, dict) and str(a.get("date", "")).startswith(date)
        ]
        schedule_str = ans.get("schedule") or "Lunes a Viernes 08:00 - 18:00"
        if booked_for_date:
            msg = (
                f"Agenda interna en PostgreSQL para {date}: "
                f"Turnos ocupados: {', '.join(booked_for_date)}. "
                f"Resto del horario ({schedule_str}) disponible."
            )
        else:
            msg = (
                f"Agenda disponible en PostgreSQL para el {date} "
                f"según horario ({schedule_str})."
            )
        return AvailabilityResponse(date=date, availability=msg, google_connected=False)


@router.post(
    "",
    response_model=AppointmentResponse,
    summary="Create appointment in PostgreSQL and Google Calendar",
)
async def create_appointment(
    payload: AppointmentCreate,
    user: UserContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
) -> AppointmentResponse:
    """
    Book a new appointment: persists in PostgreSQL for the tenant and, if Google
    Calendar is connected, creates the event directly in the primary calendar.
    """
    org_id = _require_org_id(user)
    cfg = await _get_or_create_agent_config(db, org_id)
    service = GoogleCalendarService(db, str(org_id))

    service_label = payload.service or payload.notes or "Consulta / Asesoría"
    synced_google = False
    event_id = f"pg_evt_{uuid.uuid4().hex[:8]}"
    html_link: Optional[str] = None
    cal_msg = "Cita guardada en PostgreSQL."

    try:
        res = await service.create_appointment(
            customer_name=payload.customer_name,
            customer_phone=payload.customer_phone,
            start_time=payload.start_time,
            end_time=payload.end_time,
            notes=service_label,
        )
        if isinstance(res, dict) and res.get("success"):
            synced_google = True
            event_id = res.get("event_id") or event_id
            html_link = res.get("html_link") or None
            cal_msg = "¡Cita guardada en PostgreSQL y sincronizada en Google Calendar!"
    except ValueError:
        cal_msg = "Cita guardada en PostgreSQL (Conecta Google Calendar en Integraciones para sincronización externa)."
    except Exception as exc:
        logger.warning("Google Calendar sync warning: %s", exc)

    date_part = payload.date_label or payload.start_time.split("T")[0]
    time_part = payload.time_label or (
        payload.start_time.split("T")[1][:5] if "T" in payload.start_time else payload.start_time
    )

    new_item = AppointmentItemSchema(
        id=f"apt-{uuid.uuid4().hex[:10]}",
        patientName=payload.customer_name.strip(),
        phone=payload.customer_phone.strip(),
        service=service_label,
        date=date_part,
        time=time_part,
        status="confirmed",
        googleEventId=event_id,
        htmlLink=html_link,
        createdAt=datetime.utcnow().isoformat() + "Z",
    )

    ans = dict(cfg.onboarding_answers or {})
    existing_apts = ans.get("appointments")
    if not isinstance(existing_apts, list):
        existing_apts = []
    existing_apts.insert(0, new_item.model_dump())
    ans["appointments"] = existing_apts
    cfg.onboarding_answers = ans
    flag_modified(cfg, "onboarding_answers")
    db.add(cfg)
    await db.commit()

    return AppointmentResponse(
        status="success",
        message=cal_msg,
        synced_with_google=synced_google,
        event_id=event_id,
        html_link=html_link,
        appointment=new_item,
    )


class AppointmentStatusUpdate(BaseModel):
    status: str


@router.patch(
    "/{appointment_id}/status",
    response_model=AppointmentItemSchema,
    summary="Update appointment status (confirmed, completed, cancelled)",
)
async def update_appointment_status(
    appointment_id: str,
    payload: AppointmentStatusUpdate,
    user: UserContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
) -> AppointmentItemSchema:
    org_id = _require_org_id(user)
    cfg = await _get_or_create_agent_config(db, org_id)
    ans = dict(cfg.onboarding_answers or {})
    existing_apts = ans.get("appointments")
    if not isinstance(existing_apts, list):
        existing_apts = []

    updated_item = None
    for item in existing_apts:
        if isinstance(item, dict) and item.get("id") == appointment_id:
            item["status"] = payload.status
            updated_item = item
            break

    if not updated_item:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Cita no encontrada.",
        )

    ans["appointments"] = existing_apts
    cfg.onboarding_answers = ans
    flag_modified(cfg, "onboarding_answers")
    db.add(cfg)
    await db.commit()

    return AppointmentItemSchema(**updated_item)
