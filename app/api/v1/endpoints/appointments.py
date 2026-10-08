"""
Appointment scheduling API endpoints for tenant dashboard.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.dependencies import get_current_user
from app.auth.models import UserContext
from app.core.database import get_db_session
from app.integrations.google.calendar import GoogleCalendarService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/appointments", tags=["Appointments"])


class AppointmentCreate(BaseModel):
    """Schema for manual appointment booking from dashboard."""

    customer_name: str
    customer_phone: str
    start_time: str
    end_time: str


class AppointmentResponse(BaseModel):
    """Result of booking an appointment."""

    status: str
    message: str


class AvailabilityResponse(BaseModel):
    """Availability query result."""

    date: str
    availability: str


def _require_org_id(user: UserContext) -> str:
    """Extract organization_id or raise 403."""
    if not user.organization_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="An organization context is required.",
        )
    return str(user.organization_id)


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
    """
    org_id = _require_org_id(user)
    service = GoogleCalendarService(db, org_id)

    try:
        result = await service.get_availability(date)
        return AvailabilityResponse(date=date, availability=result)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc


@router.post(
    "",
    response_model=AppointmentResponse,
    summary="Create appointment in Google Calendar",
)
async def create_appointment(
    payload: AppointmentCreate,
    user: UserContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
) -> AppointmentResponse:
    """
    Book a new appointment directly into Google Calendar.
    """
    org_id = _require_org_id(user)
    service = GoogleCalendarService(db, org_id)

    try:
        msg = await service.create_appointment(
            customer_name=payload.customer_name,
            customer_phone=payload.customer_phone,
            start_time=payload.start_time,
            end_time=payload.end_time,
        )
        return AppointmentResponse(status="success", message=msg)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
