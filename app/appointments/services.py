"""
Appointment service layer.

The single path every channel uses to book, move and cancel appointments.
PostgreSQL is the source of truth; Google Calendar mirrors it and also reports
when the business owner is busy with events of their own.
"""

import logging
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import StrEnum

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.appointments.models import Appointment
from app.integrations.google.calendar import GoogleCalendarService

logger = logging.getLogger(__name__)

BOGOTA = timezone(timedelta(hours=-5))
DEFAULT_DURATION = timedelta(hours=1)
DEFAULT_SERVICE = "Asesoría especializada"
_OVERLAP_CONSTRAINT = "appointments_no_overlap"

CALENDAR_FAILED_REPLY = (
    "Recibí tu solicitud, pero no pude actualizar la agenda en este momento, así que "
    "este cambio aún no está confirmado. Por favor intenta de nuevo en unos minutos."
)
NEED_DATETIME_REPLY = (
    "Para continuar necesito la fecha y la hora exactas de tu cita. ¿Me las confirmas?"
)
NO_APPOINTMENT_REPLY = (
    "No encontré una cita próxima registrada con este número. ¿Deseas agendar una nueva?"
)
SLOT_TAKEN_REPLY = "Ese horario ya no está disponible. ¿Te sirve otro día u otra hora?"

_COMMAND = re.compile(r"\[\[\s*(AGENDAR|REAGENDAR|CANCELAR)\b([^\]]*)\]\]", re.IGNORECASE)
_TIME = re.compile(r"^(\d{1,2})(?::(\d{2}))?\s*(am|pm|a\.m\.|p\.m\.)?$", re.IGNORECASE)


class BookingOutcome(StrEnum):
    BOOKED = "booked"
    ALREADY_BOOKED = "already_booked"
    SLOT_TAKEN = "slot_taken"
    FAILED = "failed"


class _CalendarWriteError(Exception):
    """The calendar rejected a write, so the surrounding savepoint must be undone."""


@dataclass
class BookingResult:
    outcome: BookingOutcome
    appointment: Appointment | None = None
    synced: bool = False


@dataclass
class CommandResult:
    """What a calendar command actually did. `reply_override` replaces the model's reply."""

    reply_override: str | None = None
    booked_start: datetime | None = None
    changed: bool = False
    synced: bool = False


def parse_command(reply_text: str) -> tuple[str | None, list[str]]:
    """Extract the calendar command the model emitted, if any."""
    match = _COMMAND.search(reply_text)
    if not match:
        return None, []
    raw_args = match.group(2).strip()
    if raw_args.startswith(("|", ":")):
        raw_args = raw_args[1:].strip()
    return match.group(1).upper(), [part.strip() for part in raw_args.split("|") if part.strip()]


def strip_command_tags(reply_text: str) -> str:
    """Remove command tags while keeping all the surrounding conversation text."""
    text = re.sub(r"\[\[\s*[A-Z_]+[^\]]*\]\]?", "", reply_text)
    text = re.sub(r"\[\[.*?\]\]?", "", text)
    text = re.sub(r"\[\[[A-Za-z0-9_|:\- ]*$", "", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def parse_appointment_start(date_arg: str, time_arg: str, now: datetime) -> datetime | None:
    """Parse the date and time of a command; None unless they form a future slot."""
    time_match = _TIME.match(time_arg.strip())
    if not time_match:
        return None

    hour = int(time_match.group(1))
    minute = int(time_match.group(2) or 0)
    meridiem = (time_match.group(3) or "").lower().replace(".", "")
    if meridiem == "pm" and hour < 12:
        hour += 12
    elif meridiem == "am" and hour == 12:
        hour = 0

    try:
        day = datetime.strptime(date_arg.strip(), "%Y-%m-%d")
        start = day.replace(hour=hour, minute=minute)
    except ValueError:
        return None
    return start if start > now else None


def _digits(value: str) -> str:
    return "".join(filter(str.isdigit, str(value)))


def phones_match(stored: str, customer_phone: str) -> bool:
    """Compare two phones, tolerating formatting and a missing country prefix."""
    if stored == customer_phone:
        return True
    stored_digits, customer_digits = _digits(stored), _digits(customer_phone)
    if len(stored_digits) < 7 or len(customer_digits) < 7:
        return False
    return stored_digits == customer_digits or (
        len(stored_digits) >= 10 and customer_digits.endswith(stored_digits)
    )


def to_bogota_iso(moment: datetime) -> str:
    return moment.astimezone(BOGOTA).strftime("%Y-%m-%dT%H:%M:00-05:00")


class AppointmentService:
    """Books, moves and cancels appointments for one organization."""

    def __init__(
        self,
        db: AsyncSession,
        organization_id: uuid.UUID | str,
        calendar: GoogleCalendarService,
    ) -> None:
        self.db = db
        self.organization_id = uuid.UUID(str(organization_id))
        self.calendar = calendar

    async def list_all(self) -> list[Appointment]:
        result = await self.db.execute(
            select(Appointment).order_by(Appointment.starts_at.desc())
        )
        return list(result.scalars().all())

    async def get(self, appointment_id: uuid.UUID) -> Appointment | None:
        result = await self.db.execute(
            select(Appointment).where(Appointment.id == appointment_id)
        )
        return result.scalar_one_or_none()

    async def upcoming_for_customer(self, customer_phone: str) -> list[Appointment]:
        """Return the customer's future confirmed appointments, soonest first."""
        result = await self.db.execute(
            select(Appointment)
            .where(
                Appointment.status == "confirmed",
                Appointment.starts_at > datetime.now(BOGOTA),
            )
            .order_by(Appointment.starts_at)
        )
        return [a for a in result.scalars().all() if phones_match(a.customer_phone, customer_phone)]

    async def _calendar_conflict(
        self, start: datetime, end: datetime, moving: Appointment | None
    ) -> bool:
        """True when the calendar is busy, ignoring the appointment being moved."""
        busy = await self.calendar.busy_intervals(to_bogota_iso(start), to_bogota_iso(end))
        if moving:
            # Google trims periods to the queried window, so match by containment
            busy = [
                (busy_start, busy_end)
                for busy_start, busy_end in busy
                if not (moving.starts_at <= busy_start and busy_end <= moving.ends_at)
            ]
        return bool(busy)

    async def _calendar_is_usable(
        self, start: datetime, end: datetime, moving: Appointment | None, *, required: bool
    ) -> tuple[bool, BookingOutcome | None]:
        """Return (mirror to calendar?, outcome that stops the booking, if any)."""
        try:
            if await self._calendar_conflict(start, end, moving):
                return True, BookingOutcome.SLOT_TAKEN
            return True, None
        except Exception as exc:
            logger.warning(f"Calendar availability check failed: {exc}")
            return False, BookingOutcome.FAILED if required else None

    async def _create_calendar_event(self, appointment: Appointment) -> None:
        created = await self.calendar.create_appointment(
            customer_name=appointment.customer_name,
            customer_phone=appointment.customer_phone,
            start_time=to_bogota_iso(appointment.starts_at),
            end_time=to_bogota_iso(appointment.ends_at),
            notes=appointment.service,
        )
        if not created.get("success"):
            raise _CalendarWriteError(str(created.get("message")))
        appointment.google_event_id = created.get("event_id") or None
        appointment.html_link = created.get("html_link") or None

    async def book(
        self,
        customer_name: str,
        customer_phone: str,
        start: datetime,
        end: datetime,
        service: str,
        source: str,
        *,
        require_calendar: bool = True,
    ) -> BookingResult:
        """Reserve a slot. Both the agenda and the calendar must accept it."""
        use_calendar, stop = await self._calendar_is_usable(
            start, end, None, required=require_calendar
        )
        if stop:
            return BookingResult(stop)

        appointment = Appointment(
            organization_id=self.organization_id,
            customer_name=customer_name,
            customer_phone=customer_phone,
            service=service,
            starts_at=start,
            ends_at=end,
            status="confirmed",
            source=source,
        )
        try:
            async with self.db.begin_nested():
                self.db.add(appointment)
                await self.db.flush()
                if use_calendar:
                    await self._create_calendar_event(appointment)
        except IntegrityError as exc:
            if _OVERLAP_CONSTRAINT in str(exc.orig):
                return BookingResult(BookingOutcome.SLOT_TAKEN)
            raise
        except _CalendarWriteError as exc:
            logger.error(f"Could not create calendar event: {exc}")
            return BookingResult(BookingOutcome.FAILED)

        await self.db.commit()
        return BookingResult(BookingOutcome.BOOKED, appointment, synced=use_calendar)

    async def reschedule(
        self,
        appointment: Appointment,
        start: datetime,
        end: datetime,
        *,
        require_calendar: bool = True,
    ) -> BookingResult:
        """Move an appointment. The old calendar slot is released only after the new one exists."""
        use_calendar, stop = await self._calendar_is_usable(
            start, end, appointment, required=require_calendar
        )
        if stop:
            return BookingResult(stop)

        previous_event_id = appointment.google_event_id
        try:
            async with self.db.begin_nested():
                appointment.starts_at = start
                appointment.ends_at = end
                await self.db.flush()
                if use_calendar:
                    await self._create_calendar_event(appointment)
        except IntegrityError as exc:
            if _OVERLAP_CONSTRAINT in str(exc.orig):
                return BookingResult(BookingOutcome.SLOT_TAKEN)
            raise
        except _CalendarWriteError as exc:
            logger.error(f"Could not move calendar event: {exc}")
            return BookingResult(BookingOutcome.FAILED)

        await self.db.commit()

        if use_calendar and previous_event_id:
            removed = await self.calendar.delete_appointment(previous_event_id)
            if not removed.get("success"):
                logger.error(f"Rescheduled but old event remains {previous_event_id}: {removed}")

        return BookingResult(BookingOutcome.BOOKED, appointment, synced=use_calendar)

    async def cancel(self, appointment: Appointment, *, require_calendar: bool = True) -> bool:
        """Cancel an appointment. Returns False if it had to stay on the calendar."""
        if appointment.google_event_id:
            removed = await self.calendar.delete_appointment(appointment.google_event_id)
            if not removed.get("success"):
                logger.error(f"Could not cancel event {appointment.google_event_id}: {removed}")
                if require_calendar:
                    return False

        appointment.status = "cancelled"
        await self.db.commit()
        return True

    async def set_status(self, appointment: Appointment, status: str) -> bool:
        """Change the status. Returns False if reactivating it would overlap another one."""
        try:
            async with self.db.begin_nested():
                appointment.status = status
                await self.db.flush()
        except IntegrityError as exc:
            if _OVERLAP_CONSTRAINT in str(exc.orig):
                return False
            raise
        await self.db.commit()
        return True

    async def execute_command(
        self,
        command_type: str | None,
        cmd_args: list[str],
        customer_name: str,
        customer_phone: str,
        now: datetime,
        source: str,
        *,
        require_calendar: bool = True,
    ) -> CommandResult:
        """Run a model command before anything is promised to the customer."""
        if not command_type:
            return CommandResult()

        start = None
        if command_type in ("AGENDAR", "REAGENDAR"):
            local_start = parse_appointment_start(
                cmd_args[0] if cmd_args else "",
                cmd_args[1] if len(cmd_args) > 1 else "",
                now,
            )
            if not local_start:
                logger.info(f"Calendar command {command_type} without a valid date/time: {cmd_args}")
                return CommandResult(reply_override=NEED_DATETIME_REPLY)
            start = local_start.replace(tzinfo=BOGOTA)

        try:
            upcoming = await self.upcoming_for_customer(customer_phone)

            if command_type == "CANCELAR":
                if not upcoming:
                    return CommandResult(reply_override=NO_APPOINTMENT_REPLY)
                if not await self.cancel(upcoming[0], require_calendar=require_calendar):
                    return CommandResult(reply_override=CALENDAR_FAILED_REPLY)
                return CommandResult(changed=True)

            if any(a.starts_at == start for a in upcoming):
                return CommandResult(booked_start=start)

            end = start + DEFAULT_DURATION
            if command_type == "REAGENDAR":
                if not upcoming:
                    return CommandResult(reply_override=NO_APPOINTMENT_REPLY)
                result = await self.reschedule(
                    upcoming[0], start, end, require_calendar=require_calendar
                )
            else:
                result = await self.book(
                    customer_name,
                    customer_phone,
                    start,
                    end,
                    cmd_args[3] if len(cmd_args) > 3 else DEFAULT_SERVICE,
                    source,
                    require_calendar=require_calendar,
                )
        except Exception as exc:
            logger.error(f"Calendar command {command_type} failed: {exc}")
            return CommandResult(reply_override=CALENDAR_FAILED_REPLY)

        if result.outcome == BookingOutcome.SLOT_TAKEN:
            return CommandResult(reply_override=SLOT_TAKEN_REPLY)
        if result.outcome == BookingOutcome.FAILED:
            return CommandResult(reply_override=CALENDAR_FAILED_REPLY)
        return CommandResult(booked_start=start, changed=True, synced=result.synced)
