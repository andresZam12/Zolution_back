"""
Google Calendar integration service with automatic OAuth2 refresh_token rotation.
"""

import uuid
from datetime import UTC, datetime, timedelta

import httpx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.database import bind_tenant
from app.core.time_format import format_time_12h
from app.integrations.models import IntegrationCredential

CALENDAR_NOT_CONNECTED = "Calendario no conectado aún."


class CalendarUnavailableError(Exception):
    """Google Calendar could not be reached or rejected the request."""


class GoogleCalendarService:
    """Service to interact with Google Calendar API for a specific tenant."""

    def __init__(self, db: AsyncSession, organization_id: str):
        self.db = db
        self.organization_id = str(organization_id)
        self._access_token: str | None = None

    async def _get_access_token(self) -> str:
        """
        Fetch the current access token for the organization, automatically
        refreshing it via `refresh_token` if expired or close to expiring.
        """
        if self._access_token:
            return self._access_token

        org_u = uuid.UUID(self.organization_id)
        await bind_tenant(self.db, org_u)

        stmt = select(IntegrationCredential).where(
            IntegrationCredential.organization_id == org_u,
            IntegrationCredential.provider == "google",
        )
        result = await self.db.execute(stmt)
        cred = result.scalar_one_or_none()

        if not cred:
            raise ValueError("Google Calendar no está conectado para esta organización.")

        now = datetime.now(UTC)
        is_expired = (
            cred.expires_at is not None
            and cred.expires_at <= (now + timedelta(minutes=2))
        )

        if is_expired and cred.refresh_token:
            settings = get_settings()
            try:
                async with httpx.AsyncClient(timeout=4.0) as client:
                    resp = await client.post(
                        "https://oauth2.googleapis.com/token",
                        data={
                            "client_id": settings.GOOGLE_CLIENT_ID,
                            "client_secret": settings.GOOGLE_CLIENT_SECRET,
                            "refresh_token": cred.refresh_token,
                            "grant_type": "refresh_token",
                        },
                    )
                    if resp.status_code == 200:
                        payload = resp.json()
                        cred.access_token = payload["access_token"]
                        expires_in = payload.get("expires_in", 3599)
                        cred.expires_at = datetime.now(UTC) + timedelta(seconds=expires_in)
                        self.db.add(cred)
                        await self.db.flush()
            except Exception:
                pass

        self._access_token = cred.access_token
        return cred.access_token

    async def get_availability(self, date_str: str) -> str:
        """
        Check the primary calendar's free/busy status for a given date (YYYY-MM-DD).
        """
        try:
            token = await self._get_access_token()
        except Exception:
            return CALENDAR_NOT_CONNECTED

        try:
            date_obj = datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            return f"Invalid date format: {date_str}. Expected YYYY-MM-DD."

        time_min = f"{date_str}T00:00:00-05:00"
        time_max = f"{(date_obj + timedelta(days=1)):%Y-%m-%d}T00:00:00-05:00"

        url = "https://www.googleapis.com/calendar/v3/freeBusy"
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        data = {
            "timeMin": time_min,
            "timeMax": time_max,
            "timeZone": "America/Bogota",
            "items": [{"id": "primary"}],
        }

        try:
            async with httpx.AsyncClient(timeout=4.0) as client:
                response = await client.post(url, headers=headers, json=data)

                if response.status_code != 200:
                    return "Consulta de agenda no disponible temporalmente."

                payload = response.json()
                busy_list = payload.get("calendars", {}).get("primary", {}).get("busy", [])
                if not busy_list:
                    return f"El calendario tiene disponibilidad completa el {date_str} entre las 8:00 AM y las 6:00 PM."
                busy_summary = ", ".join(
                    f"{format_time_12h(datetime.fromisoformat(b['start']))} a "
                    f"{format_time_12h(datetime.fromisoformat(b['end']))}"
                    for b in busy_list
                )
                return f"Horarios ocupados el {date_str}: {busy_summary}. El resto del horario laboral está disponible."
        except Exception:
            return "Consulta de agenda no disponible temporalmente."

    async def create_appointment(
        self, customer_name: str, customer_phone: str, start_time: str, end_time: str, notes: str = ""
    ) -> dict:
        """
        Create a new event in the primary Google Calendar and return event metadata.
        start_time and end_time should be ISO8601 strings (e.g. 2026-10-02T10:00:00-05:00).
        """
        token = await self._get_access_token()

        url = "https://www.googleapis.com/calendar/v3/calendars/primary/events"
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        description_lines = [f"Cliente: {customer_name}", f"Teléfono / WhatsApp: {customer_phone}"]
        if notes:
            description_lines.append(f"Servicio / Notas: {notes}")

        data = {
            "summary": f"Cita: {customer_name}" + (f" - {notes}" if notes else ""),
            "description": "\n".join(description_lines),
            "start": {
                "dateTime": start_time,
                "timeZone": "America/Bogota",
            },
            "end": {
                "dateTime": end_time,
                "timeZone": "America/Bogota",
            },
        }

        import logging
        logger = logging.getLogger(__name__)

        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(url, headers=headers, json=data)

            if response.status_code in (200, 201):
                event = response.json()
                logger.info("Google Calendar created event successfully: %s", event.get("id"))
                return {
                    "success": True,
                    "event_id": event.get("id", ""),
                    "html_link": event.get("htmlLink", ""),
                    "message": f"Appointment successfully scheduled! Calendar Link: {event.get('htmlLink')}",
                }
            logger.error("Google Calendar failed [HTTP %d]: %s", response.status_code, response.text)
            return {
                "success": False,
                "event_id": "",
                "html_link": "",
                "message": f"Google Calendar HTTP {response.status_code}: {response.text}",
            }

    async def delete_appointment(self, event_id: str) -> dict:
        """
        Delete an event from the primary Google Calendar by event_id.
        """
        if not event_id or event_id.startswith("local_") or event_id.startswith("pg_"):
            return {"success": True, "message": "No external Google event ID to delete"}

        try:
            token = await self._get_access_token()
        except Exception as e:
            return {"success": False, "message": f"Could not obtain Google token: {e}"}

        url = f"https://www.googleapis.com/calendar/v3/calendars/primary/events/{event_id}"
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

        import logging
        logger = logging.getLogger(__name__)

        try:
            async with httpx.AsyncClient(timeout=8.0) as client:
                response = await client.delete(url, headers=headers)
                if response.status_code in (200, 204, 404, 410):
                    logger.info("Google Calendar event deleted successfully: %s", event_id)
                    return {"success": True, "message": "Event deleted from Google Calendar"}
                logger.error("Failed to delete Google Calendar event [HTTP %d]: %s", response.status_code, response.text)
                return {"success": False, "message": f"Google Calendar HTTP {response.status_code}: {response.text}"}
        except Exception as exc:
            logger.warning("Error deleting event from Google Calendar: %s", exc)
            return {"success": False, "message": str(exc)}

    async def busy_intervals(self, start_iso: str, end_iso: str) -> list[tuple[datetime, datetime]]:
        """
        Return the busy periods of the primary calendar within a time window.

        Raises:
            ValueError: Google Calendar is not connected.
            CalendarUnavailableError: Google did not answer successfully.
        """
        token = await self._get_access_token()
        async with httpx.AsyncClient(timeout=6.0) as client:
            response = await client.post(
                "https://www.googleapis.com/calendar/v3/freeBusy",
                headers={"Authorization": f"Bearer {token}"},
                json={"timeMin": start_iso, "timeMax": end_iso, "items": [{"id": "primary"}]},
            )
        if response.status_code != 200:
            raise CalendarUnavailableError(f"Google Calendar HTTP {response.status_code}")

        busy = response.json().get("calendars", {}).get("primary", {}).get("busy", [])
        return [
            (datetime.fromisoformat(period["start"]), datetime.fromisoformat(period["end"]))
            for period in busy
        ]
