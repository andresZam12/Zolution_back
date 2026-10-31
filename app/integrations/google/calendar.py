"""
Google Calendar integration service with automatic OAuth2 refresh_token rotation.
"""

from datetime import UTC, datetime, timedelta

import httpx
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.integrations.models import IntegrationCredential


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

        try:
            await self.db.execute(
                text("SELECT set_config('app.current_organization_id', :org_id, true)"),
                {"org_id": self.organization_id},
            )
        except Exception:
            pass

        stmt = select(IntegrationCredential).where(
            IntegrationCredential.organization_id == self.organization_id,
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
            async with httpx.AsyncClient(timeout=10.0) as client:
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

        self._access_token = cred.access_token
        return cred.access_token

    async def get_availability(self, date_str: str) -> str:
        """
        Check the primary calendar's free/busy status for a given date (YYYY-MM-DD).
        """
        token = await self._get_access_token()

        try:
            date_obj = datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            return f"Invalid date format: {date_str}. Expected YYYY-MM-DD."

        time_min = date_obj.isoformat() + "Z"
        time_max = (date_obj + timedelta(days=1)).isoformat() + "Z"

        url = "https://www.googleapis.com/calendar/v3/freeBusy"
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        data = {"timeMin": time_min, "timeMax": time_max, "items": [{"id": "primary"}]}

        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(url, headers=headers, json=data)

            if response.status_code != 200:
                return "Failed to fetch calendar availability."

            payload = response.json()
            busy_list = payload.get("calendars", {}).get("primary", {}).get("busy", [])
            if not busy_list:
                return f"El calendario tiene disponibilidad completa el {date_str} entre las 08:00 y las 18:00."
            busy_summary = ", ".join(
                f"{b.get('start', '')} a {b.get('end', '')}" for b in busy_list
            )
            return f"Horarios ocupados el {date_str}: {busy_summary}. El resto del horario laboral está disponible."

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

        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(url, headers=headers, json=data)

            if response.status_code in (200, 201):
                event = response.json()
                return {
                    "success": True,
                    "event_id": event.get("id", ""),
                    "html_link": event.get("htmlLink", ""),
                    "message": f"Appointment successfully scheduled! Calendar Link: {event.get('htmlLink')}",
                }
            return {
                "success": False,
                "event_id": "",
                "html_link": "",
                "message": f"Failed to schedule appointment: {response.text}",
            }
