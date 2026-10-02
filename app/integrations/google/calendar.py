"""
Google Calendar integration service.
"""

import httpx
from datetime import datetime, timedelta

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.integrations.models import IntegrationCredential


class GoogleCalendarService:
    """Service to interact with Google Calendar API."""

    def __init__(self, db: AsyncSession, organization_id: str):
        self.db = db
        self.organization_id = organization_id
        self._access_token: str | None = None

    async def _get_access_token(self) -> str:
        """
        Fetch the current access token for the organization.
        TODO: Implement token refresh logic if expires_at is passed.
        """
        if self._access_token:
            return self._access_token

        stmt = select(IntegrationCredential).where(
            IntegrationCredential.organization_id == self.organization_id,
            IntegrationCredential.provider == "google",
        )
        result = await self.db.execute(stmt)
        cred = result.scalar_one_or_none()
        
        if not cred:
            raise ValueError("Google integration not connected.")

        # In a complete implementation, check cred.expires_at here and use
        # cred.refresh_token to get a new access_token if expired.
        # For this MVP, we assume the token is valid.
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
        data = {
            "timeMin": time_min,
            "timeMax": time_max,
            "items": [{"id": "primary"}]
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(url, headers=headers, json=data)
            
            if response.status_code != 200:
                return "Failed to fetch calendar availability."

            # Return a simple text summary for the LLM
            # In a real app, you'd parse the busy slots and map them to available slots
            return f"Raw calendar response for {date_str}: {response.text}"

    async def create_appointment(
        self, customer_name: str, customer_phone: str, start_time: str, end_time: str
    ) -> str:
        """
        Create a new event in the primary calendar.
        start_time and end_time should be ISO8601 strings (e.g. 2026-10-02T10:00:00Z).
        """
        token = await self._get_access_token()
        
        url = "https://www.googleapis.com/calendar/v3/calendars/primary/events"
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        data = {
            "summary": f"Appointment: {customer_name}",
            "description": f"Customer phone: {customer_phone}",
            "start": {
                "dateTime": start_time,
            },
            "end": {
                "dateTime": end_time,
            },
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(url, headers=headers, json=data)
            
            if response.status_code == 200:
                event = response.json()
                return f"Appointment successfully scheduled! Calendar Link: {event.get('htmlLink')}"
            else:
                return f"Failed to schedule appointment: {response.text}"
