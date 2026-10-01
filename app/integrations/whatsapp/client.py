"""
WhatsApp Cloud API client.
"""

import httpx

from app.core.config import get_settings


class WhatsAppClientError(Exception):
    pass


class WhatsAppClient:
    """Client for the Meta WhatsApp Cloud API."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.base_url = "https://graph.facebook.com/v19.0"
        self.token = self.settings.WHATSAPP_API_TOKEN

    async def send_text_message(self, phone_number_id: str, to: str, text: str) -> dict:
        """
        Sends a simple text message to a WhatsApp user.

        Args:
            phone_number_id: The WhatsApp Business Account Phone Number ID.
            to: The recipient's phone number.
            text: The text content of the message.
        """
        if not self.token:
            raise WhatsAppClientError("WHATSAPP_API_TOKEN is not configured.")

        url = f"{self.base_url}/{phone_number_id}/messages"
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }
        payload = {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": to,
            "type": "text",
            "text": {"preview_url": False, "body": text},
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(url, headers=headers, json=payload, timeout=10.0)

            if response.status_code not in (200, 201):
                raise WhatsAppClientError(f"Failed to send message: {response.text}")

            return response.json()
