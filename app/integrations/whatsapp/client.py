"""
WhatsApp Cloud API client.
"""

import logging
import httpx

from app.core.config import get_settings

logger = logging.getLogger(__name__)


class WhatsAppClientError(Exception):
    pass


class WhatsAppClient:
    """Client for the Meta WhatsApp Cloud API."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.base_url = "https://graph.facebook.com/v21.0"

    @property
    def token(self) -> str:
        return (
            self.settings.WHATSAPP_API_TOKEN
            or self.settings.WHATSAPP_ACCESS_TOKEN
            or ""
        ).strip()

    async def send_text_message(self, phone_number_id: str, to: str, text: str) -> dict:
        """
        Sends a simple text message to a WhatsApp user.

        Args:
            phone_number_id: The WhatsApp Business Account Phone Number ID.
            to: The recipient's phone number.
            text: The text content of the message.
        """
        token = self.token
        if not token:
            logger.error("Cannot send WhatsApp message: WHATSAPP_API_TOKEN is empty in environment.")
            raise WhatsAppClientError("WHATSAPP_API_TOKEN is not configured.")

        # Strip '+' and all non-numeric characters for Meta API
        clean_to = "".join(filter(str.isdigit, str(to)))
        if not clean_to:
            raise WhatsAppClientError(f"Invalid recipient phone number: {to}")

        clean_phone_id = str(phone_number_id).strip()
        url = f"{self.base_url}/{clean_phone_id}/messages"
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        payload = {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": clean_to,
            "type": "text",
            "text": {"preview_url": False, "body": text},
        }

        logger.info(f"Sending WhatsApp text message via phone_number_id {clean_phone_id}")
        async with httpx.AsyncClient() as client:
            response = await client.post(url, headers=headers, json=payload, timeout=12.0)

            if response.status_code not in (200, 201):
                logger.error(
                    f"Meta WhatsApp API returned HTTP {response.status_code}: {response.text}"
                )
                raise WhatsAppClientError(f"Failed to send message: {response.text}")

            res_json = response.json()
            logger.info(f"WhatsApp message sent successfully: {res_json.get('messages', [])}")
            return res_json
