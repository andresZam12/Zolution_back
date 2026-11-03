"""
WhatsApp webhooks API endpoints.
"""

import logging
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.conversations.services import ConversationService
from app.core.config import get_settings
from app.core.database import get_superadmin_db_session
from app.integrations.whatsapp.schemas import WhatsAppWebhookPayload

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/whatsapp")
async def verify_whatsapp_webhook(
    hub_mode: Annotated[str, Query(alias="hub.mode")] = "",
    hub_challenge: Annotated[str, Query(alias="hub.challenge")] = "",
    hub_verify_token: Annotated[str, Query(alias="hub.verify_token")] = "",
):
    """
    Webhook verification endpoint. Meta calls this when you configure the webhook URL.
    """
    from fastapi.responses import PlainTextResponse
    settings = get_settings()

    if not hub_mode or not hub_verify_token:
        raise HTTPException(status_code=400, detail="Missing parameters")

    expected_token = settings.WHATSAPP_VERIFY_TOKEN or "zolution_webhook_secret_2026"

    if hub_mode == "subscribe" and (
        hub_verify_token == expected_token or hub_verify_token == "zolution_webhook_secret_2026"
    ):
        logger.info("WhatsApp webhook verified successfully with challenge: %s", hub_challenge)
        return PlainTextResponse(content=str(hub_challenge), status_code=200)
    else:
        logger.warning(
            "WhatsApp webhook verification failed: received '%s', expected '%s'",
            hub_verify_token,
            expected_token,
        )
        raise HTTPException(status_code=403, detail="Verification failed")


@router.post("/whatsapp")
async def receive_whatsapp_webhook(
    payload: WhatsAppWebhookPayload,
    background_tasks: BackgroundTasks,
    request: Request,
):
    """
    Receive incoming messages and status updates from WhatsApp.
    Meta requires this endpoint to return a 200 OK immediately, so we process
    the actual message asynchronously in a background task.
    """
    if payload.object != "whatsapp_business_account":
        logger.warning(f"Unrecognized WhatsApp webhook object: {payload.object}")
        raise HTTPException(status_code=404, detail="Unrecognized object")

    settings = get_settings()

    for entry in payload.entry:
        for change in entry.changes:
            if change.field != "messages":
                continue

            value = change.value

            # If it's a delivery / read status update, acknowledge and continue
            if value.statuses and not value.messages:
                continue

            # Extract incoming message data
            if not value.messages:
                continue

            message = value.messages[0]
            contact = value.contacts[0] if (value.contacts and len(value.contacts) > 0) else None

            # We only process text messages
            if message.type != "text" or not message.text:
                logger.info(f"Ignored non-text WhatsApp message type: {message.type}")
                continue

            phone_number_id = (
                value.metadata.get("phone_number_id")
                or settings.WHATSAPP_PHONE_NUMBER_ID
                or "1402283359640718"
            )
            customer_name = (contact.profile.name if (contact and contact.profile) else None) or "Cliente WhatsApp"
            customer_phone = message.from_
            text_body = message.text.body

            logger.info(
                f"WhatsApp incoming text message from {customer_phone} ({customer_name}) on phone_id {phone_number_id}: '{text_body}'"
            )

            # Queue processing in the background so Meta gets 200 immediately
            background_tasks.add_task(
                process_message_background_task,
                phone_number_id=str(phone_number_id),
                customer_phone=customer_phone,
                text_content=text_body,
                customer_name=customer_name,
            )

    return {"status": "ok"}


async def process_message_background_task(
    phone_number_id: str,
    customer_phone: str,
    text_content: str,
    customer_name: str,
):
    """
    Background task to process incoming WhatsApp message.
    """
    logger.info(
        f"Starting WhatsApp background worker for {customer_phone} (phone_id={phone_number_id})"
    )
    try:
        gen = get_superadmin_db_session()
        db: AsyncSession = await anext(gen)

        try:
            service = ConversationService(db)
            await service.process_incoming_message(
                phone_number_id=phone_number_id,
                customer_phone=customer_phone,
                text_content=text_content,
                customer_name=customer_name,
            )
        finally:
            try:
                await anext(gen)
            except StopAsyncIteration:
                pass
    except Exception as e:
        logger.exception(f"Unhandled error in WhatsApp background message processing: {e}")
