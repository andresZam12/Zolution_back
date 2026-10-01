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
    settings = get_settings()

    if not hub_mode or not hub_verify_token:
        raise HTTPException(status_code=400, detail="Missing parameters")

    if hub_mode == "subscribe" and hub_verify_token == settings.WHATSAPP_VERIFY_TOKEN:
        logger.info("WhatsApp webhook verified successfully.")
        return int(hub_challenge)
    else:
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
        raise HTTPException(status_code=404, detail="Unrecognized object")

    for entry in payload.entry:
        for change in entry.changes:
            if change.field != "messages":
                continue

            value = change.value

            # Extract incoming message data
            if not value.messages or not value.contacts:
                continue

            message = value.messages[0]
            contact = value.contacts[0]

            # We only support text messages for now
            if message.type != "text" or not message.text:
                logger.info(f"Ignored non-text message type: {message.type}")
                continue

            # Queue the processing in the background to respond 200 immediately
            background_tasks.add_task(
                process_message_background_task,
                phone_number_id=value.metadata["phone_number_id"],
                customer_phone=message.from_,
                text_content=message.text.body,
                customer_name=contact.profile.name,
            )

    return {"status": "ok"}


async def process_message_background_task(
    phone_number_id: str,
    customer_phone: str,
    text_content: str,
    customer_name: str,
):
    """
    Background task to actually process the message.
    Uses the superadmin database session because the webhook has no user token,
    so we bypass standard RLS and rely on the phone_number_id mapping to
    ensure data isolation at the application layer.
    """
    try:
        # get_superadmin_db_session yields an async generator
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
            # We must drive the generator to completion to trigger commit/rollback/close
            try:
                await anext(gen)
            except StopAsyncIteration:
                pass
    except Exception as e:
        logger.exception(f"Unhandled error in background message processing: {e}")
