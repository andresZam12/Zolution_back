"""
WhatsApp webhooks API endpoints.
"""

import hashlib
import hmac
import logging
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, Request
from pydantic import ValidationError

from app.auth.dependencies import require_superadmin
from app.auth.models import UserContext
from app.conversations.services import ConversationService
from app.core.config import get_settings
from app.integrations.whatsapp.schemas import WhatsAppWebhookPayload

router = APIRouter()
logger = logging.getLogger(__name__)

# In-memory deduplication set to avoid processing duplicate webhook deliveries from Meta
_PROCESSED_MESSAGE_IDS: set[str] = set()


def _has_valid_signature(raw_body: bytes, signature_header: str, app_secret: str) -> bool:
    """Check Meta's X-Hub-Signature-256 header against the exact request bytes."""
    expected = "sha256=" + hmac.new(app_secret.encode(), raw_body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected.encode(), signature_header.encode())


@router.post("/whatsapp/subscribe")
async def subscribe_whatsapp_waba(_user: UserContext = Depends(require_superadmin)):
    """
    Calls Meta Graph API to subscribe the WABA to this app's webhook.
    This registers the app so Meta forwards incoming WhatsApp messages to our callback URL.
    """
    import httpx
    settings = get_settings()
    token = (settings.WHATSAPP_API_TOKEN or settings.WHATSAPP_ACCESS_TOKEN or "").strip()
    waba_id = settings.WHATSAPP_WABA_ID.strip()
    if not token or not waba_id:
        raise HTTPException(
            status_code=503,
            detail="WHATSAPP_API_TOKEN and WHATSAPP_WABA_ID must be configured.",
        )

    results: dict = {"waba_id": waba_id}

    async with httpx.AsyncClient() as client:
        # 1. Subscribe WABA
        try:
            r1 = await client.post(
                f"https://graph.facebook.com/v21.0/{waba_id}/subscribed_apps",
                headers={"Authorization": f"Bearer {token}"},
                timeout=12.0,
            )
            results["waba_subscription"] = {
                "status_code": r1.status_code,
                "data": r1.json() if "application/json" in r1.headers.get("content-type", "") else r1.text,
            }
        except Exception as e:
            results["waba_subscription_error"] = str(e)

        # 2. Check current subscriptions
        try:
            r2 = await client.get(
                f"https://graph.facebook.com/v21.0/{waba_id}/subscribed_apps",
                headers={"Authorization": f"Bearer {token}"},
                timeout=12.0,
            )
            results["current_subscribed_apps"] = {
                "status_code": r2.status_code,
                "data": r2.json() if "application/json" in r2.headers.get("content-type", "") else r2.text,
            }
        except Exception as e:
            results["check_subscription_error"] = str(e)

    logger.info("Subscribed WABA results: %s", results)
    return results


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

    expected_token = settings.WHATSAPP_VERIFY_TOKEN
    token_matches = bool(expected_token) and hmac.compare_digest(
        hub_verify_token.encode(), expected_token.encode()
    )

    if hub_mode == "subscribe" and token_matches:
        logger.info("WhatsApp webhook verified successfully")
        return PlainTextResponse(content=str(hub_challenge), status_code=200)

    logger.warning("WhatsApp webhook verification failed")
    raise HTTPException(status_code=403, detail="Verification failed")


@router.post("/whatsapp")
async def receive_whatsapp_webhook(
    background_tasks: BackgroundTasks,
    request: Request,
):
    """
    Receive incoming messages and status updates from WhatsApp.
    Meta requires this endpoint to return a 200 OK immediately, so we process
    the actual message asynchronously in a background task.
    """
    settings = get_settings()
    if not settings.WHATSAPP_APP_SECRET:
        logger.error("Rejected WhatsApp webhook: WHATSAPP_APP_SECRET is not configured")
        raise HTTPException(status_code=503, detail="Webhook signature validation is not configured")

    raw_body = await request.body()
    signature = request.headers.get("X-Hub-Signature-256", "")
    if not _has_valid_signature(raw_body, signature, settings.WHATSAPP_APP_SECRET):
        logger.warning("Rejected WhatsApp webhook with a missing or invalid signature")
        raise HTTPException(status_code=403, detail="Invalid signature")

    try:
        payload = WhatsAppWebhookPayload.model_validate_json(raw_body)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail="Invalid webhook payload") from exc

    if payload.object != "whatsapp_business_account":
        logger.warning(f"Unrecognized WhatsApp webhook object: {payload.object}")
        raise HTTPException(status_code=404, detail="Unrecognized object")

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
            msg_id = getattr(message, "id", None)
            if msg_id and msg_id in _PROCESSED_MESSAGE_IDS:
                logger.info(f"Ignored duplicate incoming WhatsApp message: {msg_id}")
                continue
            if msg_id:
                _PROCESSED_MESSAGE_IDS.add(msg_id)
                if len(_PROCESSED_MESSAGE_IDS) > 2000:
                    _PROCESSED_MESSAGE_IDS.clear()

            contact = value.contacts[0] if (value.contacts and len(value.contacts) > 0) else None

            # We only process text messages
            if message.type != "text" or not message.text:
                logger.info(f"Ignored non-text WhatsApp message type: {message.type}")
                continue

            phone_number_id = value.metadata.get("phone_number_id")
            if not phone_number_id:
                logger.warning("Ignored WhatsApp message without phone_number_id metadata")
                continue

            customer_name = (contact.profile.name if (contact and contact.profile) else None) or "Cliente WhatsApp"
            customer_phone = message.from_
            text_body = message.text.body

            logger.info(f"WhatsApp text message {msg_id} received on phone_id {phone_number_id}")

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
    logger.info(f"Starting WhatsApp background worker for phone_id {phone_number_id}")
    try:
        from app.core.database import AsyncSessionFactory

        async with AsyncSessionFactory() as db:
            service = ConversationService(db)
            await service.process_incoming_message(
                phone_number_id=phone_number_id,
                customer_phone=customer_phone,
                text_content=text_content,
                customer_name=customer_name,
            )
    except Exception as e:
        logger.exception(f"Unhandled error in WhatsApp background message processing: {e}")
