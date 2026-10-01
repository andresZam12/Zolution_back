import pytest
from httpx import AsyncClient

from app.core.config import get_settings
from app.main import app


@pytest.mark.asyncio
async def test_verify_webhook_success():
    """Test the Meta GET webhook verification challenge."""
    settings = get_settings()
    settings.WHATSAPP_VERIFY_TOKEN = "test-verify-token"

    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.get(
            "/api/v1/webhooks/whatsapp",
            params={
                "hub.mode": "subscribe",
                "hub.verify_token": "test-verify-token",
                "hub.challenge": "11223344",
            },
        )

    assert response.status_code == 200
    assert response.json() == 11223344


@pytest.mark.asyncio
async def test_verify_webhook_forbidden():
    """Test verification with invalid token."""
    settings = get_settings()
    settings.WHATSAPP_VERIFY_TOKEN = "test-verify-token"

    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.get(
            "/api/v1/webhooks/whatsapp",
            params={
                "hub.mode": "subscribe",
                "hub.verify_token": "wrong-token",
                "hub.challenge": "11223344",
            },
        )

    assert response.status_code == 403


@pytest.mark.asyncio
async def test_receive_webhook_text_message():
    """Test receiving a valid text message payload."""
    payload = {
        "object": "whatsapp_business_account",
        "entry": [
            {
                "id": "12345",
                "changes": [
                    {
                        "field": "messages",
                        "value": {
                            "messaging_product": "whatsapp",
                            "metadata": {
                                "display_phone_number": "123456789",
                                "phone_number_id": "phone-id-123",
                            },
                            "contacts": [{"profile": {"name": "John Doe"}, "wa_id": "19876543210"}],
                            "messages": [
                                {
                                    "from": "19876543210",
                                    "id": "msg-id-123",
                                    "timestamp": "1609459200",
                                    "type": "text",
                                    "text": {"body": "Hello Zolution"},
                                }
                            ],
                        },
                    }
                ],
            }
        ],
    }

    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post("/api/v1/webhooks/whatsapp", json=payload)

    # The endpoint should return 200 immediately
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
