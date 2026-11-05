import hashlib
import hmac
import json

import pytest
from httpx import ASGITransport, AsyncClient

from app.api.v1.endpoints import webhooks
from app.core.config import Settings, get_settings
from app.main import app, create_app
from scripts import seed_data

APP_SECRET = "test-app-secret"
WEBHOOK_URL = "/api/v1/webhooks/whatsapp"


def _client() -> AsyncClient:
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


def _sign(body: bytes, secret: str = APP_SECRET) -> str:
    return "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


def _message_body(message_id: str, messages: list[dict] | None = None) -> bytes:
    return json.dumps(
        {
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
                                "contacts": [
                                    {"profile": {"name": "John Doe"}, "wa_id": "19876543210"}
                                ],
                                "messages": messages
                                or [
                                    {
                                        "from": "19876543210",
                                        "id": message_id,
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
    ).encode()


@pytest.fixture
def whatsapp_settings(monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings, "WHATSAPP_VERIFY_TOKEN", "test-verify-token")
    monkeypatch.setattr(settings, "WHATSAPP_APP_SECRET", APP_SECRET)
    return settings


@pytest.fixture
def queued_messages(monkeypatch):
    calls: list[dict] = []

    async def record(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(webhooks, "process_message_background_task", record)
    return calls


@pytest.mark.asyncio
async def test_verify_webhook_success(whatsapp_settings):
    async with _client() as client:
        response = await client.get(
            WEBHOOK_URL,
            params={
                "hub.mode": "subscribe",
                "hub.verify_token": "test-verify-token",
                "hub.challenge": "11223344",
            },
        )

    assert response.status_code == 200
    assert response.text == "11223344"


@pytest.mark.asyncio
@pytest.mark.parametrize("token", ["wrong-token", "zolution_webhook_secret_2026"])
async def test_verify_webhook_rejects_other_tokens(whatsapp_settings, token):
    async with _client() as client:
        response = await client.get(
            WEBHOOK_URL,
            params={"hub.mode": "subscribe", "hub.verify_token": token, "hub.challenge": "1"},
        )

    assert response.status_code == 403


@pytest.mark.asyncio
async def test_verify_webhook_fails_closed_without_configured_token(whatsapp_settings, monkeypatch):
    monkeypatch.setattr(whatsapp_settings, "WHATSAPP_VERIFY_TOKEN", "")

    async with _client() as client:
        response = await client.get(
            WEBHOOK_URL,
            params={"hub.mode": "subscribe", "hub.verify_token": "anything", "hub.challenge": "1"},
        )

    assert response.status_code == 403


@pytest.mark.asyncio
async def test_signed_message_is_queued(whatsapp_settings, queued_messages):
    body = _message_body("signed-1")

    async with _client() as client:
        response = await client.post(
            WEBHOOK_URL, content=body, headers={"X-Hub-Signature-256": _sign(body)}
        )

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert queued_messages == [
        {
            "phone_number_id": "phone-id-123",
            "customer_phone": "19876543210",
            "text_content": "Hello Zolution",
            "customer_name": "John Doe",
            "whatsapp_message_id": "signed-1",
        }
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "headers",
    [
        {},
        {"X-Hub-Signature-256": "sha256=deadbeef"},
        {"X-Hub-Signature-256": "not-a-signature"},
    ],
)
async def test_unsigned_or_forged_message_is_rejected(whatsapp_settings, queued_messages, headers):
    async with _client() as client:
        response = await client.post(WEBHOOK_URL, content=_message_body("forged"), headers=headers)

    assert response.status_code == 403
    assert queued_messages == []


@pytest.mark.asyncio
async def test_signature_from_another_secret_is_rejected(whatsapp_settings, queued_messages):
    body = _message_body("other-secret")

    async with _client() as client:
        response = await client.post(
            WEBHOOK_URL, content=body, headers={"X-Hub-Signature-256": _sign(body, "other")}
        )

    assert response.status_code == 403
    assert queued_messages == []


@pytest.mark.asyncio
async def test_webhook_fails_closed_without_app_secret(
    whatsapp_settings, queued_messages, monkeypatch
):
    monkeypatch.setattr(whatsapp_settings, "WHATSAPP_APP_SECRET", "")
    body = _message_body("no-secret")

    async with _client() as client:
        response = await client.post(
            WEBHOOK_URL, content=body, headers={"X-Hub-Signature-256": _sign(body, "")}
        )

    assert response.status_code == 503
    assert queued_messages == []


@pytest.mark.asyncio
async def test_subscribe_requires_authentication():
    async with _client() as client:
        anonymous = await client.post(f"{WEBHOOK_URL}/subscribe")
        via_get = await client.get(f"{WEBHOOK_URL}/subscribe")

    assert anonymous.status_code in (401, 403)
    assert via_get.status_code == 405


def test_production_rejects_default_secret_key():
    with pytest.raises(ValueError, match="SECRET_KEY"):
        Settings(_env_file=None, APP_ENV="production")


def test_production_accepts_explicit_secret_key():
    settings = Settings(_env_file=None, APP_ENV="production", SECRET_KEY="k" * 44)

    assert settings.APP_ENV == "production"


@pytest.mark.parametrize(("environment", "status"), [("production", 404), ("development", 200)])
async def test_api_schema_is_public_only_outside_production(monkeypatch, environment, status):
    monkeypatch.setattr(get_settings(), "APP_ENV", environment)

    transport = ASGITransport(app=create_app())
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/openapi.json")

    assert response.status_code == status


async def test_message_content_and_phones_are_not_logged(whatsapp_settings, queued_messages, caplog):
    body = _message_body("logged-1")

    with caplog.at_level("INFO"):
        async with _client() as client:
            await client.post(WEBHOOK_URL, content=body, headers={"X-Hub-Signature-256": _sign(body)})

    assert "Hello Zolution" not in caplog.text
    assert "19876543210" not in caplog.text


async def test_demo_data_is_never_seeded_in_production(monkeypatch):
    monkeypatch.setattr(get_settings(), "APP_ENV", "production")

    def must_not_open_a_session():
        raise AssertionError("seed touched the database")

    monkeypatch.setattr(seed_data, "AsyncSessionFactory", must_not_open_a_session)

    await seed_data.seed()


async def test_every_message_in_a_batch_is_processed(whatsapp_settings, queued_messages):
    batch = [
        {"from": "1", "id": f"batch-{i}", "timestamp": "1", "type": "text", "text": {"body": f"m{i}"}}
        for i in range(3)
    ]
    body = _message_body("", batch)

    async with _client() as client:
        await client.post(WEBHOOK_URL, content=body, headers={"X-Hub-Signature-256": _sign(body)})

    assert [call["text_content"] for call in queued_messages] == ["m0", "m1", "m2"]


@pytest.mark.parametrize(
    ("message_type", "expected"),
    [("audio", 1), ("image", 1), ("location", 1), ("reaction", 0)],
)
async def test_unsupported_messages_get_a_reply_only_when_useful(
    whatsapp_settings, queued_messages, monkeypatch, message_type, expected
):
    unsupported: list[dict] = []

    async def record(**kwargs):
        unsupported.append(kwargs)

    monkeypatch.setattr(webhooks, "process_unsupported_background_task", record)
    body = _message_body("", [{"from": "1", "id": "m", "timestamp": "1", "type": message_type}])

    async with _client() as client:
        response = await client.post(
            WEBHOOK_URL, content=body, headers={"X-Hub-Signature-256": _sign(body)}
        )

    assert response.status_code == 200
    assert len(unsupported) == expected
    assert queued_messages == []
