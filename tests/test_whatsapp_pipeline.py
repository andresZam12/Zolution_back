"""
Regression tests for the WhatsApp pipeline.

The pipeline depends on PostgreSQL features, so tests that touch the database
run only when TEST_DATABASE_URL points at a disposable database. All its tables
are dropped and recreated for every test. LLM, Calendar and WhatsApp are fakes.

    TEST_DATABASE_URL=postgresql+asyncpg://user:pass@localhost/zolution_test pytest tests/
"""

from __future__ import annotations

import os
import uuid
from collections.abc import AsyncGenerator
from datetime import datetime, timedelta
from types import SimpleNamespace
from typing import ClassVar

import httpx
import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from app.admin import models as _admin_models  # noqa: F401
from app.agents.models import AgentConfig
from app.conversations import services
from app.conversations.models import Conversation, Message
from app.conversations.services import (
    CALENDAR_FAILED_REPLY,
    NEED_DATETIME_REPLY,
    NO_APPOINTMENT_REPLY,
    ConversationService,
    _parse_appointment_start,
)
from app.core.database import Base
from app.integrations import models as _integration_models  # noqa: F401
from app.integrations.google.calendar import CalendarUnavailableError, GoogleCalendarService
from app.tenants.models import Organization

TEST_DATABASE_URL = os.environ.get("TEST_DATABASE_URL", "")
requires_postgres = pytest.mark.skipif(not TEST_DATABASE_URL, reason="TEST_DATABASE_URL is not set")

CUSTOMER = "573000000001"
NEXT_WEEK = (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d")
FUTURE_EVENT = f"{NEXT_WEEK}T09:00:00-05:00"


class FakeCalendar:
    events: list[dict]
    created: list[dict]
    deleted: list[str]
    calls: list[str]
    lookup_error: Exception | None
    create_result: dict

    def __init__(self, db, organization_id):
        pass

    @classmethod
    def reset(cls):
        cls.events, cls.created, cls.deleted, cls.calls = [], [], [], []
        cls.lookup_error = None
        cls.create_result = {"success": True, "event_id": "new"}

    async def find_customer_events(self, customer_phone):
        if FakeCalendar.lookup_error:
            raise FakeCalendar.lookup_error
        return list(FakeCalendar.events)

    async def get_availability(self, date_str):
        return "Disponible"

    async def create_appointment(self, **kwargs):
        FakeCalendar.calls.append("create")
        FakeCalendar.created.append(kwargs)
        return FakeCalendar.create_result

    async def delete_appointment(self, event_id):
        FakeCalendar.calls.append("delete")
        FakeCalendar.deleted.append(event_id)
        return {"success": True}


class FakeLLM:
    reply = "Hola"
    histories: ClassVar[list[list[tuple[str, str]]]] = []

    async def generate(self, system_prompt, messages, temperature, max_tokens):
        FakeLLM.histories.append([(m.role, m.content) for m in messages])
        return SimpleNamespace(content=FakeLLM.reply)


class FakeFactory:
    def get_provider(self, *args, **kwargs):
        return FakeLLM()


class FakeWhatsApp:
    sent: ClassVar[list[str]] = []

    async def send_text_message(self, phone_number_id, to, text):
        FakeWhatsApp.sent.append(text)


@pytest.fixture(autouse=True)
def fakes(monkeypatch):
    FakeCalendar.reset()
    FakeLLM.reply = "Hola"
    FakeLLM.histories = []
    FakeWhatsApp.sent = []
    monkeypatch.setattr(services, "GoogleCalendarService", FakeCalendar)
    monkeypatch.setattr(services, "LLMProviderFactory", FakeFactory)
    monkeypatch.setattr(services, "WhatsAppClient", FakeWhatsApp)


@pytest_asyncio.fixture
async def sessions() -> AsyncGenerator[async_sessionmaker[AsyncSession], None]:
    engine = create_async_engine(TEST_DATABASE_URL, poolclass=NullPool)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    yield async_sessionmaker(engine, expire_on_commit=False)
    await engine.dispose()


async def _add_tenant(sessions, name: str, phone_id: str | None, status: str = "active"):
    organization_id = uuid.uuid4()
    async with sessions() as db:
        db.add(Organization(id=organization_id, name=name, slug=name.lower(), plan="starter", status="active"))
        await db.flush()
        db.add(
            AgentConfig(
                organization_id=organization_id,
                status=status,
                llm_provider="openai",
                llm_model="gpt-4o-mini",
                onboarding_answers={},
                system_prompt_generated=f"Agente de {name}",
                whatsapp_phone_number_id=phone_id,
            )
        )
        await db.commit()
    return organization_id


async def _incoming(sessions, phone_id: str, text: str, customer: str = CUSTOMER):
    async with sessions() as db:
        await ConversationService(db).process_incoming_message(phone_id, customer, text, "Ana")


async def _conversations(sessions, customer: str = CUSTOMER) -> list[Conversation]:
    async with sessions() as db:
        rows = await db.execute(select(Conversation).where(Conversation.customer_phone == customer))
        return list(rows.scalars().all())


def _event(event_id: str = "old") -> dict:
    return {"id": event_id, "start": {"dateTime": FUTURE_EVENT}}


# ---------------------------------------------------------------------------
# Tenant routing
# ---------------------------------------------------------------------------


@requires_postgres
@pytest.mark.asyncio
async def test_unknown_number_is_not_assigned_to_any_tenant(sessions):
    await _add_tenant(sessions, "Older", "111")
    await _add_tenant(sessions, "Newest", "222")

    await _incoming(sessions, "unknown", "hola")

    async with sessions() as db:
        bindings = set((await db.execute(select(AgentConfig.whatsapp_phone_number_id))).scalars())
    assert bindings == {"111", "222"}
    assert FakeWhatsApp.sent == []
    assert await _conversations(sessions) == []


@requires_postgres
@pytest.mark.asyncio
async def test_suspended_agent_stays_silent_and_suspended(sessions):
    await _add_tenant(sessions, "Paused", "111", status="suspended")

    await _incoming(sessions, "111", "hola")

    async with sessions() as db:
        status = (await db.execute(select(AgentConfig.status))).scalar_one()
    assert status == "suspended"
    assert FakeWhatsApp.sent == []


@requires_postgres
@pytest.mark.asyncio
async def test_suspended_organization_stays_silent(sessions):
    organization_id = await _add_tenant(sessions, "Unpaid", "111")
    async with sessions() as db:
        (await db.get(Organization, organization_id)).status = "suspended"
        await db.commit()

    await _incoming(sessions, "111", "hola")

    assert FakeWhatsApp.sent == []


@requires_postgres
@pytest.mark.asyncio
async def test_same_customer_keeps_separate_threads_per_tenant(sessions):
    org_a = await _add_tenant(sessions, "NegocioA", "111")
    org_b = await _add_tenant(sessions, "NegocioB", "222")

    await _incoming(sessions, "111", "secreto para A")
    await _incoming(sessions, "222", "hola B")

    assert {c.organization_id for c in await _conversations(sessions)} == {org_a, org_b}
    assert FakeLLM.histories[-1] == [("user", "hola B")]


@requires_postgres
@pytest.mark.asyncio
async def test_messages_of_a_previously_moved_thread_stay_hidden(sessions):
    org_a = await _add_tenant(sessions, "NegocioA", "111")
    org_b = await _add_tenant(sessions, "NegocioB", "222")
    conversation_id = uuid.uuid4()
    async with sessions() as db:
        db.add(Conversation(id=conversation_id, organization_id=org_b, customer_phone=CUSTOMER, status="active"))
        await db.flush()
        db.add(Message(id=uuid.uuid4(), conversation_id=conversation_id, organization_id=org_a, role="user", content="secreto para A"))
        await db.commit()

    await _incoming(sessions, "222", "hola B")

    assert FakeLLM.histories[-1] == [("user", "hola B")]


# ---------------------------------------------------------------------------
# Calendar commands
# ---------------------------------------------------------------------------


@requires_postgres
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text", ["¿mi cita del 15 sigue en pie?", "cuánto cuesta la cita para 2 personas?"]
)
async def test_questions_never_book_without_a_command(sessions, text):
    await _add_tenant(sessions, "Negocio", "111")
    FakeLLM.reply = "Claro, te cuento."

    await _incoming(sessions, "111", text)

    assert FakeCalendar.created == []
    assert FakeWhatsApp.sent == ["Claro, te cuento."]


@requires_postgres
@pytest.mark.asyncio
async def test_booking_is_confirmed_once_the_event_exists(sessions):
    await _add_tenant(sessions, "Negocio", "111")
    FakeLLM.reply = f"Listo, quedó agendada.\n[[AGENDAR|{NEXT_WEEK}|10:00]]"

    await _incoming(sessions, "111", "agéndame")

    assert FakeCalendar.created[0]["start_time"] == f"{NEXT_WEEK}T10:00:00-05:00"
    assert FakeCalendar.created[0]["end_time"] == f"{NEXT_WEEK}T11:00:00-05:00"
    assert FakeWhatsApp.sent == ["Listo, quedó agendada."]


@requires_postgres
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "setup",
    [
        lambda: setattr(FakeCalendar, "lookup_error", ValueError("not connected")),
        lambda: setattr(FakeCalendar, "create_result", {"success": False, "message": "HTTP 500"}),
    ],
)
async def test_failed_booking_is_never_confirmed(sessions, setup):
    await _add_tenant(sessions, "Negocio", "111")
    setup()
    FakeLLM.reply = f"Listo, quedó agendada.\n[[AGENDAR|{NEXT_WEEK}|10:00]]"

    await _incoming(sessions, "111", "agéndame")

    assert FakeWhatsApp.sent == [CALENDAR_FAILED_REPLY]
    async with sessions() as db:
        stored = (await db.execute(select(Message).where(Message.role == "assistant"))).scalar_one()
    assert stored.content == CALENDAR_FAILED_REPLY


@requires_postgres
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tag", ["[[AGENDAR]]", f"[[AGENDAR|{NEXT_WEEK}]]", "[[AGENDAR|2020-01-01|10:00]]", f"[[AGENDAR|{NEXT_WEEK}|25:00]]"]
)
async def test_booking_needs_a_valid_future_slot(sessions, tag):
    await _add_tenant(sessions, "Negocio", "111")
    FakeLLM.reply = f"Listo.\n{tag}"

    await _incoming(sessions, "111", "agéndame")

    assert FakeCalendar.created == []
    assert FakeWhatsApp.sent == [NEED_DATETIME_REPLY]


@requires_postgres
@pytest.mark.asyncio
async def test_reschedule_creates_the_new_slot_before_releasing_the_old_one(sessions):
    await _add_tenant(sessions, "Negocio", "111")
    FakeCalendar.events = [_event()]
    FakeLLM.reply = f"Cambiada.\n[[REAGENDAR|{NEXT_WEEK}|15:00]]"

    await _incoming(sessions, "111", "cámbiala")

    assert FakeCalendar.calls == ["create", "delete"]
    assert FakeCalendar.deleted == ["old"]


@requires_postgres
@pytest.mark.asyncio
async def test_failed_reschedule_keeps_the_original_appointment(sessions):
    await _add_tenant(sessions, "Negocio", "111")
    FakeCalendar.events = [_event()]
    FakeCalendar.create_result = {"success": False, "message": "HTTP 500"}
    FakeLLM.reply = f"Cambiada.\n[[REAGENDAR|{NEXT_WEEK}|15:00]]"

    await _incoming(sessions, "111", "cámbiala")

    assert FakeCalendar.deleted == []
    assert FakeWhatsApp.sent == [CALENDAR_FAILED_REPLY]


@requires_postgres
@pytest.mark.asyncio
async def test_cancel_without_an_appointment_says_so(sessions):
    await _add_tenant(sessions, "Negocio", "111")
    FakeLLM.reply = "Tu cita fue cancelada.\n[[CANCELAR]]"

    await _incoming(sessions, "111", "cancela mi cita")

    assert FakeCalendar.deleted == []
    assert FakeWhatsApp.sent == [NO_APPOINTMENT_REPLY]


@requires_postgres
@pytest.mark.asyncio
async def test_cancel_removes_the_soonest_appointment(sessions):
    await _add_tenant(sessions, "Negocio", "111")
    FakeCalendar.events = [_event("soonest"), _event("later")]
    FakeLLM.reply = "Tu cita fue cancelada.\n[[CANCELAR]]"

    await _incoming(sessions, "111", "cancela mi cita")

    assert FakeCalendar.deleted == ["soonest"]
    assert FakeWhatsApp.sent == ["Tu cita fue cancelada."]


# ---------------------------------------------------------------------------
# Units that need no database
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("time_arg", "expected"),
    [("14:30", (14, 30)), ("2:30 pm", (14, 30)), ("10am", (10, 0)), ("12 am", (0, 0)), ("12:15 p.m.", (12, 15))],
)
def test_parse_appointment_start_reads_times(time_arg, expected):
    start = _parse_appointment_start("2026-11-20", time_arg, datetime(2026, 11, 4))

    assert (start.hour, start.minute) == expected


@pytest.mark.parametrize(
    ("date_arg", "time_arg"),
    [("", ""), ("mañana", "10:00"), ("2026-13-40", "10:00"), ("2026-11-20", "24:00"), ("2026-11-03", "10:00")],
)
def test_parse_appointment_start_rejects_invalid_slots(date_arg, time_arg):
    assert _parse_appointment_start(date_arg, time_arg, datetime(2026, 11, 4)) is None


def _google_calendar(monkeypatch, handler) -> GoogleCalendarService:
    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        "app.integrations.google.calendar.httpx.AsyncClient",
        lambda **kwargs: real_client(transport=httpx.MockTransport(handler)),
    )
    service = GoogleCalendarService(None, str(uuid.uuid4()))
    service._access_token = "token"
    return service


@pytest.mark.asyncio
async def test_find_customer_events_matches_only_this_customers_upcoming_events(monkeypatch):
    seen: dict = {}
    items = [
        {"id": "personal", "summary": "Almuerzo con Ana", "description": "Llamar al 573000000001"},
        {"id": "other", "description": "Cliente: Ana\nTeléfono / WhatsApp: 573111111111"},
        {"id": "mine", "description": "Cliente: Ana\nTeléfono / WhatsApp: 573000000001"},
        {"id": "manual", "description": "Cliente: Ana\nTeléfono / WhatsApp: +57 300 000 0001"},
        {"id": "no-prefix", "description": "Cliente: Ana\nTeléfono / WhatsApp: 3000000001"},
        {"id": "no-description"},
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(request.url.params)
        return httpx.Response(200, json={"items": items})

    events = await _google_calendar(monkeypatch, handler).find_customer_events(CUSTOMER)

    assert [e["id"] for e in events] == ["mine", "manual", "no-prefix"]
    assert seen["timeMin"]
    assert seen["orderBy"] == "startTime"
    assert seen["singleEvents"] == "true"


@pytest.mark.asyncio
async def test_find_customer_events_reports_google_failures(monkeypatch):
    service = _google_calendar(monkeypatch, lambda request: httpx.Response(500))

    with pytest.raises(CalendarUnavailableError):
        await service.find_customer_events(CUSTOMER)
