"""
Regression tests for the WhatsApp pipeline.

Tests that touch the database need TEST_DATABASE_URL (see conftest.py). They run
on the real migrations, with RLS and the overlap constraint active. LLM, Calendar
and WhatsApp are fakes.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime, timedelta
from types import SimpleNamespace
from typing import ClassVar

import httpx
import pytest
from sqlalchemy import select

from app.admin import models as _admin_models  # noqa: F401
from app.agents.models import AgentConfig
from app.appointments.models import Appointment
from app.appointments.services import (
    BOGOTA,
    CALENDAR_FAILED_REPLY,
    NEED_DATETIME_REPLY,
    NO_APPOINTMENT_REPLY,
    SLOT_TAKEN_REPLY,
    AppointmentService,
    BookingOutcome,
    parse_appointment_start,
    parse_command,
    phones_match,
    strip_command_tags,
)
from app.conversations import services
from app.conversations.agent import MODEL_UNAVAILABLE_REPLY, NO_CATALOG_TEXT
from app.conversations.models import Conversation, Message
from app.conversations.services import UNSUPPORTED_MESSAGE_REPLY, ConversationService
from app.core.database import bind_tenant
from app.integrations import models as _integration_models  # noqa: F401
from app.integrations.google.calendar import CalendarUnavailableError, GoogleCalendarService
from app.tenants.models import Organization
from tests.conftest import requires_postgres

CUSTOMER = "573000000001"
NEXT_WEEK = (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d")
OTHER_CUSTOMER = "573111111111"


def _at(hour: int, minute: int = 0) -> datetime:
    return datetime.fromisoformat(f"{NEXT_WEEK}T{hour:02d}:{minute:02d}:00-05:00")


class FakeCalendar:
    busy: list[tuple[datetime, datetime]]
    created: list[dict]
    deleted: list[str]
    calls: list[str]
    busy_error: Exception | None
    create_result: dict | None
    delete_result: dict

    def __init__(self, db, organization_id):
        pass

    @classmethod
    def reset(cls):
        cls.busy, cls.created, cls.deleted, cls.calls = [], [], [], []
        cls.busy_error = None
        cls.create_result = None
        cls.delete_result = {"success": True}

    async def busy_intervals(self, start_iso, end_iso):
        if FakeCalendar.busy_error:
            raise FakeCalendar.busy_error
        return list(FakeCalendar.busy)

    async def get_availability(self, date_str):
        return "Disponible"

    async def create_appointment(self, **kwargs):
        FakeCalendar.calls.append("create")
        FakeCalendar.created.append(kwargs)
        return FakeCalendar.create_result or {
            "success": True,
            "event_id": f"event-{len(FakeCalendar.created)}",
        }

    async def delete_appointment(self, event_id):
        FakeCalendar.calls.append("delete")
        FakeCalendar.deleted.append(event_id)
        return FakeCalendar.delete_result


class FakeLLM:
    reply = "Hola"
    histories: ClassVar[list[list[tuple[str, str]]]] = []
    prompts: ClassVar[list[str]] = []

    async def generate(self, system_prompt, messages, temperature, max_tokens):
        FakeLLM.prompts.append(system_prompt)
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
    FakeLLM.prompts = []
    FakeWhatsApp.sent = []
    monkeypatch.setattr(services, "GoogleCalendarService", FakeCalendar)
    monkeypatch.setattr(services, "LLMProviderFactory", FakeFactory)
    monkeypatch.setattr(services, "WhatsAppClient", FakeWhatsApp)


async def _add_tenant(sessions, name: str, phone_id: str | None, status: str = "active"):
    organization_id = uuid.uuid4()
    async with sessions() as db:
        db.add(
            Organization(
                id=organization_id, name=name, slug=name.lower(), plan="starter", status="active"
            )
        )
        await db.flush()
        await bind_tenant(db, organization_id)
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


async def _conversations(sessions, organization_id: uuid.UUID) -> list[Conversation]:
    async with sessions() as db:
        await bind_tenant(db, organization_id)
        rows = await db.execute(select(Conversation).where(Conversation.customer_phone == CUSTOMER))
        return list(rows.scalars().all())


async def _agent_config(sessions, organization_id: uuid.UUID) -> AgentConfig:
    async with sessions() as db:
        await bind_tenant(db, organization_id)
        return (await db.execute(select(AgentConfig))).scalar_one()


async def _book(
    sessions,
    organization_id: uuid.UUID,
    phone: str,
    hour: int,
    event_id: str | None = None,
    status: str = "confirmed",
) -> None:
    async with sessions() as db:
        await bind_tenant(db, organization_id)
        db.add(
            Appointment(
                organization_id=organization_id,
                customer_name="Cliente",
                customer_phone=phone,
                service="Consulta",
                starts_at=_at(hour),
                ends_at=_at(hour + 1),
                status=status,
                source="dashboard",
                google_event_id=event_id,
            )
        )
        await db.commit()


async def _appointments(sessions, organization_id: uuid.UUID) -> list[Appointment]:
    async with sessions() as db:
        await bind_tenant(db, organization_id)
        return list((await db.execute(select(Appointment))).scalars().all())


# ---------------------------------------------------------------------------
# Tenant routing
# ---------------------------------------------------------------------------


@requires_postgres
@pytest.mark.asyncio
async def test_unknown_number_is_not_assigned_to_any_tenant(sessions):
    older = await _add_tenant(sessions, "Older", "111")
    newest = await _add_tenant(sessions, "Newest", "222")

    await _incoming(sessions, "unknown", "hola")

    assert (await _agent_config(sessions, older)).whatsapp_phone_number_id == "111"
    assert (await _agent_config(sessions, newest)).whatsapp_phone_number_id == "222"
    assert FakeWhatsApp.sent == []
    assert await _conversations(sessions, newest) == []


@requires_postgres
@pytest.mark.asyncio
async def test_suspended_agent_stays_silent_and_suspended(sessions):
    organization_id = await _add_tenant(sessions, "Paused", "111", status="suspended")

    await _incoming(sessions, "111", "hola")

    assert (await _agent_config(sessions, organization_id)).status == "suspended"
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

    assert len(await _conversations(sessions, org_a)) == 1
    assert len(await _conversations(sessions, org_b)) == 1
    assert FakeLLM.histories[-1] == [("user", "hola B")]


@requires_postgres
@pytest.mark.asyncio
async def test_messages_of_a_previously_moved_thread_stay_hidden(sessions):
    org_a = await _add_tenant(sessions, "NegocioA", "111")
    org_b = await _add_tenant(sessions, "NegocioB", "222")
    conversation_id = uuid.uuid4()
    async with sessions() as db:
        await bind_tenant(db, org_b)
        db.add(
            Conversation(
                id=conversation_id, organization_id=org_b, customer_phone=CUSTOMER, status="active"
            )
        )
        await db.commit()
        await bind_tenant(db, org_a)
        db.add(
            Message(
                id=uuid.uuid4(),
                conversation_id=conversation_id,
                organization_id=org_a,
                role="user",
                content="secreto para A",
            )
        )
        await db.commit()

    await _incoming(sessions, "222", "hola B")

    assert FakeLLM.histories[-1] == [("user", "hola B")]


# ---------------------------------------------------------------------------
# Booking through WhatsApp
# ---------------------------------------------------------------------------


@requires_postgres
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text", ["¿mi cita del 15 sigue en pie?", "cuánto cuesta la cita para 2 personas?"]
)
async def test_questions_never_book_without_a_command(sessions, text):
    organization_id = await _add_tenant(sessions, "Negocio", "111")
    FakeLLM.reply = "Claro, te cuento."

    await _incoming(sessions, "111", text)

    assert await _appointments(sessions, organization_id) == []
    assert FakeWhatsApp.sent == ["Claro, te cuento."]


@requires_postgres
@pytest.mark.asyncio
async def test_booking_is_stored_and_mirrored_before_confirming(sessions):
    organization_id = await _add_tenant(sessions, "Negocio", "111")
    FakeLLM.reply = f"Listo, quedó agendada.\n[[AGENDAR|{NEXT_WEEK}|10:00]]"

    await _incoming(sessions, "111", "agéndame")

    assert FakeCalendar.created[0]["start_time"] == f"{NEXT_WEEK}T10:00:00-05:00"
    assert FakeCalendar.created[0]["end_time"] == f"{NEXT_WEEK}T11:00:00-05:00"
    assert FakeWhatsApp.sent == ["Listo, quedó agendada."]
    stored = await _appointments(sessions, organization_id)
    assert [(a.source, a.status, a.google_event_id) for a in stored] == [
        ("whatsapp", "confirmed", "event-1")
    ]


@requires_postgres
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "setup",
    [
        lambda: setattr(FakeCalendar, "busy_error", ValueError("not connected")),
        lambda: setattr(FakeCalendar, "busy_error", CalendarUnavailableError("HTTP 500")),
        lambda: setattr(FakeCalendar, "create_result", {"success": False, "message": "HTTP 500"}),
    ],
)
async def test_failed_booking_is_never_confirmed_nor_kept(sessions, setup):
    organization_id = await _add_tenant(sessions, "Negocio", "111")
    setup()
    FakeLLM.reply = f"Listo, quedó agendada.\n[[AGENDAR|{NEXT_WEEK}|10:00]]"

    await _incoming(sessions, "111", "agéndame")

    assert FakeWhatsApp.sent == [CALENDAR_FAILED_REPLY]
    assert await _appointments(sessions, organization_id) == []
    async with sessions() as db:
        await bind_tenant(db, organization_id)
        stored = (await db.execute(select(Message).where(Message.role == "assistant"))).scalar_one()
    assert stored.content == CALENDAR_FAILED_REPLY


@requires_postgres
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tag",
    [
        "[[AGENDAR]]",
        f"[[AGENDAR|{NEXT_WEEK}]]",
        "[[AGENDAR|2020-01-01|10:00]]",
        f"[[AGENDAR|{NEXT_WEEK}|25:00]]",
    ],
)
async def test_booking_needs_a_valid_future_slot(sessions, tag):
    await _add_tenant(sessions, "Negocio", "111")
    FakeLLM.reply = f"Listo.\n{tag}"

    await _incoming(sessions, "111", "agéndame")

    assert FakeCalendar.created == []
    assert FakeWhatsApp.sent == [NEED_DATETIME_REPLY]


@requires_postgres
@pytest.mark.asyncio
@pytest.mark.parametrize("time_arg", ["10:00", "10:30", "09:30"])
async def test_overlapping_another_customer_is_refused(sessions, time_arg):
    organization_id = await _add_tenant(sessions, "Negocio", "111")
    await _book(sessions, organization_id, OTHER_CUSTOMER, 10)
    FakeLLM.reply = f"Listo, quedó agendada.\n[[AGENDAR|{NEXT_WEEK}|{time_arg}]]"

    await _incoming(sessions, "111", "agéndame")

    assert FakeWhatsApp.sent == [SLOT_TAKEN_REPLY]
    assert FakeCalendar.created == []
    assert len(await _appointments(sessions, organization_id)) == 1


@requires_postgres
@pytest.mark.asyncio
async def test_back_to_back_appointments_are_allowed(sessions):
    organization_id = await _add_tenant(sessions, "Negocio", "111")
    await _book(sessions, organization_id, OTHER_CUSTOMER, 10)
    FakeLLM.reply = f"Listo.\n[[AGENDAR|{NEXT_WEEK}|11:00]]"

    await _incoming(sessions, "111", "agéndame")

    assert FakeWhatsApp.sent == ["Listo."]
    assert len(await _appointments(sessions, organization_id)) == 2


@requires_postgres
@pytest.mark.asyncio
async def test_two_businesses_can_use_the_same_hour(sessions):
    await _add_tenant(sessions, "NegocioA", "111")
    org_b = await _add_tenant(sessions, "NegocioB", "222")
    await _book(sessions, org_b, OTHER_CUSTOMER, 10)
    FakeLLM.reply = f"Listo.\n[[AGENDAR|{NEXT_WEEK}|10:00]]"

    await _incoming(sessions, "111", "agéndame")

    assert FakeWhatsApp.sent == ["Listo."]


@requires_postgres
@pytest.mark.asyncio
async def test_owner_personal_events_block_the_slot(sessions):
    organization_id = await _add_tenant(sessions, "Negocio", "111")
    FakeCalendar.busy = [(_at(10), _at(12))]
    FakeLLM.reply = f"Listo.\n[[AGENDAR|{NEXT_WEEK}|10:00]]"

    await _incoming(sessions, "111", "agéndame")

    assert FakeWhatsApp.sent == [SLOT_TAKEN_REPLY]
    assert await _appointments(sessions, organization_id) == []


@requires_postgres
@pytest.mark.asyncio
async def test_repeating_a_confirmed_slot_does_not_duplicate_it(sessions):
    organization_id = await _add_tenant(sessions, "Negocio", "111")
    await _book(sessions, organization_id, CUSTOMER, 10)
    FakeLLM.reply = f"Gracias a ti.\n[[AGENDAR|{NEXT_WEEK}|10:00]]"

    await _incoming(sessions, "111", "gracias")

    assert FakeWhatsApp.sent == ["Gracias a ti."]
    assert FakeCalendar.created == []
    assert len(await _appointments(sessions, organization_id)) == 1


@requires_postgres
@pytest.mark.asyncio
async def test_simultaneous_requests_get_exactly_one_booking(sessions):
    organization_id = await _add_tenant(sessions, "Negocio", "111")

    # Several rounds: the deadlock this guards against only appeared intermittently
    for hour in (8, 10, 12, 14, 16):

        async def request_slot(phone: str, start: int = hour):
            async with sessions() as db:
                await bind_tenant(db, organization_id)
                service = AppointmentService(db, organization_id, FakeCalendar(db, organization_id))
                result = await service.book(
                    "Ana", phone, _at(start), _at(start + 1), "Consulta", "whatsapp"
                )
                return result.outcome

        outcomes = await asyncio.gather(*(request_slot(f"57300000000{i}") for i in range(5)))

        assert sorted(outcomes) == sorted([BookingOutcome.BOOKED] + [BookingOutcome.SLOT_TAKEN] * 4)

    assert len(await _appointments(sessions, organization_id)) == 5


# ---------------------------------------------------------------------------
# Rescheduling and cancelling through WhatsApp
# ---------------------------------------------------------------------------


@requires_postgres
@pytest.mark.asyncio
async def test_reschedule_creates_the_new_slot_before_releasing_the_old_one(sessions):
    organization_id = await _add_tenant(sessions, "Negocio", "111")
    await _book(sessions, organization_id, CUSTOMER, 9, "old")
    FakeLLM.reply = f"Cambiada.\n[[REAGENDAR|{NEXT_WEEK}|15:00]]"

    await _incoming(sessions, "111", "cámbiala")

    assert FakeCalendar.calls == ["create", "delete"]
    assert FakeCalendar.deleted == ["old"]
    stored = await _appointments(sessions, organization_id)
    assert [(a.starts_at, a.google_event_id) for a in stored] == [(_at(15), "event-1")]


@requires_postgres
@pytest.mark.asyncio
async def test_reschedule_can_overlap_the_appointment_it_replaces(sessions):
    organization_id = await _add_tenant(sessions, "Negocio", "111")
    await _book(sessions, organization_id, CUSTOMER, 10, "old")
    FakeCalendar.busy = [(_at(10, 30), _at(11))]  # Google trims to the queried window
    FakeLLM.reply = f"Cambiada.\n[[REAGENDAR|{NEXT_WEEK}|10:30]]"

    await _incoming(sessions, "111", "muévela media hora")

    assert FakeWhatsApp.sent == ["Cambiada."]
    assert (await _appointments(sessions, organization_id))[0].starts_at == _at(10, 30)


@requires_postgres
@pytest.mark.asyncio
async def test_reschedule_into_a_taken_slot_keeps_the_original(sessions):
    organization_id = await _add_tenant(sessions, "Negocio", "111")
    await _book(sessions, organization_id, CUSTOMER, 9, "mine")
    await _book(sessions, organization_id, OTHER_CUSTOMER, 15, "theirs")
    FakeLLM.reply = f"Cambiada.\n[[REAGENDAR|{NEXT_WEEK}|15:00]]"

    await _incoming(sessions, "111", "cámbiala")

    assert FakeWhatsApp.sent == [SLOT_TAKEN_REPLY]
    assert FakeCalendar.deleted == []
    stored = await _appointments(sessions, organization_id)
    assert sorted(a.starts_at for a in stored) == [_at(9), _at(15)]


@requires_postgres
@pytest.mark.asyncio
async def test_failed_reschedule_keeps_the_original_appointment(sessions):
    organization_id = await _add_tenant(sessions, "Negocio", "111")
    await _book(sessions, organization_id, CUSTOMER, 9, "old")
    FakeCalendar.create_result = {"success": False, "message": "HTTP 500"}
    FakeLLM.reply = f"Cambiada.\n[[REAGENDAR|{NEXT_WEEK}|15:00]]"

    await _incoming(sessions, "111", "cámbiala")

    assert FakeCalendar.deleted == []
    assert FakeWhatsApp.sent == [CALENDAR_FAILED_REPLY]
    assert (await _appointments(sessions, organization_id))[0].starts_at == _at(9)


@requires_postgres
@pytest.mark.asyncio
async def test_reschedule_without_an_appointment_says_so(sessions):
    await _add_tenant(sessions, "Negocio", "111")
    FakeLLM.reply = f"Cambiada.\n[[REAGENDAR|{NEXT_WEEK}|15:00]]"

    await _incoming(sessions, "111", "cámbiala")

    assert FakeCalendar.created == []
    assert FakeWhatsApp.sent == [NO_APPOINTMENT_REPLY]


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
async def test_cancel_removes_only_the_customers_soonest_appointment(sessions):
    organization_id = await _add_tenant(sessions, "Negocio", "111")
    await _book(sessions, organization_id, OTHER_CUSTOMER, 8, "theirs")
    await _book(sessions, organization_id, CUSTOMER, 10, "soonest")
    await _book(sessions, organization_id, CUSTOMER, 15, "later")
    FakeLLM.reply = "Tu cita fue cancelada.\n[[CANCELAR]]"

    await _incoming(sessions, "111", "cancela mi cita")

    assert FakeCalendar.deleted == ["soonest"]
    assert FakeWhatsApp.sent == ["Tu cita fue cancelada."]
    statuses = {a.google_event_id: a.status for a in await _appointments(sessions, organization_id)}
    assert statuses == {"theirs": "confirmed", "soonest": "cancelled", "later": "confirmed"}


@requires_postgres
@pytest.mark.asyncio
async def test_failed_cancel_keeps_the_appointment_and_says_so(sessions):
    organization_id = await _add_tenant(sessions, "Negocio", "111")
    await _book(sessions, organization_id, CUSTOMER, 10, "event")
    FakeCalendar.delete_result = {"success": False, "message": "HTTP 500"}
    FakeLLM.reply = "Tu cita fue cancelada.\n[[CANCELAR]]"

    await _incoming(sessions, "111", "cancela mi cita")

    assert FakeWhatsApp.sent == [CALENDAR_FAILED_REPLY]
    assert (await _appointments(sessions, organization_id))[0].status == "confirmed"


@requires_postgres
@pytest.mark.asyncio
async def test_cancelled_slot_can_be_booked_again(sessions):
    organization_id = await _add_tenant(sessions, "Negocio", "111")
    await _book(sessions, organization_id, OTHER_CUSTOMER, 10, status="cancelled")
    FakeLLM.reply = f"Listo.\n[[AGENDAR|{NEXT_WEEK}|10:00]]"

    await _incoming(sessions, "111", "agéndame")

    assert FakeWhatsApp.sent == ["Listo."]


@requires_postgres
@pytest.mark.asyncio
async def test_panel_appointment_is_found_despite_phone_formatting(sessions):
    organization_id = await _add_tenant(sessions, "Negocio", "111")
    await _book(sessions, organization_id, "+57 300 000 0001", 10, "panel")
    FakeLLM.reply = "Tu cita fue cancelada.\n[[CANCELAR]]"

    await _incoming(sessions, "111", "cancela mi cita")

    assert FakeCalendar.deleted == ["panel"]


# ---------------------------------------------------------------------------
# Units that need no database
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("time_arg", "expected"),
    [
        ("14:30", (14, 30)),
        ("2:30 pm", (14, 30)),
        ("10am", (10, 0)),
        ("12 am", (0, 0)),
        ("12:15 p.m.", (12, 15)),
    ],
)
def test_parse_appointment_start_reads_times(time_arg, expected):
    start = parse_appointment_start("2026-11-20", time_arg, datetime(2026, 11, 4))

    assert (start.hour, start.minute) == expected


@pytest.mark.parametrize(
    ("date_arg", "time_arg"),
    [
        ("", ""),
        ("mañana", "10:00"),
        ("2026-13-40", "10:00"),
        ("2026-11-20", "24:00"),
        ("2026-11-03", "10:00"),
    ],
)
def test_parse_appointment_start_rejects_invalid_slots(date_arg, time_arg):
    assert parse_appointment_start(date_arg, time_arg, datetime(2026, 11, 4)) is None


@pytest.mark.parametrize(
    ("stored", "customer", "expected"),
    [
        ("573000000001", "573000000001", True),
        ("+57 300 000 0001", "573000000001", True),
        ("3000000001", "573000000001", True),
        ("573111111111", "573000000001", False),
        ("+57-sim-sim_abc", "+57-sim-sim_abc", True),
        ("+57-sim-sim_abc", "+57-sim-sim_xyz", False),
        ("0001", "573000000001", False),
    ],
)
def test_phones_match(stored, customer, expected):
    assert phones_match(stored, customer) is expected


def test_commands_are_read_and_removed_from_replies():
    reply = "Listo, te espero.\n[[AGENDAR|2026-11-20|10:00|Ana|Corte]]"

    assert parse_command(reply) == ("AGENDAR", ["2026-11-20", "10:00", "Ana", "Corte"])
    assert parse_command("Hola, ¿cómo estás?") == (None, [])
    assert strip_command_tags(reply) == "Listo, te espero."


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
async def test_busy_intervals_reads_the_requested_window(monkeypatch):
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        busy = [{"start": "2026-11-20T15:00:00Z", "end": "2026-11-20T16:00:00Z"}]
        return httpx.Response(200, json={"calendars": {"primary": {"busy": busy}}})

    busy = await _google_calendar(monkeypatch, handler).busy_intervals(
        "2026-11-20T10:00:00-05:00", "2026-11-20T11:00:00-05:00"
    )

    assert seen["timeMin"] == "2026-11-20T10:00:00-05:00"
    assert busy == [
        (datetime(2026, 11, 20, 10, tzinfo=BOGOTA), datetime(2026, 11, 20, 11, tzinfo=BOGOTA))
    ]


@pytest.mark.asyncio
async def test_busy_intervals_reports_google_failures(monkeypatch):
    service = _google_calendar(monkeypatch, lambda request: httpx.Response(500))

    with pytest.raises(CalendarUnavailableError):
        await service.busy_intervals("2026-11-20T10:00:00-05:00", "2026-11-20T11:00:00-05:00")


@pytest.mark.asyncio
async def test_availability_covers_the_colombian_day(monkeypatch):
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(200, json={"calendars": {"primary": {"busy": []}}})

    await _google_calendar(monkeypatch, handler).get_availability("2026-12-31")

    assert seen["timeMin"] == "2026-12-31T00:00:00-05:00"
    assert seen["timeMax"] == "2027-01-01T00:00:00-05:00"
    assert seen["timeZone"] == "America/Bogota"


@pytest.mark.asyncio
async def test_availability_reports_busy_hours_in_12_hour_format(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        busy = [{"start": "2026-11-20T14:00:00-05:00", "end": "2026-11-20T15:30:00-05:00"}]
        return httpx.Response(200, json={"calendars": {"primary": {"busy": busy}}})

    availability = await _google_calendar(monkeypatch, handler).get_availability("2026-11-20")

    assert "2:00 PM a 3:30 PM" in availability


@requires_postgres
@pytest.mark.asyncio
async def test_agent_is_told_to_use_12_hour_format(sessions):
    organization_id = await _add_tenant(sessions, "Negocio", "111")
    await _book(sessions, organization_id, CUSTOMER, 15, "event")

    await _incoming(sessions, "111", "¿a qué hora es mi cita?")

    assert "a las 3:00 PM" in FakeLLM.prompts[-1]
    assert "formato de 12 horas" in FakeLLM.prompts[-1]


# ---------------------------------------------------------------------------
# Dashboard metrics
# ---------------------------------------------------------------------------


async def _last_reply(sessions, organization_id: uuid.UUID) -> Message:
    async with sessions() as db:
        await bind_tenant(db, organization_id)
        return (
            (await db.execute(select(Message).where(Message.role == "assistant"))).scalars().one()
        )


@requires_postgres
@pytest.mark.asyncio
async def test_delivered_reply_is_measured_as_resolved(sessions):
    organization_id = await _add_tenant(sessions, "Negocio", "111")

    await _incoming(sessions, "111", "hola")

    reply = await _last_reply(sessions, organization_id)
    assert reply.resolved is True
    assert reply.response_time_ms >= 0


@requires_postgres
@pytest.mark.asyncio
async def test_refusing_an_occupied_slot_still_counts_as_resolved(sessions):
    organization_id = await _add_tenant(sessions, "Negocio", "111")
    FakeCalendar.busy = [(_at(10), _at(11))]
    FakeLLM.reply = f"Listo.\n[[AGENDAR|{NEXT_WEEK}|10:00]]"

    await _incoming(sessions, "111", "agéndame")

    assert (await _last_reply(sessions, organization_id)).resolved is True


@requires_postgres
@pytest.mark.asyncio
async def test_calendar_failure_counts_as_unresolved(sessions):
    organization_id = await _add_tenant(sessions, "Negocio", "111")
    FakeCalendar.busy_error = CalendarUnavailableError("HTTP 500")
    FakeLLM.reply = f"Listo.\n[[AGENDAR|{NEXT_WEEK}|10:00]]"

    await _incoming(sessions, "111", "agéndame")

    assert (await _last_reply(sessions, organization_id)).resolved is False


@requires_postgres
@pytest.mark.asyncio
async def test_model_failure_counts_as_unresolved(sessions):
    organization_id = await _add_tenant(sessions, "Negocio", "111")
    FakeLLM.reply = ""

    await _incoming(sessions, "111", "hola")

    assert (await _last_reply(sessions, organization_id)).resolved is False


@requires_postgres
@pytest.mark.asyncio
async def test_undelivered_reply_counts_as_unresolved(sessions, monkeypatch):
    organization_id = await _add_tenant(sessions, "Negocio", "111")

    async def fail(self, phone_number_id, to, text):
        raise RuntimeError("Meta rejected the message")

    monkeypatch.setattr(FakeWhatsApp, "send_text_message", fail)

    await _incoming(sessions, "111", "hola")

    assert (await _last_reply(sessions, organization_id)).resolved is False


# ---------------------------------------------------------------------------
# Never leaving a customer without a correct answer
# ---------------------------------------------------------------------------


@requires_postgres
@pytest.mark.asyncio
async def test_a_redelivered_message_is_answered_once(sessions):
    await _add_tenant(sessions, "Negocio", "111")

    for _ in range(3):
        async with sessions() as db:
            await ConversationService(db).process_incoming_message(
                "111", CUSTOMER, "hola", "Ana", whatsapp_message_id="wamid.same"
            )

    assert len(FakeWhatsApp.sent) == 1


@requires_postgres
@pytest.mark.asyncio
async def test_simultaneous_redeliveries_are_answered_once(sessions):
    await _add_tenant(sessions, "Negocio", "111")
    await _incoming(sessions, "111", "primer mensaje")
    FakeWhatsApp.sent.clear()

    async def deliver():
        async with sessions() as db:
            await ConversationService(db).process_incoming_message(
                "111", CUSTOMER, "hola", "Ana", whatsapp_message_id="wamid.race"
            )

    await asyncio.gather(*(deliver() for _ in range(4)))

    assert len(FakeWhatsApp.sent) == 1


@requires_postgres
@pytest.mark.asyncio
async def test_different_messages_are_all_answered(sessions):
    await _add_tenant(sessions, "Negocio", "111")

    for number in range(3):
        async with sessions() as db:
            await ConversationService(db).process_incoming_message(
                "111", CUSTOMER, "hola", "Ana", whatsapp_message_id=f"wamid.{number}"
            )

    assert len(FakeWhatsApp.sent) == 3


@requires_postgres
@pytest.mark.asyncio
async def test_audio_gets_a_clear_answer_without_calling_the_model(sessions):
    organization_id = await _add_tenant(sessions, "Negocio", "111")

    async with sessions() as db:
        await ConversationService(db).process_unsupported_message(
            "111", CUSTOMER, "audio", "Ana", whatsapp_message_id="wamid.audio"
        )

    assert FakeWhatsApp.sent == [UNSUPPORTED_MESSAGE_REPLY]
    assert FakeLLM.prompts == []
    async with sessions() as db:
        await bind_tenant(db, organization_id)
        contents = (
            (await db.execute(select(Message.content).order_by(Message.role.desc())))
            .scalars()
            .all()
        )
    assert contents == ["[audio]", UNSUPPORTED_MESSAGE_REPLY]


@requires_postgres
@pytest.mark.asyncio
async def test_suspended_agent_ignores_audio_too(sessions):
    await _add_tenant(sessions, "Paused", "111", status="suspended")

    async with sessions() as db:
        await ConversationService(db).process_unsupported_message("111", CUSTOMER, "audio", "Ana")

    assert FakeWhatsApp.sent == []


@requires_postgres
@pytest.mark.asyncio
async def test_model_outage_is_explained_instead_of_greeting(sessions):
    await _add_tenant(sessions, "Negocio", "111")
    FakeLLM.reply = ""

    await _incoming(sessions, "111", "¿cuánto cuesta?")

    assert FakeWhatsApp.sent == [MODEL_UNAVAILABLE_REPLY]


@requires_postgres
@pytest.mark.asyncio
async def test_business_without_catalog_gets_no_invented_services(sessions):
    await _add_tenant(sessions, "Negocio", "111")

    await _incoming(sessions, "111", "¿qué servicios tienen?")

    assert NO_CATALOG_TEXT in FakeLLM.prompts[-1]
    assert "diseño web" not in FakeLLM.prompts[-1]


@requires_postgres
@pytest.mark.asyncio
async def test_no_transaction_stays_open_while_the_model_answers(sessions, monkeypatch):
    await _add_tenant(sessions, "Negocio", "111")
    states: list[bool] = []

    class WatchedLLM(FakeLLM):
        async def generate(self, system_prompt, messages, temperature, max_tokens):
            states.append(session.in_transaction())
            return await super().generate(system_prompt, messages, temperature, max_tokens)

    monkeypatch.setattr(FakeFactory, "get_provider", lambda self, *args, **kwargs: WatchedLLM())

    async with sessions() as session:
        await ConversationService(session).process_incoming_message("111", CUSTOMER, "hola", "Ana")

    assert states == [False]
