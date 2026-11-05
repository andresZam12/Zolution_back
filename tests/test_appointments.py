"""
Dashboard and simulator appointment tests. Requires TEST_DATABASE_URL (see conftest.py).
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select

from app.agents.models import AgentConfig
from app.api.v1.endpoints import conversations as conversations_endpoint
from app.appointments.models import Appointment
from app.appointments.services import BOGOTA
from app.core import database
from app.core.database import bind_tenant
from app.core.security import create_access_token
from app.core.time_format import format_time_12h
from app.main import app
from app.tenants.models import Organization
from tests.conftest import requires_postgres

NEXT_WEEK = (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d")
URL = "/api/v1/appointments"


@pytest.fixture
def api(sessions, monkeypatch):
    monkeypatch.setattr(database, "AsyncSessionFactory", sessions)
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


async def _add_tenant(sessions, name: str) -> uuid.UUID:
    organization_id = uuid.uuid4()
    async with sessions() as db:
        db.add(Organization(id=organization_id, name=name, slug=name.lower(), plan="starter", status="active"))
        await db.flush()
        await bind_tenant(db, organization_id)
        db.add(
            AgentConfig(
                organization_id=organization_id,
                status="active",
                llm_provider="openai",
                llm_model="gpt-4o-mini",
                onboarding_answers={},
                system_prompt_generated="Agente",
            )
        )
        await db.commit()
    return organization_id


def _headers(organization_id: uuid.UUID) -> dict[str, str]:
    user_id = str(uuid.uuid4())
    token = create_access_token(
        {
            "sub": user_id,
            "user_id": user_id,
            "email": "owner@example.com",
            "role": "owner",
            "organization_id": str(organization_id),
        }
    )
    return {"Authorization": f"Bearer {token}"}


def _payload(hour: int = 10, phone: str = "573000000001") -> dict:
    return {
        "customer_name": "Ana",
        "customer_phone": phone,
        "start_time": f"{NEXT_WEEK}T{hour:02d}:00:00-05:00",
        "end_time": f"{NEXT_WEEK}T{hour:02d}:45:00-05:00",
        "service": "Consulta",
    }


@requires_postgres
@pytest.mark.asyncio
async def test_panel_books_without_google_and_lists_the_appointment(sessions, api):
    organization_id = await _add_tenant(sessions, "Negocio")

    async with api as client:
        created = await client.post(URL, json=_payload(), headers=_headers(organization_id))
        listing = await client.get(URL, headers=_headers(organization_id))

    assert created.status_code == 200
    assert created.json()["synced_with_google"] is False
    item = listing.json()["items"][0]
    assert (item["patientName"], item["date"], item["time"], item["status"]) == (
        "Ana",
        NEXT_WEEK,
        "10:00 AM - 10:45 AM",
        "confirmed",
    )


@requires_postgres
@pytest.mark.asyncio
async def test_panel_refuses_an_occupied_slot(sessions, api):
    organization_id = await _add_tenant(sessions, "Negocio")

    async with api as client:
        await client.post(URL, json=_payload(), headers=_headers(organization_id))
        repeated = await client.post(
            URL, json=_payload(phone="573111111111"), headers=_headers(organization_id)
        )
        listing = await client.get(URL, headers=_headers(organization_id))

    assert repeated.status_code == 409
    assert "ocupado" in repeated.json()["detail"]
    assert listing.json()["total"] == 1


@requires_postgres
@pytest.mark.asyncio
async def test_status_changes_follow_the_agenda_rules(sessions, api):
    organization_id = await _add_tenant(sessions, "Negocio")
    headers = _headers(organization_id)

    async with api as client:
        first = (await client.post(URL, json=_payload(), headers=headers)).json()["appointment"]
        completed = await client.patch(f"{URL}/{first['id']}/status", json={"status": "completed"}, headers=headers)
        invalid = await client.patch(f"{URL}/{first['id']}/status", json={"status": "lost"}, headers=headers)
        cancelled = await client.patch(f"{URL}/{first['id']}/status", json={"status": "cancelled"}, headers=headers)
        replacement = await client.post(URL, json=_payload(phone="573111111111"), headers=headers)
        reactivated = await client.patch(f"{URL}/{first['id']}/status", json={"status": "confirmed"}, headers=headers)

    assert completed.json()["status"] == "completed"
    assert invalid.status_code == 422
    assert cancelled.json()["status"] == "cancelled"
    assert replacement.status_code == 200
    assert reactivated.status_code == 409


@requires_postgres
@pytest.mark.asyncio
async def test_delete_removes_the_appointment(sessions, api):
    organization_id = await _add_tenant(sessions, "Negocio")
    headers = _headers(organization_id)

    async with api as client:
        created = (await client.post(URL, json=_payload(), headers=headers)).json()["appointment"]
        deleted = await client.delete(f"{URL}/{created['id']}", headers=headers)
        listing = await client.get(URL, headers=headers)

    assert deleted.status_code == 200
    assert listing.json()["total"] == 0


@requires_postgres
@pytest.mark.asyncio
async def test_another_tenants_appointment_cannot_be_touched(sessions, api):
    org_a = await _add_tenant(sessions, "NegocioA")
    org_b = await _add_tenant(sessions, "NegocioB")

    async with api as client:
        theirs = (await client.post(URL, json=_payload(), headers=_headers(org_b))).json()["appointment"]
        listing = await client.get(URL, headers=_headers(org_a))
        changed = await client.patch(
            f"{URL}/{theirs['id']}/status", json={"status": "cancelled"}, headers=_headers(org_a)
        )
        removed = await client.delete(f"{URL}/{theirs['id']}", headers=_headers(org_a))

    assert listing.json()["total"] == 0
    assert (changed.status_code, removed.status_code) == (404, 404)


@requires_postgres
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("start", "end"),
    [("mañana", "tarde"), (f"{NEXT_WEEK}T11:00:00-05:00", f"{NEXT_WEEK}T10:00:00-05:00")],
)
async def test_invalid_time_windows_are_rejected(sessions, api, start, end):
    organization_id = await _add_tenant(sessions, "Negocio")
    payload = {**_payload(), "start_time": start, "end_time": end}

    async with api as client:
        response = await client.post(URL, json=payload, headers=_headers(organization_id))

    assert response.status_code == 422


@requires_postgres
@pytest.mark.asyncio
async def test_availability_without_google_reports_the_agenda(sessions, api):
    organization_id = await _add_tenant(sessions, "Negocio")
    headers = _headers(organization_id)

    async with api as client:
        await client.post(URL, json=_payload(), headers=headers)
        response = await client.get(f"{URL}/availability", params={"date": NEXT_WEEK}, headers=headers)

    assert response.json()["google_connected"] is False
    assert "10:00 AM - 10:45 AM (Ana)" in response.json()["availability"]


@requires_postgres
@pytest.mark.asyncio
async def test_simulator_uses_the_same_agenda(sessions, api, monkeypatch):
    class Reply:
        content = f"Listo, agendada.\n[[AGENDAR|{NEXT_WEEK}|10:00|Ana|Consulta]]"

    class Provider:
        async def generate(self, **kwargs):
            return Reply()

    class Factory:
        def get_provider(self, *args, **kwargs):
            return Provider()

    monkeypatch.setattr(conversations_endpoint, "LLMProviderFactory", Factory)
    organization_id = await _add_tenant(sessions, "Negocio")
    headers = _headers(organization_id)

    async with api as client:
        await client.post(URL, json=_payload(hour=10, phone="573111111111"), headers=headers)
        refused = await client.post(
            "/api/v1/conversations/simulate",
            json={"message": "agéndame", "session_id": "abc"},
            headers=headers,
        )
        Reply.content = f"Listo, agendada.\n[[AGENDAR|{NEXT_WEEK}|14:00|Ana|Consulta]]"
        booked = await client.post(
            "/api/v1/conversations/simulate",
            json={"message": "agéndame", "session_id": "abc"},
            headers=headers,
        )

    assert "no está disponible" in refused.json()["reply"]
    assert refused.json()["appointment_created"] is False
    assert booked.json()["reply"] == "Listo, agendada."
    assert booked.json()["appointment_created"] is True
    async with sessions() as db:
        await bind_tenant(db, organization_id)
        sources = sorted((await db.execute(select(Appointment.source))).scalars().all())
    assert sources == ["dashboard", "simulator"]


@pytest.mark.parametrize(
    ("date_text", "time_text", "start_hour", "minutes"),
    [
        ("2026-11-20", "10:00", 10, 60),
        ("2026-11-20", "10:00 (45 min)", 10, 45),
        ("2026-11-20", "3:30 PM (45 min)", 15, 45),
        ("2026-11-20", "10:00 - 10:45", 10, 45),
    ],
)
def test_legacy_appointment_times_are_understood(date_text, time_text, start_hour, minutes):
    migration = _migration()

    start, end = migration._legacy_window(date_text, time_text)

    assert start.astimezone(BOGOTA).hour == start_hour
    assert end - start == timedelta(minutes=minutes)


@pytest.mark.parametrize(("date_text", "time_text"), [("", "10:00"), ("mañana", "10:00"), ("2026-11-20", ""), ("2026-11-20", "25:00")])
def test_unreadable_legacy_appointments_are_skipped(date_text, time_text):
    assert _migration()._legacy_window(date_text, time_text) is None


def _migration():
    from importlib.util import module_from_spec, spec_from_file_location
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "alembic/versions/20240110_000000_create_appointments.py"
    spec = spec_from_file_location("create_appointments", path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("hour", "minute", "expected"),
    [(0, 5, "12:05 AM"), (9, 0, "9:00 AM"), (12, 0, "12:00 PM"), (15, 30, "3:30 PM"), (23, 59, "11:59 PM")],
)
def test_times_are_shown_in_12_hour_format(hour, minute, expected):
    assert format_time_12h(datetime(2026, 11, 20, hour, minute, tzinfo=BOGOTA)) == expected


def test_times_are_shown_in_colombian_local_time():
    assert format_time_12h(datetime(2026, 11, 20, 20, 0, tzinfo=UTC)) == "3:00 PM"
