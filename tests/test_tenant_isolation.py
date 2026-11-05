"""
Row-Level Security regression tests. Requires TEST_DATABASE_URL (see conftest.py).

The role running these tests must match production: it owns the tables and is
neither a superuser nor BYPASSRLS. `test_rls_is_forced_for_the_application_role`
fails otherwise, because the remaining assertions would prove nothing.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import pytest
from httpx import ASGITransport, AsyncClient
from jose import jwt
from sqlalchemy import select, text
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.agents.models import AgentConfig
from app.api.v1.endpoints import conversations as conversations_endpoint
from app.appointments.models import Appointment
from app.conversations.models import Conversation, Message
from app.core import database
from app.core.database import bind_tenant, resolve_whatsapp_tenant
from app.core.middleware import TenantContextMiddleware
from app.core.security import create_access_token, hash_otp_code
from app.integrations.google.calendar import GoogleCalendarService
from app.integrations.models import IntegrationCredential
from app.main import app
from app.tenants.models import Organization, User, VerificationCode
from tests.conftest import TEST_DATABASE_URL, requires_postgres

pytestmark = requires_postgres


async def _empty_jwks() -> dict:
    return {"keys": []}


TENANT_TABLES = ["agent_configs", "appointments", "conversations", "messages", "integration_credentials"]


async def _add_tenant(sessions, name: str, phone_id: str | None = None) -> uuid.UUID:
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
                onboarding_answers={"appointments": [], "secret": name},
                system_prompt_generated=f"Agente de {name}",
                whatsapp_phone_number_id=phone_id,
            )
        )
        await db.commit()
    return organization_id


async def _add_conversation(sessions, organization_id: uuid.UUID, phone: str) -> uuid.UUID:
    conversation_id = uuid.uuid4()
    async with sessions() as db:
        await bind_tenant(db, organization_id)
        db.add(Conversation(id=conversation_id, organization_id=organization_id, customer_phone=phone, status="active"))
        await db.flush()
        db.add(Message(id=uuid.uuid4(), conversation_id=conversation_id, organization_id=organization_id, role="user", content="hola"))
        await db.commit()
    return conversation_id


@pytest.fixture
def api(sessions, monkeypatch):
    """HTTP client whose requests use the migrated test database."""
    monkeypatch.setattr(database, "AsyncSessionFactory", sessions)
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


def _token(organization_id: uuid.UUID | None) -> dict[str, str]:
    user_id = str(uuid.uuid4())
    token = create_access_token(
        {
            "sub": user_id,
            "user_id": user_id,
            "email": "owner@example.com",
            "role": "owner",
            "organization_id": str(organization_id) if organization_id else None,
        }
    )
    return {"Authorization": f"Bearer {token}"}


# ---------------------------------------------------------------------------
# Database guarantees
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rls_is_forced_for_the_application_role(sessions):
    async with sessions() as db:
        role = (
            await db.execute(
                text("SELECT rolsuper, rolbypassrls FROM pg_roles WHERE rolname = current_user")
            )
        ).one()
        tables = (
            await db.execute(
                text(
                    "SELECT relname, relrowsecurity, relforcerowsecurity, "
                    "pg_get_userbyid(relowner) = current_user FROM pg_class "
                    "WHERE relname = ANY(:tables)"
                ),
                {"tables": TENANT_TABLES},
            )
        ).all()

    assert tuple(role) == (False, False)
    assert sorted(tuple(t) for t in tables) == sorted((t, True, True, True) for t in TENANT_TABLES)


@pytest.mark.asyncio
@pytest.mark.parametrize("model", [AgentConfig, Appointment, Conversation, Message, IntegrationCredential])
async def test_session_without_tenant_sees_nothing(sessions, model):
    organization_id = await _add_tenant(sessions, "NegocioA", "111")
    await _add_conversation(sessions, organization_id, "573000000001")

    async with sessions() as db:
        assert (await db.execute(select(model))).scalars().all() == []


@pytest.mark.asyncio
async def test_tenant_reads_only_its_own_rows(sessions):
    org_a = await _add_tenant(sessions, "NegocioA")
    org_b = await _add_tenant(sessions, "NegocioB")
    await _add_conversation(sessions, org_a, "573000000001")
    await _add_conversation(sessions, org_b, "573000000002")

    async with sessions() as db:
        await bind_tenant(db, org_a)
        configs = (await db.execute(select(AgentConfig))).scalars().all()
        conversations = (await db.execute(select(Conversation))).scalars().all()
        messages = (await db.execute(select(Message))).scalars().all()

    assert [c.organization_id for c in configs] == [org_a]
    assert [c.organization_id for c in conversations] == [org_a]
    assert [m.organization_id for m in messages] == [org_a]


@pytest.mark.asyncio
async def test_tenant_cannot_change_or_delete_other_rows(sessions):
    org_a = await _add_tenant(sessions, "NegocioA")
    org_b = await _add_tenant(sessions, "NegocioB")

    async with sessions() as db:
        await bind_tenant(db, org_a)
        updated = await db.execute(
            text("UPDATE agent_configs SET status = 'suspended' WHERE organization_id = :o"),
            {"o": org_b},
        )
        deleted = await db.execute(
            text("DELETE FROM agent_configs WHERE organization_id = :o"), {"o": org_b}
        )
        await db.commit()

    assert (updated.rowcount, deleted.rowcount) == (0, 0)


@pytest.mark.asyncio
async def test_tenant_cannot_insert_for_another_organization(sessions):
    org_a = await _add_tenant(sessions, "NegocioA")
    org_b = await _add_tenant(sessions, "NegocioB")

    async with sessions() as db:
        await bind_tenant(db, org_a)
        db.add(Conversation(id=uuid.uuid4(), organization_id=org_b, customer_phone="573000000001", status="active"))
        with pytest.raises(DBAPIError, match="row-level security"):
            await db.flush()


@pytest.mark.asyncio
async def test_tenant_binding_survives_commits(sessions):
    organization_id = await _add_tenant(sessions, "NegocioA")

    async with sessions() as db:
        await bind_tenant(db, organization_id)
        await db.commit()
        await db.commit()
        configs = (await db.execute(select(AgentConfig))).scalars().all()

    assert len(configs) == 1


@pytest.mark.asyncio
async def test_reused_connection_keeps_no_tenant_and_raises_no_error(sessions):
    """Covers both pool leakage and ''::uuid, which used to raise on reused connections."""
    organization_id = await _add_tenant(sessions, "NegocioA")
    engine = create_async_engine(TEST_DATABASE_URL, pool_size=1, max_overflow=0)
    pooled = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

    async with pooled() as db:
        await bind_tenant(db, organization_id)
        assert len((await db.execute(select(AgentConfig))).scalars().all()) == 1
        await db.commit()

    async with pooled() as db:
        tenant = await db.execute(text("SELECT current_setting('app.current_organization_id', true)"))
        assert tenant.scalar() in (None, "")
        assert (await db.execute(select(AgentConfig))).scalars().all() == []

    await engine.dispose()


@pytest.mark.asyncio
async def test_routing_finds_the_owner_of_a_number_only(sessions):
    org_a = await _add_tenant(sessions, "NegocioA", "111")
    await _add_tenant(sessions, "NegocioB", "222")

    async with sessions() as db:
        assert await resolve_whatsapp_tenant(db, "111") == org_a
        assert await resolve_whatsapp_tenant(db, "unknown") is None
        assert await resolve_whatsapp_tenant(db, "") is None
        assert (await db.execute(select(AgentConfig))).scalars().all() == []


# ---------------------------------------------------------------------------
# HTTP flows
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_api_lists_only_the_callers_conversations(sessions, api):
    org_a = await _add_tenant(sessions, "NegocioA")
    org_b = await _add_tenant(sessions, "NegocioB")
    mine = await _add_conversation(sessions, org_a, "573000000001")
    theirs = await _add_conversation(sessions, org_b, "573000000002")

    async with api as client:
        listing = await client.get("/api/v1/conversations", headers=_token(org_a))
        own = await client.get(f"/api/v1/conversations/{mine}", headers=_token(org_a))
        other = await client.get(f"/api/v1/conversations/{theirs}", headers=_token(org_a))

    assert [item["id"] for item in listing.json()["items"]] == [str(mine)]
    assert own.status_code == 200
    assert other.status_code == 404


def test_only_verified_tokens_provide_a_tenant():
    organization_id = str(uuid.uuid4())
    claims = {
        "sub": str(uuid.uuid4()),
        "organization_id": organization_id,
        "exp": datetime.now(UTC) + timedelta(hours=1),
    }
    forged = jwt.encode(claims, "not-the-secret-key", algorithm="HS256")
    valid = create_access_token(claims)

    assert TenantContextMiddleware._extract_org_id(valid) == organization_id
    assert TenantContextMiddleware._extract_org_id(forged) is None
    assert TenantContextMiddleware._extract_org_id("not-a-token") is None


@pytest.mark.asyncio
async def test_forged_token_cannot_read_a_tenant(sessions, api, monkeypatch):
    org_b = await _add_tenant(sessions, "NegocioB")
    forged = jwt.encode(
        {
            "sub": str(uuid.uuid4()),
            "role": "owner",
            "organization_id": str(org_b),
            "exp": datetime.now(UTC) + timedelta(hours=1),
        },
        "not-the-secret-key",
        algorithm="HS256",
    )
    monkeypatch.setattr("app.auth.dependencies._fetch_jwks", _empty_jwks)

    async with api as client:
        response = await client.get(
            "/api/v1/conversations", headers={"Authorization": f"Bearer {forged}"}
        )

    assert response.status_code == 401


@pytest.mark.asyncio
async def test_calendar_credentials_stay_inside_their_tenant(sessions):
    org_a = await _add_tenant(sessions, "NegocioA")
    org_b = await _add_tenant(sessions, "NegocioB")
    async with sessions() as db:
        await bind_tenant(db, org_a)
        db.add(IntegrationCredential(organization_id=org_a, provider="google", access_token="token-a"))
        await db.commit()

    async with sessions() as db:
        assert await GoogleCalendarService(db, str(org_a))._get_access_token() == "token-a"

    async with sessions() as db:
        with pytest.raises(ValueError, match="no está conectado"):
            await GoogleCalendarService(db, str(org_b))._get_access_token()


@pytest.mark.asyncio
async def test_organization_header_cannot_select_another_tenant(sessions, api):
    org_b = await _add_tenant(sessions, "NegocioB")

    async with api as client:
        response = await client.patch(
            "/api/v1/agents/me",
            json={"status": "suspended"},
            headers={**_token(None), "X-Organization-ID": str(org_b)},
        )

    assert response.status_code == 200
    assert response.json()["organization_id"] != str(org_b)
    async with sessions() as db:
        await bind_tenant(db, org_b)
        assert (await db.execute(select(AgentConfig))).scalar_one().status == "active"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "path", ["/agents/me", "/agents/me/status", "/appointments", "/integrations/status", "/organizations/me"]
)
async def test_dashboard_reads_work_under_rls(sessions, api, path):
    organization_id = await _add_tenant(sessions, "NegocioA")

    async with api as client:
        response = await client.get(f"/api/v1{path}", headers=_token(organization_id))

    assert response.status_code == 200


@pytest.mark.asyncio
async def test_agent_update_works_under_rls(sessions, api):
    organization_id = await _add_tenant(sessions, "NegocioA")

    async with api as client:
        response = await client.patch(
            "/api/v1/agents/me",
            json={"status": "suspended"},
            headers=_token(organization_id),
        )

    assert response.status_code == 200
    assert response.json()["status"] == "suspended"


@pytest.mark.asyncio
async def test_registration_and_login_work_under_rls(sessions, api):
    email = "dueno@example.com"
    async with sessions() as db:
        db.add(
            VerificationCode(
                email=email,
                code=hash_otp_code(email, "123456"),
                purpose="registration",
                expires_at=datetime.now(UTC) + timedelta(minutes=10),
                is_used=False,
                attempts=0,
            )
        )
        await db.commit()

    async with api as client:
        registered = await client.post(
            "/api/v1/auth/verify-code",
            json={"email": email, "code": "123456", "company_name": "Negocio Nuevo", "password": "secreto123"},
        )
        logged_in = await client.post(
            "/api/v1/auth/login", json={"email": email, "password": "secreto123"}
        )
        token = {"Authorization": f"Bearer {logged_in.json()['access_token']}"}
        agent = await client.get("/api/v1/agents/me", headers=token)

    assert registered.status_code == 201
    assert logged_in.status_code == 200
    assert agent.status_code == 200
    async with sessions() as db:
        assert (await db.execute(select(User))).scalar_one().email == email


@pytest.mark.asyncio
async def test_public_organization_registration_provisions_its_agent(sessions, api):
    async with api as client:
        response = await client.post(
            "/api/v1/organizations/register",
            json={"name": "Negocio Publico", "email": "x@example.com"},
        )

    assert response.status_code == 201
    async with sessions() as db:
        await bind_tenant(db, response.json()["id"])
        assert (await db.execute(select(AgentConfig))).scalar_one().status == "draft"


@pytest.mark.asyncio
async def test_simulator_persists_inside_the_callers_tenant(sessions, api, monkeypatch):
    class Reply:
        content = "Hola desde el simulador"

    class Provider:
        async def generate(self, **kwargs):
            return Reply()

    class Factory:
        def get_provider(self, *args, **kwargs):
            return Provider()

    monkeypatch.setattr(conversations_endpoint, "LLMProviderFactory", Factory)
    organization_id = await _add_tenant(sessions, "NegocioA")

    async with api as client:
        response = await client.post(
            "/api/v1/conversations/simulate",
            json={"message": "hola", "session_id": "abc"},
            headers=_token(organization_id),
        )

    assert response.json()["reply"] == "Hola desde el simulador."
    async with sessions() as db:
        await bind_tenant(db, organization_id)
        messages = (await db.execute(select(Message))).scalars().all()
    assert sorted(m.role for m in messages) == ["assistant", "user"]
