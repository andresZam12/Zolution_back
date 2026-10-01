"""
Unit tests for OrganizationService.

Tests verify business logic in isolation from HTTP.
No routes, no Auth0, no Postgres required — uses SQLite in-memory.

To run:
    cd backend/
    pytest tests/test_tenant_service.py -v
"""

from __future__ import annotations

import uuid

import pytest
from fastapi import HTTPException

from app.agents.models import AgentConfig
from app.tenants.schemas import OrganizationCreate, OrganizationUpdate
from app.tenants.services import OrganizationService

# Mark all tests in this file as async
pytestmark = pytest.mark.asyncio


@pytest.fixture
def service() -> OrganizationService:
    return OrganizationService()


@pytest.fixture
def create_data() -> OrganizationCreate:
    return OrganizationCreate(
        name="Clínica Dental Test",
        slug="clinica-dental-test",
        plan="free",
    )


# ---------------------------------------------------------------------------
# create()
# ---------------------------------------------------------------------------


async def test_create_organization_persists_correctly(service, create_data, db_session):
    """Creating an organization persists name, slug, and plan to the database."""
    org = await service.create(create_data, db_session)
    await db_session.flush()

    assert org.id is not None
    assert org.name == "Clínica Dental Test"
    assert org.slug == "clinica-dental-test"
    assert org.plan == "free"
    assert org.status == "pending_setup"


async def test_create_organization_creates_agent_config(service, create_data, db_session):
    """
    Creating an organization must auto-create an AgentConfig in 'draft' status.

    This ensures the onboarding flow can start immediately after the org is created.
    """
    from sqlalchemy import select

    org = await service.create(create_data, db_session)
    await db_session.flush()

    result = await db_session.execute(
        select(AgentConfig).where(AgentConfig.organization_id == org.id)
    )
    config = result.scalar_one_or_none()

    assert config is not None, "AgentConfig must be created alongside the Organization"
    assert config.status == "draft"
    assert config.llm_provider == "anthropic"


async def test_create_organization_raises_on_duplicate_slug(service, create_data, db_session):
    """Attempting to create two orgs with the same slug raises HTTPException 409."""
    await service.create(create_data, db_session)
    await db_session.flush()

    with pytest.raises(HTTPException) as exc_info:
        await service.create(create_data, db_session)

    assert exc_info.value.status_code == 409
    assert "already taken" in exc_info.value.detail


# ---------------------------------------------------------------------------
# get_by_id()
# ---------------------------------------------------------------------------


async def test_get_by_id_returns_organization(service, create_data, db_session):
    """get_by_id returns the correct organization when it exists."""
    org = await service.create(create_data, db_session)
    await db_session.flush()

    fetched = await service.get_by_id(org.id, db_session)
    assert fetched.id == org.id
    assert fetched.slug == org.slug


async def test_get_by_id_raises_404_for_unknown_id(service, db_session):
    """get_by_id raises HTTPException 404 for a non-existent organization."""
    non_existent_id = uuid.uuid4()

    with pytest.raises(HTTPException) as exc_info:
        await service.get_by_id(non_existent_id, db_session)

    assert exc_info.value.status_code == 404


# ---------------------------------------------------------------------------
# get_for_current_user()
# ---------------------------------------------------------------------------


async def test_get_for_current_user_raises_403_when_no_org(service, db_session):
    """
    Superadmins (organization_id=None) calling get_for_current_user get a 403.
    """
    with pytest.raises(HTTPException) as exc_info:
        await service.get_for_current_user(None, db_session)

    assert exc_info.value.status_code == 403


# ---------------------------------------------------------------------------
# update()
# ---------------------------------------------------------------------------


async def test_update_organization_name(service, create_data, db_session):
    """update() changes the name when provided."""
    org = await service.create(create_data, db_session)
    await db_session.flush()

    updated = await service.update(
        org.id, OrganizationUpdate(name="Nuevo Nombre Clínica"), db_session
    )
    assert updated.name == "Nuevo Nombre Clínica"
    assert updated.slug == "clinica-dental-test"  # slug unchanged


async def test_update_organization_raises_409_on_slug_conflict(service, create_data, db_session):
    """Updating a slug to one already used by another org raises 409."""
    await service.create(create_data, db_session)
    org2_data = OrganizationCreate(name="Otra Clínica", slug="otra-clinica", plan="free")
    org2 = await service.create(org2_data, db_session)
    await db_session.flush()

    with pytest.raises(HTTPException) as exc_info:
        await service.update(org2.id, OrganizationUpdate(slug="clinica-dental-test"), db_session)

    assert exc_info.value.status_code == 409


# ---------------------------------------------------------------------------
# set_status()
# ---------------------------------------------------------------------------


async def test_set_status_suspended(service, create_data, db_session):
    """set_status('suspended') updates the organization status."""
    org = await service.create(create_data, db_session)
    await db_session.flush()

    updated = await service.set_status(org.id, "suspended", db_session)
    assert updated.status == "suspended"


async def test_set_status_invalid_value_raises_400(service, create_data, db_session):
    """set_status raises 400 for an invalid status value."""
    org = await service.create(create_data, db_session)
    await db_session.flush()

    with pytest.raises(HTTPException) as exc_info:
        await service.set_status(org.id, "invalid_status", db_session)

    assert exc_info.value.status_code == 400
