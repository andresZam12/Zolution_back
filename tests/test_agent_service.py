"""
Unit tests for AgentConfigService and the prompt engineering engine.

Tests cover:
- Service business logic (status, onboarding, activation guards)
- `_build_system_prompt()` — verifies the generated prompt contains
  critical sections and business data from the onboarding answers

To run:
    cd backend/
    pytest tests/test_agent_service.py -v
"""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from app.agents.schemas import BusinessHours, OnboardingAnswers, ServiceItem
from app.agents.services import (
    AgentConfigService,
    _build_system_prompt,
    _is_onboarding_complete,
)
from app.tenants.schemas import OrganizationCreate
from app.tenants.services import OrganizationService

pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def agent_service() -> AgentConfigService:
    return AgentConfigService()


@pytest.fixture
def org_service() -> OrganizationService:
    return OrganizationService()


@pytest.fixture
def minimal_answers() -> OnboardingAnswers:
    """Minimum valid onboarding answers for prompt generation."""
    return OnboardingAnswers(
        business_name="Clínica Dental Sonríe Bien",
        business_type="clínica odontológica",
        services=[
            ServiceItem(name="Limpieza dental", price_cop=80_000, duration_minutes=45),
            ServiceItem(name="Consulta general", price_cop=60_000),
        ],
        hours={
            "monday": BusinessHours(open="08:00", close="18:00"),
            "friday": BusinessHours(open="08:00", close="17:00"),
            "saturday": BusinessHours(open="09:00", close="13:00"),
        },
        agent_name="Sofia",
        tone="cálido y profesional",
        use_emojis=True,
        cancellation_policy_hours=24,
        handles_emergencies=False,
        escalation_contact="Dr. García +573001234567",
    )


@pytest.fixture
async def org_with_config(db_session, org_service):
    """Creates an org + initial AgentConfig in one step."""
    org = await org_service.create(
        OrganizationCreate(name="Test Dental", slug="test-dental", plan="free"),
        db_session,
    )
    await db_session.flush()
    return org


# ---------------------------------------------------------------------------
# Prompt engineering engine — pure function tests (no DB needed)
# ---------------------------------------------------------------------------


class TestBuildSystemPrompt:
    """Tests for the _build_system_prompt() pure function."""

    def test_contains_agent_name(self, minimal_answers):
        prompt = _build_system_prompt(minimal_answers)
        assert "Sofia" in prompt

    def test_contains_business_name(self, minimal_answers):
        prompt = _build_system_prompt(minimal_answers)
        assert "Clínica Dental Sonríe Bien" in prompt

    def test_contains_service_names(self, minimal_answers):
        prompt = _build_system_prompt(minimal_answers)
        assert "Limpieza dental" in prompt
        assert "Consulta general" in prompt

    def test_contains_service_prices(self, minimal_answers):
        prompt = _build_system_prompt(minimal_answers)
        assert "80,000" in prompt or "80.000" in prompt  # locale formatting

    def test_contains_operating_hours(self, minimal_answers):
        prompt = _build_system_prompt(minimal_answers)
        assert "08:00" in prompt
        assert "Sábado" in prompt

    def test_contains_cancellation_policy(self, minimal_answers):
        prompt = _build_system_prompt(minimal_answers)
        assert "24" in prompt

    def test_no_emergency_handling(self, minimal_answers):
        minimal_answers = minimal_answers.model_copy(update={"handles_emergencies": False})
        prompt = _build_system_prompt(minimal_answers)
        assert "NO atendemos urgencias" in prompt

    def test_emergency_handling_enabled(self, minimal_answers):
        minimal_answers = minimal_answers.model_copy(update={"handles_emergencies": True})
        prompt = _build_system_prompt(minimal_answers)
        assert "SÍ atendemos urgencias" in prompt

    def test_contains_guardrail_fallback_phrase(self, minimal_answers):
        """The exact fallback phrase must appear verbatim (from Section 6)."""
        prompt = _build_system_prompt(minimal_answers)
        assert "Solo puedo ayudarte con citas en Clínica Dental Sonríe Bien." in prompt

    def test_contains_escalation_contact(self, minimal_answers):
        prompt = _build_system_prompt(minimal_answers)
        assert "Dr. García" in prompt

    def test_six_sections_present(self, minimal_answers):
        """All 6 sections defined in Section 6 of the project doc must be present."""
        prompt = _build_system_prompt(minimal_answers)
        assert "## 1. ROL Y ALCANCE" in prompt
        assert "## 2. DATOS DEL NEGOCIO" in prompt
        assert "## 3. REGLAS DE GUARDRAIL" in prompt
        assert "## 4. INFORMACIÓN SENSIBLE" in prompt
        assert "## 5. FORMATO DE RESPUESTA" in prompt
        assert "## 6. ESCALACIÓN" in prompt

    def test_no_emoji_rule_when_disabled(self, minimal_answers):
        minimal_answers = minimal_answers.model_copy(update={"use_emojis": False})
        prompt = _build_system_prompt(minimal_answers)
        assert "No uses emojis" in prompt


class TestOnboardingComplete:
    """Tests for _is_onboarding_complete() validation."""

    def test_complete_answers_returns_true(self, minimal_answers):
        assert _is_onboarding_complete(minimal_answers.model_dump()) is True

    def test_missing_business_name_returns_false(self, minimal_answers):
        data = minimal_answers.model_dump()
        data["business_name"] = None
        assert _is_onboarding_complete(data) is False

    def test_empty_services_returns_false(self, minimal_answers):
        data = minimal_answers.model_dump()
        data["services"] = []
        assert _is_onboarding_complete(data) is False

    def test_missing_agent_name_returns_false(self, minimal_answers):
        data = minimal_answers.model_dump()
        data["agent_name"] = None
        assert _is_onboarding_complete(data) is False


# ---------------------------------------------------------------------------
# AgentConfigService — service method tests (with DB)
# ---------------------------------------------------------------------------


async def test_get_status_returns_draft_for_new_org(agent_service, org_with_config, db_session):
    """Newly created organizations should have a draft agent status."""
    status = await agent_service.get_status(org_with_config.id, db_session)

    assert status.status == "draft"
    assert status.has_system_prompt is False
    assert status.onboarding_complete is False


async def test_update_onboarding_answers_persists(
    agent_service, org_with_config, db_session, minimal_answers
):
    """update_onboarding_answers persists the answers to the DB."""
    config = await agent_service.update_onboarding_answers(
        org_with_config.id, minimal_answers, db_session
    )
    await db_session.flush()

    assert config.onboarding_answers is not None
    assert config.onboarding_answers["business_name"] == "Clínica Dental Sonríe Bien"


async def test_generate_system_prompt_succeeds_with_complete_answers(
    agent_service, org_with_config, db_session, minimal_answers
):
    """generate_system_prompt populates system_prompt_generated."""
    await agent_service.update_onboarding_answers(org_with_config.id, minimal_answers, db_session)
    await db_session.flush()

    config = await agent_service.generate_system_prompt(org_with_config.id, db_session)

    assert config.system_prompt_generated is not None
    assert len(config.system_prompt_generated) > 100
    assert "Sofia" in config.system_prompt_generated


async def test_generate_prompt_raises_400_without_answers(
    agent_service, org_with_config, db_session
):
    """generate_system_prompt raises 400 when no answers are stored."""
    with pytest.raises(HTTPException) as exc_info:
        await agent_service.generate_system_prompt(org_with_config.id, db_session)

    assert exc_info.value.status_code == 400
    assert "onboarding answers are not set" in exc_info.value.detail


async def test_activate_raises_400_without_prompt(agent_service, org_with_config, db_session):
    """activate() raises 400 when no system prompt has been generated."""
    with pytest.raises(HTTPException) as exc_info:
        await agent_service.activate(org_with_config.id, db_session)

    assert exc_info.value.status_code == 400
    assert "generate-prompt" in exc_info.value.detail


async def test_activate_succeeds_after_prompt_generation(
    agent_service, org_with_config, db_session, minimal_answers
):
    """Full happy path: update answers → generate prompt → activate."""
    await agent_service.update_onboarding_answers(org_with_config.id, minimal_answers, db_session)
    await db_session.flush()

    await agent_service.generate_system_prompt(org_with_config.id, db_session)
    await db_session.flush()

    config = await agent_service.activate(org_with_config.id, db_session)
    assert config.status == "active"
