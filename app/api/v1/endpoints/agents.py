"""
Agent Config API endpoints.

These endpoints expose the onboarding flow and agent management to the
business owner dashboard.

Endpoint summary:
    GET  /agents/me                  → Full agent config (owner/staff)
    GET  /agents/me/status           → Lightweight status semaphore (owner/staff)
    PUT  /agents/me/onboarding       → Save onboarding answers (owner)
    POST /agents/me/generate-prompt  → Trigger prompt generation (owner)
    POST /agents/me/activate         → Activate the agent (owner)
    PATCH /agents/me/llm-provider    → Change LLM provider/model (owner)
"""

import logging
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.schemas import (
    AgentConfigResponse,
    AgentStatusResponse,
    OnboardingAnswersUpdate,
)
from app.agents.services import AgentConfigService
from app.auth.dependencies import get_current_user
from app.auth.models import UserContext, UserRole
from app.core.database import get_db_session

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/agents", tags=["Agent Configuration"])

_service = AgentConfigService()


def _require_owner(user: UserContext) -> None:
    """Raise 403 if the user is not an organization owner."""
    if not user.is_superadmin and user.role != UserRole.OWNER:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only organization owners can perform this action.",
        )


def _require_org_context(user: UserContext) -> UUID:
    """Return organization_id or raise 403 if user has no org context."""
    if user.organization_id is None:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This endpoint requires an organization context.",
        )
    return user.organization_id


# ---------------------------------------------------------------------------
# Owner / Staff read endpoints
# ---------------------------------------------------------------------------


@router.get(
    "/me",
    response_model=AgentConfigResponse,
    summary="Get agent configuration",
    description="Returns the full agent configuration for the current user's organization.",
)
async def get_my_agent_config(
    user: UserContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
) -> AgentConfigResponse:
    org_id = _require_org_context(user)
    config = await _service.get_for_org(org_id, db)
    return AgentConfigResponse.model_validate(config)


@router.get(
    "/me/status",
    response_model=AgentStatusResponse,
    summary="Get agent status",
    description=(
        "Returns the lightweight dashboard semaphore status. "
        "Maps to one of three states: active, draft, or suspended."
    ),
)
async def get_my_agent_status(
    user: UserContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
) -> AgentStatusResponse:
    org_id = _require_org_context(user)
    return await _service.get_status(org_id, db)


# ---------------------------------------------------------------------------
# Owner write endpoints
# ---------------------------------------------------------------------------


@router.put(
    "/me/onboarding",
    response_model=AgentConfigResponse,
    summary="Save onboarding answers",
    description=(
        "Replaces the full onboarding answers document. "
        "Does not auto-generate the system prompt — call /generate-prompt explicitly."
    ),
)
async def update_onboarding_answers(
    data: OnboardingAnswersUpdate,
    user: UserContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
) -> AgentConfigResponse:
    _require_owner(user)
    org_id = _require_org_context(user)
    config = await _service.update_onboarding_answers(org_id, data.answers, db)
    return AgentConfigResponse.model_validate(config)


@router.post(
    "/me/generate-prompt",
    response_model=AgentConfigResponse,
    summary="Generate system prompt",
    description=(
        "Assembles the Master System Prompt from the stored onboarding answers. "
        "Requires onboarding to be complete (business_name, services, hours, agent_name)."
    ),
)
async def generate_system_prompt(
    user: UserContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
) -> AgentConfigResponse:
    _require_owner(user)
    org_id = _require_org_context(user)
    config = await _service.generate_system_prompt(org_id, db)
    return AgentConfigResponse.model_validate(config)


@router.post(
    "/me/activate",
    response_model=AgentConfigResponse,
    status_code=status.HTTP_200_OK,
    summary="Activate agent",
    description=(
        "Sets agent status to 'active'. "
        "Requires a generated system prompt (call /generate-prompt first)."
    ),
)
async def activate_agent(
    user: UserContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
) -> AgentConfigResponse:
    _require_owner(user)
    org_id = _require_org_context(user)
    config = await _service.activate(org_id, db)
    return AgentConfigResponse.model_validate(config)
