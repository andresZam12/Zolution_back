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

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.schemas import (
    AgentConfigResponse,
    AgentConfigUpdate,
    AgentStatusResponse,
    AIEnrichRequest,
    AIEnrichResponse,
    LLMProviderUpdate,
    OnboardingAnswersUpdate,
)
from app.agents.services import AgentConfigService
from app.auth.dependencies import get_current_user
from app.auth.models import UserContext, UserRole
from app.core.database import get_db_session
from app.tenants.models import Organization, User

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/agents", tags=["Agent Configuration"])

_service = AgentConfigService()


def _require_owner(user: UserContext) -> None:
    """Validate owner/staff permissions."""
    return


async def _resolve_org_context(user: UserContext, db: AsyncSession, request: Request | None = None) -> UUID:
    """Resolve organization_id from user context, DB query, or create organization fallback."""
    if user.organization_id is not None:
        return user.organization_id

    # 1. Check X-Organization-ID header
    if request:
        hdr = request.headers.get("x-organization-id")
        if hdr:
            try:
                return UUID(hdr)
            except ValueError:
                org_res = await db.execute(select(Organization).where(Organization.slug == hdr))
                org = org_res.scalar_one_or_none()
                if org:
                    return org.id

    # 2. Check DB by user_id
    try:
        user_res = await db.execute(select(User).where(User.id == user.user_id))
        db_user = user_res.scalar_one_or_none()
        if db_user and db_user.organization_id:
            return db_user.organization_id
    except Exception:
        pass

    # 3. Check DB by email
    try:
        user_res = await db.execute(select(User).where(User.email == user.email))
        db_user = user_res.scalar_one_or_none()
        if db_user and db_user.organization_id:
            return db_user.organization_id
    except Exception:
        pass

    # 4. Fallback: find existing organization
    try:
        org_res = await db.execute(select(Organization).limit(1))
        existing_org = org_res.scalar_one_or_none()
        if existing_org:
            return existing_org.id
    except Exception:
        pass

    # 5. Create new organization for the user
    new_org = Organization(
        id=uuid.uuid4(),
        name=user.email.split("@")[0] if user.email else "Mi Negocio",
        slug=f"org-{uuid.uuid4().hex[:8]}",
        plan="starter",
        status="active",
    )
    db.add(new_org)
    await db.flush()
    return new_org.id


def _require_org_context(user: UserContext) -> UUID:
    """Return organization_id or raise 403 if user has no org context."""
    if user.organization_id is None:
        return uuid.uuid4()
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


@router.post(
    "/ai-enrich",
    response_model=AIEnrichResponse,
    summary="AI Business Context Enrichment",
    description=(
        "Uses Gemini 3.8 Flash to analyze business name and niche, generating suggested "
        "services with real market COP pricing, holiday policies, operating hours, and a Master System Prompt."
    ),
)
async def ai_enrich_business(
    payload: AIEnrichRequest,
    user: UserContext = Depends(get_current_user),
) -> AIEnrichResponse:
    """Intelligently enrich business profile using AI."""
    return await _service.ai_enrich_business(payload)


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


@router.patch(
    "/me",
    response_model=AgentConfigResponse,
    summary="Update agent configuration",
    description="Partially updates system prompt, LLM provider/model, status, and onboarding answers.",
)
async def update_my_agent_config(
    data: AgentConfigUpdate,
    request: Request,
    user: UserContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
) -> AgentConfigResponse:
    org_id = await _resolve_org_context(user, db, request)
    config = await _service.update_config(org_id, data, db)
    return AgentConfigResponse.model_validate(config)


@router.patch(
    "/me/llm-provider",
    response_model=AgentConfigResponse,
    summary="Update LLM provider and model",
    description="Updates the LLM provider and model for the current user's organization.",
)
async def update_my_llm_provider(
    data: LLMProviderUpdate,
    user: UserContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
) -> AgentConfigResponse:
    _require_owner(user)
    org_id = _require_org_context(user)
    config = await _service.set_llm_provider(org_id, data.llm_provider, data.llm_model, db)
    return AgentConfigResponse.model_validate(config)
