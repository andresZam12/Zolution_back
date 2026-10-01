"""
Pydantic schemas for the Agent Config domain.

The agent config is the central artifact produced by the onboarding flow:
a generated system prompt + the raw answers that produced it.

`OnboardingAnswers` is the typed representation of the JSONB document
stored in `agent_configs.onboarding_answers`. Typing it as a Pydantic
model rather than a plain `dict` gives us validation at the API boundary
and makes the prompt engineering engine type-safe.
"""

from datetime import datetime
from typing import Annotated
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Onboarding answers — typed representation of JSONB
# ---------------------------------------------------------------------------

class BusinessHours(BaseModel):
    """Operating hours for a single day."""
    open: str = Field(description="Opening time in HH:MM format.", examples=["08:00"])
    close: str = Field(description="Closing time in HH:MM format.", examples=["18:00"])


class ServiceItem(BaseModel):
    """A single service offered by the business."""
    name: str = Field(max_length=200, examples=["Limpieza dental"])
    price_cop: int | None = Field(
        default=None,
        ge=0,
        description="Price in Colombian pesos. Null if price varies.",
    )
    duration_minutes: int | None = Field(
        default=None,
        ge=5,
        description="Approximate service duration in minutes.",
    )


class OnboardingAnswers(BaseModel):
    """
    Typed structure for the onboarding Q&A document.

    Each field corresponds to a question asked by the AI onboarding assistant.
    Stored as JSONB in agent_configs.onboarding_answers.

    All fields are optional so that a partial onboarding save is valid —
    the agent_config.status stays 'draft' until all required fields are present.
    """

    # Business identity
    business_name: str | None = Field(default=None, max_length=255)
    business_type: str | None = Field(
        default=None,
        description="e.g. 'dental clinic', 'barbershop', 'beauty salon'",
    )
    address: str | None = Field(default=None, max_length=500)

    # Services & pricing
    services: list[ServiceItem] = Field(
        default_factory=list,
        max_length=50,
        description="List of services offered with optional pricing.",
    )

    # Operating schedule
    hours: dict[str, BusinessHours] = Field(
        default_factory=dict,
        description="Operating hours keyed by day name (monday, tuesday, etc.).",
    )

    # Policies
    cancellation_policy_hours: int | None = Field(
        default=None,
        ge=0,
        description="Minimum hours required to cancel without penalty.",
    )
    handles_emergencies: bool = Field(
        default=False,
        description="Whether the business handles emergency cases.",
    )

    # Agent persona
    agent_name: str | None = Field(
        default=None,
        max_length=100,
        description="Name of the AI assistant (e.g. 'Sofia', 'Alejandro').",
    )
    tone: str | None = Field(
        default=None,
        description="Desired tone: 'formal', 'friendly', 'professional'.",
    )
    use_emojis: bool = Field(default=True)

    # Escalation
    escalation_contact: str | None = Field(
        default=None,
        description="Phone or name to escalate to when human intervention needed.",
    )


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------

class AgentConfigResponse(BaseModel):
    """Full agent configuration as returned by the API."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    organization_id: UUID
    status: str
    llm_provider: str
    llm_model: str
    onboarding_answers: dict | None
    system_prompt_generated: str | None
    created_at: datetime
    updated_at: datetime


class AgentStatusResponse(BaseModel):
    """
    Lightweight status for the dashboard 'semaphore'.

    Maps to one of three states shown in the UI:
        active      → "Agente activo ✅"
        draft       → "En configuración 🔵"
        suspended   → "Necesita atención 🔴"
    """

    status: str
    has_system_prompt: bool
    onboarding_complete: bool
    llm_provider: str
    llm_model: str


class OnboardingAnswersUpdate(BaseModel):
    """Body for PUT /agents/me/onboarding. Replaces the full answers document."""
    answers: OnboardingAnswers
