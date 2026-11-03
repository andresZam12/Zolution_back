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
    description: str | None = Field(default=None, max_length=500)


class AIEnrichRequest(BaseModel):
    """Request to generate rich business context and pricing suggestions using AI."""

    business_name: str = Field(..., min_length=2, max_length=200)
    industry: str = Field(..., min_length=2, max_length=200)
    city_or_country: str | None = Field(default="Colombia", max_length=150)
    extra_notes: str | None = Field(default=None, max_length=1000)


class AIEnrichResponse(BaseModel):
    """Structured AI suggestions for business context and system prompt."""

    agent_name: str
    tone: str
    services: list[ServiceItem]
    schedule_weekdays: str
    schedule_saturdays: str
    holiday_policy: str
    cancellation_policy: str
    tolerance_minutes: int
    payment_methods: list[str]
    faqs: list[dict[str, str]]
    recommended_system_prompt: str


class OnboardingAnswers(BaseModel):
    """
    Typed structure for the onboarding Q&A document.

    Each field corresponds to a question asked by the AI onboarding assistant.
    Stored as JSONB in agent_configs.onboarding_answers.

    All fields are optional so that a partial onboarding save is valid —
    the agent_config.status stays 'draft' until all required fields are present.
    Also accepts frontend camelCase fields (companyName, industry, schedule, etc.).
    """

    model_config = ConfigDict(extra="allow")

    # Business identity
    business_name: str | None = Field(default=None, max_length=255)
    business_type: str | None = Field(
        default=None,
        description="e.g. 'dental clinic', 'barbershop', 'beauty salon'",
    )
    address: str | None = Field(default=None, max_length=500)

    # Services & pricing (supports structured list or free-text catalog from frontend)
    services: list[ServiceItem] | str = Field(
        default_factory=list,
        description="List of services offered with optional pricing, or free-text catalog.",
    )

    # Operating schedule (supports structured dict or free-text schedule from frontend)
    hours: dict[str, BusinessHours] | str = Field(
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
    holiday_policy: str | None = Field(
        default=None,
        description="Operating policy on holidays, e.g. 'Solo hasta mediodía (8am - 1pm)' or 'Cerrado'.",
    )
    tolerance_minutes: int | None = Field(
        default=15,
        ge=0,
        le=60,
        description="Grace period in minutes before appointment is released.",
    )
    payment_methods: list[str] | str | None = Field(
        default=None,
        description="Accepted payment methods (Nequi, Daviplata, Tarjetas, Efectivo).",
    )
    faqs: list[dict[str, str]] | None = Field(
        default=None,
        description="Common niche questions and answers.",
    )
    city_or_country: str | None = Field(
        default=None,
        max_length=150,
        description="City or country of the business.",
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

    # Frontend onboarding / settings fields
    companyName: str | None = Field(default=None, max_length=255)
    industry: str | None = Field(default=None, max_length=255)
    whatsappPhone: str | None = Field(default=None, max_length=100)
    schedule: str | None = Field(default=None)
    agentName: str | None = Field(default=None, max_length=100)
    cancellationPolicy: str | None = Field(default=None)
    contactEmail: str | None = Field(default=None, max_length=255)
    phone: str | None = Field(default=None, max_length=100)
    system_prompt_generated: str | None = Field(default=None)


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
    created_at: datetime | None = None
    updated_at: datetime | None = None


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


class LLMProviderUpdate(BaseModel):
    """Body for PATCH /agents/me/llm-provider."""

    llm_provider: str = Field(..., min_length=2, max_length=50)
    llm_model: str = Field(..., min_length=2, max_length=100)


class AgentConfigUpdate(BaseModel):
    """Body for PATCH /agents/me to update prompt, LLM provider, status, or metadata."""

    system_prompt_generated: str | None = None
    llm_provider: str | None = None
    llm_model: str | None = None
    status: str | None = None
    onboarding_answers: dict | None = None
    company_name: str | None = None
