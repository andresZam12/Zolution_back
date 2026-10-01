"""
Agent domain ORM models: AgentConfig.

Each organization has exactly one AgentConfig. This is the central artifact
produced by the onboarding flow: the generated system prompt + configuration
that drives the WhatsApp agent for that business.

`onboarding_answers` is stored as JSONB — a flexible document that captures
the business owner's answers without requiring a fixed schema per question.
This allows the onboarding question set to evolve without migrations.
"""

import uuid
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.tenants.models import Organization

from sqlalchemy import DateTime, ForeignKey, String, Text, func
from sqlalchemy.dialects.postgresql import JSON, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


class AgentConfig(Base):
    """
    Configuration for a tenant's WhatsApp AI agent.

    Created when an organization completes the onboarding flow.
    `system_prompt_generated` is the full, ready-to-use system prompt
    assembled by the prompt engineering engine from `onboarding_answers`.

    Status lifecycle: draft → active → suspended
    """

    __tablename__ = "agent_configs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=func.gen_random_uuid(),
    )
    organization_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,  # One config per organization
        index=True,
    )
    # The compiled system prompt sent to the LLM on every conversation turn
    system_prompt_generated: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Raw answers from the onboarding flow — flexible JSONB document
    onboarding_answers: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    # LLM provider in use: anthropic | google | openai
    llm_provider: Mapped[str] = mapped_column(String(50), nullable=False, default="anthropic")
    # Specific model name, e.g. "claude-haiku-4-5"
    llm_model: Mapped[str] = mapped_column(String(100), nullable=False, default="")
    # Agent lifecycle: draft | active | suspended
    status: Mapped[str] = mapped_column(String(50), nullable=False, default="draft")
    # Meta WhatsApp Business Account mapping (the phone number ID that receives messages)
    whatsapp_phone_number_id: Mapped[str | None] = mapped_column(
        String(100), nullable=True, index=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    # Relationships
    organization: Mapped["Organization"] = relationship(
        "Organization", back_populates="agent_config"
    )

    def __repr__(self) -> str:
        return f"<AgentConfig org={self.organization_id} status={self.status!r}>"
