"""
Conversation domain ORM models: Conversation and Message.

A Conversation represents a WhatsApp thread between an end customer and
a tenant's AI agent. Each turn in the conversation is a Message.

Both tables carry `organization_id` directly (no JOIN needed for RLS queries)
to keep RLS policy checks efficient.
"""

import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


class Conversation(Base):
    """
    A WhatsApp conversation thread between a customer and a tenant's agent.

    Status lifecycle: active → closed | escalated
    `escalated` means a human staff member needs to take over.
    """

    __tablename__ = "conversations"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=func.gen_random_uuid(),
    )
    organization_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    # Customer's WhatsApp number in E.164 format (e.g. "+573001234567")
    customer_phone: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    customer_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    # active | closed | escalated
    status: Mapped[str] = mapped_column(String(50), nullable=False, default="active")
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
    messages: Mapped[list["Message"]] = relationship("Message", back_populates="conversation")

    def __repr__(self) -> str:
        return f"<Conversation id={self.id} phone={self.customer_phone!r} status={self.status!r}>"


class Message(Base):
    """
    A single message turn within a Conversation.

    `role` mirrors the LLM convention: 'user' (customer) or 'assistant' (agent).
    `tokens_in` / `tokens_out` enable per-tenant cost tracking.
    `organization_id` is denormalized here (also on Conversation) so RLS
    policies on this table don't require a JOIN to conversations.
    """

    __tablename__ = "messages"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=func.gen_random_uuid(),
    )
    conversation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    # Denormalized for efficient RLS without JOIN
    organization_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    # user | assistant
    role: Mapped[str] = mapped_column(String(20), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    # Token counts for cost tracking (null for user-side messages)
    tokens_in: Mapped[int | None] = mapped_column(Integer, nullable=True)
    tokens_out: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Relationships
    conversation: Mapped["Conversation"] = relationship("Conversation", back_populates="messages")

    def __repr__(self) -> str:
        return f"<Message id={self.id} role={self.role!r} conversation={self.conversation_id}>"
