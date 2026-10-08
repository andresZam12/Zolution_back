"""
Pydantic schemas for Conversations and Messages.
"""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class MessageResponse(BaseModel):
    """Schema for returning an individual message."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    conversation_id: UUID
    organization_id: UUID
    role: str
    content: str
    tokens_in: int | None = None
    tokens_out: int | None = None
    created_at: datetime


class ConversationListItem(BaseModel):
    """Lightweight conversation schema for list views."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    organization_id: UUID
    customer_phone: str
    customer_name: str | None = None
    status: str
    created_at: datetime
    updated_at: datetime
    last_message: MessageResponse | None = None


class ConversationDetailResponse(BaseModel):
    """Full conversation details including entire message history."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    organization_id: UUID
    customer_phone: str
    customer_name: str | None = None
    status: str
    created_at: datetime
    updated_at: datetime
    messages: list[MessageResponse] = []


class ConversationListResponse(BaseModel):
    """Paginated list of conversations."""

    items: list[ConversationListItem]
    total: int
    page: int
    page_size: int
