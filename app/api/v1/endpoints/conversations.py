"""
Conversation API endpoints for the tenant dashboard and Live WhatsApp Simulator.
"""

import logging
from typing import Optional
from uuid import UUID
import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.services import AgentConfigService
from app.auth.dependencies import get_current_user
from app.auth.models import UserContext
from app.conversations.agent import run_agent_turn
from app.conversations.models import Conversation, Message
from app.conversations.schemas import (
    ConversationDetailResponse,
    ConversationListItem,
    ConversationListResponse,
    MessageResponse,
)
from app.conversations.services import ConversationService
from app.core.database import bind_tenant, get_db_session
from app.core.time_format import format_time_12h
from app.integrations.google.calendar import GoogleCalendarService
from app.llm_providers.factory import LLMProviderFactory
from app.tenants.models import Organization

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/conversations", tags=["Conversations"])


class SimulateMessageRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    customer_name: Optional[str] = Field(default="Cliente Simulador WhatsApp")
    customer_phone: Optional[str] = Field(default="+57 300 000 0000")
    session_id: Optional[str] = None
    reset_session: Optional[bool] = False
    system_prompt_override: Optional[str] = None


class SimulateMessageResponse(BaseModel):
    conversation_id: str
    reply: str
    tool_called: Optional[str] = None
    appointment_created: bool = False
    google_calendar_synced: bool = False
    used_provider: Optional[str] = None
    llm_errors: list[str] = []


def _require_org_id(user: UserContext) -> str:
    """Extract organization_id or raise 403."""
    if not user.organization_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="An organization context is required to access conversations.",
        )
    return str(user.organization_id)


@router.get(
    "",
    response_model=ConversationListResponse,
    summary="List organization conversations",
)
async def list_conversations(
    page: int = Query(default=1, ge=1, description="Page number"),
    page_size: int = Query(default=20, ge=1, le=100, description="Items per page"),
    user: UserContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
) -> ConversationListResponse:
    """
    Retrieve paginated list of conversations for the authenticated tenant.
    """
    org_id = _require_org_id(user)
    await bind_tenant(db, org_id)
    service = ConversationService(db)

    conversations, total = await service.list_conversations(
        organization_id=org_id, page=page, page_size=page_size
    )

    items: list[ConversationListItem] = []
    for conv in conversations:
        last_msg = None
        if conv.messages:
            latest = conv.messages[-1]
            last_msg = MessageResponse.model_validate(latest)

        item = ConversationListItem(
            id=conv.id,
            organization_id=conv.organization_id,
            customer_phone=conv.customer_phone,
            customer_name=conv.customer_name,
            status=conv.status,
            created_at=conv.created_at,
            updated_at=conv.updated_at,
            last_message=last_msg,
        )
        items.append(item)

    return ConversationListResponse(
        items=items,
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get(
    "/{conversation_id}",
    response_model=ConversationDetailResponse,
    summary="Get conversation detail with messages",
)
async def get_conversation(
    conversation_id: UUID,
    user: UserContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
) -> ConversationDetailResponse:
    """
    Retrieve a specific conversation thread and its full message history.
    """
    org_id = _require_org_id(user)
    await bind_tenant(db, org_id)
    service = ConversationService(db)

    conv = await service.get_conversation(
        organization_id=org_id, conversation_id=str(conversation_id)
    )

    if not conv:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found.",
        )

    return ConversationDetailResponse.model_validate(conv)


@router.post(
    "/simulate",
    response_model=SimulateMessageResponse,
    summary="Run a turn of the real agent in the simulator",
)
async def simulate_conversation_turn(
    payload: SimulateMessageRequest,
    user: UserContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
) -> SimulateMessageResponse:
    """
    Run the same agent turn as WhatsApp, without sending anything to a customer.
    Bookings are real: they use the tenant's agenda and calendar.
    """
    org_uuid = uuid.UUID(_require_org_id(user))
    await bind_tenant(db, org_uuid)

    org = (await db.execute(select(Organization).where(Organization.id == org_uuid))).scalar_one()
    agent_config = await AgentConfigService().get_for_org(org_uuid, db)

    session_id = (payload.session_id or "").strip()
    customer_phone = (
        f"+57-sim-{session_id}" if session_id else (payload.customer_phone or "+57 300 000 0000").strip()
    )
    customer_name = (payload.customer_name or "Cliente Simulador WhatsApp").strip()

    conversation = None
    if payload.reset_session:
        await db.execute(
            update(Conversation)
            .where(
                Conversation.organization_id == org_uuid,
                Conversation.customer_phone == customer_phone,
            )
            .values(status="archived")
        )
    else:
        result = await db.execute(
            select(Conversation)
            .where(
                Conversation.organization_id == org_uuid,
                Conversation.customer_phone == customer_phone,
                Conversation.status == "active",
            )
            .order_by(Conversation.updated_at.desc())
        )
        conversation = result.scalars().first()

    if conversation is None:
        conversation = Conversation(
            id=uuid.uuid4(),
            organization_id=org_uuid,
            customer_phone=customer_phone,
            customer_name=customer_name,
            status="active",
        )
        db.add(conversation)
        await db.flush()

    text_content = payload.message.strip()
    db.add(
        Message(
            id=uuid.uuid4(),
            conversation_id=conversation.id,
            organization_id=org_uuid,
            role="user",
            content=text_content,
        )
    )
    await db.flush()

    turn = await run_agent_turn(
        db,
        agent_config=agent_config,
        org_name=org.name or "tu negocio",
        conversation_id=conversation.id,
        customer_name=customer_name,
        customer_phone=customer_phone,
        text_content=text_content,
        source="simulator",
        calendar=GoogleCalendarService(db, str(org_uuid)),
        llm_factory=LLMProviderFactory(),
        require_calendar=False,
        prompt_override=payload.system_prompt_override,
    )

    db.add(
        Message(
            id=uuid.uuid4(),
            conversation_id=conversation.id,
            organization_id=org_uuid,
            role="assistant",
            content=turn.reply,
        )
    )
    await db.commit()

    tool_called = None
    appointment_created = False
    if turn.outcome.changed and turn.command_type == "CANCELAR":
        tool_called = "cancel_appointment()"
    elif turn.outcome.changed:
        appointment_created = True
        action = "reschedule" if turn.command_type == "REAGENDAR" else "book"
        tool_called = (
            f"{action}_appointment(date='{turn.outcome.booked_start:%Y-%m-%d}', "
            f"time='{format_time_12h(turn.outcome.booked_start)}')"
        )

    return SimulateMessageResponse(
        conversation_id=str(conversation.id),
        reply=turn.reply,
        tool_called=tool_called,
        appointment_created=appointment_created,
        google_calendar_synced=turn.outcome.synced,
        used_provider=turn.used_provider,
        llm_errors=turn.errors,
    )
