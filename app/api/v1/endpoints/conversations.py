"""
Conversation API endpoints for the tenant dashboard.
"""

import logging
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.dependencies import get_current_user
from app.auth.models import UserContext
from app.conversations.schemas import (
    ConversationDetailResponse,
    ConversationListItem,
    ConversationListResponse,
    MessageResponse,
)
from app.conversations.services import ConversationService
from app.core.database import get_db_session

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/conversations", tags=["Conversations"])


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
