"""
Dashboard metrics, computed from each tenant's real activity.
"""

from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import Integer, cast, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.dependencies import get_current_user
from app.auth.models import UserContext
from app.conversations.models import Conversation, Message
from app.core.database import bind_tenant, get_db_session

router = APIRouter(prefix="/analytics", tags=["Analytics"])

METRICS_WINDOW = timedelta(days=30)
NEW_CONVERSATIONS_WINDOW = timedelta(days=7)


class OverviewMetrics(BaseModel):
    """Rates are None until there are measured replies, instead of showing a made-up value."""

    new_conversations_7d: int
    measured_replies: int
    resolution_rate: float | None
    avg_response_seconds: float | None


@router.get("/overview", response_model=OverviewMetrics, summary="Get dashboard metrics")
async def get_overview(
    user: UserContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
) -> OverviewMetrics:
    if not user.organization_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="An organization context is required.",
        )
    await bind_tenant(db, user.organization_id)
    now = datetime.now(UTC)

    measured, resolved, average_ms = (
        await db.execute(
            select(
                func.count(Message.id),
                func.coalesce(func.sum(cast(Message.resolved, Integer)), 0),
                func.avg(Message.response_time_ms),
            ).where(
                Message.organization_id == user.organization_id,
                Message.resolved.is_not(None),
                Message.created_at >= now - METRICS_WINDOW,
            )
        )
    ).one()

    new_conversations = (
        await db.execute(
            select(func.count(Conversation.id)).where(
                Conversation.organization_id == user.organization_id,
                Conversation.created_at >= now - NEW_CONVERSATIONS_WINDOW,
            )
        )
    ).scalar_one()

    return OverviewMetrics(
        new_conversations_7d=new_conversations,
        measured_replies=measured,
        resolution_rate=round(resolved * 100 / measured, 1) if measured else None,
        avg_response_seconds=round(float(average_ms) / 1000, 1) if measured else None,
    )
