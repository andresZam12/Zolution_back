"""
Async database engine, session factory, and FastAPI dependencies.

Architecture decisions:
- SQLAlchemy async engine with asyncpg driver (see ADR-002)
- One AsyncSession per request, injected via FastAPI's Depends()
- RLS enforcement: `bind_tenant()` ties a session to one organization and
  `app.current_organization_id` is re-applied at the start of every transaction.
  Without a tenant, tenant tables return no rows (fail closed).

Usage:
    from app.core.database import get_db_session, Base

    # In a route or service:
    async def some_route(db: AsyncSession = Depends(get_db_session)):
        ...
"""

from collections.abc import AsyncGenerator
import uuid

from fastapi import Request
from sqlalchemy import Connection, event, select, text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, Session, SessionTransaction

from app.core.config import get_settings

# ---------------------------------------------------------------------------
# Declarative base — all ORM models inherit from this
# ---------------------------------------------------------------------------


class Base(DeclarativeBase):
    """
    SQLAlchemy declarative base for all Zolution ORM models.

    Convention:
    - Table names are snake_case plurals (e.g. `organizations`, `users`)
    - All multi-tenant tables include an `organization_id` column
    - Timestamps: `created_at` and `updated_at` on every table
    """


# ---------------------------------------------------------------------------
# Engine factory
# ---------------------------------------------------------------------------


def _create_engine(database_url: str, *, echo: bool = False) -> AsyncEngine:
    """Create a configured async SQLAlchemy engine."""
    return create_async_engine(
        database_url,
        echo=echo,  # Log SQL only when DEBUG=True
        pool_size=10,  # Connections kept open in the pool
        max_overflow=20,  # Connections beyond pool_size (burst)
        pool_pre_ping=True,  # Validate connections before use
        pool_recycle=3600,  # Recycle connections every hour
    )


settings = get_settings()

engine: AsyncEngine = _create_engine(
    settings.DATABASE_URL,
    echo=settings.DEBUG,
)

# ---------------------------------------------------------------------------
# Session factories
# ---------------------------------------------------------------------------

AsyncSessionFactory = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,  # Avoid implicit lazy loads after commit
)
AsyncSessionLocal = AsyncSessionFactory


# ---------------------------------------------------------------------------
# Tenant context
# ---------------------------------------------------------------------------

_TENANT_KEY = "tenant_organization_id"
_SET_TENANT = text("SELECT set_config('app.current_organization_id', :org_id, true)")
_SET_ROUTING_PHONE = text("SELECT set_config('app.routing_phone_number_id', :phone, true)")


@event.listens_for(Session, "after_begin")
def _apply_tenant_context(
    session: Session, transaction: SessionTransaction, connection: Connection
) -> None:
    """A transaction-local setting ends at commit, so each new transaction sets it again."""
    organization_id = session.info.get(_TENANT_KEY)
    if organization_id:
        connection.execute(_SET_TENANT, {"org_id": organization_id})


async def bind_tenant(session: AsyncSession, organization_id: uuid.UUID | str) -> None:
    """Scope this session to one organization for all its remaining transactions."""
    tenant = str(uuid.UUID(str(organization_id)))
    session.info[_TENANT_KEY] = tenant
    await session.execute(_SET_TENANT, {"org_id": tenant})


async def resolve_whatsapp_tenant(session: AsyncSession, phone_number_id: str) -> uuid.UUID | None:
    """
    Find the organization that owns a WhatsApp number.

    The number is exposed to the `whatsapp_routing` policy only for this lookup.
    """
    from app.agents.models import AgentConfig

    await session.execute(_SET_ROUTING_PHONE, {"phone": phone_number_id})
    result = await session.execute(
        select(AgentConfig.organization_id).where(
            AgentConfig.whatsapp_phone_number_id == phone_number_id
        )
    )
    organization_id = result.scalar_one_or_none()
    await session.execute(_SET_ROUTING_PHONE, {"phone": ""})
    return organization_id


# ---------------------------------------------------------------------------
# FastAPI dependencies
# ---------------------------------------------------------------------------


async def get_db_session(request: Request = None) -> AsyncGenerator[AsyncSession, None]:  # type: ignore[assignment]
    """
    Yield an async database session for a single request.

    Requests with a verified token start bound to their organization. Rolls back
    on any unhandled exception to keep the connection clean.
    """
    async with AsyncSessionFactory() as session:
        try:
            organization_id = getattr(getattr(request, "state", None), "organization_id", None)
            if organization_id:
                await bind_tenant(session, organization_id)
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
