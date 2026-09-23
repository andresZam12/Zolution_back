"""
Async database engine, session factory, and FastAPI dependencies.

Architecture decisions:
- SQLAlchemy async engine with asyncpg driver (see ADR-002)
- One AsyncSession per request, injected via FastAPI's Depends()
- RLS enforcement: `app.current_organization_id` is set at the start of
  every request by TenantContextMiddleware using the verified JWT
- Superadmin connection: separate engine with row_security disabled;
  only accessible from admin service layer, never from a request context
  that originated from a JWT

Usage:
    from app.core.database import get_db_session, Base

    # In a route or service:
    async def some_route(db: AsyncSession = Depends(get_db_session)):
        ...
"""

from collections.abc import AsyncGenerator

from sqlalchemy import event, text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

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
        echo=echo,                  # Log SQL only when DEBUG=True
        pool_size=10,               # Connections kept open in the pool
        max_overflow=20,            # Connections beyond pool_size (burst)
        pool_pre_ping=True,         # Validate connections before use
        pool_recycle=3600,          # Recycle connections every hour
    )


settings = get_settings()

engine: AsyncEngine = _create_engine(
    settings.DATABASE_URL,
    echo=settings.DEBUG,
)

# Superadmin engine — bypasses RLS via a dedicated DB user
_superadmin_engine: AsyncEngine = _create_engine(
    settings.SUPERADMIN_DATABASE_URL,
    echo=False,  # Never log superadmin queries to avoid leaking tenant data
)

# ---------------------------------------------------------------------------
# Session factories
# ---------------------------------------------------------------------------

AsyncSessionFactory = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,  # Avoid implicit lazy loads after commit
)

_SuperadminSessionFactory = async_sessionmaker(
    bind=_superadmin_engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


# ---------------------------------------------------------------------------
# FastAPI dependencies
# ---------------------------------------------------------------------------

async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Yield an async database session for a single request.

    RLS is enforced automatically — the session will only see rows matching
    the `organization_id` set by TenantContextMiddleware.

    Rolls back on any unhandled exception to keep the connection clean.
    """
    async with AsyncSessionFactory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def get_superadmin_db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Yield an async database session with RLS disabled.

    SECURITY: This dependency must ONLY be used in routes under /admin/*.
    It should NEVER be imported or used in tenant-facing endpoints.
    All calls through this session are logged to audit_log by the
    admin service layer — not here, to keep concerns separated.
    """
    async with _SuperadminSessionFactory() as session:
        try:
            # Explicitly disable row security for this session
            await session.execute(text("SET LOCAL row_security = off"))
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
