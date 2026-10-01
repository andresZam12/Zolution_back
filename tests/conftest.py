"""
Shared pytest fixtures for the Zolution backend test suite.

Key design decisions:
- `APP_ENV=test` is set via os.environ BEFORE any app imports so that
  `get_settings()` (which is called at module-level in database.py)
  receives the correct env value on first call.
- Tests use an in-memory SQLite database (via aiosqlite) so they run
  without a PostgreSQL instance. No Docker required for unit tests.
- All SQLAlchemy models are created fresh per test session, then
  each test runs in a rolled-back transaction for isolation.
- `mock_user_*` fixtures provide ready-made UserContext instances for
  different roles without requiring Auth0 or JWT validation.

Running tests:
    cd backend/
    pytest tests/ -v --tb=short
    pytest tests/ -v --cov=app --cov-report=term-missing  # with coverage
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# CRITICAL: set APP_ENV before any app imports so that get_settings() (called
# eagerly in database.py at module level) picks up "test" as the environment.
# Config defaults for SECRET_KEY / AUTH0_DOMAIN / AUTH0_AUDIENCE are used
# when not explicitly set — see app/core/config.py.
# ---------------------------------------------------------------------------
import os

os.environ.setdefault("APP_ENV", "test")
os.environ.setdefault("DATABASE_PASSWORD", "test_password")

import uuid
from collections.abc import AsyncGenerator

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

# Clear the lru_cache so get_settings() re-reads our env vars above
from app.core.config import get_settings

get_settings.cache_clear()

from app.admin import models as _admin_models  # noqa: E402, F401
from app.agents import models as _agent_models  # noqa: E402, F401
from app.auth.models import UserContext, UserRole  # noqa: E402
from app.core.database import Base  # noqa: E402
from app.tenants import models as _tenant_models  # noqa: E402, F401

# In-memory SQLite for unit tests — no Postgres, no Docker needed
SQLITE_URL = "sqlite+aiosqlite:///:memory:"


@pytest_asyncio.fixture(scope="session")
async def engine():
    """Create a shared async engine for the test session."""
    eng = create_async_engine(SQLITE_URL, echo=False)
    async with eng.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield eng
    await eng.dispose()


@pytest_asyncio.fixture
async def db_session(engine) -> AsyncGenerator[AsyncSession, None]:
    """
    Provide a transactional AsyncSession that is rolled back after each test.

    This ensures test isolation without truncating tables between tests.
    """
    async with engine.begin() as conn:
        session = AsyncSession(bind=conn, expire_on_commit=False)
        try:
            yield session
        finally:
            await session.close()
            await conn.rollback()


# ---------------------------------------------------------------------------
# UserContext fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def org_id() -> uuid.UUID:
    """A consistent test organization ID."""
    return uuid.UUID("12345678-1234-5678-1234-567812345678")


@pytest.fixture
def mock_owner(org_id: uuid.UUID) -> UserContext:
    """UserContext for an organization owner."""
    return UserContext(
        user_id=str(uuid.uuid4()),
        email="owner@testclinic.com",
        role=UserRole.OWNER,
        organization_id=org_id,
    )


@pytest.fixture
def mock_staff(org_id: uuid.UUID) -> UserContext:
    """UserContext for a staff member."""
    return UserContext(
        user_id=str(uuid.uuid4()),
        email="staff@testclinic.com",
        role=UserRole.STAFF,
        organization_id=org_id,
    )


@pytest.fixture
def mock_superadmin() -> UserContext:
    """UserContext for a superadmin (no organization)."""
    return UserContext(
        user_id=str(uuid.uuid4()),
        email="admin@zolution.app",
        role=UserRole.SUPERADMIN,
        organization_id=None,
    )
