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

import asyncio
import subprocess
import sys
import uuid
from collections.abc import AsyncGenerator
from pathlib import Path

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

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

# Real PostgreSQL for tests that depend on migrations and RLS. Point this at a
# disposable database owned by a non-superuser role: its schema is recreated.
TEST_DATABASE_URL = os.environ.get("TEST_DATABASE_URL", "")
requires_postgres = pytest.mark.skipif(not TEST_DATABASE_URL, reason="TEST_DATABASE_URL is not set")


async def _recreate_public_schema() -> None:
    engine = create_async_engine(TEST_DATABASE_URL, poolclass=NullPool)
    async with engine.begin() as conn:
        await conn.execute(text("DROP SCHEMA public CASCADE"))
        await conn.execute(text("CREATE SCHEMA public"))
    await engine.dispose()


@pytest.fixture(scope="session")
def migrated_database() -> None:
    """Apply the real migrations once, so RLS policies exist as in production."""
    asyncio.run(_recreate_public_schema())
    subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head"],
        cwd=Path(__file__).resolve().parents[1],
        env={**os.environ, "DATABASE_URL": TEST_DATABASE_URL},
        check=True,
        capture_output=True,
    )


@pytest_asyncio.fixture
async def sessions(migrated_database) -> AsyncGenerator[async_sessionmaker[AsyncSession], None]:
    """Session factory on migrated PostgreSQL, with empty tables for every test."""
    engine = create_async_engine(TEST_DATABASE_URL, poolclass=NullPool)
    async with engine.begin() as conn:
        await conn.execute(text("TRUNCATE organizations, users, verification_codes CASCADE"))
    yield async_sessionmaker(engine, expire_on_commit=False)
    await engine.dispose()


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
