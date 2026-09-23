"""
Alembic environment configuration for async SQLAlchemy migrations.

Key design decisions:
- Uses `run_async_migrations()` with `AsyncEngine` for asyncpg compatibility
- Imports `Base.metadata` for `--autogenerate` support
- DATABASE_URL is read from `app.core.config.Settings` — same source as the app
- Both online (connected) and offline (SQL script generation) modes supported

Usage:
    cd backend/
    alembic upgrade head          # Apply all pending migrations
    alembic revision --autogenerate -m "add_some_table"  # Generate new migration
    alembic downgrade -1          # Roll back one migration
    alembic history               # List migration history
"""

import asyncio
from logging.config import fileConfig

from alembic import context
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

# ---------------------------------------------------------------------------
# Import application Base and Settings
# ---------------------------------------------------------------------------

# Import Base so Alembic can detect model changes for autogenerate.
# All ORM models must be imported (directly or transitively) before this
# point so their metadata is registered on Base.
from app.core.config import get_settings
from app.core.database import Base  # noqa: F401

# Import all models so their tables are registered on Base.metadata.
# Add new model modules here as they are created.
# fmt: off
from app.tenants import models as _tenant_models          # noqa: F401
from app.agents import models as _agent_models            # noqa: F401
from app.integrations import models as _integration_models  # noqa: F401
from app.admin import models as _admin_models             # noqa: F401
# fmt: on

# ---------------------------------------------------------------------------
# Alembic configuration
# ---------------------------------------------------------------------------

alembic_config = context.config
settings = get_settings()

# Override the sqlalchemy.url from alembic.ini with the value from Settings.
# This ensures credentials always come from the environment, never from ini.
alembic_config.set_main_option("sqlalchemy.url", settings.DATABASE_URL)

# Configure Python logging from alembic.ini [loggers] section
if alembic_config.config_file_name is not None:
    fileConfig(alembic_config.config_file_name)

# Metadata object for autogenerate support
target_metadata = Base.metadata


# ---------------------------------------------------------------------------
# Offline mode — generate SQL script without a live DB connection
# ---------------------------------------------------------------------------

def run_migrations_offline() -> None:
    """
    Run migrations in 'offline' mode.

    Generates SQL statements to stdout/file without connecting to the DB.
    Useful for reviewing migrations before applying them or for DBAs who
    apply migrations manually.
    """
    url = alembic_config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )

    with context.begin_transaction():
        context.run_migrations()


# ---------------------------------------------------------------------------
# Online mode — connect to the DB and apply migrations
# ---------------------------------------------------------------------------

def do_run_migrations(connection: Connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,         # Detect column type changes in autogenerate
        compare_server_default=True,  # Detect server default changes
    )
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """Run migrations using an async engine (required for asyncpg)."""
    connectable = async_engine_from_config(
        alembic_config.get_section(alembic_config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,  # No pooling during migrations
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    """Entry point for online migrations — runs the async event loop."""
    asyncio.run(run_async_migrations())


# ---------------------------------------------------------------------------
# Entry point — called by Alembic CLI
# ---------------------------------------------------------------------------

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
