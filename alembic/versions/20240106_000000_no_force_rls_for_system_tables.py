"""no force rls for system tables so webhook and table owner can operate

Revision ID: 20240106_000000
Revises: 20240105_000000
Create Date: 2026-11-03 21:30:00.000000
"""

from collections.abc import Sequence

from alembic import op

revision: str = "20240106_000000"
down_revision: str | None = "20240105_000000"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Allow table owner to operate on tables without getting InsufficientPrivilegeError,
    # while standard RLS policies still isolate tenants.
    for table in ["agent_configs", "conversations", "messages", "integration_credentials"]:
        op.execute(f"ALTER TABLE {table} NO FORCE ROW LEVEL SECURITY")


def downgrade() -> None:
    for table in ["agent_configs", "conversations", "messages", "integration_credentials"]:
        op.execute(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY")

