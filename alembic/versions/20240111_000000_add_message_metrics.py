"""add response metrics to messages

Only replies sent to real WhatsApp customers are measured. Older messages and
simulator replies keep NULL, so they never affect the dashboard metrics.

Revision ID: 20240111_000000
Revises: 20240110_000000
Create Date: 2026-11-05 20:00:00.000000
"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa

revision: str = "20240111_000000"
down_revision: str | None = "20240110_000000"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("messages", sa.Column("response_time_ms", sa.Integer(), nullable=True))
    op.add_column("messages", sa.Column("resolved", sa.Boolean(), nullable=True))


def downgrade() -> None:
    op.drop_column("messages", "resolved")
    op.drop_column("messages", "response_time_ms")
