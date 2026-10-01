"""add whatsapp_phone_number_id to agent_config

Revision ID: 20240102_000000
Revises: 20240101_000000_initial_schema
Create Date: 2026-10-01 17:18:00.000000

"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "20240102_000000"
down_revision: str | None = "20240101_000000"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "agent_configs", sa.Column("whatsapp_phone_number_id", sa.String(length=100), nullable=True)
    )
    op.create_index(
        op.f("ix_agent_configs_whatsapp_phone_number_id"),
        "agent_configs",
        ["whatsapp_phone_number_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_agent_configs_whatsapp_phone_number_id"), table_name="agent_configs")
    op.drop_column("agent_configs", "whatsapp_phone_number_id")
