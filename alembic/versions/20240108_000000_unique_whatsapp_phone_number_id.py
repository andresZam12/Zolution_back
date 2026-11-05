"""one tenant per whatsapp phone number id

Revision ID: 20240108_000000
Revises: 20240107_000000
Create Date: 2026-11-04 12:00:00.000000
"""

from collections.abc import Sequence

from alembic import op

revision: str = "20240108_000000"
down_revision: str | None = "20240107_000000"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

INDEX_NAME = "ix_agent_configs_whatsapp_phone_number_id"


def upgrade() -> None:
    # A number can only route to one tenant: keep the most recently updated
    # binding and clear the rest so the unique index can be created.
    op.execute("""
        UPDATE agent_configs
        SET whatsapp_phone_number_id = NULL
        WHERE id IN (
            SELECT id FROM (
                SELECT id, ROW_NUMBER() OVER (
                    PARTITION BY whatsapp_phone_number_id
                    ORDER BY updated_at DESC, id
                ) AS position
                FROM agent_configs
                WHERE whatsapp_phone_number_id IS NOT NULL
            ) ranked
            WHERE position > 1
        )
    """)
    op.drop_index(INDEX_NAME, table_name="agent_configs")
    op.create_index(INDEX_NAME, "agent_configs", ["whatsapp_phone_number_id"], unique=True)


def downgrade() -> None:
    op.drop_index(INDEX_NAME, table_name="agent_configs")
    op.create_index(INDEX_NAME, "agent_configs", ["whatsapp_phone_number_id"], unique=False)
