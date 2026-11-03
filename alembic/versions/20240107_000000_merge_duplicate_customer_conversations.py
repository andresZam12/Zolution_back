"""merge duplicate customer conversations

Revision ID: 20240107_000000
Revises: 20240106_000000
Create Date: 2026-11-03 23:05:00.000000
"""

from collections.abc import Sequence
from alembic import op

revision: str = "20240107_000000"
down_revision: str | None = "20240106_000000"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # 1. Merge messages from duplicate conversations for the same phone number into the oldest conversation
    op.execute("""
        DO $$
        DECLARE
            rec RECORD;
            master_id UUID;
        BEGIN
            FOR rec IN 
                SELECT regexp_replace(customer_phone, '\\D', '', 'g') AS clean_phone, organization_id
                FROM conversations
                GROUP BY regexp_replace(customer_phone, '\\D', '', 'g'), organization_id
                HAVING COUNT(*) > 1
            LOOP
                -- Get the master conversation (first created)
                SELECT id INTO master_id
                FROM conversations
                WHERE regexp_replace(customer_phone, '\\D', '', 'g') = rec.clean_phone 
                  AND organization_id = rec.organization_id
                ORDER BY created_at ASC
                LIMIT 1;

                -- Move all messages from other conversations to the master conversation
                UPDATE messages
                SET conversation_id = master_id
                WHERE conversation_id IN (
                    SELECT id FROM conversations
                    WHERE regexp_replace(customer_phone, '\\D', '', 'g') = rec.clean_phone 
                      AND organization_id = rec.organization_id
                      AND id != master_id
                );

                -- Delete the duplicate conversation rows
                DELETE FROM conversations
                WHERE regexp_replace(customer_phone, '\\D', '', 'g') = rec.clean_phone 
                  AND organization_id = rec.organization_id
                  AND id != master_id;
            END LOOP;
        END $$;
    """)


def downgrade() -> None:
    pass
