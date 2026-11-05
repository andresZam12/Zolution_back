"""deduplicate WhatsApp messages and encrypt stored Google tokens

Revision ID: 20240112_000000
Revises: 20240111_000000
Create Date: 2026-11-06 10:00:00.000000
"""

from collections.abc import Callable, Sequence

from alembic import op
import sqlalchemy as sa

from app.core.crypto import decrypt_token, encrypt_token

revision: str = "20240112_000000"
down_revision: str | None = "20240111_000000"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

INDEX_NAME = "ix_messages_whatsapp_message_id"


def _rewrite_tokens(transform: Callable[[str], str]) -> None:
    """Rewrite each tenant's credentials. FORCE RLS requires setting the tenant first."""
    bind = op.get_bind()
    set_tenant = sa.text("SELECT set_config('app.current_organization_id', :org, true)")
    update = sa.text(
        "UPDATE integration_credentials SET access_token = :access, refresh_token = :refresh "
        "WHERE id = :id"
    )

    for organization_id in bind.execute(sa.text("SELECT id FROM organizations")).scalars().all():
        bind.execute(set_tenant, {"org": str(organization_id)})
        rows = bind.execute(
            sa.text("SELECT id, access_token, refresh_token FROM integration_credentials")
        ).all()
        for row in rows:
            bind.execute(
                update,
                {
                    "id": row.id,
                    "access": transform(row.access_token),
                    "refresh": transform(row.refresh_token) if row.refresh_token else None,
                },
            )
    bind.execute(set_tenant, {"org": ""})


def upgrade() -> None:
    op.add_column("messages", sa.Column("whatsapp_message_id", sa.String(255), nullable=True))
    op.create_index(INDEX_NAME, "messages", ["whatsapp_message_id"], unique=True)
    _rewrite_tokens(lambda token: encrypt_token(decrypt_token(token)))


def downgrade() -> None:
    _rewrite_tokens(decrypt_token)
    op.drop_index(INDEX_NAME, table_name="messages")
    op.drop_column("messages", "whatsapp_message_id")
