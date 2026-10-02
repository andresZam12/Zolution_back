"""add integration_credentials table

Revision ID: 20240103_000000
Revises: 20240102_000000
Create Date: 2026-10-02 17:03:00.000000

"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "20240103_000000"
down_revision: str | None = "20240102_000000"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "integration_credentials",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("organization_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("provider", sa.String(length=50), nullable=False),
        sa.Column("access_token", sa.Text(), nullable=False),
        sa.Column("refresh_token", sa.Text(), nullable=True),
        sa.Column("scopes", sa.Text(), nullable=True),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("NOW()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("NOW()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["organization_id"], ["organizations.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_integration_credentials_organization_id"),
        "integration_credentials",
        ["organization_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_integration_credentials_provider"),
        "integration_credentials",
        ["provider"],
        unique=False,
    )

    # RLS Policies
    op.execute("ALTER TABLE integration_credentials ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE integration_credentials FORCE ROW LEVEL SECURITY")
    op.execute("""
        CREATE POLICY tenant_isolation ON integration_credentials
        USING (
            organization_id = current_setting('app.current_organization_id', TRUE)::uuid
        )
    """)
    op.execute("""
        CREATE POLICY tenant_insert ON integration_credentials
        FOR INSERT
        WITH CHECK (
            organization_id = current_setting('app.current_organization_id', TRUE)::uuid
        )
    """)

    # Trigger for updated_at
    op.execute("""
        CREATE TRIGGER trigger_integration_credentials_updated_at
        BEFORE UPDATE ON integration_credentials
        FOR EACH ROW EXECUTE FUNCTION set_updated_at();
    """)


def downgrade() -> None:
    op.execute(
        "DROP TRIGGER IF EXISTS trigger_integration_credentials_updated_at ON integration_credentials"
    )
    op.execute("DROP POLICY IF EXISTS tenant_isolation ON integration_credentials")
    op.execute("DROP POLICY IF EXISTS tenant_insert ON integration_credentials")
    op.execute("ALTER TABLE integration_credentials DISABLE ROW LEVEL SECURITY")

    op.drop_index(op.f("ix_integration_credentials_provider"), table_name="integration_credentials")
    op.drop_index(
        op.f("ix_integration_credentials_organization_id"), table_name="integration_credentials"
    )
    op.drop_table("integration_credentials")
