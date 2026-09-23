"""Initial database schema with RLS policies.

Creates all core tables for the Zolution multi-tenant platform and applies
Row-Level Security (RLS) policies to enforce tenant data isolation at the
database level (see ADR-002).

Tables created:
    organizations   — tenant registry
    users           — platform users (staff, owners, superadmins)
    agent_configs   — AI agent configuration per tenant
    conversations   — WhatsApp conversation threads
    messages        — individual message turns
    audit_log       — immutable superadmin action log

RLS policies are applied to all tables that contain tenant data.
The superadmin bypasses RLS via a separate DB connection (see database.py).

Revision ID: 20240101_000000_initial_schema
Revises: None
Create Date: 2024-01-01 00:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers
revision: str = "20240101_000000"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # -------------------------------------------------------------------------
    # Enable pgcrypto for gen_random_uuid() (requires PostgreSQL 13+, or
    # use the built-in gen_random_uuid() from PostgreSQL 14+)
    # -------------------------------------------------------------------------
    op.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto")

    # -------------------------------------------------------------------------
    # organizations — tenant registry (no RLS: public-ish registry)
    # -------------------------------------------------------------------------
    op.create_table(
        "organizations",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("slug", sa.String(100), nullable=False, unique=True),
        sa.Column("plan", sa.String(50), nullable=False, server_default="free"),
        sa.Column("status", sa.String(50), nullable=False, server_default="pending_setup"),
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
    )
    op.create_index("ix_organizations_slug", "organizations", ["slug"], unique=True)

    # -------------------------------------------------------------------------
    # users
    # -------------------------------------------------------------------------
    op.create_table(
        "users",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("auth0_id", sa.String(255), nullable=False, unique=True),
        sa.Column("email", sa.String(255), nullable=False, unique=True),
        sa.Column("role", sa.String(50), nullable=False),
        sa.Column("is_active", sa.Boolean, nullable=False, server_default=sa.text("TRUE")),
        sa.Column(
            "organization_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("organizations.id", ondelete="CASCADE"),
            nullable=True,
        ),
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
    )
    op.create_index("ix_users_auth0_id", "users", ["auth0_id"], unique=True)
    op.create_index("ix_users_email", "users", ["email"], unique=True)
    op.create_index("ix_users_organization_id", "users", ["organization_id"])

    # -------------------------------------------------------------------------
    # agent_configs
    # -------------------------------------------------------------------------
    op.create_table(
        "agent_configs",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "organization_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("organizations.id", ondelete="CASCADE"),
            nullable=False,
            unique=True,
        ),
        sa.Column("system_prompt_generated", sa.Text, nullable=True),
        sa.Column("onboarding_answers", postgresql.JSON, nullable=True),
        sa.Column("llm_provider", sa.String(50), nullable=False, server_default="anthropic"),
        sa.Column("llm_model", sa.String(100), nullable=False, server_default=""),
        sa.Column("status", sa.String(50), nullable=False, server_default="draft"),
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
    )
    op.create_index("ix_agent_configs_organization_id", "agent_configs", ["organization_id"], unique=True)

    # -------------------------------------------------------------------------
    # conversations
    # -------------------------------------------------------------------------
    op.create_table(
        "conversations",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "organization_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("organizations.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("customer_phone", sa.String(20), nullable=False),
        sa.Column("customer_name", sa.String(255), nullable=True),
        sa.Column("status", sa.String(50), nullable=False, server_default="active"),
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
    )
    op.create_index("ix_conversations_organization_id", "conversations", ["organization_id"])
    op.create_index("ix_conversations_customer_phone", "conversations", ["customer_phone"])

    # -------------------------------------------------------------------------
    # messages
    # -------------------------------------------------------------------------
    op.create_table(
        "messages",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "conversation_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("conversations.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "organization_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("organizations.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("role", sa.String(20), nullable=False),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("tokens_in", sa.Integer, nullable=True),
        sa.Column("tokens_out", sa.Integer, nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("NOW()"),
            nullable=False,
        ),
    )
    op.create_index("ix_messages_conversation_id", "messages", ["conversation_id"])
    op.create_index("ix_messages_organization_id", "messages", ["organization_id"])

    # -------------------------------------------------------------------------
    # audit_log
    # -------------------------------------------------------------------------
    op.create_table(
        "audit_log",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "actor_user_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="SET NULL"),
            nullable=False,
        ),
        sa.Column(
            "organization_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("organizations.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("action", sa.String(100), nullable=False),
        sa.Column("resource_type", sa.String(100), nullable=True),
        sa.Column("resource_id", sa.String(255), nullable=True),
        sa.Column("metadata", postgresql.JSON, nullable=True),
        sa.Column("ip_address", sa.String(45), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("NOW()"),
            nullable=False,
        ),
    )
    op.create_index("ix_audit_log_actor_user_id", "audit_log", ["actor_user_id"])
    op.create_index("ix_audit_log_organization_id", "audit_log", ["organization_id"])
    op.create_index("ix_audit_log_action", "audit_log", ["action"])
    op.create_index("ix_audit_log_created_at", "audit_log", ["created_at"])

    # =========================================================================
    # Row-Level Security (RLS) policies
    #
    # Each RLS policy checks the current session variable
    # `app.current_organization_id` (set by TenantContextMiddleware) against
    # the row's organization_id.
    #
    # The superadmin connection uses `SET LOCAL row_security = off` to bypass
    # these policies (see app/core/database.py :: get_superadmin_db_session).
    # =========================================================================

    rls_tables = ["users", "agent_configs", "conversations", "messages", "audit_log"]

    for table in rls_tables:
        # Enable RLS on the table
        op.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")

        # Force RLS even for the table owner (defense in depth)
        op.execute(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY")

        # SELECT / UPDATE / DELETE policy — tenant can only see their own rows
        op.execute(f"""
            CREATE POLICY tenant_isolation ON {table}
            USING (
                organization_id = current_setting('app.current_organization_id', TRUE)::uuid
            )
        """)

    # INSERT policy for tables where tenants create rows
    # (audit_log is insert-only from the admin service — handled separately)
    for table in ["users", "agent_configs", "conversations", "messages"]:
        op.execute(f"""
            CREATE POLICY tenant_insert ON {table}
            FOR INSERT
            WITH CHECK (
                organization_id = current_setting('app.current_organization_id', TRUE)::uuid
            )
        """)

    # -------------------------------------------------------------------------
    # Updated_at trigger function (shared across tables)
    # -------------------------------------------------------------------------
    op.execute("""
        CREATE OR REPLACE FUNCTION set_updated_at()
        RETURNS TRIGGER AS $$
        BEGIN
            NEW.updated_at = NOW();
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
    """)

    for table in ["organizations", "users", "agent_configs", "conversations"]:
        op.execute(f"""
            CREATE TRIGGER trigger_{table}_updated_at
            BEFORE UPDATE ON {table}
            FOR EACH ROW EXECUTE FUNCTION set_updated_at();
        """)


def downgrade() -> None:
    # Drop triggers first
    for table in ["organizations", "users", "agent_configs", "conversations"]:
        op.execute(f"DROP TRIGGER IF EXISTS trigger_{table}_updated_at ON {table}")

    op.execute("DROP FUNCTION IF EXISTS set_updated_at()")

    # Drop RLS policies
    rls_tables = ["users", "agent_configs", "conversations", "messages", "audit_log"]
    for table in rls_tables:
        op.execute(f"DROP POLICY IF EXISTS tenant_isolation ON {table}")
        op.execute(f"DROP POLICY IF EXISTS tenant_insert ON {table}")
        op.execute(f"ALTER TABLE {table} DISABLE ROW LEVEL SECURITY")

    # Drop tables in reverse dependency order
    op.drop_table("audit_log")
    op.drop_table("messages")
    op.drop_table("conversations")
    op.drop_table("agent_configs")
    op.drop_table("users")
    op.drop_table("organizations")
