"""enforce row level security for the application role

The application role owns the tenant tables, so RLS only applies to it with
FORCE. Policies now tolerate an empty tenant variable: PostgreSQL leaves it as
'' after a transaction-local set_config, and ''::uuid raises instead of
matching nothing.

Data migrations on these tables must set `app.current_organization_id`, because
FORCE applies to the role running Alembic as well.

Revision ID: 20240109_000000
Revises: 20240108_000000
Create Date: 2026-11-05 12:00:00.000000
"""

from collections.abc import Sequence

from alembic import op

revision: str = "20240109_000000"
down_revision: str | None = "20240108_000000"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

FORCED_TABLES = ["agent_configs", "conversations", "messages", "integration_credentials"]
INSERT_POLICY_TABLES = FORCED_TABLES
TENANT_TABLES = [*FORCED_TABLES, "audit_log"]

CURRENT_TENANT = "NULLIF(current_setting('app.current_organization_id', TRUE), '')::uuid"
LEGACY_CURRENT_TENANT = "current_setting('app.current_organization_id', TRUE)::uuid"


def _replace_policies(tenant_expression: str) -> None:
    for table in TENANT_TABLES:
        op.execute(f"DROP POLICY IF EXISTS tenant_isolation ON {table}")
        op.execute(f"""
            CREATE POLICY tenant_isolation ON {table}
            USING (organization_id = {tenant_expression})
        """)
    for table in INSERT_POLICY_TABLES:
        op.execute(f"DROP POLICY IF EXISTS tenant_insert ON {table}")
        op.execute(f"""
            CREATE POLICY tenant_insert ON {table}
            FOR INSERT
            WITH CHECK (organization_id = {tenant_expression})
        """)


def upgrade() -> None:
    _replace_policies(CURRENT_TENANT)

    # The webhook must find the tenant from the receiving number before any
    # tenant is known. This grants read access to that one matching row only.
    op.execute("""
        CREATE POLICY whatsapp_routing ON agent_configs
        FOR SELECT
        USING (
            whatsapp_phone_number_id =
                NULLIF(current_setting('app.routing_phone_number_id', TRUE), '')
        )
    """)

    for table in FORCED_TABLES:
        op.execute(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY")


def downgrade() -> None:
    for table in FORCED_TABLES:
        op.execute(f"ALTER TABLE {table} NO FORCE ROW LEVEL SECURITY")

    op.execute("DROP POLICY IF EXISTS whatsapp_routing ON agent_configs")
    _replace_policies(LEGACY_CURRENT_TENANT)
