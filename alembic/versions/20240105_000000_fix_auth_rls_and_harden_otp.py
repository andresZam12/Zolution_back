"""fix auth: global users identity table and hardened OTP codes

Revision ID: 20240105_000000
Revises: 20240104_000000
Create Date: 2026-10-14 22:30:00.000000

Why:
- `users` had FORCE ROW LEVEL SECURITY bound to `app.current_organization_id`.
  Login and registration must look a user up by email BEFORE any tenant
  context exists, so every lookup returned nothing and every insert was
  rejected (500 -> "Failed to fetch" in the browser).
  `users` is the global identity table (like `organizations`), so RLS is
  removed here. Tenant isolation for business data (agent_configs,
  conversations, messages, integration_credentials) stays untouched.
- `verification_codes.code` now stores an HMAC-SHA256 hash (64 hex chars)
  instead of the plain code, and `attempts` limits brute-force guessing.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20240105_000000"
down_revision: str | None = "20240104_000000"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # 1. users: global identity table — no tenant RLS
    op.execute("DROP POLICY IF EXISTS tenant_isolation ON users")
    op.execute("DROP POLICY IF EXISTS tenant_insert ON users")
    op.execute("ALTER TABLE users NO FORCE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE users DISABLE ROW LEVEL SECURITY")

    # 2. verification_codes: invalidate old plain-text codes, store hashes, count attempts
    op.execute("UPDATE verification_codes SET is_used = TRUE WHERE is_used = FALSE")
    op.alter_column(
        "verification_codes",
        "code",
        type_=sa.String(length=64),
        existing_type=sa.String(length=10),
        existing_nullable=False,
    )
    op.add_column(
        "verification_codes",
        sa.Column("attempts", sa.Integer(), server_default=sa.text("0"), nullable=False),
    )


def downgrade() -> None:
    op.drop_column("verification_codes", "attempts")
    op.execute("DELETE FROM verification_codes")
    op.alter_column(
        "verification_codes",
        "code",
        type_=sa.String(length=10),
        existing_type=sa.String(length=64),
        existing_nullable=False,
    )

    op.execute("ALTER TABLE users ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE users FORCE ROW LEVEL SECURITY")
    op.execute("""
        CREATE POLICY tenant_isolation ON users
        USING (organization_id = current_setting('app.current_organization_id', TRUE)::uuid)
    """)
    op.execute("""
        CREATE POLICY tenant_insert ON users
        FOR INSERT
        WITH CHECK (organization_id = current_setting('app.current_organization_id', TRUE)::uuid)
    """)
