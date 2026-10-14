"""add user auth fields and verification_codes table

Revision ID: 20240104_000000
Revises: 20240103_000000
Create Date: 2026-10-14 18:00:00.000000

"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "20240104_000000"
down_revision: str | None = "20240103_000000"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # 1. Update users table with native auth and Google login fields
    op.alter_column("users", "auth0_id", nullable=True)
    op.add_column("users", sa.Column("password_hash", sa.String(length=255), nullable=True))
    op.add_column("users", sa.Column("google_id", sa.String(length=255), nullable=True))
    op.add_column(
        "users",
        sa.Column("is_verified", sa.Boolean(), server_default=sa.text("FALSE"), nullable=False),
    )
    op.create_index(op.f("ix_users_google_id"), "users", ["google_id"], unique=True)

    # 2. Create verification_codes table for OTP delivery
    op.create_table(
        "verification_codes",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("email", sa.String(length=255), nullable=False),
        sa.Column("code", sa.String(length=10), nullable=False),
        sa.Column("purpose", sa.String(length=50), server_default="registration", nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("is_used", sa.Boolean(), server_default=sa.text("FALSE"), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("NOW()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_verification_codes_email"),
        "verification_codes",
        ["email"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_verification_codes_email"), table_name="verification_codes")
    op.drop_table("verification_codes")

    op.drop_index(op.f("ix_users_google_id"), table_name="users")
    op.drop_column("users", "is_verified")
    op.drop_column("users", "google_id")
    op.drop_column("users", "password_hash")
    op.alter_column("users", "auth0_id", nullable=False)
