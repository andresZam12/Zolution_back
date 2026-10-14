"""
Tenant domain ORM models: Organization and User.

These are the root entities of the multi-tenant data model (see ADR-002).
Every other table in the system references `organizations.id` via an
`organization_id` FK and is protected by an RLS policy.

Model conventions followed throughout the codebase:
- All PKs are UUIDs (uuid4, server-default `gen_random_uuid()`)
- All timestamps are `TIMESTAMP WITH TIME ZONE` stored as UTC
- `created_at` is server-defaulted; `updated_at` uses an `onupdate` trigger
- Every multi-tenant table has `organization_id` as a non-nullable FK
  (except `users`, where it is nullable for the superadmin)
"""

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    # Imported only for type annotations — avoids circular import at runtime.
    # AgentConfig is defined in app.agents.models and references Organization.
    from app.agents.models import AgentConfig

from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    String,
    func,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


class Organization(Base):
    """
    Represents a business tenant on the platform.

    Each organization has exactly one agent configuration and can have
    multiple users (staff) associated with it.
    """

    __tablename__ = "organizations"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=func.gen_random_uuid(),
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    # URL-safe identifier used in routes and display (e.g. "dra-garcia-dental")
    slug: Mapped[str] = mapped_column(String(100), nullable=False, unique=True, index=True)
    # Subscription plan: free | starter | professional | enterprise
    plan: Mapped[str] = mapped_column(String(50), nullable=False, default="free")
    # Lifecycle status: active | suspended | pending_setup | deleted
    status: Mapped[str] = mapped_column(String(50), nullable=False, default="pending_setup")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    # Relationships
    users: Mapped[list["User"]] = relationship("User", back_populates="organization")
    agent_config: Mapped[Optional["AgentConfig"]] = relationship(
        "AgentConfig", back_populates="organization", uselist=False
    )

    def __repr__(self) -> str:
        return f"<Organization id={self.id} slug={self.slug!r}>"


class User(Base):
    """
    Platform user — can be a business owner, staff member, or superadmin.

    `organization_id` is nullable: superadmin users have no organization.
    The `role` column maps directly to `UserRole` in app/auth/models.py.
    """

    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=func.gen_random_uuid(),
    )
    # Auth0 subject identifier — nullable for native email / Google logins
    auth0_id: Mapped[str | None] = mapped_column(String(255), nullable=True, unique=True, index=True)
    google_id: Mapped[str | None] = mapped_column(String(255), nullable=True, unique=True, index=True)
    email: Mapped[str] = mapped_column(String(255), nullable=False, unique=True, index=True)
    password_hash: Mapped[str | None] = mapped_column(String(255), nullable=True)
    # owner | staff | superadmin — matches UserRole enum
    role: Mapped[str] = mapped_column(String(50), nullable=False, default="owner")
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    is_verified: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    organization_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    # Relationships
    organization: Mapped[Optional["Organization"]] = relationship(
        "Organization", back_populates="users"
    )

    def __repr__(self) -> str:
        return f"<User id={self.id} email={self.email!r} role={self.role!r}>"


class VerificationCode(Base):
    """
    Temporary one-time verification codes (OTP) sent to email for authentication
    and user verification.
    """

    __tablename__ = "verification_codes"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=func.gen_random_uuid(),
    )
    email: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    code: Mapped[str] = mapped_column(String(10), nullable=False)
    purpose: Mapped[str] = mapped_column(String(50), nullable=False, default="registration")
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    is_used: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    def __repr__(self) -> str:
        return f"<VerificationCode email={self.email!r} code={self.code!r} used={self.is_used}>"
