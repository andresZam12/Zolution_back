"""
Admin domain ORM models: AuditLog.

Every action performed by a superadmin (impersonation, account suspension,
data access) must be recorded here. This is a compliance and security
requirement, not an optional feature.

The `audit_log` table is NOT protected by RLS for tenant staff — staff
users can read their own org's audit entries (filtered by organization_id).
Superadmins can read all entries via the service connection.
"""

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import DateTime, ForeignKey, String, func
from sqlalchemy.dialects.postgresql import JSON, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class AuditLog(Base):
    """
    Immutable record of a privileged or notable action in the system.

    Design decisions:
    - No `updated_at` — audit entries are append-only, never modified.
    - `organization_id` is nullable: superadmin actions not tied to a
      specific tenant have null organization_id.
    - `metadata` JSONB holds action-specific context (e.g. which tenant
      was impersonated, what fields were changed).
    - `ip_address` is stored for security investigations.
    """

    __tablename__ = "audit_log"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=func.gen_random_uuid(),
    )
    # The user who performed the action (FK to users.id)
    actor_user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=False,
        index=True,
    )
    # The organization context of the action (null for platform-wide actions)
    organization_id: Mapped[Optional[uuid.UUID]] = mapped_column(  # noqa: UP007
        UUID(as_uuid=True),
        ForeignKey("organizations.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    # Human-readable action name, e.g. "impersonate_tenant", "suspend_account"
    action: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    # The type of resource acted upon, e.g. "organization", "user", "agent_config"
    resource_type: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)  # noqa: UP007
    # The ID of the specific resource (as string to handle different ID types)
    resource_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)  # noqa: UP007
    # Flexible context: previous values, new values, reason, etc.
    metadata_: Mapped[Optional[dict]] = mapped_column(  # noqa: UP007
        "metadata", JSON, nullable=True
    )
    # Requester's IP address for security audit trail
    ip_address: Mapped[Optional[str]] = mapped_column(String(45), nullable=True)  # noqa: UP007
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False, index=True
    )

    def __repr__(self) -> str:
        return (
            f"<AuditLog action={self.action!r} actor={self.actor_user_id} "
            f"org={self.organization_id}>"
        )
