"""
Auth domain models.

Defines the UserContext dataclass that represents an authenticated user
throughout the application. All code outside of app/auth/ works with
UserContext — never with raw JWT payloads or Auth0-specific types.
This is the isolation boundary described in ADR-003.
"""

from dataclasses import dataclass
from enum import StrEnum
from uuid import UUID


class UserRole(StrEnum):
    """
    Application-level user roles.

    Maps to the `role` claim in the Auth0 JWT (set via Auth0 Actions).
    """

    OWNER = "owner"
    STAFF = "staff"
    SUPERADMIN = "superadmin"


@dataclass(frozen=True)
class UserContext:
    """
    Immutable representation of an authenticated, verified user.

    This object is injected into route handlers via FastAPI's dependency
    injection system (see dependencies.py). It is the single source of
    truth for "who is making this request" throughout the application.

    Attributes:
        user_id:         Internal UUID from the `users` table.
        auth0_id:        Auth0 subject identifier (e.g. "auth0|abc123").
        email:           Verified email from the JWT.
        role:            Application role (owner, staff, superadmin).
        organization_id: UUID of the tenant this user belongs to.
                         None for superadmins.
        is_superadmin:   Convenience flag to avoid role string comparisons.
    """

    user_id: UUID
    auth0_id: str
    email: str
    role: UserRole
    organization_id: UUID | None

    @property
    def is_superadmin(self) -> bool:
        """Return True if the user is a platform superadmin."""
        return self.role == UserRole.SUPERADMIN

    def __str__(self) -> str:
        return f"UserContext(email={self.email}, role={self.role}, org={self.organization_id})"
