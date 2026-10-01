"""
Pydantic schemas for the tenant (Organization) domain.

These schemas define the HTTP contract for the organizations API.
They are intentionally separate from the ORM models in models.py —
the ORM model represents the database row, the schema represents
what enters and leaves the API.

Naming convention:
    <Entity>Create   — body for POST (creation)
    <Entity>Update   — body for PATCH (partial update)
    <Entity>Response — shape returned by any endpoint
"""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------

class OrganizationCreate(BaseModel):
    """Body for POST /organizations (superadmin creates a new tenant)."""

    name: str = Field(
        min_length=2,
        max_length=255,
        description="Display name of the business.",
        examples=["Clínica Dental Sonríe Bien"],
    )
    slug: str = Field(
        min_length=2,
        max_length=100,
        pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$",
        description="URL-safe identifier. Lowercase letters, numbers, hyphens only.",
        examples=["clinica-dental-sonrie-bien"],
    )
    plan: str = Field(
        default="free",
        description="Subscription plan tier.",
        examples=["free", "starter", "professional"],
    )

    @field_validator("slug")
    @classmethod
    def slug_must_be_lowercase(cls, v: str) -> str:
        return v.lower()


class OrganizationUpdate(BaseModel):
    """Body for PATCH /organizations/{id}. All fields optional."""

    name: str | None = Field(
        default=None,
        min_length=2,
        max_length=255,
    )
    slug: str | None = Field(
        default=None,
        min_length=2,
        max_length=100,
        pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$",
    )

    @field_validator("slug")
    @classmethod
    def slug_must_be_lowercase(cls, v: str | None) -> str | None:
        return v.lower() if v else v


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------

class OrganizationResponse(BaseModel):
    """Organization resource as returned by the API."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    name: str
    slug: str
    plan: str
    status: str
    created_at: datetime
    updated_at: datetime


class OrganizationListResponse(BaseModel):
    """Paginated list of organizations (superadmin endpoint)."""

    items: list[OrganizationResponse]
    total: int
    page: int
    page_size: int
    has_next: bool
