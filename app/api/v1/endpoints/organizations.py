"""
Organizations API endpoints.

Route responsibilities:
- Parse and validate incoming requests (delegated to Pydantic schemas)
- Verify authentication and authorization (delegated to auth dependencies)
- Call the service layer with validated data
- Return serialized responses

Routes must NOT contain business logic. Keep them thin.

Endpoint summary:
    GET  /organizations/me          → Current user's organization (owner/staff)
    PATCH /organizations/{id}       → Update org fields (owner only)
    GET  /organizations             → List all orgs, paginated (superadmin)
    POST /organizations             → Create new org (superadmin)
    POST /organizations/{id}/suspend   → Suspend org (superadmin)
    POST /organizations/{id}/activate  → Activate org (superadmin)
"""

import logging
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.dependencies import get_current_user, require_superadmin
from app.auth.models import UserContext, UserRole
from app.core.database import get_db_session
from app.tenants.schemas import (
    OrganizationCreate,
    OrganizationListResponse,
    OrganizationResponse,
    OrganizationUpdate,
)
from app.tenants.services import OrganizationService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/organizations", tags=["Organizations"])

# Shared service instance (stateless — safe to share)
_service = OrganizationService()


# ---------------------------------------------------------------------------
# Owner / Staff endpoints
# ---------------------------------------------------------------------------


@router.get(
    "/me",
    response_model=OrganizationResponse,
    summary="Get current user's organization",
    description="Returns the organization that the authenticated user belongs to.",
)
async def get_my_organization(
    user: UserContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
) -> OrganizationResponse:
    org = await _service.get_for_current_user(user.organization_id, db)
    return OrganizationResponse.model_validate(org)


@router.patch(
    "/{org_id}",
    response_model=OrganizationResponse,
    summary="Update organization",
    description="Update mutable organization fields. Restricted to the organization owner.",
)
async def update_organization(
    org_id: UUID,
    data: OrganizationUpdate,
    user: UserContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
) -> OrganizationResponse:
    # Owners can only update their own org
    if not user.is_superadmin:
        if user.organization_id != org_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You can only update your own organization.",
            )
        if user.role != UserRole.OWNER:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only organization owners can update organization details.",
            )

    org = await _service.update(org_id, data, db)
    return OrganizationResponse.model_validate(org)


# ---------------------------------------------------------------------------
# Superadmin endpoints
# ---------------------------------------------------------------------------


@router.get(
    "",
    response_model=OrganizationListResponse,
    summary="List all organizations",
    description="Paginated list of all tenant organizations. Superadmin only.",
)
async def list_organizations(
    page: int = Query(default=1, ge=1, description="Page number (1-based)."),
    page_size: int = Query(default=20, ge=1, le=100, description="Items per page."),
    _user: UserContext = Depends(require_superadmin),
    db: AsyncSession = Depends(get_db_session),
) -> OrganizationListResponse:
    return await _service.list_all(db, page=page, page_size=page_size)


@router.post(
    "",
    response_model=OrganizationResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create organization",
    description="Create a new tenant organization. Superadmin only.",
)
async def create_organization(
    data: OrganizationCreate,
    _user: UserContext = Depends(require_superadmin),
    db: AsyncSession = Depends(get_db_session),
) -> OrganizationResponse:
    org = await _service.create(data, db)
    return OrganizationResponse.model_validate(org)


@router.post(
    "/{org_id}/suspend",
    response_model=OrganizationResponse,
    summary="Suspend organization",
    description="Set organization status to 'suspended'. Superadmin only.",
)
async def suspend_organization(
    org_id: UUID,
    _user: UserContext = Depends(require_superadmin),
    db: AsyncSession = Depends(get_db_session),
) -> OrganizationResponse:
    org = await _service.set_status(org_id, "suspended", db)
    return OrganizationResponse.model_validate(org)


@router.post(
    "/{org_id}/activate",
    response_model=OrganizationResponse,
    summary="Activate organization",
    description="Set organization status to 'active'. Superadmin only.",
)
async def activate_organization(
    org_id: UUID,
    _user: UserContext = Depends(require_superadmin),
    db: AsyncSession = Depends(get_db_session),
) -> OrganizationResponse:
    org = await _service.set_status(org_id, "active", db)
    return OrganizationResponse.model_validate(org)
