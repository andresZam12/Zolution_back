"""
Tenant service layer — business logic for Organization management.

This module is the ONLY place where organization business logic lives.
Routes call this service; this service calls the ORM. No SQL in routes.
No HTTP concepts (status codes, Request objects) here.

Design notes:
- All methods are async and accept an `AsyncSession` parameter.
  This makes them trivially testable without HTTP (pass a mock session).
- Raises `HTTPException` for domain-level errors (not found, conflict).
  This is a pragmatic choice for a FastAPI app — keeps the service layer
  thin while keeping error semantics clear. In a larger system, raise
  domain exceptions and translate to HTTP in an error handler.
- Slug uniqueness is enforced at the DB level (unique index) AND here
  (check before insert) to give a clear error message instead of a PG error.
"""

import logging
from uuid import UUID

from fastapi import HTTPException, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.tenants.models import Organization
from app.tenants.schemas import OrganizationCreate, OrganizationListResponse, OrganizationResponse, OrganizationUpdate

logger = logging.getLogger(__name__)


class OrganizationService:
    """
    Business logic for Organization (tenant) management.

    Instantiate per-request via FastAPI's dependency injection:

        service = OrganizationService()
        org = await service.get_by_id(org_id, db)
    """

    async def get_by_id(
        self,
        org_id: UUID,
        db: AsyncSession,
    ) -> Organization:
        """
        Fetch an organization by its primary key.

        Raises:
            HTTPException 404: Organization not found.
        """
        result = await db.execute(
            select(Organization).where(Organization.id == org_id)
        )
        org = result.scalar_one_or_none()
        if org is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Organization {org_id} not found.",
            )
        return org

    async def get_for_current_user(
        self,
        organization_id: UUID | None,
        db: AsyncSession,
    ) -> Organization:
        """
        Fetch the organization that the authenticated user belongs to.

        Args:
            organization_id: From `UserContext.organization_id`.

        Raises:
            HTTPException 403: User has no organization (superadmin without org context).
            HTTPException 404: Organization not found (data integrity issue).
        """
        if organization_id is None:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="This endpoint requires an organization context.",
            )
        return await self.get_by_id(organization_id, db)

    async def create(
        self,
        data: OrganizationCreate,
        db: AsyncSession,
    ) -> Organization:
        """
        Create a new organization (tenant).

        Also creates an initial AgentConfig with status='draft' so the
        organization is immediately ready for the onboarding flow.

        Raises:
            HTTPException 409: Slug already taken.
        """
        # Check slug uniqueness before insert to provide a clear error
        existing = await db.execute(
            select(Organization).where(Organization.slug == data.slug)
        )
        if existing.scalar_one_or_none() is not None:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Slug '{data.slug}' is already taken. Choose a different slug.",
            )

        org = Organization(
            name=data.name,
            slug=data.slug,
            plan=data.plan,
            status="pending_setup",
        )
        db.add(org)
        await db.flush()  # Flush to get the generated ID without committing

        # Create the initial agent config so onboarding can start immediately
        from app.agents.models import AgentConfig  # noqa: PLC0415 — avoid circular import

        agent_config = AgentConfig(
            organization_id=org.id,
            status="draft",
            llm_provider="anthropic",
            llm_model="claude-haiku-4-5",
        )
        db.add(agent_config)
        # Commit is handled by get_db_session() dependency

        logger.info("Created organization: id=%s slug=%s", org.id, org.slug)
        return org

    async def update(
        self,
        org_id: UUID,
        data: OrganizationUpdate,
        db: AsyncSession,
    ) -> Organization:
        """
        Partially update an organization's mutable fields.

        Only fields explicitly set in `data` (non-None) are updated.

        Raises:
            HTTPException 404: Organization not found.
            HTTPException 409: New slug already taken by another org.
        """
        org = await self.get_by_id(org_id, db)

        if data.slug is not None and data.slug != org.slug:
            existing = await db.execute(
                select(Organization).where(
                    Organization.slug == data.slug,
                    Organization.id != org_id,
                )
            )
            if existing.scalar_one_or_none() is not None:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail=f"Slug '{data.slug}' is already taken.",
                )
            org.slug = data.slug

        if data.name is not None:
            org.name = data.name

        db.add(org)
        logger.info("Updated organization: id=%s", org_id)
        return org

    async def set_status(
        self,
        org_id: UUID,
        new_status: str,
        db: AsyncSession,
    ) -> Organization:
        """
        Set organization lifecycle status.

        Valid values: 'active' | 'suspended' | 'pending_setup'

        Raises:
            HTTPException 404: Organization not found.
        """
        valid_statuses = {"active", "suspended", "pending_setup"}
        if new_status not in valid_statuses:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid status '{new_status}'. Must be one of: {sorted(valid_statuses)}",
            )

        org = await self.get_by_id(org_id, db)
        org.status = new_status
        db.add(org)
        logger.info("Organization %s status → %s", org_id, new_status)
        return org

    async def list_all(
        self,
        db: AsyncSession,
        page: int = 1,
        page_size: int = 20,
    ) -> OrganizationListResponse:
        """
        List all organizations (superadmin only).

        Args:
            page:      1-based page number.
            page_size: Items per page (max 100).

        Returns:
            Paginated OrganizationListResponse.
        """
        page_size = min(page_size, 100)
        offset = (page - 1) * page_size

        total_result = await db.execute(select(func.count(Organization.id)))
        total = total_result.scalar_one()

        orgs_result = await db.execute(
            select(Organization)
            .order_by(Organization.created_at.desc())
            .offset(offset)
            .limit(page_size)
        )
        orgs = list(orgs_result.scalars().all())

        return OrganizationListResponse(
            items=[OrganizationResponse.model_validate(o) for o in orgs],
            total=total,
            page=page,
            page_size=page_size,
            has_next=(offset + page_size) < total,
        )
