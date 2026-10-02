"""
Integrations API endpoints.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user
from app.core.database import get_db_session
from app.integrations.google.auth import GoogleOAuthService
from app.integrations.models import IntegrationCredential
from app.integrations.schemas import (
    GoogleOAuthCallbackRequest,
    GoogleOAuthUrlResponse,
    IntegrationStatusResponse,
)

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/google/authorize", response_model=GoogleOAuthUrlResponse)
async def authorize_google(
    request: Request,
    user: dict = Depends(get_current_user),
):
    """
    Generate the Google OAuth2 authorization URL.
    The frontend should redirect the user to this URL.
    """
    organization_id = request.state.organization_id
    if not organization_id:
        raise HTTPException(status_code=400, detail="No organization context found")

    oauth_service = GoogleOAuthService()
    url = oauth_service.get_authorization_url(organization_id)
    return GoogleOAuthUrlResponse(auth_url=url)


@router.post("/google/callback")
async def google_callback(
    payload: GoogleOAuthCallbackRequest,
    request: Request,
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
):
    """
    Exchange the authorization code for tokens and save them.
    This is called by the frontend after Google redirects back to it.
    """
    organization_id = request.state.organization_id

    oauth_service = GoogleOAuthService()
    try:
        tokens = await oauth_service.exchange_code_for_tokens(payload.code)
    except Exception as e:
        logger.error(f"Failed to exchange Google OAuth code: {e}")
        raise HTTPException(status_code=400, detail="Failed to exchange authorization code") from e

    # Check if credentials already exist for this provider
    stmt = select(IntegrationCredential).where(
        IntegrationCredential.provider == "google",
    )
    result = await db.execute(stmt)
    credential = result.scalar_one_or_none()

    if credential:
        # Update existing
        credential.access_token = tokens["access_token"]
        if tokens.get("refresh_token"):
            credential.refresh_token = tokens["refresh_token"]
        credential.expires_at = tokens["expires_at"]
        credential.scopes = tokens.get("scopes")
    else:
        # Create new
        credential = IntegrationCredential(
            organization_id=organization_id,
            provider="google",
            access_token=tokens["access_token"],
            refresh_token=tokens.get("refresh_token"),
            expires_at=tokens["expires_at"],
            scopes=tokens.get("scopes"),
        )
        db.add(credential)

    await db.commit()
    return {"status": "success", "message": "Google integration connected"}


@router.get("/status", response_model=list[IntegrationStatusResponse])
async def get_integrations_status(
    db: AsyncSession = Depends(get_db_session),
):
    """Get the connection status of all available integrations."""
    stmt = select(IntegrationCredential)
    result = await db.execute(stmt)
    credentials = result.scalars().all()

    connected_providers = {cred.provider: cred for cred in credentials}

    return [
        IntegrationStatusResponse(
            provider="google",
            is_connected="google" in connected_providers,
        )
    ]
