"""
Integrations API endpoints.
Manages third-party OAuth2 connections (Google Calendar & Google Sheets) per tenant.
"""

import logging
from typing import Optional

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.models import AgentConfig
from app.auth.dependencies import get_current_user
from app.auth.models import UserContext
from app.core.config import get_settings
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


async def _ensure_rls_context(db: AsyncSession, organization_id: str) -> None:
    try:
        await db.execute(
            text("SELECT set_config('app.current_organization_id', :org_id, true)"),
            {"org_id": str(organization_id)},
        )
    except Exception:
        pass


@router.get("/google/authorize", response_model=GoogleOAuthUrlResponse)
async def authorize_google(
    request: Request,
    redirect_uri: Optional[str] = None,
    state: Optional[str] = None,
    user: UserContext = Depends(get_current_user),
):
    """
    Generate the Google OAuth2 authorization URL with Calendar & Sheets scopes.
    The frontend redirects the user to this URL.
    """
    organization_id = str(user.organization_id or getattr(request.state, "organization_id", "") or "")
    if not organization_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No se encontró el contexto de organización del usuario.",
        )

    settings = get_settings()
    target_redirect = redirect_uri or (
        settings.GOOGLE_REDIRECT_URI
        if settings.GOOGLE_REDIRECT_URI and "localhost" not in settings.GOOGLE_REDIRECT_URI
        else "https://zolution-agent.vercel.app/login"
    )
    target_state = state or f"calendar_connect_{organization_id}"

    oauth_service = GoogleOAuthService(scopes=GoogleOAuthService.INTEGRATION_SCOPES)
    url = oauth_service.get_authorization_url(
        organization_id=target_state,
        redirect_uri=target_redirect,
    )
    return GoogleOAuthUrlResponse(auth_url=url)


@router.post("/google/callback")
async def google_callback(
    payload: GoogleOAuthCallbackRequest,
    request: Request,
    user: UserContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
):
    """
    Exchange the authorization code for Google Calendar tokens and persist them
    in `integration_credentials` protected by tenant RLS.
    """
    org_uuid = user.organization_id or getattr(request.state, "organization_id", None)
    if not org_uuid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No se encontró la organización activa para vincular Google Calendar.",
        )

    await _ensure_rls_context(db, str(org_uuid))

    settings = get_settings()
    target_redirect = payload.redirect_uri or (
        settings.GOOGLE_REDIRECT_URI
        if settings.GOOGLE_REDIRECT_URI and "localhost" not in settings.GOOGLE_REDIRECT_URI
        else "https://zolution-agent.vercel.app/login"
    )

    oauth_service = GoogleOAuthService(scopes=GoogleOAuthService.INTEGRATION_SCOPES)
    try:
        tokens = await oauth_service.exchange_code_for_tokens(
            payload.code,
            redirect_uri=target_redirect,
        )
    except Exception as e:
        logger.error("Failed to exchange Google OAuth code for Calendar: %s", e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No se pudo intercambiar el código de autorización con Google Calendar.",
        ) from e

    # Fetch connected Google account email for display in the dashboard
    connected_email: Optional[str] = None
    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            userinfo_resp = await client.get(
                "https://www.googleapis.com/oauth2/v3/userinfo",
                headers={"Authorization": f"Bearer {tokens['access_token']}"},
            )
            if userinfo_resp.status_code == 200:
                connected_email = userinfo_resp.json().get("email")
    except Exception:
        pass

    stmt = select(IntegrationCredential).where(
        IntegrationCredential.organization_id == org_uuid,
        IntegrationCredential.provider == "google",
    )
    result = await db.execute(stmt)
    credential = result.scalar_one_or_none()

    if credential:
        credential.access_token = tokens["access_token"]
        if tokens.get("refresh_token"):
            credential.refresh_token = tokens["refresh_token"]
        credential.expires_at = tokens["expires_at"]
        credential.scopes = tokens.get("scopes")
    else:
        credential = IntegrationCredential(
            organization_id=org_uuid,
            provider="google",
            access_token=tokens["access_token"],
            refresh_token=tokens.get("refresh_token"),
            expires_at=tokens["expires_at"],
            scopes=tokens.get("scopes"),
        )
        db.add(credential)

    # Also store connected_email in AgentConfig.onboarding_answers metadata
    if connected_email:
        agent_stmt = select(AgentConfig).where(AgentConfig.organization_id == org_uuid)
        agent_cfg = (await db.execute(agent_stmt)).scalar_one_or_none()
        if agent_cfg:
            ans = dict(agent_cfg.onboarding_answers or {})
            ans["google_calendar_email"] = connected_email
            agent_cfg.onboarding_answers = ans
            db.add(agent_cfg)

    await db.commit()
    return {
        "status": "success",
        "message": "Google Calendar conectado exitosamente.",
        "email_address": connected_email or user.email,
    }


@router.get("/status", response_model=list[IntegrationStatusResponse])
async def get_integrations_status(
    request: Request,
    user: UserContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
):
    """Get the real connection status of all available integrations for this tenant."""
    org_uuid = user.organization_id or getattr(request.state, "organization_id", None)
    if not org_uuid:
        return [IntegrationStatusResponse(provider="google", is_connected=False)]

    await _ensure_rls_context(db, str(org_uuid))

    stmt = select(IntegrationCredential).where(
        IntegrationCredential.organization_id == org_uuid
    )
    result = await db.execute(stmt)
    credentials = result.scalars().all()

    connected_providers = {cred.provider: cred for cred in credentials}
    google_cred = connected_providers.get("google")

    connected_email: Optional[str] = None
    if google_cred:
        agent_stmt = select(AgentConfig).where(AgentConfig.organization_id == org_uuid)
        agent_cfg = (await db.execute(agent_stmt)).scalar_one_or_none()
        if agent_cfg and isinstance(agent_cfg.onboarding_answers, dict):
            connected_email = agent_cfg.onboarding_answers.get("google_calendar_email")
        if not connected_email:
            connected_email = user.email

    return [
        IntegrationStatusResponse(
            provider="google",
            is_connected=google_cred is not None,
            email_address=connected_email if google_cred else None,
            scopes=google_cred.scopes if google_cred else None,
            updated_at=google_cred.updated_at if google_cred else None,
        )
    ]


@router.delete("/google")
async def disconnect_google(
    request: Request,
    user: UserContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
):
    """Disconnect Google Calendar integration for the current tenant."""
    org_uuid = user.organization_id or getattr(request.state, "organization_id", None)
    if not org_uuid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No se encontró la organización activa.",
        )

    await _ensure_rls_context(db, str(org_uuid))

    stmt = select(IntegrationCredential).where(
        IntegrationCredential.organization_id == org_uuid,
        IntegrationCredential.provider == "google",
    )
    cred = (await db.execute(stmt)).scalar_one_or_none()
    if cred:
        await db.delete(cred)
        await db.commit()

    return {"status": "disconnected", "message": "Google Calendar desconectado."}
