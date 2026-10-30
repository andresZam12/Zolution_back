"""
Authentication and Onboarding API endpoints.
Provides:
- Self-service registration request with 6-digit email OTP.
- OTP verification and automatic User + Organization + AgentConfig provisioning.
- Password login with JWT issuance.
- Google OAuth 2.0 Sign-In + automatic Google Calendar token pairing.
- Current user profile (/me).
"""

import asyncio
import logging
import re
import uuid
from datetime import UTC, datetime, timedelta
from typing import Optional

import httpx
from email_validator import EmailNotValidError, validate_email
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.models import AgentConfig
from app.auth.dependencies import get_current_user
from app.auth.models import UserContext
from app.core.config import get_settings
from app.core.database import get_db_session
from app.core.email import send_verification_email
from app.core.security import (
    create_access_token,
    generate_otp_code,
    hash_otp_code,
    hash_password,
    verify_otp_code,
    verify_password,
)
from app.integrations.google.auth import GoogleOAuthService
from app.integrations.models import IntegrationCredential
from app.tenants.models import Organization, User, VerificationCode

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/auth", tags=["Authentication"])


# ---------------------------------------------------------------------------
# Request & Response Schemas
# ---------------------------------------------------------------------------


class RegisterRequest(BaseModel):
    email: str = Field(..., min_length=3, max_length=255)
    company_name: str = Field(..., min_length=2, max_length=255)
    industry: str = Field(default="Servicios Profesionales")


class VerifyOtpRequest(BaseModel):
    email: str = Field(..., min_length=3, max_length=255)
    code: str = Field(..., min_length=6, max_length=10)
    company_name: str = Field(..., min_length=2, max_length=255)
    password: str = Field(..., min_length=6)
    industry: str = Field(default="Servicios Profesionales")


class LoginRequest(BaseModel):
    email: str = Field(..., min_length=3, max_length=255)
    password: str = Field(..., min_length=1)


class GoogleAuthUrlResponse(BaseModel):
    auth_url: str


class GoogleCallbackRequest(BaseModel):
    code: str
    redirect_uri: Optional[str] = None


class UserProfileResponse(BaseModel):
    id: str
    email: str
    role: str
    is_verified: bool


class OrganizationProfileResponse(BaseModel):
    id: str
    name: str
    slug: str
    industry: Optional[str] = None


class AuthResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserProfileResponse
    organization: OrganizationProfileResponse


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


MAX_OTP_ATTEMPTS = 5
OTP_TTL_MINUTES = 10


async def _normalize_and_validate_email(raw_email: str) -> str:
    """
    Validate the email syntax AND that its domain can actually receive mail
    (DNS MX lookup). Returns the normalized, lower-cased address.
    """
    try:
        info = await asyncio.to_thread(validate_email, raw_email.strip(), check_deliverability=True)
    except EmailNotValidError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="El correo electrónico no es válido o su dominio no puede recibir correos.",
        ) from exc
    return info.normalized.lower()


async def _set_tenant_context(db: AsyncSession, organization_id: uuid.UUID) -> None:
    """
    Set the RLS tenant variable for the current transaction so inserts into
    tenant-scoped tables (agent_configs, integration_credentials) pass policy checks.
    """
    await db.execute(
        text("SELECT set_config('app.current_organization_id', :org_id, true)"),
        {"org_id": str(organization_id)},
    )


@router.post("/register-request", status_code=status.HTTP_200_OK)
async def request_registration(
    payload: RegisterRequest,
    db: AsyncSession = Depends(get_db_session),
):
    """
    Step 1: Validate the email, generate a 6-digit OTP and send it by email.
    The code is NEVER returned in the response — it only travels by email.
    """
    email = await _normalize_and_validate_email(payload.email)

    existing = await db.execute(
        select(User).where(User.email == email, User.is_verified == True)  # noqa: E712
    )
    if existing.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Ya existe una cuenta verificada con este correo electrónico. Por favor, inicia sesión.",
        )

    # Invalidate previous unused codes for this email
    prev_codes = await db.execute(
        select(VerificationCode).where(
            VerificationCode.email == email,
            VerificationCode.is_used == False,  # noqa: E712
        )
    )
    for c in prev_codes.scalars().all():
        c.is_used = True

    otp_code = generate_otp_code(6)
    verif_record = VerificationCode(
        email=email,
        code=hash_otp_code(email, otp_code),
        purpose="registration",
        expires_at=datetime.now(UTC) + timedelta(minutes=OTP_TTL_MINUTES),
        is_used=False,
        attempts=0,
    )
    db.add(verif_record)
    await db.commit()

    sent, detail_msg = await send_verification_email(
        to_email=email,
        code=otp_code,
        company_name=payload.company_name,
    )
    if not sent:
        verif_record.is_used = True
        await db.commit()
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"No se pudo enviar el correo de verificación ({detail_msg}).",
        )

    return {
        "status": "code_sent",
        "email": email,
        "message": "Te enviamos un código de 6 dígitos. Revisa tu bandeja de entrada (y spam).",
    }


@router.post("/verify-code", response_model=AuthResponse, status_code=status.HTTP_201_CREATED)
async def verify_code_and_register(
    payload: VerifyOtpRequest,
    db: AsyncSession = Depends(get_db_session),
):
    """
    Step 2: Validate OTP code, persist User + Organization in PostgreSQL,
    create default AgentConfig, and return access token.
    """
    email = payload.email.strip().lower()
    submitted_code = payload.code.strip()

    # Latest active code for this email
    stmt = (
        select(VerificationCode)
        .where(
            VerificationCode.email == email,
            VerificationCode.is_used == False,  # noqa: E712
            VerificationCode.expires_at > datetime.now(UTC),
        )
        .order_by(VerificationCode.created_at.desc())
    )
    code_record = (await db.execute(stmt)).scalars().first()

    if not code_record:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="El código expiró o ya fue usado. Solicita uno nuevo.",
        )

    if not verify_otp_code(email, submitted_code, code_record.code):
        code_record.attempts += 1
        remaining = MAX_OTP_ATTEMPTS - code_record.attempts
        if remaining <= 0:
            code_record.is_used = True
        await db.commit()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Código incorrecto. Te quedan {remaining} intento(s)."
                if remaining > 0
                else "Demasiados intentos fallidos. Solicita un código nuevo."
            ),
        )

    # Guard against a race where the account was verified meanwhile
    already = await db.execute(
        select(User).where(User.email == email, User.is_verified == True)  # noqa: E712
    )
    if already.scalar_one_or_none():
        code_record.is_used = True
        await db.commit()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Esta cuenta ya fue verificada. Inicia sesión.",
        )

    # Mark code as used
    code_record.is_used = True

    # Generate clean unique slug for organization
    raw_slug = re.sub(r"[^a-z0-9]+", "-", payload.company_name.lower().strip()).strip("-")
    if not raw_slug:
        raw_slug = f"org-{uuid.uuid4().hex[:6]}"

    base_slug = raw_slug
    idx = 1
    while True:
        check_slug = await db.execute(select(Organization).where(Organization.slug == raw_slug))
        if check_slug.scalar_one_or_none() is None:
            break
        raw_slug = f"{base_slug}-{idx}"
        idx += 1

    # Create Organization
    org = Organization(
        name=payload.company_name,
        slug=raw_slug,
        plan="starter",
        status="active",
    )
    db.add(org)
    await db.flush()  # Populates org.id

    # agent_configs is protected by RLS: bind this transaction to the new tenant
    await _set_tenant_context(db, org.id)

    # Create default AgentConfig
    agent = AgentConfig(
        organization_id=org.id,
        llm_provider="openai",
        llm_model="gpt-4o-mini",
        system_prompt_generated=None,
        status="active",
    )
    db.add(agent)

    # Check if an unverified user record exists or create new
    user_stmt = select(User).where(User.email == email)
    existing_user = (await db.execute(user_stmt)).scalar_one_or_none()

    if existing_user:
        existing_user.password_hash = hash_password(payload.password)
        existing_user.is_verified = True
        existing_user.organization_id = org.id
        existing_user.role = "owner"
        user = existing_user
    else:
        user = User(
            email=email,
            password_hash=hash_password(payload.password),
            role="owner",
            is_active=True,
            is_verified=True,
            organization_id=org.id,
        )
        db.add(user)

    await db.commit()
    await db.refresh(user)
    await db.refresh(org)

    # Issue JWT token
    token = create_access_token(
        {
            "sub": str(user.id),
            "user_id": str(user.id),
            "email": user.email,
            "role": user.role,
            "organization_id": str(org.id),
        }
    )

    return AuthResponse(
        access_token=token,
        token_type="bearer",
        user=UserProfileResponse(
            id=str(user.id),
            email=user.email,
            role=user.role,
            is_verified=user.is_verified,
        ),
        organization=OrganizationProfileResponse(
            id=str(org.id),
            name=org.name,
            slug=org.slug,
            industry=payload.industry,
        ),
    )


@router.post("/login", response_model=AuthResponse)
async def login(
    payload: LoginRequest,
    db: AsyncSession = Depends(get_db_session),
):
    """
    Authenticate an existing user with email and password.
    Returns access token with tenant context.
    """
    stmt = select(User).where(User.email == payload.email.strip().lower())
    result = await db.execute(stmt)
    user = result.scalar_one_or_none()

    if not user or not user.password_hash or not verify_password(payload.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Correo o contraseña incorrectos.",
        )

    if not user.is_verified:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Tu correo no ha sido verificado. Por favor realiza el proceso de verificación.",
        )

    # Load organization
    org = None
    if user.organization_id:
        org_stmt = select(Organization).where(Organization.id == user.organization_id)
        org = (await db.execute(org_stmt)).scalar_one_or_none()

    if not org:
        # Fallback dummy organization if none linked
        org_name = user.email.split("@")[0].capitalize()
        org_slug = user.email.split("@")[0].lower()
        org_id = str(user.organization_id or uuid.uuid4())
    else:
        org_name = org.name
        org_slug = org.slug
        org_id = str(org.id)

    token = create_access_token(
        {
            "sub": str(user.id),
            "user_id": str(user.id),
            "email": user.email,
            "role": user.role,
            "organization_id": org_id,
        }
    )

    return AuthResponse(
        access_token=token,
        token_type="bearer",
        user=UserProfileResponse(
            id=str(user.id),
            email=user.email,
            role=user.role,
            is_verified=user.is_verified,
        ),
        organization=OrganizationProfileResponse(
            id=org_id,
            name=org_name,
            slug=org_slug,
        ),
    )


# ---------------------------------------------------------------------------
# Google OAuth 2.0 (Sign In + Google Calendar Sync)
# ---------------------------------------------------------------------------


@router.get("/google/url", response_model=GoogleAuthUrlResponse)
async def get_google_auth_url(redirect_uri: Optional[str] = None):
    """
    Generate Google OAuth URL for seamless Sign-In and Google Calendar permission grant.
    """
    settings = get_settings()
    if not settings.GOOGLE_CLIENT_ID or not settings.GOOGLE_CLIENT_ID.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Google OAuth no está configurado aún. Falta agregar GOOGLE_CLIENT_ID y GOOGLE_CLIENT_SECRET en las variables de Render.",
        )
    oauth_service = GoogleOAuthService()
    target_redirect = redirect_uri or (
        settings.GOOGLE_REDIRECT_URI
        if settings.GOOGLE_REDIRECT_URI and "localhost" not in settings.GOOGLE_REDIRECT_URI
        else "https://zolution-front.vercel.app/login"
    )
    auth_url = oauth_service.get_authorization_url(organization_id="auth_login", redirect_uri=target_redirect)
    return GoogleAuthUrlResponse(auth_url=auth_url)


@router.post("/google/callback", response_model=AuthResponse)
async def google_auth_callback(
    payload: GoogleCallbackRequest,
    db: AsyncSession = Depends(get_db_session),
):
    """
    Handle Google OAuth callback:
    1. Exchange code for tokens (includes Calendar scopes).
    2. Fetch Google profile (email, name, sub).
    3. Find or create Organization and User in PostgreSQL.
    4. Automatically store Google Calendar refresh token in integration_credentials.
    5. Issue Zolution access token.
    """
    settings = get_settings()
    target_redirect = payload.redirect_uri or (
        settings.GOOGLE_REDIRECT_URI
        if settings.GOOGLE_REDIRECT_URI and "localhost" not in settings.GOOGLE_REDIRECT_URI
        else "https://zolution-front.vercel.app/login"
    )
    oauth_service = GoogleOAuthService()

    try:
        tokens = await oauth_service.exchange_code_for_tokens(payload.code, redirect_uri=target_redirect)
    except Exception as exc:
        logger.error("Failed exchanging Google OAuth code: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Error al procesar el código de autorización de Google.",
        ) from exc

    # Fetch Google user info using the access token
    async with httpx.AsyncClient(timeout=8.0) as client:
        userinfo_resp = await client.get(
            "https://www.googleapis.com/oauth2/v3/userinfo",
            headers={"Authorization": f"Bearer {tokens['access_token']}"},
        )
        if userinfo_resp.status_code != 200:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No se pudo obtener la información de perfil de Google.",
            )
        google_profile = userinfo_resp.json()

    google_email = google_profile.get("email")
    google_sub = google_profile.get("sub")
    google_name = google_profile.get("name") or google_email.split("@")[0]

    if not google_email:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="La cuenta de Google no devolvió un correo electrónico.",
        )

    google_email = google_email.strip().lower()

    # Check if user already exists by google_id or email
    user_stmt = select(User).where((User.google_id == google_sub) | (User.email == google_email))
    user = (await db.execute(user_stmt)).scalar_one_or_none()

    if user:
        user.google_id = google_sub
        user.is_verified = True
        # Fetch organization
        org_stmt = select(Organization).where(Organization.id == user.organization_id)
        org = (await db.execute(org_stmt)).scalar_one_or_none()
    else:
        # Create new Organization for this user
        clean_slug = re.sub(r"[^a-z0-9]+", "-", google_name.lower().strip()).strip("-")
        if not clean_slug:
            clean_slug = f"org-{uuid.uuid4().hex[:6]}"

        base_slug = clean_slug
        idx = 1
        while True:
            chk = await db.execute(select(Organization).where(Organization.slug == clean_slug))
            if chk.scalar_one_or_none() is None:
                break
            clean_slug = f"{base_slug}-{idx}"
            idx += 1

        org = Organization(
            name=f"Empresa de {google_name}",
            slug=clean_slug,
            plan="starter",
            status="active",
        )
        db.add(org)
        await db.flush()
        await _set_tenant_context(db, org.id)

        # Provision initial AgentConfig
        agent = AgentConfig(
            organization_id=org.id,
            llm_provider="openai",
            llm_model="gpt-4o-mini",
            status="active",
        )
        db.add(agent)

        user = User(
            email=google_email,
            google_id=google_sub,
            role="owner",
            is_active=True,
            is_verified=True,
            organization_id=org.id,
        )
        db.add(user)

    await db.flush()

    # Automatically save Google Calendar integration credentials ONLY if calendar permissions were granted
    granted_scopes = tokens.get("scopes") or ""
    if org and "calendar" in granted_scopes:
        await _set_tenant_context(db, org.id)
        cred_stmt = select(IntegrationCredential).where(
            IntegrationCredential.organization_id == org.id,
            IntegrationCredential.provider == "google",
        )
        credential = (await db.execute(cred_stmt)).scalar_one_or_none()

        if credential:
            credential.access_token = tokens["access_token"]
            if tokens.get("refresh_token"):
                credential.refresh_token = tokens["refresh_token"]
            credential.expires_at = tokens["expires_at"]
            credential.scopes = tokens.get("scopes")
        else:
            credential = IntegrationCredential(
                organization_id=org.id,
                provider="google",
                access_token=tokens["access_token"],
                refresh_token=tokens.get("refresh_token"),
                expires_at=tokens["expires_at"],
                scopes=tokens.get("scopes"),
            )
            db.add(credential)

    await db.commit()
    await db.refresh(user)
    if org:
        await db.refresh(org)

    token = create_access_token(
        {
            "sub": str(user.id),
            "user_id": str(user.id),
            "email": user.email,
            "role": user.role,
            "organization_id": str(org.id) if org else None,
        }
    )

    return AuthResponse(
        access_token=token,
        token_type="bearer",
        user=UserProfileResponse(
            id=str(user.id),
            email=user.email,
            role=user.role,
            is_verified=user.is_verified,
        ),
        organization=OrganizationProfileResponse(
            id=str(org.id) if org else str(uuid.uuid4()),
            name=org.name if org else google_name,
            slug=org.slug if org else "empresa",
        ),
    )


@router.get("/me", response_model=UserProfileResponse)
async def get_me(
    current_user: UserContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
):
    """
    Get profile of the currently authenticated user.
    """
    stmt = select(User).where(User.id == current_user.user_id)
    user = (await db.execute(stmt)).scalar_one_or_none()

    if not user:
        return UserProfileResponse(
            id=str(current_user.user_id),
            email=current_user.email,
            role=str(current_user.role),
            is_verified=True,
        )

    return UserProfileResponse(
        id=str(user.id),
        email=user.email,
        role=user.role,
        is_verified=user.is_verified,
    )
