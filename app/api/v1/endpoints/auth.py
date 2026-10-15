"""
Authentication and Onboarding API endpoints.
Provides:
- Self-service registration request with 6-digit email OTP.
- OTP verification and automatic User + Organization + AgentConfig provisioning.
- Password login with JWT issuance.
- Google OAuth 2.0 Sign-In + automatic Google Calendar token pairing.
- Current user profile (/me).
"""

import logging
import re
import uuid
from datetime import UTC, datetime, timedelta
from typing import Optional

import httpx
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.models import AgentConfig
from app.auth.dependencies import get_current_user
from app.auth.models import UserContext
from app.core.config import get_settings
from app.core.database import get_db_session
from app.core.email import send_verification_email
from app.core.security import create_access_token, generate_otp_code, hash_password, verify_password
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


@router.post("/register-request", status_code=status.HTTP_200_OK)
async def request_registration(
    payload: RegisterRequest,
    db: AsyncSession = Depends(get_db_session),
):
    """
    Step 1: Check user existence, generate 6-digit OTP code, and dispatch verification email.
    """
    settings = get_settings()

    # Check if a verified user with this email already exists
    existing = await db.execute(
        select(User).where(User.email == payload.email, User.is_verified == True)  # noqa: E712
    )
    if existing.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Ya existe una cuenta verificada con este correo electrónico. Por favor, inicia sesión.",
        )

    # Invalidate previous unused codes for this email
    prev_codes = await db.execute(
        select(VerificationCode).where(
            VerificationCode.email == payload.email,
            VerificationCode.is_used == False,  # noqa: E712
        )
    )
    for c in prev_codes.scalars().all():
        c.is_used = True

    # Generate new 6-digit OTP
    otp_code = generate_otp_code(6)
    expires_at = datetime.now(UTC) + timedelta(minutes=10)

    verif_record = VerificationCode(
        email=payload.email,
        code=otp_code,
        purpose="registration",
        expires_at=expires_at,
        is_used=False,
    )
    db.add(verif_record)
    await db.commit()

    # Send verification email asynchronously
    await send_verification_email(
        to_email=payload.email,
        code=otp_code,
        company_name=payload.company_name,
    )

    response_data = {
        "status": "code_sent",
        "email": payload.email,
        "message": "Código de verificación de 6 dígitos enviado exitosamente a tu correo.",
    }

    # In development mode or when no real email provider is configured, provide dev_code for instant testing
    if settings.APP_ENV != "production" or not settings.RESEND_API_KEY:
        response_data["dev_code"] = otp_code

    return response_data


@router.post("/verify-code", response_model=AuthResponse, status_code=status.HTTP_201_CREATED)
async def verify_code_and_register(
    payload: VerifyOtpRequest,
    db: AsyncSession = Depends(get_db_session),
):
    """
    Step 2: Validate OTP code, persist User + Organization in PostgreSQL,
    create default AgentConfig, and return access token.
    """
    # Look for matching active verification code
    now = datetime.now(UTC)
    stmt = (
        select(VerificationCode)
        .where(
            VerificationCode.email == payload.email,
            VerificationCode.code == payload.code.strip(),
            VerificationCode.is_used == False,  # noqa: E712
            VerificationCode.expires_at > now,
        )
        .order_by(VerificationCode.created_at.desc())
    )
    result = await db.execute(stmt)
    code_record = result.scalars().first()

    if not code_record:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Código de verificación incorrecto o expirado. Solicita uno nuevo.",
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
    user_stmt = select(User).where(User.email == payload.email)
    existing_user = (await db.execute(user_stmt)).scalar_one_or_none()

    if existing_user:
        existing_user.password_hash = hash_password(payload.password)
        existing_user.is_verified = True
        existing_user.organization_id = org.id
        existing_user.role = "owner"
        user = existing_user
    else:
        user = User(
            email=payload.email,
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
    stmt = select(User).where(User.email == payload.email)
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
async def get_google_auth_url():
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
    # Use 'auth_login' as state to distinguish from pure integration link
    auth_url = oauth_service.get_authorization_url(organization_id="auth_login")
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
    oauth_service = GoogleOAuthService()

    try:
        tokens = await oauth_service.exchange_code_for_tokens(payload.code)
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

    # Automatically save Google Calendar integration credentials for this organization
    if org:
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
