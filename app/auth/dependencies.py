"""
Auth0 JWT verification and FastAPI dependency injection.

This module is the ONLY place in the codebase that knows about Auth0.
Everything else depends on UserContext (see models.py), not on Auth0.

Flow:
  1. Request arrives with `Authorization: Bearer <token>`
  2. `get_current_user()` extracts and verifies the JWT
  3. JWKS public keys are fetched from Auth0 and cached in memory
  4. The validated payload is mapped to a `UserContext`
  5. `UserContext` is injected into the route handler

JWKS caching strategy:
  Keys are fetched once and cached. On `jose.ExpiredSignatureError` or
  `jose.JWKError`, the cache is cleared and keys are re-fetched once
  (handles key rotation without downtime).
"""

import logging
from typing import Any
from uuid import UUID

import httpx
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import ExpiredSignatureError, JWTError, jwt

from app.auth.models import UserContext, UserRole
from app.core.config import get_settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# HTTP Bearer scheme — extracts token from Authorization header
# ---------------------------------------------------------------------------

_bearer_scheme = HTTPBearer(auto_error=True)

# ---------------------------------------------------------------------------
# JWKS cache — module-level, cleared on key rotation
# ---------------------------------------------------------------------------

_jwks_cache: dict[str, Any] | None = None


async def _fetch_jwks() -> dict[str, Any]:
    """
    Fetch Auth0 JSON Web Key Set and cache it.

    The JWKS is fetched once and cached for the lifetime of the process.
    On key rotation (JWKError), callers should clear `_jwks_cache` and
    call this function again.
    """
    global _jwks_cache  # noqa: PLW0603

    if _jwks_cache is not None:
        return _jwks_cache

    settings = get_settings()
    async with httpx.AsyncClient(timeout=5.0) as client:
        response = await client.get(settings.AUTH0_JWKS_URL)
        response.raise_for_status()

    _jwks_cache = response.json()
    logger.info("JWKS fetched and cached from %s", settings.AUTH0_JWKS_URL)
    return _jwks_cache


def _clear_jwks_cache() -> None:
    """Clear the JWKS cache to force a re-fetch (called on key rotation)."""
    global _jwks_cache  # noqa: PLW0603
    _jwks_cache = None


# ---------------------------------------------------------------------------
# JWT verification
# ---------------------------------------------------------------------------

async def _verify_token(token: str) -> dict[str, Any]:
    """
    Verify an Auth0 JWT and return the decoded payload.

    Validates: RS256 signature, expiry, audience, issuer.
    On JWKError (possible key rotation), clears cache and retries once.

    Raises:
        HTTPException 401: Invalid, expired, or malformed token.
    """
    settings = get_settings()
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials.",
        headers={"WWW-Authenticate": "Bearer"},
    )

    for attempt in range(2):  # Allow one retry on key rotation
        try:
            jwks = await _fetch_jwks()
            payload: dict[str, Any] = jwt.decode(
                token,
                jwks,
                algorithms=settings.AUTH0_ALGORITHMS,
                audience=settings.AUTH0_AUDIENCE,
                issuer=settings.AUTH0_ISSUER,
                options={"verify_at_hash": False},
            )
            return payload

        except ExpiredSignatureError:
            logger.warning("Rejected expired JWT token")
            raise credentials_exception from None

        except JWTError as exc:
            if attempt == 0:
                # Could be a key rotation — clear cache and retry
                logger.warning("JWTError on attempt %d — clearing JWKS cache: %s", attempt, exc)
                _clear_jwks_cache()
                continue
            logger.warning("JWTError on retry — rejecting token: %s", exc)
            raise credentials_exception from exc

    raise credentials_exception  # Should not be reached


# ---------------------------------------------------------------------------
# Payload → UserContext mapping
# ---------------------------------------------------------------------------

def _extract_user_context(payload: dict[str, Any]) -> UserContext:
    """
    Map a validated JWT payload to a UserContext.

    Expected JWT claims (set via Auth0 Actions / custom namespace):
        sub                  Auth0 subject ID
        email                Verified email
        {namespace}/role     Application role (owner | staff | superadmin)
        {namespace}/user_id  Internal UUID from the users table
        {namespace}/org_id   Organization UUID (None for superadmin)

    Raises:
        HTTPException 401: Missing or invalid required claims.
    """
    # Auth0 custom claims must be namespaced to avoid conflicts
    namespace = f"https://api.zolution.app"

    try:
        auth0_id: str = payload["sub"]
        email: str = payload["email"]
        role_str: str = payload[f"{namespace}/role"]
        user_id_str: str = payload[f"{namespace}/user_id"]
        org_id_str: str | None = payload.get(f"{namespace}/org_id")

        role = UserRole(role_str)
        user_id = UUID(user_id_str)
        organization_id = UUID(org_id_str) if org_id_str else None

    except (KeyError, ValueError, AttributeError) as exc:
        logger.warning("Invalid JWT claims: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token claims.",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc

    return UserContext(
        user_id=user_id,
        auth0_id=auth0_id,
        email=email,
        role=role,
        organization_id=organization_id,
    )


# ---------------------------------------------------------------------------
# FastAPI dependencies
# ---------------------------------------------------------------------------

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(_bearer_scheme),
) -> UserContext:
    """
    FastAPI dependency — verify JWT and return the authenticated UserContext.

    Inject this into any route that requires authentication:

        @router.get("/me")
        async def get_me(user: UserContext = Depends(get_current_user)):
            ...
    """
    payload = await _verify_token(credentials.credentials)
    return _extract_user_context(payload)


async def require_superadmin(
    user: UserContext = Depends(get_current_user),
) -> UserContext:
    """
    FastAPI dependency — require superadmin role.

    Inject into routes under /admin/* to restrict access:

        @router.get("/admin/tenants")
        async def list_tenants(user: UserContext = Depends(require_superadmin)):
            ...
    """
    if not user.is_superadmin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Superadmin access required.",
        )
    return user
