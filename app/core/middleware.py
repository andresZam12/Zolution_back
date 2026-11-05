"""
Tenant context middleware.

Extracts the `organization_id` from a signature-verified Zolution JWT so the
database session can bind itself to that tenant. Tokens that fail verification
get no tenant, and RLS then returns no tenant rows.

This middleware does NOT authenticate: routes still use `get_current_user()`.
"""

import logging

from jose import JWTError, jwt
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from app.core.config import get_settings

logger = logging.getLogger(__name__)

# Routes that bypass the middleware (no JWT required)
_EXCLUDED_PATHS: frozenset[str] = frozenset(
    {
        "/health",
        "/docs",
        "/redoc",
        "/openapi.json",
    }
)


class TenantContextMiddleware(BaseHTTPMiddleware):
    """
    Starlette middleware that makes the verified tenant available early, so
    `get_db_session()` can bind the session to it.
    """

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        # Skip middleware for excluded paths
        if request.url.path in _EXCLUDED_PATHS:
            return await call_next(request)

        organization_id: str | None = None

        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header.removeprefix("Bearer ")
            organization_id = self._extract_org_id(token)

        # Store on request state — the DB session dependency reads this
        request.state.organization_id = organization_id

        return await call_next(request)

    @staticmethod
    def _extract_org_id(token: str) -> str | None:
        """
        Return the tenant of a valid Zolution token, or None.

        Auth0 tokens are not resolved here; their routes bind the session with
        the organization from the `UserContext` built by `get_current_user()`.
        """
        try:
            payload = jwt.decode(token, get_settings().SECRET_KEY, algorithms=["HS256"])
        except JWTError:
            return None
        namespace = "https://api.zolution.app"
        org_id = payload.get("organization_id") or payload.get(f"{namespace}/org_id")
        return str(org_id) if org_id else None
