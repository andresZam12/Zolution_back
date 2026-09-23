"""
Tenant context middleware.

Extracts the `organization_id` from a verified JWT and sets it as a
PostgreSQL session-level variable (`app.current_organization_id`).
This variable is the mechanism by which Row-Level Security (RLS) policies
enforce tenant data isolation at the database level.

How RLS uses this:
    CREATE POLICY tenant_isolation ON some_table
        USING (organization_id = current_setting('app.current_organization_id')::uuid);

This middleware runs AFTER auth token verification. Routes that do not
require authentication (e.g. /health, /webhook) are excluded.

Superadmin requests: `organization_id` will be None. The middleware sets
the variable to an empty string, which means no RLS policy will match —
but superadmin routes use `get_superadmin_db_session()` which disables
RLS entirely via `SET LOCAL row_security = off`.
"""

import logging

from jose import JWTError, jwt
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from app.core.config import get_settings

logger = logging.getLogger(__name__)

# Routes that bypass the middleware (no JWT required)
_EXCLUDED_PATHS: frozenset[str] = frozenset({
    "/health",
    "/docs",
    "/redoc",
    "/openapi.json",
})


class TenantContextMiddleware(BaseHTTPMiddleware):
    """
    Starlette middleware that injects tenant context into each request.

    Reads the JWT from the Authorization header (without full verification —
    the FastAPI dependency does full verification). Extracts `org_id` and
    stores it in `request.state.organization_id` for use by the DB session
    dependency.

    Note: This middleware does NOT replace auth dependency verification.
    It only makes `organization_id` available on the request state early,
    so the DB session can set the RLS context variable.
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
        Decode JWT payload without signature verification to extract org_id.

        Verification is done by `get_current_user()` in the route handler.
        Here we only need the claim value to set the DB context variable.
        """
        try:
            # options={"verify_signature": False} is intentional here —
            # we just need the payload to set DB context, not to authenticate.
            payload = jwt.decode(
                token,
                key="",
                options={"verify_signature": False, "verify_exp": False},
                algorithms=get_settings().AUTH0_ALGORITHMS,
            )
            namespace = "https://api.zolution.app"
            return payload.get(f"{namespace}/org_id")
        except JWTError:
            return None
