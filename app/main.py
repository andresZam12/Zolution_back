"""
Zolution FastAPI application factory.

This module creates and configures the FastAPI application instance.
Application-level concerns (lifespan, middleware, routers, exception handlers)
live here. Business logic never belongs in this file.
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.config import get_settings
from app.core.logging import configure_logging


# ---------------------------------------------------------------------------
# Lifespan — startup / shutdown hooks
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Manage application lifespan events.

    Startup: configure logging, verify DB connectivity.
    Shutdown: close connection pools gracefully.
    """
    settings = get_settings()
    configure_logging(level=settings.LOG_LEVEL)

    # Import here to avoid circular imports at module load time
    from app.core.database import engine  # noqa: PLC0415

    # Verify the database is reachable before accepting traffic
    async with engine.connect() as conn:
        await conn.execute(__import__("sqlalchemy").text("SELECT 1"))

    yield  # Application runs here

    # Gracefully close the async engine connection pool
    await engine.dispose()


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title="Zolution API",
        description=(
            "Backend API for Zolution — AI-powered appointment scheduling agents "
            "for local businesses."
        ),
        version="0.1.0",
        docs_url="/docs" if settings.APP_ENV != "production" else None,
        redoc_url="/redoc" if settings.APP_ENV != "production" else None,
        lifespan=lifespan,
    )

    _register_middleware(app, settings)
    _register_routers(app)
    _register_exception_handlers(app)

    return app


def _register_middleware(app: FastAPI, settings: "Settings") -> None:  # type: ignore[name-defined]  # noqa: F821
    """Attach all middleware to the application."""
    from app.core.middleware import TenantContextMiddleware  # noqa: PLC0415

    # CORS — restrict to configured origins in production
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["Authorization", "Content-Type"],
    )

    # Tenant context — sets organization_id on DB session from JWT
    app.add_middleware(TenantContextMiddleware)


def _register_routers(app: FastAPI) -> None:
    """Mount all versioned API routers."""
    from app.api.v1 import router as v1_router  # noqa: PLC0415

    app.include_router(v1_router, prefix="/api/v1")


def _register_exception_handlers(app: FastAPI) -> None:
    """Register application-wide exception handlers."""

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        """Catch-all handler — never leak internal details to the client."""
        import logging  # noqa: PLC0415

        logger = logging.getLogger(__name__)
        logger.exception("Unhandled exception on %s %s", request.method, request.url)

        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "An internal error occurred. Please try again later."},
        )


# ---------------------------------------------------------------------------
# Application instance
# ---------------------------------------------------------------------------

app = create_app()


# ---------------------------------------------------------------------------
# Health check — outside /api/v1 to allow load balancer probes without auth
# ---------------------------------------------------------------------------

@app.get(
    "/health",
    tags=["Health"],
    summary="Liveness probe",
    response_description="Returns 200 if the service is up.",
)
async def health_check() -> dict[str, str]:
    """Lightweight liveness endpoint for container health checks."""
    return {"status": "ok"}
