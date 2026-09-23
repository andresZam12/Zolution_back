"""
Application configuration.

All settings are loaded from environment variables (or a .env file) via
Pydantic Settings. Import `get_settings()` anywhere in the app — it returns
a cached singleton so the .env file is parsed only once.

Usage:
    from app.core.config import get_settings

    settings = get_settings()
    print(settings.DATABASE_URL)
"""

from functools import lru_cache
from typing import Literal

from pydantic import AnyHttpUrl, Field, PostgresDsn, computed_field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Typed application configuration.

    Every field maps 1-to-1 to a variable in backend/.env.example.
    Pydantic validates types at startup — misconfiguration fails fast
    instead of causing runtime errors deep in a request.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",  # Ignore unexpected env vars (e.g. from CI)
    )

    # -------------------------------------------------------------------------
    # Application
    # -------------------------------------------------------------------------
    APP_ENV: Literal["development", "staging", "production"] = "development"
    DEBUG: bool = False
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    SECRET_KEY: str = Field(min_length=32)

    # CORS — comma-separated list of allowed origins
    # In development: ["http://localhost:4200"]
    # In production: the real frontend domain
    CORS_ORIGINS: list[AnyHttpUrl] = Field(
        default=["http://localhost:4200"]
    )

    # -------------------------------------------------------------------------
    # PostgreSQL — individual fields to build the async DSN
    # -------------------------------------------------------------------------
    DATABASE_HOST: str = "localhost"
    DATABASE_PORT: int = 5432
    DATABASE_NAME: str = "zolution"
    DATABASE_USER: str = "zolution"
    DATABASE_PASSWORD: str

    @computed_field  # type: ignore[prop-decorator]
    @property
    def DATABASE_URL(self) -> str:
        """Async PostgreSQL DSN assembled from individual fields."""
        return (
            f"postgresql+asyncpg://{self.DATABASE_USER}:{self.DATABASE_PASSWORD}"
            f"@{self.DATABASE_HOST}:{self.DATABASE_PORT}/{self.DATABASE_NAME}"
        )

    # Superadmin DB connection (bypasses RLS — never expose to frontend)
    SUPERADMIN_DB_USER: str = "zolution_admin"
    SUPERADMIN_DB_PASSWORD: str = ""

    @computed_field  # type: ignore[prop-decorator]
    @property
    def SUPERADMIN_DATABASE_URL(self) -> str:
        """Async PostgreSQL DSN for the superadmin service connection."""
        return (
            f"postgresql+asyncpg://{self.SUPERADMIN_DB_USER}:{self.SUPERADMIN_DB_PASSWORD}"
            f"@{self.DATABASE_HOST}:{self.DATABASE_PORT}/{self.DATABASE_NAME}"
        )

    # -------------------------------------------------------------------------
    # Auth0
    # -------------------------------------------------------------------------
    AUTH0_DOMAIN: str
    AUTH0_AUDIENCE: str
    AUTH0_ALGORITHMS: list[str] = ["RS256"]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def AUTH0_ISSUER(self) -> str:
        return f"https://{self.AUTH0_DOMAIN}/"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def AUTH0_JWKS_URL(self) -> str:
        return f"https://{self.AUTH0_DOMAIN}/.well-known/jwks.json"

    # -------------------------------------------------------------------------
    # LLM Providers (all optional — only fill what you use)
    # -------------------------------------------------------------------------
    ANTHROPIC_API_KEY: str = ""
    GOOGLE_API_KEY: str = ""
    OPENAI_API_KEY: str = ""

    # -------------------------------------------------------------------------
    # WhatsApp Cloud API
    # -------------------------------------------------------------------------
    WHATSAPP_API_TOKEN: str = ""
    WHATSAPP_PHONE_NUMBER_ID: str = ""
    WHATSAPP_VERIFY_TOKEN: str = ""

    # -------------------------------------------------------------------------
    # Google Calendar OAuth2
    # -------------------------------------------------------------------------
    GOOGLE_CLIENT_ID: str = ""
    GOOGLE_CLIENT_SECRET: str = ""
    GOOGLE_REDIRECT_URI: str = "http://localhost:8000/api/v1/integrations/google-calendar/callback"

    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------
    @model_validator(mode="after")
    def validate_production_settings(self) -> "Settings":
        """Enforce stricter requirements in production."""
        if self.APP_ENV == "production":
            if self.DEBUG:
                raise ValueError("DEBUG must be False in production")
            if not self.SECRET_KEY or len(self.SECRET_KEY) < 64:
                raise ValueError("SECRET_KEY must be at least 64 characters in production")
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return the cached application settings singleton.

    The @lru_cache ensures .env is parsed exactly once at startup.
    In tests, call `get_settings.cache_clear()` before overriding settings.
    """
    return Settings()
