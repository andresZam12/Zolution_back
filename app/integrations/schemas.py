"""
Schemas for integrations.
"""

from pydantic import BaseModel, HttpUrl


class GoogleOAuthUrlResponse(BaseModel):
    """Response containing the Google OAuth2 authorization URL."""

    auth_url: HttpUrl


class GoogleOAuthCallbackRequest(BaseModel):
    """The data received from the frontend after Google redirects back."""

    code: str


class IntegrationStatusResponse(BaseModel):
    """Status of an integration for the tenant dashboard."""

    provider: str
    is_connected: bool
    email_address: str | None = None
