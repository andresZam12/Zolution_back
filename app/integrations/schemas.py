"""
Schemas for integrations.
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class GoogleOAuthUrlResponse(BaseModel):
    """Response containing the Google OAuth2 authorization URL."""

    auth_url: str


class GoogleOAuthCallbackRequest(BaseModel):
    """The data received from the frontend after Google redirects back."""

    code: str
    redirect_uri: Optional[str] = None


class IntegrationStatusResponse(BaseModel):
    """Status of an integration for the tenant dashboard."""

    provider: str
    is_connected: bool
    email_address: Optional[str] = None
    scopes: Optional[str] = None
    updated_at: Optional[datetime] = None
