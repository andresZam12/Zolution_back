"""
Google Workspace OAuth2 authentication flow.
"""

from datetime import UTC, datetime, timedelta
from urllib.parse import urlencode

import httpx

from app.core.config import get_settings


class GoogleOAuthService:
    """Handles Google OAuth2 authorization and token exchange."""

    def __init__(self):
        self.settings = get_settings()
        self.client_id = self.settings.GOOGLE_CLIENT_ID
        self.client_secret = self.settings.GOOGLE_CLIENT_SECRET
        self.redirect_uri = self.settings.GOOGLE_REDIRECT_URI

        # We request scopes for Calendar and Sheets
        self.scopes = [
            "openid",
            "email",
            "https://www.googleapis.com/auth/calendar.events",
            "https://www.googleapis.com/auth/spreadsheets",
        ]

    def get_authorization_url(self, organization_id: str) -> str:
        """
        Generate the URL to redirect the user to Google for consent.
        We pass the organization_id in the 'state' parameter to map the callback
        back to the correct tenant.
        """
        base_url = "https://accounts.google.com/o/oauth2/v2/auth"
        params = {
            "client_id": self.client_id,
            "redirect_uri": self.redirect_uri,
            "response_type": "code",
            "scope": " ".join(self.scopes),
            "access_type": "offline",
            "prompt": "consent",  # Force consent to ensure we get a refresh token
            "state": organization_id,
        }
        return f"{base_url}?{urlencode(params)}"

    async def exchange_code_for_tokens(self, code: str) -> dict:
        """
        Exchange the authorization code for access and refresh tokens.
        """
        url = "https://oauth2.googleapis.com/token"
        data = {
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "code": code,
            "redirect_uri": self.redirect_uri,
            "grant_type": "authorization_code",
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(url, data=data)
            if response.status_code != 200:
                raise ValueError(f"Failed to exchange token: {response.text}")

            payload = response.json()

            # Calculate absolute expiration time
            expires_in = payload.get("expires_in", 3599)
            expires_at = datetime.now(UTC) + timedelta(seconds=expires_in)

            return {
                "access_token": payload["access_token"],
                "refresh_token": payload.get("refresh_token"),  # May be None if not first time
                "expires_at": expires_at,
                "scopes": payload.get("scope", " ".join(self.scopes)),
            }
