"""
Encryption at rest for third-party credentials.

The key derives from SECRET_KEY, which lives outside the database, so a database
leak alone does not expose tokens. Changing SECRET_KEY makes stored tokens
unreadable: affected businesses must reconnect their Google account.
"""

import base64
import hashlib
import logging

from cryptography.fernet import Fernet, InvalidToken
from sqlalchemy import Text
from sqlalchemy.types import TypeDecorator

from app.core.config import get_settings

logger = logging.getLogger(__name__)

_PREFIX = "enc:v1:"


def _fernet() -> Fernet:
    secret = f"zolution-integration-tokens:{get_settings().SECRET_KEY}".encode()
    return Fernet(base64.urlsafe_b64encode(hashlib.sha256(secret).digest()))


def encrypt_token(value: str) -> str:
    return _PREFIX + _fernet().encrypt(value.encode()).decode()


def decrypt_token(value: str) -> str:
    """Return the token. Values stored before encryption pass through unchanged."""
    if not value.startswith(_PREFIX):
        return value
    try:
        return _fernet().decrypt(value.removeprefix(_PREFIX).encode()).decode()
    except InvalidToken:
        logger.error("A stored token cannot be decrypted; the Google account must be reconnected")
        return ""


class EncryptedText(TypeDecorator):
    """Text column that is encrypted in the database and plain in Python."""

    impl = Text
    cache_ok = True

    def process_bind_param(self, value: str | None, dialect) -> str | None:
        return encrypt_token(value) if value else value

    def process_result_value(self, value: str | None, dialect) -> str | None:
        return decrypt_token(value) if value else value
