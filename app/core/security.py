"""
Security and authentication utilities:
- Secure password hashing and verification using PBKDF2-HMAC-SHA256 (standard library).
- Cryptographic JWT access token issuance using HMAC-SHA256 and app SECRET_KEY.
- Cryptographically secure 6-digit OTP code generation.
"""

import hashlib
import hmac
import os
import secrets
from datetime import UTC, datetime, timedelta
from typing import Any

from jose import jwt

from app.core.config import get_settings


def hash_password(password: str) -> str:
    """Hash a password using PBKDF2 with SHA-256 and a random 16-byte salt."""
    salt = os.urandom(16)
    iterations = 100_000
    key = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return f"{salt.hex()}${iterations}${key.hex()}"


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plain password against the stored PBKDF2 hash."""
    try:
        salt_hex, iterations_str, key_hex = hashed_password.split("$")
        salt = bytes.fromhex(salt_hex)
        iterations = int(iterations_str)
        target_key = bytes.fromhex(key_hex)

        test_key = hashlib.pbkdf2_hmac("sha256", plain_password.encode("utf-8"), salt, iterations)
        return hmac.compare_digest(target_key, test_key)
    except Exception:
        return False


def generate_otp_code(length: int = 6) -> str:
    """Generate a cryptographically secure numeric OTP verification code."""
    digits = "0123456789"
    return "".join(secrets.choice(digits) for _ in range(length))


def create_access_token(
    data: dict[str, Any],
    expires_delta: timedelta | None = None,
) -> str:
    """
    Encode an application JWT containing identity and tenant context claims.
    Compatible with both TenantContextMiddleware and get_current_user dependencies.
    """
    settings = get_settings()
    to_encode = data.copy()

    if expires_delta:
        expire = datetime.now(UTC) + expires_delta
    else:
        expire = datetime.now(UTC) + timedelta(days=7)

    to_encode.update({"exp": expire})

    # Ensure namespaced claims for seamless integration
    namespace = "https://api.zolution.app"
    if "user_id" in to_encode and f"{namespace}/user_id" not in to_encode:
        to_encode[f"{namespace}/user_id"] = str(to_encode["user_id"])
    if "organization_id" in to_encode and f"{namespace}/org_id" not in to_encode:
        to_encode[f"{namespace}/org_id"] = str(to_encode["organization_id"]) if to_encode["organization_id"] else None
    if "role" in to_encode and f"{namespace}/role" not in to_encode:
        to_encode[f"{namespace}/role"] = str(to_encode["role"])

    encoded_jwt = jwt.encode(
        to_encode,
        settings.SECRET_KEY,
        algorithm="HS256",
    )
    return encoded_jwt
