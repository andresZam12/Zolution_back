"""
Common API dependencies re-exports.
"""

from app.auth.dependencies import (
    get_current_user,
    require_permission,
    require_role,
)
from app.auth.models import UserContext, UserRole
from app.core.database import get_db_session

__all__ = [
    "UserContext",
    "UserRole",
    "get_current_user",
    "get_db_session",
    "require_permission",
    "require_role",
]
