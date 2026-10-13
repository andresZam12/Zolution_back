"""
Common API dependencies re-exports.
"""

from app.auth.dependencies import (
    get_current_user,
    require_role,
    require_permission,
)
from app.auth.models import UserContext, UserRole
from app.core.database import get_db_session, get_superadmin_db_session

__all__ = [
    "get_current_user",
    "require_role",
    "require_permission",
    "UserContext",
    "UserRole",
    "get_db_session",
    "get_superadmin_db_session",
]
