"""
API v1 root router.

Each feature module registers its own sub-router here.
This file only wires routers together — no business logic.
"""

from fastapi import APIRouter

router = APIRouter()

# Feature routers will be added here as they are implemented, e.g.:
# from app.api.v1.endpoints import tenants, agents, conversations
# router.include_router(tenants.router, prefix="/tenants", tags=["Tenants"])
