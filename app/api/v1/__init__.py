"""
API v1 root router.

Each feature module registers its own sub-router here.
This file only wires routers together — no business logic.
"""

from fastapi import APIRouter

from app.api.v1.endpoints import agents, organizations, webhooks

router = APIRouter()

router.include_router(organizations.router)
router.include_router(agents.router)
router.include_router(webhooks.router, prefix="/webhooks", tags=["Webhooks"])
