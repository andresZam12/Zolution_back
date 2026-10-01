"""
API v1 root router.

Each feature module registers its own sub-router here.
This file only wires routers together — no business logic.
"""

from fastapi import APIRouter

from app.api.v1.endpoints import agents, organizations

router = APIRouter()

router.include_router(organizations.router)
router.include_router(agents.router)

# Future routers will be registered here, e.g.:
# from app.api.v1.endpoints import conversations, webhooks
# router.include_router(conversations.router)
