"""
Tenant domain ORM models.

This module will contain the SQLAlchemy models for:
- Organization (tenant)
- User

Models are defined here (not in the migration file) so that Alembic's
--autogenerate can detect schema changes by comparing Base.metadata
against the live database.

Models are imported in alembic/env.py to register them on Base.metadata.
"""

# Models will be added in Commit 11 (initial schema migration)
