"""create appointments table and copy existing appointments into it

Appointments used to live in agent_configs.onboarding_answers. That JSON is left
untouched as a backup. Entries that cannot be read, or that overlap an earlier
one, are skipped and reported.

Revision ID: 20240110_000000
Revises: 20240109_000000
Create Date: 2026-11-05 18:00:00.000000
"""

from collections.abc import Sequence
from datetime import datetime, timedelta, timezone
import json
import re

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID

revision: str = "20240110_000000"
down_revision: str | None = "20240109_000000"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

BOGOTA = timezone(timedelta(hours=-5))
CURRENT_TENANT = "NULLIF(current_setting('app.current_organization_id', TRUE), '')::uuid"

_DATE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})")
_CLOCK = re.compile(r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)?", re.IGNORECASE)
_MINUTES = re.compile(r"(\d+)\s*min", re.IGNORECASE)


def _clock(match: re.Match) -> tuple[int, int]:
    hour = int(match.group(1))
    meridiem = (match.group(3) or "").lower()
    if meridiem == "pm" and hour < 12:
        hour += 12
    elif meridiem == "am" and hour == 12:
        hour = 0
    return hour, int(match.group(2) or 0)


def _legacy_window(date_text: str, time_text: str) -> tuple[datetime, datetime] | None:
    """Read 'YYYY-MM-DD' with '10:00', '3:30 PM (45 min)' or '10:00 - 10:45'."""
    date_match = _DATE.match(date_text.strip())
    clocks = list(_CLOCK.finditer(time_text.split("(")[0]))
    if not date_match or not clocks:
        return None

    try:
        hour, minute = _clock(clocks[0])
        day = datetime(*map(int, date_match.groups()), tzinfo=BOGOTA)
        start = day.replace(hour=hour, minute=minute)
        if len(clocks) > 1:
            end_hour, end_minute = _clock(clocks[1])
            end = day.replace(hour=end_hour, minute=end_minute)
        else:
            minutes = _MINUTES.search(time_text)
            end = start + timedelta(minutes=int(minutes.group(1)) if minutes else 60)
    except ValueError:
        return None
    return (start, end) if end > start else None


def _copy_legacy_appointments() -> None:
    bind = op.get_bind()
    set_tenant = sa.text("SELECT set_config('app.current_organization_id', :org, true)")
    insert = sa.text("""
        INSERT INTO appointments (
            organization_id, customer_name, customer_phone, service, starts_at, ends_at,
            status, source, google_event_id, html_link
        )
        VALUES (
            :org, :name, :phone, :service, :starts_at, :ends_at,
            :status, 'legacy', :event_id, :link
        )
        ON CONFLICT ON CONSTRAINT appointments_no_overlap DO NOTHING
    """)

    copied = skipped = 0
    organizations = bind.execute(sa.text("SELECT id FROM organizations")).scalars().all()
    for organization_id in organizations:
        bind.execute(set_tenant, {"org": str(organization_id)})
        raw = bind.execute(sa.text("SELECT onboarding_answers::text FROM agent_configs")).scalar()
        items = (json.loads(raw) or {}).get("appointments") if raw else None
        for item in items if isinstance(items, list) else []:
            window = None
            if isinstance(item, dict):
                window = _legacy_window(str(item.get("date", "")), str(item.get("time", "")))
            if not window:
                skipped += 1
                continue

            event_id = str(item.get("googleEventId") or "")
            status = str(item.get("status") or "confirmed")
            result = bind.execute(
                insert,
                {
                    "org": organization_id,
                    "name": str(item.get("patientName") or "Cliente")[:255],
                    "phone": str(item.get("phone") or "")[:50],
                    "service": str(item.get("service") or "Consulta")[:255],
                    "starts_at": window[0],
                    "ends_at": window[1],
                    "status": status if status in ("confirmed", "completed") else "cancelled",
                    "event_id": None
                    if event_id.startswith(("pg_", "local_"))
                    else event_id or None,
                    "link": item.get("htmlLink"),
                },
            )
            copied += result.rowcount
            skipped += 1 - result.rowcount

    bind.execute(set_tenant, {"org": ""})
    print(f"Appointments copied: {copied}, skipped: {skipped}")


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS btree_gist")

    op.create_table(
        "appointments",
        sa.Column(
            "id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")
        ),
        sa.Column(
            "organization_id",
            UUID(as_uuid=True),
            sa.ForeignKey("organizations.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("customer_name", sa.String(255), nullable=False),
        sa.Column("customer_phone", sa.String(50), nullable=False),
        sa.Column("service", sa.String(255), nullable=False),
        sa.Column("starts_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("ends_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("status", sa.String(50), nullable=False, server_default="confirmed"),
        sa.Column("source", sa.String(50), nullable=False),
        sa.Column("google_event_id", sa.String(255), nullable=True),
        sa.Column("html_link", sa.String(1024), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.CheckConstraint("ends_at > starts_at", name="appointments_valid_range"),
    )
    op.create_index("ix_appointments_organization_id", "appointments", ["organization_id"])
    op.create_index("ix_appointments_starts_at", "appointments", ["organization_id", "starts_at"])

    op.execute("""
        ALTER TABLE appointments ADD CONSTRAINT appointments_no_overlap
        EXCLUDE USING gist (
            organization_id WITH =,
            tstzrange(starts_at, ends_at) WITH &&
        )
        WHERE (status = 'confirmed')
    """)

    op.execute("ALTER TABLE appointments ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE appointments FORCE ROW LEVEL SECURITY")
    op.execute(f"""
        CREATE POLICY tenant_isolation ON appointments
        USING (organization_id = {CURRENT_TENANT})
    """)
    op.execute(f"""
        CREATE POLICY tenant_insert ON appointments
        FOR INSERT
        WITH CHECK (organization_id = {CURRENT_TENANT})
    """)
    op.execute("""
        CREATE TRIGGER trigger_appointments_updated_at
        BEFORE UPDATE ON appointments
        FOR EACH ROW EXECUTE FUNCTION set_updated_at()
    """)

    _copy_legacy_appointments()


def downgrade() -> None:
    op.drop_table("appointments")
