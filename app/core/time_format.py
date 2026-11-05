"""
Business time zone and the time format shown to people.
"""

from datetime import datetime, timedelta, timezone

BOGOTA = timezone(timedelta(hours=-5))


def format_time_12h(moment: datetime) -> str:
    """Format an aware datetime as Colombian local time, e.g. '3:05 PM'."""
    local = moment.astimezone(BOGOTA)
    return f"{local.hour % 12 or 12}:{local.minute:02d} {'PM' if local.hour >= 12 else 'AM'}"
