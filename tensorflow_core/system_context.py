"""Invisible runtime context supplied to Ori's intelligence layer."""
from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo


def current_time_context(timezone: str = "America/New_York") -> dict[str, str]:
    """Return current local time without exposing a clock in the UI."""
    now = datetime.now(ZoneInfo(timezone))
    return {
        "iso": now.isoformat(),
        "date": now.date().isoformat(),
        "day_of_week": now.strftime("%A"),
        "time": now.strftime("%I:%M:%S %p").lstrip("0"),
        "timezone": timezone,
        "time_of_day": (
            "morning" if now.hour < 12 else
            "afternoon" if now.hour < 17 else
            "evening"
        ),
    }
