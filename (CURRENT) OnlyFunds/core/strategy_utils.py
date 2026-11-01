"""Trading-hours window (optional; crypto trades 24/7 so it is off by default)."""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

import pytz

logger = logging.getLogger(__name__)


def within_trading_hours(config: dict, now: Optional[datetime] = None) -> bool:
    """True when entries are allowed. Supports windows that cross midnight (e.g. 22:00-06:00)."""
    window = config.get("trading_windows", {}) or {}
    if not window.get("enabled", False):
        return True
    try:
        tz = pytz.timezone(window.get("timezone", "UTC"))
        now = (now or datetime.now(tz)).astimezone(tz).time()
        start = datetime.strptime(str(window["start_time"]), "%H:%M").time()
        end = datetime.strptime(str(window["end_time"]), "%H:%M").time()
    except Exception as exc:
        # Misconfigured window: block entries (fail safe for capital, exits are unaffected).
        logger.error("Invalid trading window %s: %s; blocking new entries", window, exc)
        return False
    if start <= end:
        return start <= now <= end
    return now >= start or now <= end
