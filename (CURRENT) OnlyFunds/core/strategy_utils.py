# core/strategy_utils.py

from datetime import datetime
import pytz

def within_trading_hours(config: dict) -> bool:
    try:
        window = config.get("trading_windows", {})
        if not window.get("enabled", False):
            return True

        tz = pytz.timezone(window["timezone"])
        now = datetime.now(tz).time()
        start = datetime.strptime(window["start_time"], "%H:%M").time()
        end = datetime.strptime(window["end_time"], "%H:%M").time()

        return start <= now <= end
    except Exception as e:
        print(f"⛔ Error parsing trading window: {e}")
        return True  # Fail-safe: allow trade
