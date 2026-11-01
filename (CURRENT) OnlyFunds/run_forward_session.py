"""Long-running forward PAPER session on live Binance public data. No orders, no keys.

    python run_forward_session.py --deployment binance-1h --interval 1h --every 60 --websocket
    python run_forward_session.py --deployment binance-1h --report daily --date 2026-11-02
    python run_forward_session.py --deployment binance-1h --trial-report --since 2026-11-01T08:00:00Z

Stop gracefully: create state/forward_deployments/<deployment>/STOP (or Ctrl+C).
For unattended operation use scripts/forward_supervisor.py (restarts after crashes with back-off),
e.g. via START_FORWARD_PAPER.bat.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, APP_ROOT)

import pandas as pd  # noqa: E402

from core import forward_reports as R  # noqa: E402
from core.forward import ForwardPaperEngine  # noqa: E402
from core.live_data import LiveMarket  # noqa: E402
from core.session import ForwardSession, configure_logging  # noqa: E402
from core.trade_execution import LIVE_TRADING_LOCKED  # noqa: E402
from utils.config import build_settings, load_config  # noqa: E402

STATE_ROOT = os.environ.get("OF2_STATE_DIR") or os.path.join(APP_ROOT, "state")
DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "LTCUSDT", "ADAUSDT", "XLMUSDT", "TRXUSDT"]


def deployment_dir(name: str) -> str:
    return os.path.join(STATE_ROOT, "forward_deployments", name)


def settings_dict(settings) -> dict:
    d = dataclasses.asdict(settings)
    d["risk"].pop("kill_switch_file", None)
    return d


def main(argv=None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--deployment", default="binance-1h")
    ap.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    ap.add_argument("--interval", default="1h")
    ap.add_argument("--every", type=float, default=60.0)
    ap.add_argument("--websocket", action="store_true")
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--run-for", type=float, default=0, help="seconds (0 = until stopped)")
    ap.add_argument("--report", choices=["daily", "weekly"])
    ap.add_argument("--date", help="UTC date for --report (week start for weekly)")
    ap.add_argument("--trial-report", action="store_true")
    ap.add_argument("--since", help="UTC start for --trial-report")
    a = ap.parse_args(argv)
    assert LIVE_TRADING_LOCKED
    dep = deployment_dir(a.deployment)
    if a.report or a.trial_report:
        configure_logging()
        if a.trial_report:
            from core.trial import trial_report
            text = trial_report(dep, pd.Timestamp(a.since).tz_convert("UTC") if a.since else None, a.every)
            print(text)
            path = R.write_report(dep, "trial", pd.Timestamp.now(tz="UTC").strftime("%Y%m%dT%H%M%SZ"), text)
            print(f"\nwritten: {path}")
            return 0
        start = pd.Timestamp(a.date, tz="UTC")
        span = pd.Timedelta(days=1 if a.report == "daily" else 7)
        print(R.period_report(dep, a.report, start, start + span, a.every))
        return 0
    os.makedirs(dep, exist_ok=True)
    cfg = load_config()
    _, settings = build_settings(cfg, None, {"interval": a.interval, "kill_switch_file": os.path.join(dep, "KILL_SWITCH")})

    def make_engine():
        live = LiveMarket(a.symbols, a.interval, history=settings.lookback + 50, use_websocket=a.websocket)
        try:
            live.client.sync_time()
        except Exception:
            pass
        return ForwardPaperEngine(a.symbols, settings, dep, live)

    session = ForwardSession(dep, a.symbols, a.interval, a.every, make_engine, settings_dict(settings))
    session.start()
    counters = session.run(max_steps=a.steps or None, run_for_s=a.run_for or None)
    print(json.dumps({"session_id": session.session_id, **counters}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
