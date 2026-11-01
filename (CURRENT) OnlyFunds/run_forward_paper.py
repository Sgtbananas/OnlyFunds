"""Forward paper trading on live Binance public data. NO real orders, no API keys.

    python run_forward_paper.py                                  # config pairs, 1h, forever
    python run_forward_paper.py --symbols BTCUSDT ETHUSDT --interval 5m --every 30 --steps 20
    python run_forward_paper.py --report                         # print the separated performance monitor

State: state/forward/<interval>/ (paper ledger, shadow opportunities, observations, feed health).
Entries require a strategy approved by a v2 validation run; today there is none, so OF2 observes,
logs and shadow-tracks every opportunity, and does not trade.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, APP_ROOT)

from core import monitoring  # noqa: E402
from core.forward import ForwardPaperEngine  # noqa: E402
from core.live_data import LiveMarket  # noqa: E402
from core.paper_trader import LedgerBusy, LedgerCorrupt  # noqa: E402
from core.run_store import DEFAULT_ROOT  # noqa: E402
from core.trade_execution import LIVE_TRADING_LOCKED  # noqa: E402
from utils.config import build_settings, load_config  # noqa: E402

STATE_ROOT = os.environ.get("OF2_STATE_DIR") or os.path.join(APP_ROOT, "state")


def main(argv=None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="+")
    ap.add_argument("--interval", default="1h")
    ap.add_argument("--every", type=int, default=60, help="seconds between steps")
    ap.add_argument("--steps", type=int, default=0, help="0 = run until stopped")
    ap.add_argument("--websocket", action="store_true", help="stream quotes/candles over WebSocket as well as REST")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    assert LIVE_TRADING_LOCKED
    state_dir = os.path.join(STATE_ROOT, "forward", a.interval)
    if a.report:
        print(monitoring.report(state_dir, DEFAULT_ROOT))
        return 0
    cfg = load_config()
    _, settings = build_settings(cfg, None, {"interval": a.interval,
                                             "kill_switch_file": os.path.join(state_dir, "KILL_SWITCH")})
    symbols = a.symbols or cfg.trading_pairs
    live = LiveMarket(symbols, a.interval, history=settings.lookback + 50, use_websocket=a.websocket)
    try:
        engine = ForwardPaperEngine(symbols, settings, state_dir, live)
    except LedgerCorrupt as exc:
        logging.critical("%s", exc)
        return 1
    logging.info("Forward paper: %s on %s every %ss; approved strategies: %d (no approval -> no trades)",
                 symbols, a.interval, a.every, len(engine.selector.approvals))
    n = 0
    try:
        while True:
            try:
                out = engine.step()
                acts = [f"{d['pair']}:{d['action']}" for d in out["trader"] if d["action"] not in ("WAIT", "HOLD")]
                logging.info("step %d: observed %s, trader %s, shadow resolved %d, errors %s", n + 1,
                             out["observed"], acts or "-", len(out["shadow_resolved"]), out["errors"] or "-")
            except LedgerBusy as exc:
                logging.warning("%s; skipping step", exc)
            except LedgerCorrupt as exc:
                logging.critical("%s", exc)
                return 1
            except Exception:
                logging.exception("forward step failed; retrying next interval")
            n += 1
            if a.steps and n >= a.steps:
                break
            time.sleep(a.every)
    except KeyboardInterrupt:
        logging.info("stopped")
    finally:
        live.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
