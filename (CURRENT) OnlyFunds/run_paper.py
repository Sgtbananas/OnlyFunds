"""Headless paper trading loop (simulated orders on live CoinEx data).

    python run_paper.py                      # config pairs, forever
    python run_paper.py --pairs BTCUSDT ETHUSDT --interval 15m --steps 10
    python run_paper.py --metrics-port 8000  # expose Prometheus metrics

Shares its ledger with the Streamlit app (state/paper/paper_ledger.json);
a lock file stops both writing at the same time. Stop with Ctrl+C.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, APP_ROOT)

from core.core_data import get_client  # noqa: E402
from core.paper_trader import LedgerBusy, LedgerCorrupt, PaperTrader  # noqa: E402
from core.trade_execution import LIVE_TRADING_LOCKED, PaperExecutor  # noqa: E402
from utils.config import load_config, build_settings  # noqa: E402

STATE_DIR = os.environ.get("OF2_STATE_DIR") or os.path.join(APP_ROOT, "state")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="OnlyFunds 2 paper trader")
    p.add_argument("--pairs", nargs="+")
    p.add_argument("--interval")
    p.add_argument("--profile", choices=["Conservative", "Normal", "Aggressive"])
    p.add_argument("--strategy", choices=["auto", "trend_following", "mean_reversion", "stacked"])
    p.add_argument("--steps", type=int, default=0, help="0 = run forever")
    p.add_argument("--every", type=int, help="seconds between steps (default: config paper_refresh_seconds)")
    p.add_argument("--metrics-port", type=int, default=0)
    return p.parse_args(argv)


def start_metrics(port: int):
    try:
        from prometheus_client import Gauge, start_http_server
    except ImportError:
        logging.warning("prometheus_client not installed; metrics disabled")
        return None
    start_http_server(port)
    return {k: Gauge(f"onlyfunds_paper_{k}", f"OnlyFunds paper {k}") for k in
            ("equity", "cash", "open_positions", "trades", "unrealised_pnl", "heartbeat")}


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    assert LIVE_TRADING_LOCKED, "run_paper.py only supports paper trading"
    try:
        from dotenv import load_dotenv
        load_dotenv(os.path.join(APP_ROOT, ".env"))
    except ImportError:
        pass
    cfg = load_config()
    overrides = {k: v for k, v in (("interval", args.interval), ("strategy", args.strategy)) if v}
    overrides["kill_switch_file"] = os.path.join(STATE_DIR, "KILL_SWITCH")
    _, settings = build_settings(cfg, args.profile, overrides)
    pairs = args.pairs or cfg.trading_pairs
    every = args.every or int(cfg.trading.get("paper_refresh_seconds", 60))
    client = get_client()
    try:
        markets = client.get_markets()
    except Exception as exc:
        logging.warning("market metadata unavailable (%s); precision checks use defaults", exc)
        markets = None
    executor = PaperExecutor(settings.fee_rate, settings.slippage_pct, markets=markets,
                             min_notional=settings.risk.min_notional)
    bundle = None
    if settings.use_ml:
        from core.ml_filter import load_bundle
        bundle = load_bundle(os.path.join(STATE_DIR, "ml_filter.pkl"))
        if bundle is None:
            logging.warning("ml.enabled is true but no compatible model found; running without ML gate")
            settings.use_ml = False
    trader = PaperTrader(settings, os.path.join(STATE_DIR, "paper"),
                         price_fn=lambda p: float(client.get_ticker(p)["last"]), executor=executor, ml_bundle=bundle)
    gauges = start_metrics(args.metrics_port) if args.metrics_port else None
    logging.info("Paper trading %s on %s every %ss (profile %s, strategy %s)", pairs, settings.interval, every,
                 args.profile or cfg.profile, settings.strategy)
    n = 0
    try:
        while True:
            try:
                for d in trader.step(pairs):
                    if d["action"] not in ("WAIT", "HOLD"):
                        logging.info("%s %s: %s", d["pair"], d["action"], d["reason"])
                issues = trader.reconcile()
                if issues:
                    logging.error("ledger reconciliation problems: %s", issues)
                m = trader.metrics()
                logging.info("equity %.2f (%.2f%%), open %d, trades %d", m["equity"], m["total_return_pct"],
                             m["open_positions"], m["trades"])
                if gauges:
                    for k in ("equity", "cash", "open_positions", "trades", "unrealised_pnl"):
                        gauges[k].set(m[k])
                    gauges["heartbeat"].set(time.time())
            except LedgerBusy as exc:
                logging.warning("%s; skipping this step", exc)
            except LedgerCorrupt as exc:
                logging.critical("%s. Stopping: fix or quarantine the ledger before restarting.", exc)
                return 1
            except Exception:
                logging.exception("paper step failed; will retry next interval")
            n += 1
            if args.steps and n >= args.steps:
                break
            time.sleep(every)
    except KeyboardInterrupt:
        logging.info("stopped by user")
    return 0


if __name__ == "__main__":
    sys.exit(main())
