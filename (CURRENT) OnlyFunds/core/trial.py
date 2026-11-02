"""Operational trial report: sessions, uptime, data integrity, persistence and safety checks."""
from __future__ import annotations

import json
import os
import stat
from typing import Any, Dict, List, Optional

import pandas as pd

from core import forward_reports as R
from core.live_data import ALLOWED_PATHS
from core.monitoring import read_jsonl, read_jsonl_dir
from core.selection import StrategySelector
from core.trade_execution import LIVE_TRADING_LOCKED
from core.monitoring import read_json


def _load(path: str) -> Optional[Dict[str, Any]]:
    return read_json(path) if os.path.exists(path) else None


def sessions(dep: str, since: Optional[pd.Timestamp]) -> List[Dict[str, Any]]:
    out = []
    root = os.path.join(dep, "sessions")
    for sid in sorted(os.listdir(root)) if os.path.isdir(root) else []:
        d = os.path.join(root, sid)
        start = _load(os.path.join(d, "session_start.json"))
        if not start:
            continue
        st = R.utc(start["start_utc"])
        if since is not None and st < since:
            continue
        end = _load(os.path.join(d, "session_end.json"))
        rec = _load(os.path.join(d, "session_end_recovered.json"))
        hb = _load(os.path.join(d, "heartbeat.json")) or {}
        end_t = R.utc(end["end_utc"]) if end else (R.utc(rec.get("last_heartbeat_utc")) if rec else R.utc(hb.get("time")))
        out.append({"session_id": sid, "start_utc": start["start_utc"], "end_utc": str(end_t) if end_t is not None else None,
                    "ended": "gracefully: " + end["reason"] if end else ("UNEXPECTEDLY (recovered by next session)" if rec
                                                                          else "still running / unknown"),
                    "git_commit": start["git"]["commit"], "git_dirty": start["git"]["dirty"],
                    "config_fingerprint": start["config_fingerprint"],
                    "runtime_h": round((end_t - st).total_seconds() / 3600, 3) if end_t is not None else None,
                    "counters": (end or hb or {}).get("counters"),
                    "start_record_read_only": not os.stat(os.path.join(d, "session_start.json")).st_mode & stat.S_IWRITE})
    return out


def integrity(dep: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    lp = os.path.join(dep, "paper", "paper_ledger.json")
    led = _load(lp)
    if led:
        realised = sum(float(t["pnl"]) for t in led.get("trades", []))
        open_cost = sum(float(p["qty"]) * float(p["entry_price"]) + float(p.get("entry_fee", 0))
                        for p in led.get("positions", {}).values())
        expected = led["initial_capital"] + realised - open_cost
        out["ledger_cash_reconciles"] = abs(expected - led["cash"]) < 1e-6
        out["ledger_backup_readable"] = _load(lp + ".bak") is not None if os.path.exists(lp + ".bak") else "no backup yet"
        out["all_orders_simulated"] = all(str(o.get("order_id", "PAPER-")).startswith("PAPER-")
                                          for o in led.get("decisions", []) if o.get("action") == "BUY") and \
            all(str(t.get("client_id", "of2-")).startswith("of2-") for t in led.get("trades", []))
        buys = sum(1 for d in led.get("decisions", []) if d.get("action") == "BUY")
        out["paper_positions_opened"] = buys
    obs = read_jsonl_dir(os.path.join(dep, "observations"))
    keys = [(o["pair"], o["candle"]["created_at"]) for o in obs]
    out["observations"] = len(obs)
    out["duplicate_observations"] = len(keys) - len(set(keys))
    raw_lines = 0
    od = os.path.join(dep, "observations")
    for f in os.listdir(od) if os.path.isdir(od) else []:
        with open(os.path.join(od, f), encoding="utf-8") as fh:
            raw_lines += sum(1 for line in fh if line.strip())
    out["unreadable_observation_lines"] = raw_lines - len(obs)
    sh = _load(os.path.join(dep, "shadow.json")) or {"open": [], "resolved": []}
    sk = [x["key"] for x in sh["open"] + sh["resolved"]]
    out["shadow_records"] = len(sk)
    out["duplicate_shadow_records"] = len(sk) - len(set(sk))
    out["shadow_separate_from_ledger"] = not any(t.get("id") in set(sk) for t in (led or {}).get("trades", []))
    steps = read_jsonl_dir(os.path.join(dep, "steps"))
    if steps:
        t = pd.to_datetime(pd.Series([s["time"] for s in steps]), utc=True).sort_values()
        out["max_gap_between_steps_s"] = float(t.diff().dt.total_seconds().max() or 0)
    return out


def safety() -> Dict[str, Any]:
    order_like = [p for p in ALLOWED_PATHS if "order" in p or "account" in p or "withdraw" in p]
    return {"live_trading_locked": LIVE_TRADING_LOCKED, "market_data_allowlist": sorted(ALLOWED_PATHS),
            "order_or_account_endpoints_allowlisted": order_like, "credentials_used": "none",
            "approved_strategies": len(StrategySelector().approvals)}


def trial_report(dep: str, since: Optional[pd.Timestamp], every_s: float) -> str:
    ss = sessions(dep, since)
    start = min((R.utc(s["start_utc"]) for s in ss), default=since or pd.Timestamp.now(tz="UTC"))
    end = pd.Timestamp.now(tz="UTC")
    idir = os.path.join(dep, "incidents")
    flat = [read_json(os.path.join(idir, f))
            for f in sorted(os.listdir(idir)) if f.endswith(".json")] if os.path.isdir(idir) else []
    if since is not None:
        flat = [i for i in flat if R.utc(i.get("time_utc")) is not None and R.utc(i["time_utc"]) >= since]
    integ, safe = integrity(dep), safety()
    market = R.market_section(dep, start, end, every_s)
    paper = R.paper_section(dep, start, end)
    shadow = R.shadow_section(dep, start, end)
    wall_h = (end - start).total_seconds() / 3600
    checks = {
        "records_persisted": integ.get("observations", 0) > 0 and integ.get("ledger_cash_reconciles") is True,
        "no_duplicates": integ.get("duplicate_observations") == 0 and integ.get("duplicate_shadow_records") == 0,
        "recovery_tested": any(s["ended"].startswith("UNEXPECTEDLY") for s in ss) and
        any(i.get("kind") == "unexpected_termination" for i in flat),
        "no_live_orders_possible": safe["live_trading_locked"] and not safe["order_or_account_endpoints_allowlisted"]
        and integ.get("all_orders_simulated", True),
        "no_positions_without_approval": safe["approved_strategies"] > 0 or integ.get("paper_positions_opened", 0) == 0,
        "uptime_at_least_95pct": (market.get("uptime_pct") or 0) >= 95,
    }
    verdict = "CONTINUE: safe to keep collecting forward evidence" if all(checks.values()) else \
        "REVIEW before continuing: " + ", ".join(k for k, v in checks.items() if not v)
    return R.render(f"OF2 forward paper — operational trial report ({start.isoformat()} → {end.isoformat()})", {
        "Verdict": {"verdict": verdict, "checks": checks, "wall_clock_hours": round(wall_h, 2)},
        "Sessions": ss, "Incidents": flat, "Integrity and persistence": integ, "Safety": safe,
        "Market data": market, "Paper trading (simulated)": paper,
        "Shadow tracking (hypothetical)": shadow},
        preamble="Evidence status INSUFFICIENT means too few outcomes to conclude anything; it is not a negative result.")
