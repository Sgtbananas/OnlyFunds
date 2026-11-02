"""Reports for forward paper deployments: startup, daily, weekly, incidents and the trial report.

Rules:
- Evidence categories are reported separately: market data, paper trading, shadow (hypothetical).
- Sample size decides the wording: < 100 outcomes is INSUFFICIENT evidence; >= 100 is judged with a
  bootstrap interval (NEGATIVE if the whole interval is below zero, POSITIVE only if wholly above,
  otherwise INCONCLUSIVE). Forward evidence is never presented as validation.
- Reports are write-once; nothing is overwritten or deleted.
"""
from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from core.monitoring import r_stats, read_jsonl, read_jsonl_dir
from core.run_store import ImmutableError, write_once
from core.monitoring import read_json

MIN_SAMPLE = 100
PAPER_ACTIONS_REJECTED = ("REJECTED", "BLOCKED", "SKIP", "FILTERED", "DATA_GAP", "DATA_STALE", "BUY_FAILED")


def utc(ts: Any) -> Optional[pd.Timestamp]:
    try:
        t = pd.Timestamp(ts)
        return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
    except (ValueError, TypeError):
        return None


def evidence_status(r: pd.Series, seed: int = 7) -> Dict[str, Any]:
    r = pd.Series(r, dtype=float).dropna()
    n = len(r)
    if n < MIN_SAMPLE:
        return {"n": n, "status": "INSUFFICIENT", "note": f"{n} outcomes < {MIN_SAMPLE} minimum: no conclusion possible"}
    rng = np.random.default_rng(seed)
    means = rng.choice(r.to_numpy(), size=(10_000, n), replace=True).mean(axis=1)
    lo, hi = float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))
    status = "NEGATIVE" if hi < 0 else ("POSITIVE (forward sample, not a validation)" if lo > 0 else "INCONCLUSIVE")
    return {"n": n, "status": status, "ci95_r": [lo, hi], "mean_r": float(r.mean())}


def within(rows: List[Dict[str, Any]], key: str, start: pd.Timestamp, end: pd.Timestamp) -> List[Dict[str, Any]]:
    out = []
    for r in rows:
        t = utc(r.get(key))
        if t is not None and start <= t < end:
            out.append(r)
    return out


# ------------------------------------------------------------------ sections
def market_section(dep: str, start: pd.Timestamp, end: pd.Timestamp, every_s: float) -> Dict[str, Any]:
    steps = within(read_jsonl_dir(os.path.join(dep, "steps")), "time", start, end)
    health = within(read_jsonl_dir(os.path.join(dep, "feed_health")), "time", start, end)
    obs = within(read_jsonl_dir(os.path.join(dep, "observations")), "observed_utc", start, end)
    span = (min(end, pd.Timestamp.now(tz="UTC")) - start).total_seconds()
    ok_steps = sum(1 for s in steps if s.get("ok"))
    out: Dict[str, Any] = {
        "steps": len(steps), "successful_steps": ok_steps,
        "uptime_pct": round(100 * ok_steps * every_s / span, 2) if span > 0 else None,
        "completed_candles_processed": len(obs),
        "candles_per_symbol": pd.Series([o["pair"] for o in obs]).value_counts().to_dict() if obs else {},
        "step_errors": sum(len(s.get("errors", [])) for s in steps),
    }
    if health:
        first, last = health[0], health[-1]
        delta = lambda k: (last.get(k) or 0) - (first.get(k) or 0) if isinstance(last.get(k), (int, float)) else None
        out.update({"ws_reconnects": delta("ws_reconnects"), "ws_malformed": delta("ws_malformed"),
                    "stale_quote_incidents": delta("stale_quotes"), "rest_backfill_requests": delta("rest_backfills"),
                    "recovered_candles": delta("recovered_candles"),
                    "rest": {k: (last.get("rest", {}).get(k, 0) - first.get("rest", {}).get(k, 0))
                             for k in ("requests", "retries", "rate_limited", "errors")},
                    "missing_candles_in_buffers": {s: b.get("missing_candles") for s, b in last.get("buffers", {}).items()},
                    "note": "counters reset when a new session starts; deltas are within-session"})
    if obs:
        sp = pd.DataFrame([{"pair": o["pair"], "spread_bps": (o.get("snapshot") or {}).get("spread_bps"),
                            "depth10": ((o.get("snapshot") or {}).get("depth_bid_usd") or {}).get("10bps"),
                            "lower_bound50": ((o.get("snapshot") or {}).get("depth_lower_bound") or {}).get("50bps"),
                            "snap_error": "error" in (o.get("snapshot") or {})} for o in obs])
        out["spread_bps"] = {p: {"median": float(g["spread_bps"].median()), "p95": float(g["spread_bps"].quantile(0.95))}
                             for p, g in sp.dropna(subset=["spread_bps"]).groupby("pair")}
        out["depth_bid_10bps_usd_median"] = {p: float(g["depth10"].median()) for p, g in sp.dropna(subset=["depth10"]).groupby("pair")}
        out["depth50_lower_bound_share"] = float(sp["lower_bound50"].fillna(False).astype(bool).mean())
        out["snapshot_errors"] = int(sp["snap_error"].sum())
    return out


def _ledger(dep: str) -> Dict[str, Any]:
    p = os.path.join(dep, "paper", "paper_ledger.json")
    return read_json(p) if os.path.exists(p) else {}


def paper_section(dep: str, start: pd.Timestamp, end: pd.Timestamp) -> Dict[str, Any]:
    led = _ledger(dep)
    if not led:
        return {"note": "no paper ledger"}
    decisions = read_jsonl(os.path.join(dep, "paper", "archive", "decisions.jsonl")) + led.get("decisions", [])
    dec = within(decisions, "time", start, end)
    trades = pd.DataFrame(led.get("trades", []))
    closed = trades[[start <= utc(t) < end for t in trades["exit_time"]]] if len(trades) else trades
    prices = led.get("last_prices", {})
    positions = led.get("positions", {})
    unreal = sum(float(p["qty"]) * float(prices.get(k, p["entry_price"])) - float(p["qty"]) * float(p["entry_price"])
                 - float(p.get("entry_fee", 0)) for k, p in positions.items())
    equity = led.get("cash", 0) + sum(float(p["qty"]) * float(prices.get(k, p["entry_price"])) for k, p in positions.items())
    exposure = sum(float(p["qty"]) * float(prices.get(k, p["entry_price"])) for k, p in positions.items())
    d = pd.DataFrame(dec)
    buys = d[d["action"] == "BUY"] if len(d) else d
    out = {"starting_equity": led.get("initial_capital"), "current_equity": equity,
           "open_positions": len(positions), "closed_trades_in_period": int(len(closed)),
           "opened_trades_in_period": int(len(buys)),
           "realised_pnl_in_period": float(closed["pnl"].sum()) if len(closed) else 0.0,
           "unrealised_pnl_now": unreal, "fees_in_period": float(closed["fees"].sum()) if len(closed) else 0.0,
           "exposure_pct_of_equity": 100 * exposure / equity if equity else 0.0,
           "decisions_in_period": d["action"].value_counts().to_dict() if len(d) else {},
           "accepted_signals": int(len(buys)),
           "rejected_signals": int(d["action"].isin(PAPER_ACTIONS_REJECTED).sum()) if len(d) else 0,
           "rejection_reasons": d[d["action"].isin(PAPER_ACTIONS_REJECTED)]["reason"].str.slice(0, 100)
           .value_counts().to_dict() if len(d) else {},
           "risk_limit_events": d[d["action"] == "BLOCKED"]["reason"].str.slice(0, 80).value_counts().to_dict() if len(d) else {},
           "kill_switch_or_halt_events": int((d["action"] == "HALT").sum()) if len(d) else 0}
    if len(buys) and "ask" in buys:
        s = (buys["price"] / buys["ask"] - 1).dropna()
        out["entry_slippage_vs_ask_bps_mean"] = float(s.mean() * 1e4) if len(s) else None
    if len(trades):
        out["all_time"] = {**r_stats(trades["r_multiple"]), "evidence": evidence_status(trades["r_multiple"])}
    else:
        out["all_time"] = {"trades": 0, "evidence": evidence_status(pd.Series([], dtype=float))}
    hist = pd.DataFrame(led.get("equity_history", []))
    if len(hist):
        c = hist["equity"].astype(float)
        out["max_drawdown_pct_all_time_in_ledger"] = float(((c.cummax() - c) / c.cummax()).max() * 100)
    return out


def shadow_section(dep: str, start: pd.Timestamp, end: pd.Timestamp) -> Dict[str, Any]:
    p = os.path.join(dep, "shadow.json")
    if not os.path.exists(p):
        return {"note": "no shadow records"}
    s = read_json(p)
    recorded = within(s.get("open", []) + s.get("resolved", []), "recorded_utc", start, end)
    resolved_all = pd.DataFrame(s.get("resolved", []))
    resolved_p = within(s.get("resolved", []), "resolved_utc", start, end)
    rp = pd.DataFrame(resolved_p)
    out = {"recorded_in_period": len(recorded), "resolved_in_period": len(rp), "open_now": len(s.get("open", [])),
           "rejection_reasons_in_period": pd.Series([str(x.get("reason", ""))[:100] for x in recorded]).value_counts().to_dict()
           if recorded else {},
           "examples": [{k: x.get(k) for k in ("pair", "strategy", "regime", "reason", "entry", "exit", "exit_reason",
                                               "r_multiple", "recorded_utc", "resolved_utc")} for x in resolved_p[-10:]]}
    if len(resolved_all):
        out["all_time"] = {**r_stats(resolved_all["r_multiple"]), "evidence": evidence_status(resolved_all["r_multiple"]),
                           "by_strategy": {k: {**r_stats(g["r_multiple"]), "evidence": evidence_status(g["r_multiple"])["status"]}
                                           for k, g in resolved_all.groupby(resolved_all["strategy"].fillna("unknown"))}}
    return out


# ------------------------------------------------------------------ rendering
def render(title: str, sections: Dict[str, Any], preamble: str = "") -> str:
    out = [f"# {title}", "", preamble.strip(), "",
           "Market data, paper trading and shadow (hypothetical) results are separate kinds of evidence and are",
           "never combined. Forward results are not validation; historical results live in docs/.", ""]
    for name, body in sections.items():
        out += [f"## {name}", "", "```json", json.dumps(body, indent=1, default=str), "```", ""]
    return "\n".join(out)


def write_report(dep: str, kind: str, name: str, text: str) -> Optional[str]:
    path = os.path.join(dep, "reports", kind, f"{name}.md")
    try:
        return write_once(path, text)
    except ImmutableError:
        return None   # already written (e.g. by an earlier session): never overwrite


def period_report(dep: str, kind: str, start: pd.Timestamp, end: pd.Timestamp, every_s: float) -> str:
    return render(f"OF2 forward paper — {kind} report {start.isoformat()} → {end.isoformat()}", {
        "Market data": market_section(dep, start, end, every_s),
        "Paper trading (simulated)": paper_section(dep, start, end),
        "Shadow tracking (hypothetical outcomes of rejected opportunities)": shadow_section(dep, start, end)})


def incident(dep: str, session_id: str, kind: str, details: Dict[str, Any]) -> str:
    now = datetime.now(timezone.utc)
    rec = {"time_utc": now.isoformat(timespec="seconds"), "session_id": session_id, "kind": kind, **details}
    name = f"{now.strftime('%Y%m%dT%H%M%S%fZ')}-{kind}-{session_id[-8:]}.json"
    return write_once(os.path.join(dep, "incidents", name), rec)


def disk_status(dep: str) -> Dict[str, Any]:
    total = 0
    for root, _, files in os.walk(dep):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    u = shutil.disk_usage(dep)
    return {"deployment_mb": round(total / 1e6, 2), "disk_free_gb": round(u.free / 1e9, 1)}
