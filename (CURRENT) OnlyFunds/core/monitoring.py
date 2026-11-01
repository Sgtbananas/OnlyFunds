"""Performance monitoring with strictly separated evidence categories.

Categories are never pooled into one number:
  historical_backtest  - committed v1 evaluation (read from docs, not recomputed)
  validation_runs      - v2 runs: eligibility and single final tests
  forward_paper        - simulated trades on live data (PaperTrader ledger)
  shadow               - hypothetical outcomes of opportunities OF2 rejected (counterfactual)
  data_feed            - reliability of the live feed
  execution            - paper fill quality vs observed quotes
All PnL figures are net of fees and slippage; expectancy is in R.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List

import numpy as np
import pandas as pd


def r_stats(r: pd.Series, pnl: pd.Series = None) -> Dict[str, Any]:
    r = r.dropna().astype(float)
    if r.empty:
        return {"trades": 0}
    wins, losses = r[r > 0], r[r <= 0]
    gross_loss = -losses.sum()
    return {"trades": int(len(r)), "expectancy_r": float(r.mean()), "win_rate_pct": float((r > 0).mean() * 100),
            "avg_win_r": float(wins.mean()) if len(wins) else 0.0, "avg_loss_r": float(losses.mean()) if len(losses) else 0.0,
            "profit_factor_r": float(wins.sum() / gross_loss) if gross_loss > 0 else float("inf") if len(wins) else 0.0,
            "total_r": float(r.sum()), "max_drawdown_r": float((r.cumsum().cummax() - r.cumsum()).max())}


def grouped(df: pd.DataFrame, by: str) -> Dict[str, Any]:
    if df.empty or by not in df:
        return {}
    return {str(k): r_stats(g["r_multiple"]) for k, g in df.groupby(df[by].fillna("unknown"))}


def forward_paper(state_dir: str) -> Dict[str, Any]:
    path = os.path.join(state_dir, "paper", "paper_ledger.json")
    if not os.path.exists(path):
        return {"note": "no forward paper ledger yet"}
    led = json.load(open(path, encoding="utf-8"))
    t = pd.DataFrame(led.get("trades", []))
    d = pd.DataFrame(led.get("decisions", []))
    out: Dict[str, Any] = {"cash": led.get("cash"), "initial_capital": led.get("initial_capital"),
                           "open_positions": len(led.get("positions", {})), "failed_orders": len(led.get("failed_orders", []))}
    if len(t):
        out.update({"overall": r_stats(t["r_multiple"]), "fees_paid": float(t["fees"].sum()),
                    "net_pnl": float(t["pnl"].sum()), "by_pair": grouped(t, "pair"), "by_regime": grouped(t, "regime"),
                    "by_strategy": grouped(t, "strategy")})
        span_days = (pd.to_datetime(t["exit_time"], utc=True).max() - pd.to_datetime(t["entry_time"], utc=True).min()).days
        out["trades_per_day"] = len(t) / max(span_days, 1)
    else:
        out["overall"] = {"trades": 0}
    if len(d):
        out["decisions"] = d["action"].value_counts().to_dict()
        rej = d[d["action"].isin(["REJECTED", "BLOCKED", "SKIP", "FILTERED", "DATA_GAP", "DATA_STALE"])]
        out["rejection_reasons"] = rej["reason"].str.slice(0, 90).value_counts().head(15).to_dict()
        buys = d[d["action"] == "BUY"]
        if len(buys) and "ask" in buys:
            slip = (buys["price"] / buys["ask"] - 1).dropna()
            out["execution"] = {"buys": int(len(buys)), "mean_fill_vs_ask_bps": float(slip.mean() * 1e4) if len(slip) else None}
    return out


def shadow(state_dir: str) -> Dict[str, Any]:
    path = os.path.join(state_dir, "shadow.json")
    if not os.path.exists(path):
        return {"note": "no shadow opportunities yet"}
    s = json.load(open(path, encoding="utf-8"))
    res = pd.DataFrame(s.get("resolved", []))
    out = {"open": len(s.get("open", [])), "resolved": len(res)}
    if len(res):
        res["reason_kind"] = res["reason"].fillna("").str.split(":").str[0].str.slice(0, 60)
        out.update({"overall": r_stats(res["r_multiple"]), "by_strategy": grouped(res, "strategy"),
                    "by_regime": grouped(res, "regime"), "by_pair": grouped(res, "pair"),
                    "by_rejection": grouped(res, "reason_kind"),
                    "exit_reasons": res["exit_reason"].value_counts().to_dict()})
    return out


def data_feed(state_dir: str) -> Dict[str, Any]:
    path = os.path.join(state_dir, "feed_health.jsonl")
    if not os.path.exists(path):
        return {"note": "no feed health records yet"}
    rows = [json.loads(x) for x in open(path, encoding="utf-8") if x.strip()]
    last = rows[-1]
    buf = last.get("buffers", {})
    return {"samples": len(rows), "first": rows[0]["time"], "last": last["time"],
            "ws_reconnects": last.get("ws_reconnects"), "ws_malformed": last.get("ws_malformed"),
            "stale_quotes": last.get("stale_quotes"), "rest_backfills": last.get("rest_backfills"),
            "rest": last.get("rest"),
            "candles": {s: {k: b.get(k) for k in ("rows", "gaps", "missing_candles", "rejected", "duplicates")}
                        for s, b in buf.items()}}


def validation_runs(root: str) -> List[Dict[str, Any]]:
    out = []
    if not os.path.isdir(root):
        return out
    for plan in sorted(os.listdir(root)):
        if plan.startswith("_"):
            continue
        for run in sorted(os.listdir(os.path.join(root, plan))):
            rd = os.path.join(root, plan, run)
            rec = {"plan": plan, "run": run}
            el = os.path.join(rd, "stage1", "eligibility.json")
            if os.path.exists(el):
                e = json.load(open(el, encoding="utf-8"))
                rec["eligible"] = [k for k, v in e.items() if v.get("eligible")]
                rec["candidates"] = len(e)
            res = os.path.join(rd, "stage2", "result.json")
            if os.path.exists(res):
                r = json.load(open(res, encoding="utf-8"))
                rec["test"] = [{"candidate": x["candidate"], "credible_edge": x["credible_edge"]} for x in r.get("evaluated", [])] \
                    or r.get("note")
            out.append(rec)
    return out


def report(state_dir: str, runs_root: str) -> str:
    def block(title, obj):
        return f"## {title}\n\n```json\n{json.dumps(obj, indent=1, default=str)}\n```\n"
    return ("# OF2 performance monitor\n\nEach section is a separate kind of evidence; they are never combined.\n"
            "Historical backtest: see docs/REAL_MARKET_EVALUATION.md (v1, frozen).\n\n"
            + block("Validation runs (v2)", validation_runs(runs_root))
            + block("Forward paper trading (simulated, live data)", forward_paper(state_dir))
            + block("Shadow: outcomes of rejected opportunities (hypothetical)", shadow(state_dir))
            + block("Data feed reliability", data_feed(state_dir)))
