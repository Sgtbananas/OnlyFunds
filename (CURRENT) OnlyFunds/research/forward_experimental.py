"""Prospective tracking of EXPERIMENTAL T50 variants X1/X2 (ROUND2_PROGRAMME.md section A). Research only.

No orders, no keys, no capital. This is separate from the frozen T50/RS3 tracker: research/forward_shadow.py and
research/shadow/FROZEN.json are never modified, and their fingerprints are unaffected (this file is not part of
that implementation list). Everything here is written to state/shadow_experimental/.

    python research/forward_experimental.py freeze          # once: research/shadow/EXPERIMENTAL_FROZEN.json
    python research/forward_experimental.py update [RUN]    # after a frozen-tracker run (default: latest "ok" run)

Each update reads the data snapshot of one frozen-tracker run (read-only; its SHA-256 hashes are re-verified) and
recomputes, from the same start date:
- X1 T50-PM: frozen T50 signals + concentration band (15% -> 10%) + profit lock (2R, 6 x ATR14)
- X2 T50-PM-B: X1 + no new entries while BTC close <= its 100-day SMA
- the comparator: the frozen T50 portfolio book, recomputed here and cross-checked against the frozen run's own
  portfolio equity (any difference is logged as COMPARATOR_MISMATCH)
"""
import hashlib
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (ROOT, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
import forward_shadow as FS  # noqa: E402  (read-only use of its constants and helpers)

EXP_FROZEN = os.path.join(ROOT, "research", "shadow", "EXPERIMENTAL_FROZEN.json")


def exp_state():
    return os.path.join(os.path.dirname(FS.STATE), "shadow_experimental")
IMPLEMENTATION = ["research/forward_experimental.py", "research/portfolio_variants.py"] + FS.IMPLEMENTATION
VARIANTS = {
    "X1_T50_PM": {"trim_band": [0.15, 0.10], "profit_lock": [2.0, 6.0], "btc_gate": False},
    "X2_T50_PM_B": {"trim_band": [0.15, 0.10], "profit_lock": [2.0, 6.0], "btc_gate": True},
}


def spec():
    fz = FS.load_frozen()[0]
    return {"purpose": "EXPERIMENTAL prospective tracking only; not approved for trading; no orders; live trading locked",
            "preregistration": "docs/research/ROUND2_PROGRAMME.md section A",
            "start_first_fill_utc": FS.START, "signals": "frozen T50 entries and exits, unchanged",
            "universe": fz["strategies"]["T50"]["universe"], "costs": fz["costs"],
            "risk_conservative": fz["risk_conservative"], "frozen_T50_fingerprint": fz["fingerprints"]["T50"],
            "variants": VARIANTS, "btc_gate_rule": "skip new entries when BTC close <= 100-day SMA (round1.btc_filter)",
            "comparator": "frozen T50 portfolio book (same data, limits, costs, start)",
            "decision": "after >= 24 months and >= 40 closed positions per book: max DD >= 2 pts lower, return/DD not "
                        "lower, return >= 80% of T50's, top-5 T50 positions keep >= 70% of their profit"}


def fingerprint(s, name):
    sub = {k: v for k, v in s.items() if k not in ("variants", "fingerprints", "frozen_at_utc", "git_commit")}
    return hashlib.sha256(FS.canonical({**sub, "variant": s["variants"][name]})).hexdigest()


def cmd_freeze(path=None):
    path = path or EXP_FROZEN
    if os.path.exists(path):
        print(f"REFUSED: {path} exists")
        return 2
    s = spec()
    s["implementation_sha256"] = {rel: FS.file_hash(rel) for rel in IMPLEMENTATION}
    s["frozen_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    s["fingerprints"] = {n: fingerprint(s, n) for n in VARIANTS}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(s, f, indent=1, sort_keys=True)
    print(json.dumps(s["fingerprints"], indent=1))
    return 0


def load_frozen(path=None):
    path = path or EXP_FROZEN
    with open(path, encoding="utf-8") as f:
        fz = json.load(f)
    problems = [f"{rel} changed since freezing" for rel, h in fz["implementation_sha256"].items()
                if not os.path.exists(os.path.join(ROOT, rel)) or FS.file_hash(rel) != h]
    problems += [f"{n} spec fingerprint mismatch" for n, fp in fz["fingerprints"].items() if fingerprint(fz, n) != fp]
    return fz, problems


def _latest_ok_run():
    d = os.path.join(FS.STATE, "runs")
    for rid in sorted(os.listdir(d), reverse=True) if os.path.isdir(d) else []:
        p = os.path.join(d, rid, "run.json")
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                if json.load(f).get("status") == "ok":
                    return rid
    return None


def _write(run_dir, rec):
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "run.json"), "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=1, default=str)


def cmd_update(run_id=None):
    import round1 as R1
    import portfolio_variants as PV
    from core.risk_manager import RiskLimits
    from portfolio import Costs, stats
    fz, problems = load_frozen()
    run_id = run_id or _latest_ok_run()
    out_dir = os.path.join(exp_state(), "runs", run_id or datetime.now(timezone.utc).strftime("none_%Y%m%dT%H%M%SZ"))
    rec = {"frozen_run": run_id, "fingerprints": fz["fingerprints"], "problems": problems,
           "computed_utc": datetime.now(timezone.utc).isoformat()}
    if problems:
        rec["status"] = "STOPPED: experimental implementation or spec changed (results preserved)"
        _write(out_dir, rec)
        return 3
    if run_id is None:
        rec["status"] = "waiting: no frozen-tracker run with status ok yet"
        _write(out_dir, rec)
        return 0
    src = os.path.join(FS.STATE, "runs", run_id)
    with open(os.path.join(src, "run.json"), encoding="utf-8") as f:
        frozen = json.load(f)
    if frozen.get("status") != "ok":
        rec["status"] = f"skipped: frozen run status is {frozen.get('status')!r}"
        _write(out_dir, rec)
        return 0
    data = {}
    for s in fz["universe"]:
        st = frozen["data_status"].get(s)
        p = os.path.join(src, "data", f"{s}_1d.csv")
        if not isinstance(st, dict) or not os.path.exists(p):
            continue
        with open(p, "rb") as f:
            if hashlib.sha256(f.read()).hexdigest() != st["sha256"]:
                rec["status"] = f"STOPPED: snapshot {s} does not match its recorded hash"
                _write(out_dir, rec)
                return 6
        df = pd.read_csv(p)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        data[s] = df
    end = (pd.Timestamp(frozen["evaluated_through_utc_day"], tz="UTC") + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    btc_ok = R1.btc_filter(data["BTCUSDT"])
    frames, ent, ex, atr = {}, {}, {}, {}
    for pair, df in data.items():
        sig = R1.signals("T50", df, {"btc_ok": btc_ok, "pair": pair})
        idx = pd.to_datetime(df["timestamp"], utc=True)
        frames[pair] = df.set_index(idx)[["Open", "High", "Low", "Close", "Volume"]]
        ent[pair] = pd.Series(sig["entry"].to_numpy(bool), index=idx)
        ex[pair] = pd.Series(sig["exit"].to_numpy(bool), index=idx)
        atr[pair] = pd.Series(sig["ATR"].to_numpy(float), index=idx)
    rk, co = fz["risk_conservative"], fz["costs"]
    lim = RiskLimits(risk_per_trade=rk["risk_per_trade"], max_position_pct=rk["max_position_pct"],
                     max_exposure_pct=rk["max_exposure_pct"], max_open_positions=rk["max_open_positions"],
                     max_daily_loss_pct=rk["max_daily_loss_pct"], max_drawdown_pct=rk["max_drawdown_pct"],
                     min_notional=1.0, max_cost_to_risk=co["max_cost_to_risk"], kill_switch_file="")
    costs = Costs(co["fee_per_side"], co["slippage"], co["stop_slippage"])
    cap = rk["portfolio_capital_usdt"]
    books, anomalies = {}, []
    for name, v in {"T50_comparator": None, **fz["variants"]}.items():
        e = ent
        kw = {}
        if v is not None:
            kw = {"trim_band": tuple(v["trim_band"]), "profit_lock": tuple(v["profit_lock"])}
            if v["btc_gate"]:
                e = {p: s & btc_ok.reindex(s.index).fillna(False).astype(bool) for p, s in ent.items()}
        res, risk = PV.simulate(frames, e, ex, atr, FS.START, end, lim, costs, capital=cap, close_at_end=False, **kw)
        bdir = os.path.join(out_dir, name)
        os.makedirs(bdir, exist_ok=True)
        res.trades.to_csv(os.path.join(bdir, "portfolio_trades.csv"), index=False)
        pd.DataFrame({"equity": res.equity, "exposure": res.exposure, "open_positions": res.n_open}).to_csv(
            os.path.join(bdir, "portfolio_equity.csv"))
        risk.to_csv(os.path.join(bdir, "risk_decomposition.csv"))
        s = stats(res, cap) if len(res.equity) else {}
        s["max_position_pct"] = float(risk["max_position_pct"].max()) if len(risk) else 0.0
        s["max_open_risk_to_stop_pct"] = float(risk["open_risk_to_stop_pct"].max()) if len(risk) else 0.0
        books[name] = s
        if name == "T50_comparator":
            fe = pd.read_csv(os.path.join(src, "T50", "portfolio_equity.csv"))
            if len(fe) and len(res.equity) and not np.isclose(fe["equity"].iloc[-1], res.equity.iloc[-1]):
                anomalies.append(f"COMPARATOR_MISMATCH: frozen {fe['equity'].iloc[-1]:.4f} vs recomputed "
                                 f"{res.equity.iloc[-1]:.4f}")
    rec.update(status="ok", evaluated_through_utc_day=frozen["evaluated_through_utc_day"], books=books,
               anomalies=anomalies, universe_available=sorted(data))
    _write(out_dir, rec)
    _report(rec)
    return 0


def _report(rec):
    lines = [f"# Experimental forward tracking (X1, X2): run {rec['frozen_run']}\n",
             "**Experimental, prospective, not approved for trading. No orders, no capital. Not combined with "
             "historical results. The decision is due only after 24 months and 40 closed positions per book.**\n",
             f"Evaluated through {rec['evaluated_through_utc_day']} (first possible fill {FS.START}).\n",
             "| book | return % | max DD % | trades | max position % | max open risk % |", "|---|---|---|---|---|---|"]
    for n, s in rec["books"].items():
        lines.append(f"| {n} | {s.get('total_return_pct', float('nan')):+.2f} | {s.get('max_dd_pct', float('nan')):.2f} | "
                     f"{s.get('trades', 0)} | {s['max_position_pct']:.1f} | {s['max_open_risk_to_stop_pct']:.1f} |")
    lines += ["", "Anomalies: " + ("; ".join(rec["anomalies"]) if rec["anomalies"] else "none")]
    os.makedirs(os.path.join(exp_state(), "reports"), exist_ok=True)
    txt = "\n".join(lines) + "\n"
    for p in (os.path.join(exp_state(), "reports", f"{rec['frozen_run']}.md"), os.path.join(exp_state(), "EXPERIMENTAL_LATEST.md")):
        with open(p, "w", encoding="utf-8") as f:
            f.write(txt)


def main(argv=None):
    a = list(argv if argv is not None else sys.argv[1:]) or ["update"]
    if a[0] == "freeze":
        return cmd_freeze()
    return cmd_update(a[1] if len(a) > 1 else None)


if __name__ == "__main__":
    sys.exit(main())
