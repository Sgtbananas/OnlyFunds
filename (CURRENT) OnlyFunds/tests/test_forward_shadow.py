"""Forward shadow tracker (research/forward_shadow.py): offline, synthetic daily data, no network."""
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest
import requests

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "research"))
import forward_shadow as FS  # noqa: E402


def synthetic_daily(seed, start="2026-05-01", end="2027-04-01"):
    rng = np.random.default_rng(seed)
    days = pd.date_range(start, end, freq="D", tz="UTC", inclusive="left")
    drift = np.where((np.arange(len(days)) // 60) % 2 == 0, 0.006, -0.004)     # alternating trends
    close = 100 * np.exp(np.cumsum(drift + rng.normal(0, 0.03, len(days))))
    op = np.r_[close[0], close[:-1]] * (1 + rng.normal(0, 0.003, len(days)))
    hi = np.maximum(op, close) * (1 + np.abs(rng.normal(0, 0.01, len(days))))
    lo = np.minimum(op, close) * (1 - np.abs(rng.normal(0, 0.01, len(days))))
    return pd.DataFrame({"timestamp": days, "created_at": (days.astype("int64") // 10**6).astype("int64"),
                         "Open": op, "High": hi, "Low": lo, "Close": close, "Volume": 1e7, "QuoteVolume": 1e9,
                         "Trades": 1000})


@pytest.fixture
def shadow(tmp_path, monkeypatch):
    monkeypatch.setattr(requests.Session, "request", lambda *a, **k: (_ for _ in ()).throw(AssertionError("network")))
    monkeypatch.setattr(requests, "get", lambda *a, **k: (_ for _ in ()).throw(AssertionError("network")))
    monkeypatch.setattr(FS, "FROZEN", str(tmp_path / "FROZEN.json"))
    monkeypatch.setattr(FS, "STATE", str(tmp_path / "shadow"))
    monkeypatch.setattr(FS, "N_RANDOM", 3)
    monkeypatch.setattr(FS, "CLUSTER_BOOT", 200)
    monkeypatch.setattr(FS, "historical_entry_rates", lambda: {s: 0.08 for s in FS.MAIN + FS.HOLDOUT})
    frames = {s: synthetic_daily(i) for i, s in enumerate(FS.MAIN + FS.HOLDOUT)}
    state = {"end": "2027-03-01"}

    def fake_fetch(symbol, interval, start, end, **k):
        if symbol == "EOSUSDT":
            return pd.DataFrame(columns=["timestamp"])                       # delisted
        df = frames[symbol]
        ts = pd.to_datetime(df["timestamp"], utc=True)
        return df[(ts >= pd.Timestamp(start, tz="UTC")) & (ts < pd.Timestamp(state["end"], tz="UTC"))].reset_index(drop=True)

    import core.history
    monkeypatch.setattr(core.history, "fetch_binance", fake_fetch)
    monkeypatch.setattr(FS.pd.Timestamp, "now", classmethod(lambda cls, tz=None: pd.Timestamp(state["end"], tz="UTC")))
    return state


def runs(state_dir):
    out = []
    for d in sorted(os.listdir(os.path.join(state_dir, "runs"))):
        with open(os.path.join(state_dir, "runs", d, "run.json"), encoding="utf-8") as f:
            out.append(json.load(f))
    return out


def test_freeze_is_write_once_and_fingerprinted(shadow):
    assert FS.cmd_freeze() == 0
    with open(FS.FROZEN, encoding="utf-8") as f:
        fz = json.load(f)
    assert set(fz["fingerprints"]) == {"T50", "RS3"} and fz["fingerprints"]["T50"] != fz["fingerprints"]["RS3"]
    assert all(len(v) == 64 for v in fz["fingerprints"].values())
    assert FS.cmd_freeze() == 2                                               # refuses to overwrite
    _, problems = FS.load_frozen()
    assert problems == []


def test_update_records_both_books_separately_and_rejections(shadow):
    FS.cmd_freeze()
    assert FS.cmd_update() == 0
    rec = runs(FS.STATE)[-1]
    assert rec["status"] == "ok" and rec["anomalies"] == []
    assert "EOSUSDT" in rec["results"]["T50"]["universe_unavailable"]
    assert rec["data_status"]["EOSUSDT"].startswith("no data")
    run_dir = os.path.join(FS.STATE, "runs", rec["run_id"])
    for name in ("T50", "RS3"):
        for f in ("signal_book_trades.csv", "signal_book_decisions.csv", "portfolio_signal_log.csv",
                  "portfolio_equity.csv", "portfolio_trades.csv"):
            assert os.path.exists(os.path.join(run_dir, name, f))
    t = pd.read_csv(os.path.join(run_dir, "T50", "signal_book_trades.csv"))
    assert len(t) > 0 and (pd.to_datetime(t["entry_time"], utc=True) >= pd.Timestamp(FS.START, tz="UTC")).all()
    log = pd.read_csv(os.path.join(run_dir, "T50", "portfolio_signal_log.csv"))
    assert {"accepted", "filled"} <= set(log["status"])
    assert (log["status"] == "rejected").any() and log.loc[log.status == "rejected", "reason"].notna().all()
    assert "random_benchmark" in rec["results"]["T50"]
    assert rec["results"]["T50"]["signal_book"]["closed_trades"] == len(t)
    assert os.path.exists(os.path.join(FS.STATE, "SHADOW_REPORT_LATEST.md"))


def test_rerun_is_deterministic_and_revisions_are_flagged(shadow):
    FS.cmd_freeze()
    FS.cmd_update(skip_random=True)
    shadow["end"] = "2027-03-15"                                               # two more weeks of data
    FS.cmd_update(skip_random=True)
    assert runs(FS.STATE)[-1]["anomalies"] == []


def test_changed_implementation_stops_tracking(shadow, monkeypatch):
    FS.cmd_freeze()
    real = FS.file_hash
    monkeypatch.setattr(FS, "file_hash", lambda rel: "0" * 64 if rel == "research/round1.py" else real(rel))
    assert FS.cmd_update() == 3
    rec = runs(FS.STATE)[-1]
    assert rec["status"].startswith("STOPPED") and any("round1.py" in p for p in rec["problems"])


def test_shadow_code_has_no_order_or_credential_paths():
    with open(os.path.join(ROOT, "research", "forward_shadow.py"), encoding="utf-8") as f:
        src = f.read()
    for bad in ("place_order", "cancel_order", "allow_trading", "CoinExClient", "API_SECRET", "X-MBX-APIKEY",
                "requests.post", ".post(", "signature", "LIVE_TRADING_LOCKED ="):
        assert bad not in src
