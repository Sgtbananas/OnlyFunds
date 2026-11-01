"""Historical data pipeline + real-market evaluation helpers (all offline: HTTP is faked)."""
import json

import numpy as np
import pandas as pd
import pytest

from core import history
from core.backtester import BacktestConfig, run_backtest
from core.core_signals import generate_signals
from core.positions import ExitRules, planned_risk
from core.risk_manager import RiskLimits
from tests.synthetic import synthetic_ohlcv

H = 3_600_000


def kline(t, o=100.0, h=101.0, l=99.0, c=100.5, v=10.0):
    return [t, str(o), str(h), str(l), str(c), str(v), t + H - 1, str(v * c), 42, "0", "0", "0"]


class FakeGet:
    """Serves a fixed candle list the way Binance pages it (startTime/endTime/limit)."""

    def __init__(self, rows):
        self.rows, self.calls = rows, []

    def __call__(self, url, params=None, timeout=None):
        self.calls.append(dict(params))
        sel = [r for r in self.rows if params["startTime"] <= r[0] <= params["endTime"]][: params["limit"]]

        class R:
            status_code = 200
            text = ""

            def json(self_inner):
                return sel
        return R()


def test_binance_pagination_and_gaps_not_filled(monkeypatch):
    start = history._ms("2024-01-01")
    times = [start + k * H for k in range(2500) if k not in (1000, 1001, 1777)]   # 3 candles genuinely missing
    get = FakeGet([kline(t) for t in times])
    monkeypatch.setattr(history.time, "sleep", lambda s: None)
    df = history.fetch_binance("BTCUSDT", "1h", "2024-01-01", pd.Timestamp(start + 2500 * H, unit="ms", tz="UTC")
                               .isoformat(), get=get)
    assert len(df) == 2497 and len(get.calls) == 3
    q = history.quality_report(df, "1h")
    assert q["missing_candles"] == 3 and q["gap_count"] == 2 and q["ok"]
    assert q["expected_rows"] == 2500


def test_quality_report_rejects_corrupt_candles():
    start = history._ms("2024-01-01")
    df = history.binance_rows_to_frame([kline(start + k * H) for k in range(10)], "1h")
    df.loc[3, "High"] = 50.0            # high below open/close
    df.loc[5, "Close"] = -1.0           # non-positive price
    df = pd.concat([df, df.iloc[[7]]])  # duplicate
    q = history.quality_report(df, "1h")
    assert not q["ok"]
    assert {"duplicates", "non_positive_prices", "high_below_open_close_or_low"} <= set(q["problems"])


def test_unclosed_candle_dropped():
    start = history._ms("2024-01-01")
    df = history.binance_rows_to_frame([kline(start), kline(start + H)], "1h", now_ms=start + H + 10)
    assert len(df) == 1


def test_manifest_checksum_detects_tampering(tmp_path):
    start = history._ms("2024-01-01")
    df = history.binance_rows_to_frame([kline(start + k * H) for k in range(50)], "1h")
    m = history.save_dataset(df, "BTCUSDT", "1h", {"test": True}, root=str(tmp_path))
    assert m["rows"] == 50 and m["quality"]["ok"] and len(m["sha256"]) == 64
    back = history.load_dataset("BTCUSDT", "1h", root=str(tmp_path))
    assert len(back) == 50 and back["timestamp"].dt.tz is not None
    csv, _ = history.dataset_paths("BTCUSDT", "1h", root=str(tmp_path))
    with open(csv, "a") as f:
        f.write("tampered\n")
    with pytest.raises(ValueError):
        history.load_dataset("BTCUSDT", "1h", root=str(tmp_path))


def test_cross_check():
    p = pd.DataFrame({"created_at": [0, 1, 2], "Close": [100.0, 101.0, 102.0]})
    c = history.cross_check(p, pd.Series([100.0, 101.0, 112.2], index=[0, 1, 2]))
    assert c["overlap"] == 3 and c["share_over_tolerance"] == pytest.approx(1 / 3)


# ----------------------------------------------------------------- backtest additions
def test_r_multiple_matches_planned_risk():
    df = synthetic_ohlcv(3000, seed=4)
    cfg = BacktestConfig(risk=RiskLimits(kill_switch_file="", max_cost_to_risk=10.0), close_at_end=True)
    res = run_backtest(df, cfg, "S")
    t = res.trades
    assert len(t) > 5
    assert (t["r_multiple"] == t["pnl"] / t["risk_amount"]).all()
    assert res.metrics["expectancy_r"] == pytest.approx(t["r_multiple"].mean())
    stops = t[(t["exit_reason"] == "stop_loss")]
    no_gap = stops[stops["exit_price"] >= stops["stop_loss"] * (1 - cfg.stop_slippage_pct) - 1e-9]
    assert (no_gap["r_multiple"] >= -1.0 - 1e-9).all()        # a planned stop-out costs at most 1R
    assert planned_risk(2, 100, 99, 0.0, 0.0) == pytest.approx(2.0)


def test_trade_from_keeps_history_for_indicators_but_no_trades_before():
    df = synthetic_ohlcv(3000, seed=4)
    cut = df["timestamp"].iloc[2000]
    cfg = BacktestConfig(risk=RiskLimits(kill_switch_file="", max_cost_to_risk=10.0), trade_from=str(cut))
    res = run_backtest(df, cfg, "S")
    assert (pd.to_datetime(res.trades["entry_time"]) >= cut).all()
    assert res.metrics["tradable_bars"] == 1000
    expected_bh = df["Close"].iloc[-1] * (1 - cfg.slippage_pct) * (1 - cfg.fee_rate) / (
        df["Open"].iloc[2000] * (1 + cfg.slippage_pct) * (1 + cfg.fee_rate)) - 1
    assert res.metrics["buy_and_hold_return_pct"] == pytest.approx(expected_bh * 100)


def test_backtest_blocks_entries_after_missing_candles():
    df = synthetic_ohlcv(3000, seed=4).drop(index=[1500]).reset_index(drop=True)
    cfg = BacktestConfig(risk=RiskLimits(kill_switch_file="", max_cost_to_risk=10.0))
    res = run_backtest(df, cfg, "S")
    d = res.decisions
    buys = d[d["action"] == "BUY"]["bar"]
    assert not ((buys > 1500) & (buys <= 1500 + 100)).any()
    assert (d["action"] == "DATA_GAP").any()


def test_existing_dataset_is_never_silently_replaced(tmp_path):
    start = history._ms("2024-01-01")
    df = history.binance_rows_to_frame([kline(start + k * H) for k in range(20)], "1h")
    history.save_dataset(df, "BTCUSDT", "1h", {}, root=str(tmp_path))
    with pytest.raises(history.DatasetExists):
        history.save_dataset(df.iloc[:5], "BTCUSDT", "1h", {}, root=str(tmp_path))
    assert len(history.load_dataset("BTCUSDT", "1h", root=str(tmp_path))) == 20
    m = history.save_dataset(df.iloc[:5], "BTCUSDT", "1h", {}, root=str(tmp_path), overwrite=True)
    assert m["rows"] == 5


def test_manifest_separates_exchange_facts_from_assumptions(tmp_path):
    start = history._ms("2024-01-01")
    df = history.binance_rows_to_frame([kline(start + k * H) for k in range(20)], "1h")
    m = history.save_dataset(df, "ETHUSDT", "1h", {}, root=str(tmp_path))
    ca = m["cost_assumptions"]
    assert "not verified" in ca["exchange"]["fee_rate_per_side"]["status"]
    assert ca["exchange"]["slippage_pct"]["status"].startswith("assumption")
    assert ca["evaluation_default"]["fee_rate_per_side"]["value"] == 0.002
