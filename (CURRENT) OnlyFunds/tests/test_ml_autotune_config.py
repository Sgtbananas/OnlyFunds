import numpy as np
import pandas as pd
import pytest
import yaml

from core import ml_filter
from core.autotune import grid_combinations, optimise, walk_forward, ml_out_of_sample_backtest, _apply
from core.backtester import BacktestConfig
from core.meta_learner import StrategyRotation, select_strategy
from core.positions import ExitRules
from core.risk_manager import RiskLimits
from tests.synthetic import synthetic_ohlcv
from utils.config import Config, build_settings


# ------------------------------------------------------------------- ML
def test_triple_barrier_labels_handcrafted():
    n = 10
    df = pd.DataFrame({"Open": [100.0] * n, "High": [100.5] * n, "Low": [99.5] * n, "Close": [100.0] * n})
    df.loc[3, "High"] = 103          # TP for entries at t=0..2 (entry = open[t+1] = 100, tp = 102)
    df.loc[6, "Low"] = 97            # stop for entries at t=3..5
    atr = pd.Series(1.0, index=df.index)
    y = ml_filter.triple_barrier_labels(df, atr, ExitRules(1.0, 2.0, 1.0), horizon=3)
    assert list(y[:6]) == [1, 1, 1, 0, 0, 0]
    assert y[7:].isna().all()        # not enough future bars


def test_dataset_purge_gap():
    df = synthetic_ohlcv(2000, seed=2)
    Xtr, ytr, Xva, yva = ml_filter.build_dataset({"A": df}, ExitRules(), horizon=48, candidates_only=False)
    assert len(Xtr) and len(Xva)
    assert not Xtr.isna().any().any()
    assert len(Xtr) <= int(2000 * 0.7) - 48


def test_ml_does_not_claim_edge_on_random_walk(tmp_path):
    frames = {f"P{i}": synthetic_ohlcv(3000, seed=100 + i, cycle=0) for i in range(3)}
    bundle = ml_filter.train_filter(frames, save_path=str(tmp_path / "m.pkl"))
    rep = bundle["report"]
    assert rep["validation_samples"] > 30
    assert rep["useful"] is False, f"random-walk data must not show an edge: {rep}"
    loaded = ml_filter.load_bundle(str(tmp_path / "m.pkl"))
    assert loaded["report"] == rep
    sig = ml_filter.generate_signals(frames["P0"])
    p = ml_filter.predict_proba(loaded, sig)
    assert p.dropna().between(0, 1).all() and len(p) == len(sig)


def test_ml_too_little_data():
    with pytest.raises(ValueError):
        ml_filter.train_filter({"A": synthetic_ohlcv(200)}, save_path=None)


def test_ml_oos_backtest_runs():
    r = ml_out_of_sample_backtest(synthetic_ohlcv(3000, seed=8), BacktestConfig(), "S")
    assert set(r) == {"report", "rule_only", "ml_gated"}
    assert r["ml_gated"]["trades"] <= r["rule_only"]["trades"]


# -------------------------------------------------------------- autotune
SMALL_GRID = {"threshold": [0.3, 0.45], "exit_rules.stop_atr_mult": [1.0, 2.0]}


def test_grid_and_apply():
    assert len(grid_combinations(SMALL_GRID)) == 4
    c = _apply(BacktestConfig(), {"exit_rules.stop_atr_mult": 2.5, "threshold": 0.4,
                                  "indicator_params.ema_fast": 5})
    assert c.exit_rules.stop_atr_mult == 2.5 and c.threshold == 0.4 and c.indicator_params["ema_fast"] == 5
    assert BacktestConfig().exit_rules.stop_atr_mult == 1.5  # base untouched


def test_optimise_selects_on_train_only():
    df = synthetic_ohlcv(3000, seed=4)
    base = BacktestConfig(risk=RiskLimits(kill_switch_file="", max_cost_to_risk=10.0))  # 5m data: cost filter off
    r = optimise(df, base, grid=SMALL_GRID, min_trades=3)
    assert r["best_params"] is not None
    top = r["table"].iloc[0]
    assert top["threshold"] == r["best_params"]["threshold"]
    assert r["in_sample"]["bars"] == int(3000 * 0.7)
    assert isinstance(r["confirmed"], bool) and r["reason"]


def test_walk_forward():
    base = BacktestConfig(risk=RiskLimits(kill_switch_file="", max_cost_to_risk=10.0))
    r = walk_forward(synthetic_ohlcv(3000, seed=4), base, grid=SMALL_GRID, n_folds=3, min_trades=2)
    assert len(r["folds"]) == 3 and np.isfinite(r["oos_compounded_return_pct"])


# ------------------------------------------------------------ meta / rotation
def test_strategy_selection_is_regime_based():
    assert select_strategy("trending_up") == "trend_following"
    assert select_strategy("ranging") == "mean_reversion"
    assert select_strategy("trending_down") is None
    assert select_strategy("ranging", "stacked") == "stacked"
    with pytest.raises(ValueError):
        select_strategy("ranging", "nope")


def test_rotation_benches_losing_strategy():
    rot = StrategyRotation(min_trades=3, lookback_trades=5)
    trades = [{"strategy": "mean_reversion", "pnl": -1}] * 4 + [{"strategy": "trend_following", "pnl": 2}] * 4
    assert rot.update(trades) == ["mean_reversion"]
    assert rot.is_benched("mean_reversion") and not rot.is_benched("trend_following")
    rot2 = StrategyRotation()
    rot2.load(rot.to_dict())
    assert rot2.is_benched("mean_reversion")


# ----------------------------------------------------------------- config
def test_shipped_config_builds_valid_settings():
    cfg = Config()
    assert cfg.trading["mode"] == "paper"
    for profile in ("Conservative", "Normal", "Aggressive"):
        bt, paper = build_settings(cfg, profile)
        assert bt.risk.kill_switch_file == ""        # backtests never touch the real kill switch
        assert paper.risk.kill_switch_file            # paper trading honours it
    c_bt, _ = build_settings(cfg, "Conservative")
    a_bt, _ = build_settings(cfg, "Aggressive")
    assert c_bt.risk.risk_per_trade < a_bt.risk.risk_per_trade
    assert c_bt.threshold > a_bt.threshold


def test_overrides_applied():
    bt, paper = build_settings(Config(), "Normal", {"interval": "1h", "strategy": "stacked", "stop_atr_mult": 2.2,
                                                    "max_open_positions": 2})
    assert bt.interval == paper.interval == "1h"
    assert bt.strategy == "stacked" and bt.exit_rules.stop_atr_mult == 2.2
    assert paper.risk.max_open_positions == 2


def test_config_never_saves_secrets_or_live(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump({"api": {"api_key": "leak", "timeout": 5}, "trading": {"mode": "paper"}}))
    cfg = Config(str(path))
    assert "api_key" not in cfg.section("api")
    with pytest.raises(ValueError):
        cfg.save({"api": {"api_secret": "x"}})
    with pytest.raises(ValueError):
        cfg.save({"trading": {"mode": "live"}})
    cfg.save({"trading": {"profile": "Aggressive"}})
    on_disk = yaml.safe_load(path.read_text())
    assert on_disk["trading"]["profile"] == "Aggressive" and "api_key" not in on_disk["api"]


# ----------------------------------------------------------- trading window
def test_trading_window():
    from datetime import datetime, timezone
    from core.strategy_utils import within_trading_hours
    w = {"trading_windows": {"enabled": True, "start_time": "09:00", "end_time": "18:00", "timezone": "UTC"}}
    at = lambda h: datetime(2026, 1, 1, h, 0, tzinfo=timezone.utc)
    assert within_trading_hours(w, at(10)) and not within_trading_hours(w, at(20))
    night = {"trading_windows": {"enabled": True, "start_time": "22:00", "end_time": "06:00", "timezone": "UTC"}}
    assert within_trading_hours(night, at(23)) and within_trading_hours(night, at(3))
    assert not within_trading_hours(night, at(12))
    assert within_trading_hours({}, at(3))
    assert not within_trading_hours({"trading_windows": {"enabled": True, "start_time": "bad"}}, at(3))
