"""Cost-to-risk filter and strategy-logic sanity checks."""
import numpy as np
import pytest

from core.backtester import BacktestConfig, run_backtest
from core.risk_manager import RiskLimits, RiskManager
from tests.synthetic import _frame, synthetic_ohlcv


def test_cost_to_risk_formula():
    # price 100, stop 99 (1R = 1.0); costs = 100*(2*0.002 + 0.0005) + 99*0.002 = 0.648
    c = RiskManager.cost_to_risk(100, 99, 0.002, 0.0005, 0.002)
    assert c == pytest.approx(0.648)
    ok, why = RiskManager(RiskLimits(kill_switch_file="")).costs_acceptable(100, 99, 0.002, 0.0005, 0.002)
    assert not ok and "stop distance" in why
    assert RiskManager(RiskLimits(kill_switch_file="")).costs_acceptable(100, 95, 0.002, 0.0005, 0.002)[0]


def test_backtest_blocks_uneconomic_trades_on_tight_5m_stops():
    res = run_backtest(synthetic_ohlcv(3000, seed=1), BacktestConfig(), "S")
    blocked = res.decisions[res.decisions["action"] == "BLOCKED"]
    assert len(blocked) and blocked["reason"].str.contains("stop distance").all()
    assert res.metrics["trades"] < 5


def hourly(drift, seed, n=3000):
    rng = np.random.default_rng(seed)
    return _frame(100 * np.exp(np.cumsum(rng.normal(drift, 0.012, n))), rng, 0.006, "1h")


@pytest.mark.parametrize("seed", range(4))
def test_trend_following_logic_profits_in_economic_trend(seed):
    """Logic check, not market evidence: with stops wide relative to costs, trend following must
    make money in a persistent drift and lose on a driftless random walk."""
    cfg = BacktestConfig(strategy="trend_following", interval="1h")
    assert run_backtest(hourly(0.0012, seed), cfg, "T").metrics["total_return_pct"] > 0
    assert run_backtest(hourly(0.0, seed), cfg, "R").metrics["total_return_pct"] < 0
