from core.backtester import BacktestConfig
from core.evaluation import evaluate, verdict
from tests.synthetic import synthetic_regime

GRID = {"threshold": [0.3, 0.45]}


def test_evaluation_structure_and_random_walk_not_credible():
    frames = {"rw": synthetic_regime("random_walk", 2500, seed=3), "range": synthetic_regime("range", 2500, seed=3)}
    ev = evaluate(frames, BacktestConfig(), strategies=("auto", "mean_reversion"), walk_forward_folds=2, grid=GRID)
    assert len(ev["fixed"]) == 4 and len(ev["walk_forward"]) == 4
    v = ev["verdict"]
    assert set(v.columns) == {"dataset", "strategy", "credible_edge", "reasons"}
    assert not v.loc[v["dataset"] == "rw", "credible_edge"].any(), "no edge can exist on a random walk"


def test_verdict_rules():
    import pandas as pd
    fixed = pd.DataFrame([{"dataset": "d", "strategy": "s", "trades": 100, "profit_factor": 1.5,
                           "total_return_pct": 10.0, "buy_and_hold_return_pct": 2.0}])
    wf = pd.DataFrame([{"dataset": "d", "strategy": "s", "oos_compounded_return_pct": 5.0,
                        "oos_buy_hold_compounded_pct": 1.0, "profitable_folds": 3, "folds": 4}])
    assert verdict(fixed, wf)["credible_edge"].iloc[0]
    wf.loc[0, "profitable_folds"] = 2
    assert not verdict(fixed, wf)["credible_edge"].iloc[0]
    fixed.loc[0, "profit_factor"] = 1.05
    assert "profit factor" in verdict(fixed, wf)["reasons"].iloc[0]
