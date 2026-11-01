"""Audit: are the negative results robust to the modelling assumptions?  DEV and VAL only.

Scenarios change ONE assumption at a time (then all optimistic ones together), applied to the
shipped (fixed) strategies. This is a robustness check of the conclusion, not a tuning step:
nothing here is used to select or change a strategy, and the TEST period is not touched.

    python scripts/audit/sensitivity.py --out docs/audit/sensitivity.json
"""
import argparse
import copy
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from core import validation as V  # noqa: E402
from core.history import DEFAULT_UNIVERSE, load_dataset  # noqa: E402
from scripts.real_market_validation import base_config  # noqa: E402


def scenarios(base):
    def mod(**kw):
        c = copy.deepcopy(base)
        for k, v in kw.items():
            if k.startswith("risk."):
                setattr(c.risk, k[5:], v)
            else:
                setattr(c, k, v)
        return c
    return {
        "baseline (as evaluated)": base,
        "no drawdown halt": mod(**{"risk.max_drawdown_pct": 1.0, "risk.max_daily_loss_pct": 1.0}),
        "no cost-to-risk filter": mod(**{"risk.max_cost_to_risk": 1e9}),
        "Binance fee 0.1%, no slippage": mod(fee_rate=0.001, slippage_pct=0.0, stop_slippage_pct=0.0),
        "zero fees, zero slippage": mod(fee_rate=0.0, slippage_pct=0.0, stop_slippage_pct=0.0),
        "all optimistic (no halts, no filter, 0.1% fee, no slippage)": mod(
            fee_rate=0.001, slippage_pct=0.0, stop_slippage_pct=0.0,
            **{"risk.max_drawdown_pct": 1.0, "risk.max_daily_loss_pct": 1.0, "risk.max_cost_to_risk": 1e9}),
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    data = {s: load_dataset(s) for s in DEFAULT_UNIVERSE}
    rows = []
    for name, cfg in scenarios(base_config()).items():
        for strat in V.STRATEGIES:
            for per in ("DEV", "VAL"):
                p = V.pooled(V.run_period(data, cfg, strat, None, *V.PERIODS[per]))
                rows.append({"scenario": name, "strategy": strat, "period": per, "trades": p["trades"],
                             "expectancy_r": p["expectancy_r"], "ci95_low_r": p["ci_low"],
                             "ci95_high_r": p["ci_high"], "profit_factor": p["profit_factor"],
                             "breadth": p["breadth"]})
                print(rows[-1], flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=1, default=str)
    return 0


if __name__ == "__main__":
    sys.exit(main())
