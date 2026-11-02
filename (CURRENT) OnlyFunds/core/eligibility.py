"""Promotion rules for validation candidates (audit fixes P3 and P4).

P3: a candidate reaches the final TEST only if it passed every validation criterion on VAL with a
sufficient sample, plus walk-forward. Every decision records its reasons and evidence. There is
no other path to TEST (ML included).

P4: an ML-gated variant is promoted only if (a) it is itself eligible under P3, (b) the model's own
held-out check said "useful", (c) on VAL it beats the same strategy without ML on net expectancy
with a bootstrap lower bound on the improvement above zero, (d) its drawdown and risk-adjusted
return are no worse, and (e) the improvement holds in every VAL sub-period with enough trades.
Otherwise ML stays disabled; the simpler rule-based system is preferred.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from core import validation as V

DEFAULT_RULES = {
    "min_trades": 100,               # pooled trades in the VAL period
    "min_coin_trades": 10,           # per coin, for breadth
    "min_breadth": 0.60,
    "min_profit_factor": 1.10,
    "require_ci_low_positive": True,
    "require_without_best_coin_positive": True,
    "min_walk_forward_positive": 3,  # of 4 folds
    "ml_min_trades": 100,
    "ml_min_trades_per_subperiod": 20,
    "ml_require_model_useful": True,
}


def candidate_eligibility(val: Dict[str, Any], walk_forward: Optional[pd.DataFrame],
                          rules: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """P3 gate on the VAL period. Returns {'eligible', 'reasons', 'evidence'}."""
    r = {**DEFAULT_RULES, **(rules or {})}
    reasons = []
    if val["trades"] < r["min_trades"]:
        reasons.append(f"sample too small: {val['trades']} VAL trades < {r['min_trades']}")
    if not (val["expectancy_r"] > 0):
        reasons.append(f"VAL expectancy {val['expectancy_r']:+.3f}R is not positive")
    if r["require_ci_low_positive"] and not (val["ci_low"] > 0):
        reasons.append(f"VAL 95% lower bound {val['ci_low']:+.3f}R is not above zero")
    if not (val["profit_factor"] >= r["min_profit_factor"]):
        reasons.append(f"VAL profit factor {val['profit_factor']:.2f} < {r['min_profit_factor']}")
    if not (val["breadth"] >= r["min_breadth"]):
        reasons.append(f"positive on {val['breadth']:.0%} of {val['eligible_coins']} coins < {r['min_breadth']:.0%}")
    if r["require_without_best_coin_positive"] and not (val["exp_r_without_best_coin"] > 0):
        reasons.append(f"depends on one coin: {val['exp_r_without_best_coin']:+.3f}R without the best coin")
    if walk_forward is None or len(walk_forward) == 0:
        reasons.append("no walk-forward evidence")
    else:
        pos = int((walk_forward["expectancy_r"] > 0).sum())
        if pos < r["min_walk_forward_positive"]:
            reasons.append(f"walk-forward positive in {pos}/{len(walk_forward)} folds "
                           f"< {r['min_walk_forward_positive']}")
    evidence = {k: val[k] for k in ("trades", "expectancy_r", "ci_low", "ci_high", "profit_factor", "breadth",
                                    "eligible_coins", "exp_r_without_best_coin", "worst_max_dd_pct",
                                    "mean_net_return_pct")}
    if walk_forward is not None and len(walk_forward):
        evidence["walk_forward_expectancy_r"] = walk_forward["expectancy_r"].round(4).tolist()
    return {"eligible": not reasons, "reasons": reasons or ["all VAL criteria met"], "evidence": evidence}


def bootstrap_diff_lower(a: np.ndarray, b: np.ndarray, n: int = 10_000, seed: int = 7) -> Tuple[float, float]:
    """95% bootstrap interval of mean(a) - mean(b) for two independent samples."""
    if len(a) < 2 or len(b) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    da = rng.choice(a, size=(n, len(a)), replace=True).mean(axis=1)
    db = rng.choice(b, size=(n, len(b)), replace=True).mean(axis=1)
    d = da - db
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def _risk_adjusted(p: Dict[str, Any]) -> float:
    c = p.get("per_coin")
    if c is None or len(c) == 0:
        return float("nan")
    dd = c["max_dd_pct"].replace(0, np.nan)
    return float((c["net_return_pct"] / dd).mean())


def ml_promotion(base: Dict[str, Any], ml: Dict[str, Any], base_r: np.ndarray, ml_r: np.ndarray,
                 model_report: Dict[str, Any], ml_eligibility: Dict[str, Any],
                 rules: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """P4 gate. ``base``/``ml`` are ``validation.pooled`` results on VAL; ``*_r`` the trade R-multiples."""
    r = {**DEFAULT_RULES, **(rules or {})}
    reasons = []
    if ml["trades"] < r["ml_min_trades"]:
        reasons.append(f"ML sample too small: {ml['trades']} VAL trades < {r['ml_min_trades']}")
    if r["ml_require_model_useful"] and not model_report.get("useful"):
        reasons.append(f"model's own held-out check says not useful (AUC {model_report.get('validation_auc', float('nan')):.3f})")
    if not ml_eligibility["eligible"]:
        reasons.append("ML variant fails the P3 validation criteria: " + "; ".join(ml_eligibility["reasons"]))
    lo, hi = bootstrap_diff_lower(np.asarray(ml_r, float), np.asarray(base_r, float))
    if not (lo > 0):
        reasons.append(f"no measurable expectancy improvement: ML − base 95% interval [{lo:+.3f}, {hi:+.3f}]R")
    ra_ml, ra_base = _risk_adjusted(ml), _risk_adjusted(base)
    if not (ra_ml >= ra_base):
        reasons.append(f"risk-adjusted return not better (ML {ra_ml:.3f} vs base {ra_base:.3f})")
    if not (ml.get("worst_max_dd_pct", np.inf) <= base.get("worst_max_dd_pct", -np.inf)):
        reasons.append(f"worse drawdown (ML {ml.get('worst_max_dd_pct'):.2f}% vs base {base.get('worst_max_dd_pct'):.2f}%)")
    by_ml, by_base = ml.get("by_year"), base.get("by_year")
    sub = []
    if by_ml is None or len(by_ml) == 0 or by_base is None or len(by_base) == 0:
        reasons.append("no per-period breakdown to check consistency")
    else:
        for period in by_base.index:
            n_ml = int(by_ml.loc[period, "trades"]) if period in by_ml.index else 0
            e_ml = float(by_ml.loc[period, "expectancy_r"]) if period in by_ml.index else float("nan")
            e_b = float(by_base.loc[period, "expectancy_r"])
            sub.append({"period": str(period), "ml_trades": n_ml, "ml_expectancy_r": e_ml, "base_expectancy_r": e_b})
            if n_ml < r["ml_min_trades_per_subperiod"]:
                reasons.append(f"{period}: only {n_ml} ML trades (< {r['ml_min_trades_per_subperiod']})")
            elif not (e_ml > e_b):
                reasons.append(f"{period}: ML {e_ml:+.3f}R not better than base {e_b:+.3f}R")
    evidence = {"ml_trades": ml["trades"], "base_trades": base["trades"],
                "ml_expectancy_r": ml["expectancy_r"], "base_expectancy_r": base["expectancy_r"],
                "improvement_ci95": [lo, hi], "risk_adjusted_ml": ra_ml, "risk_adjusted_base": ra_base,
                "worst_dd_ml": ml.get("worst_max_dd_pct"), "worst_dd_base": base.get("worst_max_dd_pct"),
                "model_useful": model_report.get("useful"), "model_auc": model_report.get("validation_auc"),
                "subperiods": sub}
    return {"promoted": not reasons, "reasons": reasons or ["ML demonstrably improves the same strategy"],
            "evidence": evidence}
