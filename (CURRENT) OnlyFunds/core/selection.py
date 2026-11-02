"""Evidence-gated strategy selection.

A strategy may be selected only if a v2 validation run approved it: it passed VAL eligibility
(P3/P4) AND its single final TEST (`credible_edge` true). Approvals carry their evidence and,
optionally, the regimes in which that evidence was established. Selection uses only information
available at decision time (the current causal regime) and validated evidence, never recent wins.

If nothing is approved, or the regime is uncertain, or no approval covers the current regime, the
answer is NO_TRADE with a reason. Today no strategy is approved, so OF2 stays inactive.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from core.run_store import DEFAULT_ROOT
from core.monitoring import read_json

NO_TRADE = None


@dataclass
class Approval:
    strategy: str
    variant: str
    params: Optional[Dict[str, Any]]
    plan: str
    run_id: str
    ci_low_r: float                      # validated lower bound of expectancy (TEST)
    trades: int
    regimes: List[str] = field(default_factory=list)   # trend classes with sufficient evidence; empty = none
    min_regime_confidence: float = 0.5


@dataclass
class Selection:
    strategy: Optional[str]
    reason: str
    approval: Optional[Approval] = None


def load_approvals(root: str = DEFAULT_ROOT) -> List[Approval]:
    """Approvals from immutable v2 runs only: stage2 credible_edge AND stage1 eligibility."""
    out: List[Approval] = []
    if not os.path.isdir(root):
        return out
    for plan in sorted(os.listdir(root)):
        pdir = os.path.join(root, plan)
        if plan.startswith("_") or not os.path.isdir(pdir):
            continue
        for run in sorted(os.listdir(pdir)):
            res_p = os.path.join(pdir, run, "stage2", "result.json")
            el_p = os.path.join(pdir, run, "stage1", "eligibility.json")
            if not (os.path.exists(res_p) and os.path.exists(el_p)):
                continue
            res, elig = read_json(res_p), read_json(el_p)
            for row in res.get("evaluated", []):
                d = elig.get(row["candidate"], {})
                if row.get("credible_edge") and d.get("eligible"):
                    out.append(Approval(d["strategy"], d["variant"], d.get("params"), plan, run,
                                        float(row["ci95_low_r"]), int(row["trades"]),
                                        regimes=list(d.get("approved_regimes", []))))
    return out


class StrategySelector:
    def __init__(self, approvals: Optional[List[Approval]] = None, min_trades: int = 100):
        self.approvals = [a for a in (approvals if approvals is not None else load_approvals())
                          if a.trades >= min_trades and a.ci_low_r > 0]

    def select(self, regime: Dict[str, Any]) -> Selection:
        if not self.approvals:
            return Selection(NO_TRADE, "no strategy has passed validation (no approved candidates)")
        label, trend = regime.get("label"), regime.get("trend")
        if label in (None, "uncertain") or trend is None:
            return Selection(NO_TRADE, "market regime uncertain")
        covering = [a for a in self.approvals if trend in a.regimes]
        if not covering:
            return Selection(NO_TRADE, f"no approved strategy has validated evidence for regime '{trend}'")
        covering = [a for a in covering if (regime.get("confidence") or 0) >= a.min_regime_confidence]
        if not covering:
            return Selection(NO_TRADE, "regime confidence below the level required by the approvals")
        best = max(covering, key=lambda a: a.ci_low_r)          # strongest validated evidence, not recent wins
        return Selection(best.strategy, f"approved by {best.plan}/{best.run_id} for regime '{trend}' "
                                        f"(TEST lower bound {best.ci_low_r:+.3f}R)", best)
