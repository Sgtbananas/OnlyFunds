"""Strategy selection and rotation.

Selection is driven by the *current market regime* (see
``core.core_signals.REGIME_STRATEGY``), not by which strategy happened to be
most profitable in the past. Recent performance is only used defensively:
``StrategyRotation`` benches a strategy whose recent live/paper trades are
losing, for a cooldown period, and the engine stands aside instead.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, Iterable, List, Optional

from core.core_signals import REGIME_STRATEGY, STRATEGIES


def select_strategy(regime: Optional[str], requested: str = "auto") -> Optional[str]:
    """Strategy to use for ``regime``. ``None`` means stand aside."""
    if requested != "auto":
        if requested not in STRATEGIES:
            raise ValueError(f"unknown strategy {requested}")
        return requested
    return REGIME_STRATEGY.get(regime) if regime else None


@dataclass
class StrategyRotation:
    lookback_trades: int = 10
    min_trades: int = 5
    min_profit_factor: float = 0.8
    cooldown_hours: float = 24.0
    benched_until: Dict[str, str] = field(default_factory=dict)

    def update(self, closed_trades: Iterable[Dict], now: Optional[datetime] = None) -> List[str]:
        """Re-evaluate every strategy; returns the strategies newly benched."""
        now = now or datetime.now(timezone.utc)
        by_strat: Dict[str, List[float]] = {}
        for t in closed_trades:
            by_strat.setdefault(t.get("strategy", ""), []).append(float(t.get("pnl", 0.0)))
        newly = []
        for strat, pnls in by_strat.items():
            recent = pnls[-self.lookback_trades:]
            if len(recent) < self.min_trades or self.is_benched(strat, now):
                continue
            gains = sum(p for p in recent if p > 0)
            losses = -sum(p for p in recent if p <= 0)
            pf = gains / losses if losses > 0 else float("inf")
            if pf < self.min_profit_factor:
                self.benched_until[strat] = (now + timedelta(hours=self.cooldown_hours)).isoformat()
                newly.append(strat)
        return newly

    def is_benched(self, strategy: str, now: Optional[datetime] = None) -> bool:
        until = self.benched_until.get(strategy)
        if not until:
            return False
        now = now or datetime.now(timezone.utc)
        return now < datetime.fromisoformat(until)

    def to_dict(self) -> Dict:
        return {"benched_until": dict(self.benched_until)}

    def load(self, d: Optional[Dict]) -> None:
        self.benched_until = dict((d or {}).get("benched_until", {}))
