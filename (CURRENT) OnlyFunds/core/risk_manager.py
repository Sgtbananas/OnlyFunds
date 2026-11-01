"""Capital protection rules shared by backtesting and paper trading.

Sizing: risk a fixed fraction of equity between entry and stop
(``risk_per_trade``), then cap the position by ``max_position_pct`` of equity,
by remaining room under ``max_exposure_pct``, and by available cash. Orders
below the exchange minimum (or ``min_notional`` USDT) are rejected, never
rounded up.

Halts: kill-switch file, daily loss limit, and max drawdown all block new
entries. Exits are always allowed.
"""
from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass, asdict
from typing import Any, Dict, Mapping, Optional, Tuple

logger = logging.getLogger(__name__)

APP_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_KILL_SWITCH = os.path.join(APP_ROOT, "state", "KILL_SWITCH")


@dataclass
class RiskLimits:
    risk_per_trade: float = 0.005
    max_position_pct: float = 0.10
    max_exposure_pct: float = 0.30
    max_open_positions: int = 3
    max_daily_loss_pct: float = 0.03
    max_drawdown_pct: float = 0.10
    min_notional: float = 1.0
    max_cost_to_risk: float = 0.25   # skip trades whose round-trip costs exceed 25% of the stop distance
    kill_switch_file: str = DEFAULT_KILL_SWITCH

    @classmethod
    def from_mapping(cls, risk: Optional[Mapping[str, Any]] = None, **overrides) -> "RiskLimits":
        risk = dict(risk or {})
        risk.update({k: v for k, v in overrides.items() if v is not None})
        fields = cls.__dataclass_fields__
        kwargs = {k: risk[k] for k in fields if k in risk}
        # Legacy config keys
        if "risk_per_trade" not in kwargs and "per_trade" in risk:
            kwargs["risk_per_trade"] = risk["per_trade"]
        lim = cls(**kwargs)
        lim.validate()
        return lim

    def validate(self) -> None:
        for name in ("risk_per_trade", "max_position_pct", "max_exposure_pct", "max_daily_loss_pct",
                     "max_drawdown_pct"):
            v = getattr(self, name)
            if not (0 < v <= 1):
                raise ValueError(f"risk.{name} must be in (0, 1], got {v}")
        if self.risk_per_trade > 0.05:
            raise ValueError("risk.risk_per_trade above 5% of equity is not allowed")
        if self.max_open_positions < 1:
            raise ValueError("risk.max_open_positions must be >= 1")

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SizeDecision:
    qty: float
    notional: float
    reason: str

    @property
    def ok(self) -> bool:
        return self.qty > 0


class RiskManager:
    def __init__(self, limits: Optional[RiskLimits] = None):
        if limits is not None and not isinstance(limits, RiskLimits):
            # Accept a raw config mapping or utils.config.Config for compatibility.
            risk = limits.get("risk", {}) if hasattr(limits, "get") else {}
            limits = RiskLimits.from_mapping(risk)
        self.limits = limits or RiskLimits()

    # ------------------------------------------------------------- sizing
    def position_size(self, equity: float, cash: float, price: float, stop_price: float,
                      open_exposure: float = 0.0, fee_rate: float = 0.0,
                      min_amount: float = 0.0, amount_precision: Optional[int] = None,
                      stop_slippage_pct: float = 0.0, max_qty: Optional[float] = None) -> SizeDecision:
        """Size a long so that a stop-out loses at most ``risk_per_trade`` of equity.

        The per-unit loss includes the entry fee, the exit fee and stop slippage,
        not just the stop distance. Gaps through the stop can still lose more;
        that risk is capped separately by ``max_position_pct``.
        """
        L = self.limits
        if equity <= 0 or price <= 0 or cash <= 0:
            return SizeDecision(0.0, 0.0, "no equity/cash or invalid price")
        if stop_price >= price:
            return SizeDecision(0.0, 0.0, "stop must be below entry for a long")
        loss_per_unit = price * (1 + fee_rate) - stop_price * (1 - stop_slippage_pct) * (1 - fee_rate)
        qty = (equity * L.risk_per_trade) / loss_per_unit
        caps = {
            "max_position_pct": (equity * L.max_position_pct) / price,
            "max_exposure_pct": max(0.0, equity * L.max_exposure_pct - open_exposure) / price,
            "cash": cash / (price * (1 + fee_rate)),
        }
        if max_qty is not None:
            caps["liquidity"] = max(0.0, max_qty)
        limiting = "risk_per_trade"
        for name, cap in caps.items():
            if cap < qty:
                qty, limiting = cap, name
        if amount_precision is not None:
            q = 10 ** amount_precision
            qty = math.floor(qty * q + 1e-9) / q
        notional = qty * price
        if qty <= 0:
            return SizeDecision(0.0, 0.0, f"size is zero (limited by {limiting})")
        if notional < L.min_notional:
            return SizeDecision(0.0, 0.0, f"notional {notional:.4f} below min_notional {L.min_notional}")
        if qty < min_amount:
            return SizeDecision(0.0, 0.0, f"qty {qty} below exchange min_amount {min_amount}")
        return SizeDecision(qty, notional, f"limited by {limiting}")

    # ------------------------------------------------------------- halts
    def kill_switch_active(self) -> bool:
        return bool(self.limits.kill_switch_file) and os.path.exists(self.limits.kill_switch_file)

    def engage_kill_switch(self, reason: str) -> None:
        path = self.limits.kill_switch_file
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(reason)
        logger.critical("Kill switch engaged: %s", reason)

    def clear_kill_switch(self) -> None:
        if os.path.exists(self.limits.kill_switch_file):
            os.remove(self.limits.kill_switch_file)

    @staticmethod
    def cost_to_risk(price: float, stop_price: float, fee_rate: float, slippage_pct: float,
                     stop_slippage_pct: float) -> float:
        """Round-trip trading costs as a fraction of the stop distance (the 1R risk)."""
        risk = price - stop_price
        if risk <= 0:
            return float("inf")
        costs = price * (2 * fee_rate + slippage_pct) + stop_price * stop_slippage_pct
        return costs / risk

    def costs_acceptable(self, price: float, stop_price: float, fee_rate: float, slippage_pct: float,
                         stop_slippage_pct: float) -> Tuple[bool, str]:
        c = self.cost_to_risk(price, stop_price, fee_rate, slippage_pct, stop_slippage_pct)
        if c > self.limits.max_cost_to_risk:
            return False, (f"costs are {c:.0%} of the stop distance (max {self.limits.max_cost_to_risk:.0%}); "
                           "stop too tight for the fees — use a longer interval")
        return True, "ok"

    def daily_loss_hit(self, equity: float, day_start_equity: float) -> bool:
        if day_start_equity <= 0:
            return False
        return (equity / day_start_equity - 1.0) <= -self.limits.max_daily_loss_pct

    def drawdown_hit(self, equity: float, peak_equity: float) -> bool:
        if peak_equity <= 0:
            return False
        return (equity / peak_equity - 1.0) <= -self.limits.max_drawdown_pct

    def can_open(self, pair: str, open_pairs, equity: float, day_start_equity: float,
                 peak_equity: float, check_kill_switch: bool = True) -> Tuple[bool, str]:
        if check_kill_switch and self.kill_switch_active():
            return False, "kill switch active"
        if pair in open_pairs:
            return False, "position already open for pair (duplicate protection)"
        if len(open_pairs) >= self.limits.max_open_positions:
            return False, "max open positions reached"
        if self.daily_loss_hit(equity, day_start_equity):
            return False, "daily loss limit reached"
        if self.drawdown_hit(equity, peak_equity):
            return False, "max drawdown reached"
        return True, "ok"
