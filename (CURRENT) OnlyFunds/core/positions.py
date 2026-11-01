"""Position model and exit rules shared by the backtester and paper trader.

Exit order within a bar (OHLC gives no intrabar path, so be conservative):
1. Gap check at the open: if the bar opens through the stop or target, fill at the open.
2. Stop (max of hard stop and trailing stop) is checked before take-profit.
   If a bar touches both, the stop is assumed to hit first.
3. Only after the checks does the bar's high ratchet the trailing stop up,
   so a bar can never tighten its own stop and then hit it.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional, Tuple


@dataclass
class ExitRules:
    stop_atr_mult: float = 1.5
    tp_atr_mult: float = 3.0
    trail_atr_mult: float = 2.0
    use_trailing: bool = True
    max_hold_bars: int = 0  # 0 = no time stop

    def validate(self) -> None:
        if self.stop_atr_mult <= 0 or self.tp_atr_mult <= 0:
            raise ValueError("stop/take-profit ATR multipliers must be positive")
        if self.use_trailing and self.trail_atr_mult <= 0:
            raise ValueError("trailing ATR multiplier must be positive")


@dataclass
class Position:
    pair: str
    qty: float
    entry_price: float
    entry_time: str
    entry_fee: float
    atr_at_entry: float
    stop_loss: float
    take_profit: float
    trail_mult: float = 0.0
    highest: float = 0.0
    trailing_stop: float = 0.0
    strategy: str = ""
    regime: str = ""
    confidence: float = 0.0
    bars_held: int = 0
    risk_amount: float = 0.0  # planned loss if stopped out (incl. fees + stop slippage) = 1R
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    @property
    def cost_basis(self) -> float:
        return self.qty * self.entry_price + self.entry_fee

    @property
    def effective_stop(self) -> float:
        return max(self.stop_loss, self.trailing_stop) if self.trailing_stop > 0 else self.stop_loss

    def market_value(self, price: float) -> float:
        return self.qty * price

    def unrealised_pnl(self, price: float, fee_rate: float) -> float:
        """PnL if closed at ``price`` now, including both fees."""
        return self.qty * price * (1 - fee_rate) - self.cost_basis

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Position":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def open_position(pair: str, qty: float, fill_price: float, fee_rate: float, atr_value: float,
                  rules: ExitRules, entry_time: str, strategy: str = "", regime: str = "",
                  confidence: float = 0.0) -> Position:
    stop = fill_price - rules.stop_atr_mult * atr_value
    tp = fill_price + rules.tp_atr_mult * atr_value
    trail = fill_price - rules.trail_atr_mult * atr_value if rules.use_trailing else 0.0
    return Position(pair=pair, qty=qty, entry_price=fill_price, entry_time=entry_time,
                    entry_fee=qty * fill_price * fee_rate, atr_at_entry=atr_value, stop_loss=stop,
                    take_profit=tp, trail_mult=rules.trail_atr_mult if rules.use_trailing else 0.0,
                    highest=fill_price, trailing_stop=max(0.0, trail), strategy=strategy, regime=regime,
                    confidence=confidence)


def check_exit(pos: Position, bar_open: float, bar_high: float, bar_low: float,
               rules: Optional[ExitRules] = None) -> Tuple[Optional[str], Optional[float]]:
    """Return (reason, raw_fill_price) if the bar triggers an exit, else (None, None).

    Updates the trailing stop and ``bars_held`` when no exit happens.
    """
    stop = pos.effective_stop
    stop_reason = "trailing_stop" if pos.trailing_stop > pos.stop_loss else "stop_loss"
    if bar_open <= stop:
        return stop_reason, bar_open
    if bar_open >= pos.take_profit:
        return "take_profit", bar_open
    if bar_low <= stop:
        return stop_reason, stop
    if bar_high >= pos.take_profit:
        return "take_profit", pos.take_profit
    # Survived the bar: ratchet trailing stop with this bar's high.
    pos.bars_held += 1
    if bar_high > pos.highest:
        pos.highest = bar_high
    if pos.trail_mult > 0:
        pos.trailing_stop = max(pos.trailing_stop, pos.highest - pos.trail_mult * pos.atr_at_entry)
    if rules is not None and rules.max_hold_bars and pos.bars_held >= rules.max_hold_bars:
        return "time_stop", None  # caller exits at next open
    return None, None


def planned_risk(qty: float, entry: float, stop: float, fee_rate: float, stop_slippage_pct: float) -> float:
    """Loss if the stop fills as planned: same formula the risk manager sizes with (1R)."""
    return qty * (entry * (1 + fee_rate) - stop * (1 - stop_slippage_pct) * (1 - fee_rate))


def apply_slippage(price: float, side: str, slippage_pct: float) -> float:
    """Adverse slippage: buys fill higher, sells fill lower."""
    return price * (1 + slippage_pct) if side == "buy" else price * (1 - slippage_pct)


def close_trade_record(pos: Position, exit_price: float, fee_rate: float, exit_time: str,
                       reason: str) -> Dict[str, Any]:
    exit_fee = pos.qty * exit_price * fee_rate
    proceeds = pos.qty * exit_price - exit_fee
    pnl = proceeds - pos.cost_basis
    return {
        "id": pos.id, "pair": pos.pair, "strategy": pos.strategy, "regime": pos.regime,
        "entry_time": pos.entry_time, "exit_time": exit_time, "qty": pos.qty,
        "entry_price": pos.entry_price, "exit_price": exit_price, "stop_loss": pos.stop_loss,
        "take_profit": pos.take_profit, "entry_fee": pos.entry_fee, "exit_fee": exit_fee,
        "fees": pos.entry_fee + exit_fee, "pnl": pnl, "return_pct": pnl / pos.cost_basis * 100.0,
        "bars_held": pos.bars_held, "exit_reason": reason, "confidence": pos.confidence,
        "risk_amount": pos.risk_amount,
        "r_multiple": pnl / pos.risk_amount if pos.risk_amount > 0 else float("nan"),
    }
