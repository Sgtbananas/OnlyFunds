"""Order execution.

LIVE TRADING IS LOCKED. ``LIVE_TRADING_LOCKED`` is a code constant, not a
config flag: no config file, sidebar toggle or environment variable can
enable real orders. Unlocking requires a deliberate code change after
independent verification of the CoinEx order path.

``PaperExecutor`` simulates market orders against a reference price with
adverse slippage, exchange fees, precision/minimum checks, optional random
failures (to exercise failure handling) and duplicate client-id protection.
"""
from __future__ import annotations

import logging
import random
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from core.coinex_client import MarketInfo
from core.positions import apply_slippage

logger = logging.getLogger(__name__)

LIVE_TRADING_LOCKED = True


class LiveTradingDisabledError(RuntimeError):
    pass


@dataclass
class OrderResult:
    ok: bool
    client_id: str
    pair: str
    side: str
    qty: float
    price: float
    fee: float
    status: str           # "filled" | "rejected" | "failed" | "duplicate"
    error: str = ""
    order_id: str = ""
    timestamp: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def new_client_id(prefix: str = "of2") -> str:
    # CoinEx client_id: <= 32 chars, letters/digits/hyphen/underscore.
    return f"{prefix}-{uuid.uuid4().hex[:24]}"


class PaperExecutor:
    """Simulated exchange. Never sends anything to CoinEx."""

    is_live = False

    def __init__(self, fee_rate: float = 0.002, slippage_pct: float = 0.0005, failure_rate: float = 0.0,
                 seed: Optional[int] = None, markets: Optional[Dict[str, MarketInfo]] = None,
                 min_notional: float = 1.0):
        if not 0 <= failure_rate < 1:
            raise ValueError("failure_rate must be in [0, 1)")
        self.fee_rate = fee_rate
        self.slippage_pct = slippage_pct
        self.failure_rate = failure_rate
        self.markets = markets or {}
        self.min_notional = min_notional
        self._rng = random.Random(seed)
        self._seen_client_ids: set = set()

    def remember(self, client_ids) -> None:
        self._seen_client_ids.update(client_ids)

    def fee_for(self, pair: str) -> float:
        info = self.markets.get(pair)
        return max(self.fee_rate, info.taker_fee_rate) if info else self.fee_rate

    def market_order(self, pair: str, side: str, qty: float, ref_price: float,
                     client_id: Optional[str] = None) -> OrderResult:
        client_id = client_id or new_client_id()
        now = datetime.now(timezone.utc).isoformat()
        base = dict(client_id=client_id, pair=pair, side=side, timestamp=now)
        if client_id in self._seen_client_ids:
            return OrderResult(ok=False, qty=0.0, price=0.0, fee=0.0, status="duplicate",
                               error="duplicate client_id; order not re-sent", **base)
        self._seen_client_ids.add(client_id)
        if side not in ("buy", "sell"):
            return OrderResult(ok=False, qty=0, price=0, fee=0, status="rejected", error="bad side", **base)
        if not (ref_price > 0) or not (qty > 0):
            return OrderResult(ok=False, qty=0, price=0, fee=0, status="rejected",
                               error="non-positive price or quantity", **base)
        info = self.markets.get(pair)
        if info is not None:
            if not info.tradable:
                return OrderResult(ok=False, qty=0, price=0, fee=0, status="rejected",
                                   error=f"market not tradable ({info.status})", **base)
            qty = info.floor_amount(qty)
            if qty < info.min_amount:
                return OrderResult(ok=False, qty=0, price=0, fee=0, status="rejected",
                                   error=f"qty below min_amount {info.min_amount}", **base)
        if qty * ref_price < self.min_notional and side == "buy":
            return OrderResult(ok=False, qty=0, price=0, fee=0, status="rejected",
                               error=f"notional below {self.min_notional} USDT", **base)
        if self.failure_rate and self._rng.random() < self.failure_rate:
            return OrderResult(ok=False, qty=0, price=0, fee=0, status="failed",
                               error="simulated exchange failure", **base)
        fill = apply_slippage(ref_price, side, self.slippage_pct)
        fee = qty * fill * self.fee_for(pair)
        return OrderResult(ok=True, qty=qty, price=fill, fee=fee, status="filled",
                           order_id=f"PAPER-{uuid.uuid4().hex[:10]}", **base)


def place_live_order(*_args, **_kwargs):
    """Real orders are disabled. Kept so old call sites fail loudly, not silently."""
    raise LiveTradingDisabledError(
        "Live trading is locked in OnlyFunds 2 (core.trade_execution.LIVE_TRADING_LOCKED).")


def get_executor(mode: str = "paper", **kwargs) -> PaperExecutor:
    if mode == "live":
        if LIVE_TRADING_LOCKED:
            raise LiveTradingDisabledError("Live trading is locked; only paper mode is available.")
        raise LiveTradingDisabledError("Live executor not implemented pending independent verification.")
    return PaperExecutor(**kwargs)
