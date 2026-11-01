"""Paper trading engine.

Uses real CoinEx market data and the same signal engine, exit rules and risk
manager as the backtester. Orders go to ``PaperExecutor`` only; this module
has no code path to a real exchange order.

Each ``step()``:
1. Fetches completed candles for each pair and computes signals.
2. For an open position, replays every new completed bar that opened after
   the entry through the stop / trailing / take-profit rules, then checks the
   latest price, then the signal exit.
3. For a flat pair, decides once per new completed bar whether to enter
   (signal + confirmation, ML gate if enabled, rotation, risk limits).
4. Persists the ledger atomically and records every decision.
"""
from __future__ import annotations

import json
import logging
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from core.core_signals import generate_signals
from core.indicators import validate_ohlcv
from core.meta_learner import StrategyRotation
from core.positions import ExitRules, Position, close_trade_record, check_exit
from core.risk_manager import RiskLimits, RiskManager
from core.trade_execution import PaperExecutor, new_client_id

logger = logging.getLogger(__name__)

LEDGER_VERSION = 1
MAX_DECISIONS = 3000
MAX_EQUITY_POINTS = 5000
LOCK_STALE_SECONDS = 300


class LedgerBusy(RuntimeError):
    pass


@contextmanager
def ledger_lock(path: str, stale_after: float = LOCK_STALE_SECONDS):
    """Cross-process lock so the UI and ``run_paper.py`` never write the ledger at once."""
    lock = path + ".lock"
    os.makedirs(os.path.dirname(lock) or ".", exist_ok=True)
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        if time.time() - os.path.getmtime(lock) > stale_after:
            os.remove(lock)  # holder crashed
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        else:
            raise LedgerBusy("paper ledger is being updated by another process")
    try:
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        yield
    finally:
        try:
            os.remove(lock)
        except FileNotFoundError:
            pass


@dataclass
class PaperSettings:
    interval: str = "5m"
    lookback: int = 500
    strategy: str = "auto"
    threshold: float = 0.35
    initial_capital: float = 1000.0
    fee_rate: float = 0.002
    slippage_pct: float = 0.0005
    use_ml: bool = False
    ml_threshold: float = 0.55
    exit_rules: ExitRules = field(default_factory=ExitRules)
    risk: RiskLimits = field(default_factory=RiskLimits)
    signal_params: Dict[str, Any] = field(default_factory=dict)
    indicator_params: Dict[str, Any] = field(default_factory=dict)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _atomic_write_json(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1, default=str)
    os.replace(tmp, path)


def new_ledger(initial_capital: float) -> Dict[str, Any]:
    now = _now()
    return {
        "version": LEDGER_VERSION, "created": now.isoformat(), "initial_capital": float(initial_capital),
        "cash": float(initial_capital), "positions": {}, "pending_exits": {}, "trades": [], "decisions": [],
        "failed_orders": [], "equity_history": [], "last_bar": {}, "last_prices": {},
        "day": now.date().isoformat(), "day_start_equity": float(initial_capital),
        "peak_equity": float(initial_capital), "client_ids": [], "rotation": {},
    }


class PaperTrader:
    def __init__(self, settings: PaperSettings, state_dir: str,
                 data_fn: Optional[Callable[[str, str, int], pd.DataFrame]] = None,
                 price_fn: Optional[Callable[[str], float]] = None,
                 executor: Optional[PaperExecutor] = None, ml_bundle: Optional[Dict] = None,
                 now_fn: Callable[[], datetime] = _now):
        self.settings = settings
        self.ledger_path = os.path.join(state_dir, "paper_ledger.json")
        if data_fn is None:
            from core.core_data import fetch_klines
            data_fn = fetch_klines
        self.data_fn = data_fn
        self.price_fn = price_fn
        self.executor = executor or PaperExecutor(settings.fee_rate, settings.slippage_pct,
                                                  min_notional=settings.risk.min_notional)
        self.risk = RiskManager(settings.risk)
        self.ml_bundle = ml_bundle
        self.now_fn = now_fn
        self.rotation = StrategyRotation()
        self.ledger = self._load()
        self.executor.remember(self.ledger.get("client_ids", []))
        self.rotation.load(self.ledger.get("rotation"))

    # ------------------------------------------------------------ ledger
    def _load(self) -> Dict[str, Any]:
        if os.path.exists(self.ledger_path):
            try:
                with open(self.ledger_path, "r", encoding="utf-8") as f:
                    led = json.load(f)
                base = new_ledger(led.get("initial_capital", self.settings.initial_capital))
                base.update(led)
                return base
            except (OSError, ValueError) as exc:
                bad = self.ledger_path + ".corrupt"
                os.replace(self.ledger_path, bad)
                logger.error("Paper ledger unreadable (%s); moved to %s and starting fresh", exc, bad)
        return new_ledger(self.settings.initial_capital)

    def save(self) -> None:
        led = self.ledger
        led["decisions"] = led["decisions"][-MAX_DECISIONS:]
        led["equity_history"] = led["equity_history"][-MAX_EQUITY_POINTS:]
        led["client_ids"] = led["client_ids"][-5000:]
        led["rotation"] = self.rotation.to_dict()
        _atomic_write_json(self.ledger_path, led)

    def reset(self, initial_capital: Optional[float] = None) -> None:
        with ledger_lock(self.ledger_path):
            self.ledger = new_ledger(initial_capital or self.settings.initial_capital)
            self.rotation = StrategyRotation()
            self.save()

    # ----------------------------------------------------------- helpers
    @property
    def positions(self) -> Dict[str, Position]:
        return {p: Position.from_dict(d) for p, d in self.ledger["positions"].items()}

    def _store_position(self, pos: Position) -> None:
        self.ledger["positions"][pos.pair] = pos.to_dict()

    def equity(self, prices: Optional[Dict[str, float]] = None) -> float:
        prices = prices or self.ledger["last_prices"]
        total = self.ledger["cash"]
        for pair, pos in self.positions.items():
            total += pos.qty * float(prices.get(pair, pos.entry_price))
        return total

    def open_exposure(self) -> float:
        prices = self.ledger["last_prices"]
        return sum(p.qty * float(prices.get(k, p.entry_price)) for k, p in self.positions.items())

    def _decide(self, pair: str, action: str, reason: str, **extra) -> Dict[str, Any]:
        rec = {"time": self.now_fn().isoformat(), "pair": pair, "action": action, "reason": reason, **extra}
        self.ledger["decisions"].append(rec)
        logger.info("paper %s %s: %s", pair, action, reason)
        return rec

    def _roll_day(self) -> None:
        today = self.now_fn().date().isoformat()
        if self.ledger["day"] != today:
            self.ledger["day"] = today
            self.ledger["day_start_equity"] = self.equity()

    # ------------------------------------------------------------ orders
    def _sell(self, pos: Position, ref_price: float, reason: str) -> Dict[str, Any]:
        cid = new_client_id()
        self.ledger["client_ids"].append(cid)
        res = self.executor.market_order(pos.pair, "sell", pos.qty, ref_price, client_id=cid)
        if not res.ok:
            self.ledger["pending_exits"][pos.pair] = reason
            self.ledger["failed_orders"].append(res.to_dict())
            return self._decide(pos.pair, "SELL_FAILED", f"{reason}: {res.error}", price=ref_price)
        fee_rate = res.fee / (res.qty * res.price) if res.qty and res.price else self.executor.fee_rate
        sold_all = abs(res.qty - pos.qty) <= 1e-12
        pos_for_record = pos
        if not sold_all:  # precision floor left dust; record the sold part
            pos_for_record = Position.from_dict({**pos.to_dict(), "qty": res.qty,
                                                 "entry_fee": pos.entry_fee * res.qty / pos.qty})
        rec = close_trade_record(pos_for_record, res.price, fee_rate, res.timestamp, reason)
        rec["client_id"] = res.client_id
        self.ledger["cash"] += res.qty * res.price - res.fee
        self.ledger["trades"].append(rec)
        self.ledger["pending_exits"].pop(pos.pair, None)
        if sold_all:
            self.ledger["positions"].pop(pos.pair, None)
        else:
            rest = Position.from_dict({**pos.to_dict(), "qty": pos.qty - res.qty,
                                       "entry_fee": pos.entry_fee - pos_for_record.entry_fee})
            self._store_position(rest)
        self.rotation.update(self.ledger["trades"], self.now_fn())
        return self._decide(pos.pair, "SELL", reason, price=res.price, qty=res.qty, pnl=rec["pnl"],
                            client_id=res.client_id)

    def _buy(self, pair: str, row: pd.Series, ref_price: float) -> Dict[str, Any]:
        rules = self.settings.exit_rules
        atr = float(row["ATR"])
        fee = self.executor.fee_for(pair)
        stop = ref_price - rules.stop_atr_mult * atr
        info = self.executor.markets.get(pair)
        size = self.risk.position_size(self.equity(), self.ledger["cash"], ref_price, stop,
                                       self.open_exposure(), fee,
                                       min_amount=info.min_amount if info else 0.0,
                                       amount_precision=info.base_precision if info else None)
        meta = dict(signal=float(row["signal"]), confidence=float(row["confidence"]),
                    regime=str(row["regime"]), strategy=str(row["strategy"]), price=ref_price)
        if not size.ok:
            return self._decide(pair, "SKIP", f"sizing: {size.reason}", **meta)
        cid = new_client_id()
        self.ledger["client_ids"].append(cid)
        res = self.executor.market_order(pair, "buy", size.qty, ref_price, client_id=cid)
        if not res.ok:
            self.ledger["failed_orders"].append(res.to_dict())
            return self._decide(pair, "BUY_FAILED", res.error, **meta)
        # Levels are set from the actual fill price.
        stop_fill = res.price - rules.stop_atr_mult * atr
        pos = Position(pair=pair, qty=res.qty, entry_price=res.price, entry_time=res.timestamp, entry_fee=res.fee,
                       atr_at_entry=atr, stop_loss=stop_fill, take_profit=res.price + rules.tp_atr_mult * atr,
                       trail_mult=rules.trail_atr_mult if rules.use_trailing else 0.0, highest=res.price,
                       trailing_stop=max(0.0, res.price - rules.trail_atr_mult * atr) if rules.use_trailing else 0.0,
                       strategy=meta["strategy"], regime=meta["regime"], confidence=meta["confidence"])
        self.ledger["cash"] -= res.qty * res.price + res.fee
        self._store_position(pos)
        meta["price"] = res.price
        return self._decide(pair, "BUY", size.reason, qty=res.qty, client_id=res.client_id,
                            stop_loss=pos.stop_loss, take_profit=pos.take_profit, **meta)

    # -------------------------------------------------------------- step
    def reload(self) -> None:
        """Re-read the ledger from disk (another process may have advanced it)."""
        self.ledger = self._load()
        self.executor.remember(self.ledger.get("client_ids", []))
        self.rotation.load(self.ledger.get("rotation"))

    def step(self, pairs: List[str]) -> List[Dict[str, Any]]:
        with ledger_lock(self.ledger_path):
            self.reload()
            return self._step(pairs)

    def _step(self, pairs: List[str]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        self._roll_day()
        for pair in pairs:
            try:
                out.extend(self._process_pair(pair))
            except Exception as exc:  # one bad pair must not stop the others
                logger.exception("paper step failed for %s", pair)
                out.append(self._decide(pair, "ERROR", f"{type(exc).__name__}: {exc}"))
        eq = self.equity()
        self.ledger["peak_equity"] = max(self.ledger["peak_equity"], eq)
        self.ledger["equity_history"].append({"time": self.now_fn().isoformat(), "equity": eq,
                                              "cash": self.ledger["cash"]})
        if self.risk.drawdown_hit(eq, self.ledger["peak_equity"]) and not self.risk.kill_switch_active():
            self.risk.engage_kill_switch(f"max drawdown hit: equity {eq:.2f}")
            out.append(self._decide("*", "HALT", "max drawdown reached; kill switch engaged"))
        self.save()
        return out

    def _process_pair(self, pair: str) -> List[Dict[str, Any]]:
        s = self.settings
        df = self.data_fn(pair, s.interval, s.lookback)
        if not validate_ohlcv(df) or len(df) < 60:
            return [self._decide(pair, "DATA_ERROR", f"insufficient data ({0 if df is None else len(df)} bars)")]
        sig = generate_signals(df, strategy=s.strategy, threshold=s.threshold, params=s.signal_params,
                               indicator_params=s.indicator_params)
        last = sig.iloc[-1]
        last_ms = int(df["created_at"].iloc[-1]) if "created_at" in df.columns else len(df)
        prev_ms = self.ledger["last_bar"].get(pair)
        new_bar = prev_ms is None or last_ms > prev_ms
        ref = None
        if self.price_fn is not None:
            try:
                ref = float(self.price_fn(pair))
            except Exception as exc:
                logger.warning("ticker failed for %s: %s; using last close", pair, exc)
        if not ref or ref <= 0:
            ref = float(df["Close"].iloc[-1])
        self.ledger["last_prices"][pair] = ref
        decisions: List[Dict[str, Any]] = []

        pos = self.positions.get(pair)
        if pos is not None:
            if pair in self.ledger["pending_exits"]:
                decisions.append(self._sell(pos, ref, "retry:" + self.ledger["pending_exits"][pair]))
                self.ledger["last_bar"][pair] = last_ms
                return decisions
            entry_ms = int(pd.Timestamp(pos.entry_time).value // 10**6)
            new_bars = df[(df["created_at"] > (prev_ms or 0)) & (df["created_at"] >= entry_ms)] \
                if "created_at" in df.columns else df.tail(0)
            for _, bar in new_bars.iterrows():
                reason, raw = check_exit(pos, bar["Open"], bar["High"], bar["Low"], s.exit_rules)
                if reason:
                    self._store_position(pos)
                    decisions.append(self._sell(pos, raw if raw is not None else ref, reason))
                    self.ledger["last_bar"][pair] = last_ms
                    return decisions
            self._store_position(pos)
            if ref <= pos.effective_stop:
                decisions.append(self._sell(pos, ref, "trailing_stop" if pos.trailing_stop > pos.stop_loss
                                            else "stop_loss"))
            elif ref >= pos.take_profit:
                decisions.append(self._sell(pos, ref, "take_profit"))
            elif new_bar and bool(last["exit"]):
                decisions.append(self._sell(pos, ref, "signal_exit"))
            elif new_bar:
                decisions.append(self._decide(pair, "HOLD", "position open", signal=float(last["signal"]),
                                              price=ref, regime=str(last["regime"])))
            self.ledger["last_bar"][pair] = last_ms
            return decisions

        if not new_bar:
            return decisions
        self.ledger["last_bar"][pair] = last_ms
        meta = dict(signal=float(last["signal"]), confidence=float(last["confidence"]),
                    regime=str(last["regime"]), strategy=str(last["strategy"]), price=ref)
        if not bool(last["entry"]):
            return [self._decide(pair, "WAIT", "no entry signal", **meta)]
        if self.rotation.is_benched(meta["strategy"], self.now_fn()):
            return [self._decide(pair, "SKIP", f"strategy {meta['strategy']} benched by rotation", **meta)]
        if s.use_ml and self.ml_bundle is not None:
            from core.ml_filter import predict_proba
            proba = float(predict_proba(self.ml_bundle, sig).iloc[-1])
            meta["ml_proba"] = proba
            if not proba >= s.ml_threshold:
                return [self._decide(pair, "FILTERED", f"ml proba {proba:.2f} < {s.ml_threshold}", **meta)]
        ok, why = self.risk.can_open(pair, list(self.ledger["positions"]), self.equity(),
                                     self.ledger["day_start_equity"], self.ledger["peak_equity"])
        if not ok:
            return [self._decide(pair, "BLOCKED", why, **meta)]
        return [self._buy(pair, last, ref)]

    # ---------------------------------------------------- safety tooling
    def emergency_stop(self, reason: str = "manual emergency stop") -> List[Dict[str, Any]]:
        """Close every paper position at the last known price and engage the kill switch."""
        self.risk.engage_kill_switch(reason)
        with ledger_lock(self.ledger_path):
            self.reload()
            return self._close_all(reason)

    def _close_all(self, reason: str) -> List[Dict[str, Any]]:
        out = []
        for pair, pos in self.positions.items():
            out.append(self._sell(pos, float(self.ledger["last_prices"].get(pair, pos.entry_price)),
                                  "emergency_stop"))
        self.save()
        return out

    def reconcile(self) -> List[str]:
        """Check the ledger's internal consistency. Returns a list of problems (empty = OK)."""
        led = self.ledger
        issues = []
        realised = sum(float(t["pnl"]) for t in led["trades"])
        open_cost = sum(p.cost_basis for p in self.positions.values())
        expected_cash = led["initial_capital"] + realised - open_cost
        if abs(expected_cash - led["cash"]) > max(1e-6, 1e-9 * led["initial_capital"]):
            issues.append(f"cash {led['cash']:.8f} != expected {expected_cash:.8f}")
        if led["cash"] < -1e-9:
            issues.append(f"negative cash {led['cash']}")
        for pair, pos in self.positions.items():
            if pos.pair != pair:
                issues.append(f"position key {pair} != position pair {pos.pair}")
            if pos.qty <= 0:
                issues.append(f"{pair}: non-positive qty {pos.qty}")
            if pos.stop_loss >= pos.entry_price:
                issues.append(f"{pair}: stop {pos.stop_loss} not below entry {pos.entry_price}")
        ids = [t.get("client_id") for t in led["trades"] if t.get("client_id")]
        if len(ids) != len(set(ids)):
            issues.append("duplicate client_id in trade history")
        return issues

    def metrics(self) -> Dict[str, Any]:
        trades = pd.DataFrame(self.ledger["trades"])
        eq = self.equity()
        init = self.ledger["initial_capital"]
        out = {"equity": eq, "cash": self.ledger["cash"], "initial_capital": init,
               "total_return_pct": (eq / init - 1) * 100 if init else 0.0,
               "open_positions": len(self.ledger["positions"]), "trades": len(trades),
               "unrealised_pnl": sum(p.unrealised_pnl(float(self.ledger["last_prices"].get(k, p.entry_price)),
                                                      self.executor.fee_rate)
                                     for k, p in self.positions.items()),
               "kill_switch": self.risk.kill_switch_active()}
        if len(trades):
            out.update({"realised_pnl": float(trades["pnl"].sum()),
                        "win_rate_pct": float((trades["pnl"] > 0).mean() * 100),
                        "avg_return_pct": float(trades["return_pct"].mean()),
                        "fees_paid": float(trades["fees"].sum())})
        else:
            out.update({"realised_pnl": 0.0, "win_rate_pct": 0.0, "avg_return_pct": 0.0, "fees_paid": 0.0})
        hist = pd.DataFrame(self.ledger["equity_history"])
        if len(hist):
            curve = hist["equity"].astype(float)
            out["max_drawdown_pct"] = float(((curve.cummax() - curve) / curve.cummax()).max() * 100)
        else:
            out["max_drawdown_pct"] = 0.0
        return out
