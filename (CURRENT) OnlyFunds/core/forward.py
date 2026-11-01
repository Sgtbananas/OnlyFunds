"""Forward paper trading on live public market data (no real orders, ever).

ForwardPaperEngine, per step:
1. Refreshes COMPLETED candles and a live quote/depth snapshot for each symbol (LiveMarket).
2. Classifies the regime (core.regime, causal) and asks the evidence-gated StrategySelector.
3. Logs one observation per symbol per new completed candle: market snapshot, regime, every
   strategy's signal, the selection decision -> enough to reconstruct what OF2 saw and decided.
4. Records every entry opportunity that is NOT taken (any strategy, any reason) in the
   ShadowTracker, which later resolves it with the same stop/TP/cost rules on subsequent completed
   candles. This shows whether OF2 rejects poor trades or good ones, kept separate from paper results.
5. Runs the PaperTrader (bid/ask fills, stale-quote and gap guards, risk limits, persistent ledger);
   its entries must pass the selector gate. With no approved strategy, nothing is ever entered.

Forward results are stored under state/forward/ and are never mixed with historical or validation results.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd

from core.coinex_client import INTERVAL_SECONDS
from core.core_signals import STRATEGY_NAMES, generate_signals
from core.paper_trader import PaperSettings, PaperTrader, _atomic_write_json, ledger_lock
from core.positions import ExitRules, Position, apply_slippage, check_exit, planned_risk
from core.regime import RegimeParams, detect
from core.selection import StrategySelector
from core.trade_execution import LIVE_TRADING_LOCKED, PaperExecutor

logger = logging.getLogger(__name__)
STOP_REASONS = ("stop_loss", "trailing_stop")


def _utc() -> datetime:
    return datetime.now(timezone.utc)


class ShadowTracker:
    """Hypothetical outcomes of opportunities OF2 did not take (counterfactual, never real or paper PnL)."""

    def __init__(self, path: str, rules: ExitRules, fee_rate: float, slippage_pct: float, stop_slippage_pct: float,
                 max_bars: int = 240):
        self.path, self.rules = path, rules
        self.fee, self.slip, self.stop_slip, self.max_bars = fee_rate, slippage_pct, stop_slippage_pct, max_bars
        self.state = {"open": [], "resolved": []}
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                self.state = json.load(f)

    def save(self) -> None:
        _atomic_write_json(self.path, self.state)

    def _key(self, rec: Dict[str, Any]) -> str:
        return f"{rec.get('pair')}|{rec.get('strategy')}|{rec.get('candle_ms')}"

    def record(self, rec: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        atr, ref = rec.get("atr"), rec.get("ref_buy") or rec.get("ask") or rec.get("price")
        if not atr or not ref or atr <= 0 or rec.get("candle_ms") is None:
            return None
        key = self._key(rec)
        if any(o["key"] == key for o in self.state["open"]) or any(o["key"] == key for o in self.state["resolved"]):
            return None   # duplicate opportunity
        entry = apply_slippage(float(ref), "buy", self.slip)
        stop = entry - self.rules.stop_atr_mult * atr
        item = {"key": key, "pair": rec["pair"], "strategy": rec.get("strategy"), "regime": rec.get("regime"),
                "action": rec.get("action"), "reason": rec.get("reason"), "interval": rec.get("interval"),
                "candle_ms": int(rec["candle_ms"]), "recorded_utc": _utc().isoformat(), "entry": entry, "atr": atr,
                "stop": stop, "tp": entry + self.rules.tp_atr_mult * atr,
                "risk_per_unit": planned_risk(1.0, entry, stop, self.fee, self.stop_slip), "bars": 0}
        self.state["open"].append(item)
        return item

    def update(self, pair: str, candles: pd.DataFrame) -> List[Dict[str, Any]]:
        done = []
        for item in [o for o in self.state["open"] if o["pair"] == pair]:
            step = INTERVAL_SECONDS.get(item["interval"] or "1h", 3600) * 1000
            pos = Position(pair=pair, qty=1.0, entry_price=item["entry"], entry_time=item["recorded_utc"], entry_fee=0.0,
                           atr_at_entry=item["atr"], stop_loss=item["stop"], take_profit=item["tp"],
                           trail_mult=self.rules.trail_atr_mult if self.rules.use_trailing else 0.0,
                           highest=item.get("highest", item["entry"]),
                           trailing_stop=item.get("trailing", max(0.0, item["entry"] - self.rules.trail_atr_mult * item["atr"])
                                                  if self.rules.use_trailing else 0.0))
            after = candles[candles["created_at"] > item.get("last_ms", item["candle_ms"])]
            after = after[after["created_at"] >= item["candle_ms"] + step]          # bars after the decision bar
            outcome = None
            for _, bar in after.iterrows():
                item["bars"] += 1
                reason, raw = check_exit(pos, bar["Open"], bar["High"], bar["Low"], self.rules)
                item["last_ms"] = int(bar["created_at"])
                if reason and raw is not None:
                    outcome = (reason, apply_slippage(raw, "sell", self.stop_slip if reason in STOP_REASONS else self.slip))
                    break
                if item["bars"] >= self.max_bars:
                    outcome = ("expired", apply_slippage(float(bar["Close"]), "sell", self.slip))
                    break
            item["highest"], item["trailing"] = pos.highest, pos.trailing_stop
            if outcome:
                reason, px = outcome
                net = px * (1 - self.fee) - item["entry"] * (1 + self.fee)
                item.update({"exit_reason": reason, "exit": px, "r_multiple": net / item["risk_per_unit"],
                             "resolved_utc": _utc().isoformat()})
                self.state["open"].remove(item)
                self.state["resolved"].append(item)
                done.append(item)
        return done


class ForwardPaperEngine:
    def __init__(self, symbols: List[str], settings: PaperSettings, state_dir: str, live,
                 selector: Optional[StrategySelector] = None, regime_params: Optional[RegimeParams] = None,
                 now_fn=_utc):
        assert LIVE_TRADING_LOCKED, "forward engine is paper-only"
        self.symbols, self.settings, self.live, self.now_fn = list(symbols), settings, live, now_fn
        self.dir = state_dir
        os.makedirs(state_dir, exist_ok=True)
        self.selector = selector or StrategySelector()
        self.regime_params = regime_params or RegimeParams(persistence=4)
        self.shadow = ShadowTracker(os.path.join(state_dir, "shadow.json"), settings.exit_rules, settings.fee_rate,
                                    settings.slippage_pct, settings.stop_slippage_pct)
        self.state_path = os.path.join(state_dir, "engine_state.json")
        self.state = json.load(open(self.state_path, encoding="utf-8")) if os.path.exists(self.state_path) \
            else {"last_observed_ms": {}, "steps": 0, "errors": 0}
        self._regime: Dict[str, Dict[str, Any]] = {}
        self.trader = PaperTrader(settings, os.path.join(state_dir, "paper"),
                                  data_fn=lambda p, iv, n: self.live.completed_klines(p, iv, n),
                                  quote_fn=self.live.quote,
                                  executor=PaperExecutor(settings.fee_rate, settings.slippage_pct,
                                                         min_notional=settings.risk.min_notional),
                                  now_fn=now_fn, entry_gate=self._gate, on_rejected=self._on_rejected)

    # --- gate used by the PaperTrader before any entry
    def _gate(self, pair: str, last: pd.Series, df: pd.DataFrame):
        reg = self._regime.get(pair) or detect(df, self.regime_params).iloc[-1].to_dict()
        sel = self.selector.select(reg)
        if sel.strategy is None:
            return False, sel.reason
        if sel.strategy != str(last["strategy"]):
            return False, f"signal from '{last['strategy']}' but the approved strategy for this regime is '{sel.strategy}'"
        return True, sel.reason

    def _on_rejected(self, rec: Dict[str, Any]) -> None:
        self.shadow.record(rec)

    def _observe(self, pair: str, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        last_ms = int(df["created_at"].iloc[-1])
        if self.state["last_observed_ms"].get(pair) == last_ms:
            return None
        reg = detect(df, self.regime_params).iloc[-1].to_dict()
        self._regime[pair] = reg
        sel = self.selector.select(reg)
        signals = {}
        for strat in STRATEGY_NAMES:
            s = generate_signals(df, strategy=strat, threshold=self.settings.threshold,
                                 params=self.settings.signal_params, indicator_params=self.settings.indicator_params).iloc[-1]
            signals[strat] = {"signal": float(s["signal"]), "entry": bool(s["entry"]), "exit": bool(s["exit"]),
                              "sub_strategy": str(s["strategy"])}
            # opportunities of strategies the paper trader is not running are rejected here
            if s["entry"] and strat != self.settings.strategy:
                q = self.live.quotes.get(pair)
                self.shadow.record({"pair": pair, "strategy": str(s["strategy"]) if strat == "auto" else strat,
                                    "regime": str(s["regime"]), "action": "NOT_RUN",
                                    "reason": f"strategy '{strat}' not configured for paper trading; selector: {sel.reason}",
                                    "interval": self.settings.interval, "candle_ms": last_ms,
                                    "atr": float(s["ATR"]), "ref_buy": q.ask if q else float(df["Close"].iloc[-1])})
        snap = None
        try:
            snap = self.live.snapshot(pair, with_depth=True).to_dict()
        except Exception as exc:
            snap = {"error": f"{type(exc).__name__}: {exc}"}
        obs = {"observed_utc": self.now_fn().isoformat(), "pair": pair, "interval": self.settings.interval,
               "candle": {k: (float(v) if k != "created_at" else int(v)) for k, v in df.iloc[-1][
                   ["created_at", "Open", "High", "Low", "Close", "Volume"]].items()},
               "candle_open_utc": str(df["timestamp"].iloc[-1]), "snapshot": snap,
               "regime": {k: (v if not isinstance(v, float) or v == v else None) for k, v in reg.items()},
               "signals": signals, "selection": {"strategy": sel.strategy, "reason": sel.reason}}
        day = self.now_fn().strftime("%Y-%m-%d")
        path = os.path.join(self.dir, "observations", f"{day}.jsonl")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(obs, default=str) + "\n")
        self.state["last_observed_ms"][pair] = last_ms
        return obs

    def step(self) -> Dict[str, Any]:
        out = {"time": self.now_fn().isoformat(), "observed": [], "trader": [], "shadow_resolved": [], "errors": []}
        with ledger_lock(self.state_path):
            for pair in self.symbols:
                try:
                    df = self.live.completed_klines(pair, self.settings.interval, self.settings.lookback)
                    if self._observe(pair, df):
                        out["observed"].append(pair)
                    out["shadow_resolved"] += self.shadow.update(pair, df)
                except Exception as exc:
                    self.state["errors"] += 1
                    out["errors"].append(f"{pair}: {type(exc).__name__}: {exc}")
                    logger.exception("forward observation failed for %s", pair)
            self.shadow.save()
        out["trader"] = self.trader.step(self.symbols)   # holds its own ledger lock
        with ledger_lock(self.state_path):
            self.state["steps"] += 1
            self.state["last_step_utc"] = out["time"]
            self.state["feed_health"] = self.live.feed_health()
            _atomic_write_json(self.state_path, self.state)
            health_path = os.path.join(self.dir, "feed_health.jsonl")
            with open(health_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"time": out["time"], **self.state["feed_health"]}, default=str) + "\n")
        return out
