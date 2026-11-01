"""Strategy-aware, single-pair backtester.

Timeline for bar ``i`` (no look-ahead):
1. At the open of bar i, execute what was decided at the close of bar i-1
   (entry or signal/time exit), with adverse slippage and fees.
2. During bar i, check stop / trailing stop / take-profit with the bar's OHLC
   (see ``core.positions.check_exit``).
3. At the close of bar i, mark equity to market and decide using the signal
   computed from bars <= i.

The same signal engine, exit rules and risk manager are used by the paper
trader, so backtest and paper results are directly comparable.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from core.core_signals import generate_signals
from core.data_quality import missing_candles
from core.indicators import validate_ohlcv, warmup_bars
from core.positions import (ExitRules, Position, apply_slippage, check_exit, close_trade_record, open_position,
                            planned_risk)
from core.coinex_client import INTERVAL_SECONDS
from core.risk_manager import RiskLimits, RiskManager

logger = logging.getLogger(__name__)

STOP_REASONS = ("stop_loss", "trailing_stop")

BARS_PER_YEAR = {"1m": 525600, "3m": 175200, "5m": 105120, "15m": 35040, "30m": 17520, "1h": 8760,
                 "2h": 4380, "4h": 2190, "6h": 1460, "12h": 730, "1d": 365, "3d": 122, "1w": 52}


@dataclass
class BacktestConfig:
    initial_capital: float = 1000.0
    fee_rate: float = 0.002          # CoinEx default spot taker fee
    slippage_pct: float = 0.0005
    stop_slippage_pct: float = 0.002  # stops trigger in fast markets: assume a worse fill than normal exits
    max_volume_participation: Optional[float] = 0.10  # max fraction of the signal bar's base volume
    strategy: str = "auto"
    threshold: float = 0.35
    ml_threshold: float = 0.55
    interval: str = "5m"
    exit_rules: ExitRules = field(default_factory=ExitRules)
    risk: RiskLimits = field(default_factory=lambda: RiskLimits(kill_switch_file=""))
    signal_params: Dict[str, Any] = field(default_factory=dict)
    indicator_params: Dict[str, Any] = field(default_factory=dict)
    close_at_end: bool = False
    trade_from: Optional[str] = None  # earlier rows are indicator history only (no trading, no metrics window)
    gap_check_bars: int = 100         # no entries if a candle is missing in this many recent bars (as paper)

    def as_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["risk"].pop("kill_switch_file", None)
        return d


@dataclass
class BacktestResult:
    pair: str
    trades: pd.DataFrame
    equity: pd.DataFrame
    metrics: Dict[str, Any]
    decisions: pd.DataFrame
    open_position: Optional[Dict[str, Any]] = None

    def summary_frame(self) -> pd.DataFrame:
        return pd.DataFrame([{"pair": self.pair, **self.metrics}])


def _ts(df: pd.DataFrame, i: int) -> str:
    if "timestamp" in df.columns:
        return str(df["timestamp"].iloc[i])
    return str(df.index[i])


def run_backtest(data: pd.DataFrame, config: Optional[BacktestConfig] = None, pair: str = "PAIR",
                 ml_proba: Optional[pd.Series] = None, signals: Optional[pd.DataFrame] = None) -> BacktestResult:
    """Run one backtest.

    ``ml_proba`` (optional) is a probability series aligned to ``data``; an
    entry also needs ``ml_proba >= config.ml_threshold``. It must come from a
    model trained only on data *before* this segment (see ``core.ml_filter``).
    """
    cfg = config or BacktestConfig()
    cfg.exit_rules.validate()
    if not validate_ohlcv(data):
        raise ValueError("backtest needs non-empty OHLC data with positive prices")
    df = data.reset_index(drop=True)
    sig = signals if signals is not None else generate_signals(
        df, strategy=cfg.strategy, threshold=cfg.threshold, params=cfg.signal_params,
        indicator_params=cfg.indicator_params)
    sig = sig.reset_index(drop=True)
    if ml_proba is not None:
        ml = pd.Series(np.asarray(ml_proba, dtype=float), index=df.index)
    else:
        ml = None

    risk = RiskManager(cfg.risk)
    fee, slip = cfg.fee_rate, cfg.slippage_pct
    opens, highs, lows, closes = (df[c].to_numpy(float) for c in ("Open", "High", "Low", "Close"))
    volumes = df["Volume"].to_numpy(float) if "Volume" in df.columns else None
    stop_slip = cfg.stop_slippage_pct
    atr = sig["ATR"].to_numpy(float)
    days = (pd.to_datetime(df["timestamp"], utc=True).dt.date.to_numpy() if "timestamp" in df.columns
            else np.zeros(len(df), dtype=int))

    cash = cfg.initial_capital
    pos: Optional[Position] = None
    pending: Optional[str] = None       # "enter" | "exit:<reason>"
    pending_meta: Dict[str, Any] = {}
    trades: List[Dict[str, Any]] = []
    decisions: List[Dict[str, Any]] = []
    equity_rows: List[Dict[str, Any]] = []
    peak = cfg.initial_capital
    day_start_equity = cfg.initial_capital
    current_day = days[0] if len(days) else None
    halted_reason = ""
    fees_paid = 0.0
    start = max(1, warmup_bars(cfg.indicator_params))
    if cfg.trade_from is not None and "timestamp" in df.columns:
        tf = pd.Timestamp(cfg.trade_from)
        tf = tf.tz_localize("UTC") if tf.tzinfo is None else tf.tz_convert("UTC")
        ts_utc = pd.to_datetime(df["timestamp"], utc=True)
        first = int((ts_utc < tf).sum())  # timestamps are sorted: index of the first bar at/after trade_from
        start = max(start, first - 1)  # decision at bar first-1's close -> first fill at bar `first`
    recent_gap = _recent_gap_flags(df, cfg.interval, cfg.gap_check_bars)

    for i in range(len(df)):
        if days[i] != current_day:
            current_day = days[i]
            day_start_equity = cash + (pos.market_value(opens[i]) if pos else 0.0)

        # 1) execute pending action at this bar's open
        if pending == "enter" and pos is None:
            fill = apply_slippage(opens[i], "buy", slip)
            atr_v = pending_meta["atr"]
            rules = cfg.exit_rules
            stop = fill - rules.stop_atr_mult * atr_v
            max_qty = None
            if cfg.max_volume_participation and volumes is not None and np.isfinite(volumes[i - 1]):
                max_qty = cfg.max_volume_participation * volumes[i - 1]  # signal bar: known at decision time
            size = risk.position_size(cash, cash, fill, stop, 0.0, fee, stop_slippage_pct=stop_slip,
                                      max_qty=max_qty)
            if size.ok:
                pos = open_position(pair, size.qty, fill, fee, atr_v, rules, _ts(df, i),
                                    pending_meta["strategy"], pending_meta["regime"], pending_meta["confidence"])
                pos.risk_amount = planned_risk(size.qty, fill, pos.stop_loss, fee, stop_slip)
                cash -= size.qty * fill + pos.entry_fee
                fees_paid += pos.entry_fee
            decisions.append({"bar": i, "time": _ts(df, i), "action": "BUY" if size.ok else "SKIP",
                              "reason": size.reason, "price": fill})
        elif pending and pending.startswith("exit:") and pos is not None:
            fill = apply_slippage(opens[i], "sell", slip)
            rec = close_trade_record(pos, fill, fee, _ts(df, i), pending.split(":", 1)[1])
            cash += pos.qty * fill - rec["exit_fee"]
            fees_paid += rec["exit_fee"]
            trades.append(rec)
            decisions.append({"bar": i, "time": _ts(df, i), "action": "SELL", "reason": rec["exit_reason"],
                              "price": fill})
            pos = None
        pending, pending_meta = None, {}

        # 2) intrabar stop / target / trailing
        if pos is not None:
            reason, raw = check_exit(pos, opens[i], highs[i], lows[i], cfg.exit_rules)
            if reason and raw is not None:
                # Market-order exit, same as the paper executor: always adverse slippage,
                # wider for stops (they trigger when price is moving against us).
                fill = apply_slippage(raw, "sell", stop_slip if reason in STOP_REASONS else slip)
                rec = close_trade_record(pos, fill, fee, _ts(df, i), reason)
                cash += pos.qty * fill - rec["exit_fee"]
                fees_paid += rec["exit_fee"]
                trades.append(rec)
                decisions.append({"bar": i, "time": _ts(df, i), "action": "SELL", "reason": reason, "price": fill})
                pos = None
            elif reason == "time_stop":
                pending = "exit:time_stop"

        # 3) mark to market and decide at the close
        equity = cash + (pos.market_value(closes[i]) if pos else 0.0)
        equity_low = cash + (pos.market_value(lows[i]) if pos else 0.0)  # worst mark inside the bar
        peak = max(peak, equity)
        equity_rows.append({"bar": i, "time": _ts(df, i), "equity": equity, "equity_low": min(equity, equity_low),
                            "cash": cash, "in_position": pos is not None})
        if i < start or i == len(df) - 1 or pending:
            continue
        if risk.drawdown_hit(equity, peak) and not halted_reason:
            halted_reason = f"max drawdown {cfg.risk.max_drawdown_pct:.0%} hit at bar {i}"
            decisions.append({"bar": i, "time": _ts(df, i), "action": "HALT", "reason": halted_reason,
                              "price": closes[i]})
        row = sig.iloc[i]
        if pos is not None:
            if bool(row["exit"]):
                pending = "exit:signal_exit"
        elif bool(row["entry"]) and not halted_reason and np.isfinite(atr[i]) and atr[i] > 0:
            if recent_gap[i]:
                decisions.append({"bar": i, "time": _ts(df, i), "action": "DATA_GAP",
                                  "reason": f"missing candle within the last {cfg.gap_check_bars} bars",
                                  "price": closes[i]})
                continue
            if ml is not None and not (ml.iloc[i] >= cfg.ml_threshold):
                decisions.append({"bar": i, "time": _ts(df, i), "action": "FILTERED",
                                  "reason": f"ml proba {ml.iloc[i]:.2f} < {cfg.ml_threshold}", "price": closes[i]})
                continue
            ok, why = risk.can_open(pair, [], equity, day_start_equity, peak, check_kill_switch=False)
            if ok:
                ok, why = risk.costs_acceptable(closes[i], closes[i] - cfg.exit_rules.stop_atr_mult * atr[i],
                                                fee, slip, stop_slip)
            if not ok:
                decisions.append({"bar": i, "time": _ts(df, i), "action": "BLOCKED", "reason": why,
                                  "price": closes[i]})
                continue
            pending = "enter"
            pending_meta = {"atr": float(atr[i]), "strategy": str(row["strategy"]), "regime": str(row["regime"]),
                            "confidence": float(row["confidence"])}

    last_close = closes[-1]
    open_pos_dict = None
    unrealised = 0.0
    if pos is not None:
        if cfg.close_at_end:
            fill = apply_slippage(last_close, "sell", slip)
            rec = close_trade_record(pos, fill, fee, _ts(df, len(df) - 1), "end_of_data")
            cash += pos.qty * fill - rec["exit_fee"]
            fees_paid += rec["exit_fee"]
            trades.append(rec)
            equity_rows[-1]["equity"] = cash
            equity_rows[-1]["equity_low"] = min(equity_rows[-1]["equity_low"], cash)
            equity_rows[-1]["cash"] = cash
            pos = None
        else:
            unrealised = pos.unrealised_pnl(last_close, fee)
            open_pos_dict = pos.to_dict()

    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(equity_rows)
    metrics = compute_metrics(trades_df, equity_df, cfg, df, unrealised, fees_paid, start=start)
    metrics["halted"] = halted_reason
    return BacktestResult(pair=pair, trades=trades_df, equity=equity_df, metrics=metrics,
                          decisions=pd.DataFrame(decisions), open_position=open_pos_dict)


def _recent_gap_flags(df: pd.DataFrame, interval: str, window: int) -> np.ndarray:
    """flags[i] = a candle is missing between bar i-window and bar i (same rule as the paper trader)."""
    n = len(df)
    if "created_at" not in df.columns or interval not in INTERVAL_SECONDS or not window or n < 2:
        return np.zeros(n, dtype=bool)
    step = INTERVAL_SECONDS[interval] * 1000
    gap_after = np.r_[False, np.diff(df["created_at"].to_numpy("int64")) > step]  # bar i follows a gap
    c = np.cumsum(gap_after)
    lagged = np.r_[np.zeros(window, dtype=c.dtype), c[:-window]] if n > window else np.zeros(n, dtype=c.dtype)
    return (c - lagged) > 0


def compute_metrics(trades: pd.DataFrame, equity: pd.DataFrame, cfg: BacktestConfig, df: pd.DataFrame,
                    unrealised: float = 0.0, fees_paid: float = 0.0, start: int = 0) -> Dict[str, Any]:
    init = cfg.initial_capital
    final_equity = float(equity["equity"].iloc[-1]) if not equity.empty else init
    window = equity.iloc[start:] if len(equity) > start + 1 else equity  # trading window only
    curve = window["equity"].to_numpy(float) if not window.empty else np.array([init])
    lows = window["equity_low"].to_numpy(float) if "equity_low" in window else curve
    # Drawdown: running peak of closing equity vs the worst intrabar mark of each bar.
    prev_peak = np.maximum.accumulate(np.r_[init, curve[:-1]])
    max_dd = float(np.max(np.maximum(prev_peak - lows, 0) / prev_peak) * 100) if len(curve) else 0.0
    bar_ret = np.diff(curve) / curve[:-1] if len(curve) > 1 else np.array([])
    ann = BARS_PER_YEAR.get(cfg.interval, 105120)
    sd = np.std(bar_ret) if len(bar_ret) > 1 else 0.0
    sharpe = float(np.mean(bar_ret) / sd * np.sqrt(ann)) if sd > 0 else 0.0
    downside = bar_ret[bar_ret < 0]
    dsd = np.sqrt(np.mean(downside ** 2)) if len(downside) else 0.0
    sortino = float(np.mean(bar_ret) / dsd * np.sqrt(ann)) if dsd > 0 else 0.0
    n = len(trades)
    by_regime = {}
    if n:
        pnl = trades["pnl"]
        wins, losses = pnl[pnl > 0], pnl[pnl <= 0]
        gross_loss = -losses.sum()
        profit_factor = float(wins.sum() / gross_loss) if gross_loss > 0 else (float("inf") if len(wins) else 0.0)
        realised = float(pnl.sum())
        win_rate = float(len(wins) / n * 100)
        avg_ret = float(trades["return_pct"].mean())
        avg_win = float(trades.loc[pnl > 0, "return_pct"].mean()) if len(wins) else 0.0
        avg_loss = float(trades.loc[pnl <= 0, "return_pct"].mean()) if len(losses) else 0.0
        exit_reasons = trades["exit_reason"].value_counts().to_dict()
        expectancy = float(pnl.mean())
        r = trades["r_multiple"].astype(float) if "r_multiple" in trades else pd.Series(dtype=float)
        expectancy_r = float(r.mean()) if r.notna().any() else 0.0
        avg_win_r = float(r[pnl > 0].mean()) if (pnl > 0).any() else 0.0
        avg_loss_r = float(r[pnl <= 0].mean()) if (pnl <= 0).any() else 0.0
        for reg, g in trades.groupby(trades["regime"].fillna("unknown")):
            by_regime[str(reg)] = {"trades": int(len(g)), "pnl": float(g["pnl"].sum()),
                                   "expectancy_r": float(g["r_multiple"].mean()) if "r_multiple" in g else 0.0,
                                   "win_rate_pct": float((g["pnl"] > 0).mean() * 100),
                                   "expectancy": float(g["pnl"].mean())}
    else:
        profit_factor = realised = win_rate = avg_ret = avg_win = avg_loss = expectancy = 0.0
        expectancy_r = avg_win_r = avg_loss_r = 0.0
        exit_reasons = {}
    # Buy-and-hold over the same tradable window, paying the same entry/exit costs.
    first = min(start + 1, len(df) - 1)
    o = float(df["Open"].iloc[first])
    c = float(df["Close"].iloc[-1])
    s_, f_ = cfg.slippage_pct, cfg.fee_rate
    bh = (c * (1 - s_) * (1 - f_) / (o * (1 + s_) * (1 + f_)) - 1) * 100 if o > 0 else 0.0
    exposure = float(equity["in_position"].iloc[start:].mean() * 100) if len(equity) > start else 0.0
    days = _span_days(df, first)
    total_ret = (final_equity / init - 1) * 100
    return {
        "initial_capital": init, "final_equity": final_equity, "total_return_pct": total_ret,
        "realised_pnl": realised, "unrealised_pnl": float(unrealised), "fees_paid": float(fees_paid),
        "trades": n, "trades_per_day": n / days if days > 0 else 0.0,
        "win_rate_pct": win_rate, "avg_return_pct": avg_ret, "avg_win_pct": avg_win, "avg_loss_pct": avg_loss,
        "expectancy": expectancy, "expectancy_r": expectancy_r, "avg_win_r": avg_win_r, "avg_loss_r": avg_loss_r,
        "profit_factor": profit_factor,
        "max_drawdown_pct": max_dd, "return_over_max_dd": total_ret / max_dd if max_dd > 0 else 0.0,
        "sharpe": sharpe, "sortino": sortino, "exposure_pct": exposure,
        "buy_and_hold_return_pct": bh, "bars": len(df), "tradable_bars": max(0, len(df) - first),
        "missing_candles": missing_candles(df, cfg.interval), "exit_reasons": exit_reasons,
        "by_regime": by_regime,
    }


def _span_days(df: pd.DataFrame, first: int) -> float:
    if "timestamp" in df.columns and len(df) > first:
        t = pd.to_datetime(df["timestamp"], utc=True)
        return max((t.iloc[-1] - t.iloc[first]).total_seconds() / 86400, 0.0)
    return 0.0


def train_test_split_time(df: pd.DataFrame, train_frac: float = 0.7, gap: int = 0):
    """Chronological split. ``gap`` bars are dropped between train and test."""
    if not 0 < train_frac < 1:
        raise ValueError("train_frac must be in (0, 1)")
    cut = int(len(df) * train_frac)
    return df.iloc[:cut].reset_index(drop=True), df.iloc[cut + gap:].reset_index(drop=True)


def run_out_of_sample(data: pd.DataFrame, config: Optional[BacktestConfig] = None, pair: str = "PAIR",
                      train_frac: float = 0.7) -> Dict[str, BacktestResult]:
    """Run the same fixed config on the in-sample and out-of-sample segments.

    The test segment keeps ``warmup`` bars of history before it so indicators
    are defined, but trading only starts in the test segment itself.
    """
    cfg = config or BacktestConfig()
    cut = int(len(data) * train_frac)
    warm = warmup_bars(cfg.indicator_params)
    train = data.iloc[:cut].reset_index(drop=True)
    test_with_warmup = data.iloc[max(0, cut - warm):].reset_index(drop=True)
    return {"in_sample": run_backtest(train, cfg, pair), "out_of_sample": run_backtest(test_with_warmup, cfg, pair)}
