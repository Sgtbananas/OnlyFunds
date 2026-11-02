"""Single-account portfolio simulation for daily long-only signals across many coins.

Unlike the per-coin engine (one independent 1,000 USDT account per coin), this enforces the Conservative
limits ACROSS coins: max 3 open positions, max 30% total exposure, 10% max per position, 0.5% risk per trade,
3% daily loss limit, 10% drawdown halt. Sizing, the cost-to-risk filter and stop fills reuse the engine's
own code (RiskManager.position_size / costs_acceptable, positions.check_exit semantics).

Execution: decisions at the daily close, fills at the next open with slippage; stops checked intrabar
(gap through the stop fills at the open); stop fills get stop slippage; fees on both sides.
When more coins signal than there are free slots, coins are taken in fixed universe order.
"""
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd

from core.risk_manager import RiskLimits, RiskManager


@dataclass
class Costs:
    fee: float = 0.002
    slip: float = 0.0005
    stop_slip: float = 0.002


@dataclass
class Result:
    equity: pd.Series
    exposure: pd.Series                 # open notional / equity, at each close
    n_open: pd.Series
    trades: pd.DataFrame
    halted_on: object = None
    notes: List[str] = field(default_factory=list)


def simulate(frames: Dict[str, pd.DataFrame], entries: Dict[str, pd.Series], exits: Dict[str, pd.Series],
             atr: Dict[str, pd.Series], start: str, end: str, limits: RiskLimits, costs: Costs = Costs(),
             stop_atr: float = 3.0, capital: float = 9000.0, vol_cap: float = 0.10, halt_permanent: bool = True,
             close_at_end: bool = True, log: list = None) -> Result:
    """``frames``/``entries``/``exits``/``atr``: per coin, indexed by UTC day (signals known at that day's close).

    ``log`` (optional list): every entry signal and its fate is appended (accepted, filled, or rejected with the
    reason). Passing it never changes the simulation.
    """
    def note(**kw):
        if log is not None:
            log.append(kw)
    rm = RiskManager(limits)
    days = sorted(set().union(*[f.index for f in frames.values()]))
    days = [d for d in days if pd.Timestamp(start, tz="UTC") <= d < pd.Timestamp(end, tz="UTC")]
    order = list(frames)
    cash, peak, halted = capital, capital, None
    pos: Dict[str, dict] = {}
    pend_in: List[tuple] = []
    pend_out: List[str] = []
    eq_rows, ex_rows, no_rows, trades = [], [], [], []
    last_close: Dict[str, float] = {}
    prev_equity = capital

    def close(pair, d, raw, reason, slip):
        nonlocal cash
        p = pos.pop(pair)
        fill = raw * (1 - slip)
        fee = p["qty"] * fill * costs.fee
        cash += p["qty"] * fill - fee
        pnl = p["qty"] * fill - fee - p["cost"]
        trades.append({"pair": pair, "entry_day": p["day"], "exit_day": d, "entry": p["entry"], "exit": fill,
                       "exit_raw": raw, "exit_fee": fee, "stop": p["stop"], "initial_risk": p["risk"],
                       "qty": p["qty"], "pnl": pnl, "r": pnl / p["risk"], "reason": reason,
                       "notional_pct_at_entry": p["notional_pct"], "bars": p["bars"]})

    for d in days:
        bars = {s: f.loc[d] for s, f in frames.items() if d in f.index}
        # 1) pending exits, then pending entries, at the open
        for pair in pend_out:
            if pair in pos and pair in bars:
                close(pair, d, float(bars[pair]["Open"]), "signal_exit", costs.slip)
        pend_out = []
        open_val = sum(p["qty"] * float(bars[s]["Open"]) if s in bars else p["qty"] * last_close[s]
                       for s, p in pos.items())
        equity_open = cash + open_val
        for pair, a in pend_in:
            if pair in pos or pair not in bars or halted is not None or len(pos) >= limits.max_open_positions:
                why = ("already in position" if pair in pos else "no candle at fill time" if pair not in bars
                       else "drawdown halt" if halted is not None else "max open positions at fill")
                note(day=d, pair=pair, stage="fill", status="rejected", reason=why)
                continue
            o = float(bars[pair]["Open"])
            fill = o * (1 + costs.slip)
            stop = fill - stop_atr * a
            prev = frames[pair].loc[:d].iloc[-2] if len(frames[pair].loc[:d]) >= 2 else None
            max_qty = vol_cap * float(prev["Volume"]) if prev is not None else None
            size = rm.position_size(equity_open, cash, fill, stop, open_val, costs.fee,
                                    stop_slippage_pct=costs.stop_slip, max_qty=max_qty)
            if not size.ok:
                note(day=d, pair=pair, stage="fill", status="rejected", reason=f"sizing: {size.reason}",
                     signal_open=o, fill_price=fill, stop=stop, equity=equity_open, open_exposure=open_val)
                continue
            fee = size.qty * fill * costs.fee
            risk = size.qty * (fill * (1 + costs.fee) - stop * (1 - costs.stop_slip) * (1 - costs.fee))
            cash -= size.qty * fill + fee
            open_val += size.qty * fill
            pos[pair] = {"qty": size.qty, "entry": fill, "stop": stop, "cost": size.qty * fill + fee, "risk": risk,
                         "day": d, "bars": 0, "notional_pct": size.qty * fill / equity_open * 100}
            note(day=d, pair=pair, stage="fill", status="filled", reason=f"sizing: {size.reason}", open_price=o,
                 fill_price=fill, slippage_cost=size.qty * (fill - o), qty=size.qty, notional=size.qty * fill,
                 stop=stop, initial_risk=risk, entry_fee=fee, equity=equity_open)
        pend_in = []
        # 2) intrabar stops
        for pair in list(pos):
            if pair not in bars:
                continue
            b, p = bars[pair], pos[pair]
            if float(b["Open"]) <= p["stop"]:
                close(pair, d, float(b["Open"]), "stop_loss", costs.stop_slip)
            elif float(b["Low"]) <= p["stop"]:
                close(pair, d, p["stop"], "stop_loss", costs.stop_slip)
            else:
                p["bars"] += 1
        # 3) mark to market at the close
        for s, b in bars.items():
            last_close[s] = float(b["Close"])
        val = sum(p["qty"] * last_close[s] for s, p in pos.items())
        equity = cash + val
        peak = max(peak, equity)
        eq_rows.append((d, equity))
        ex_rows.append((d, val / equity if equity > 0 else 0.0))
        no_rows.append((d, len(pos)))
        if halted is None and rm.drawdown_hit(equity, peak):
            halted = d
        if halted is not None and not halt_permanent and equity >= peak * (1 - limits.max_drawdown_pct / 2):
            halted = None
        day_loss = rm.daily_loss_hit(equity, prev_equity)
        prev_equity = equity
        # 4) decisions at the close (known only from data <= d)
        for pair in list(pos):
            if pair in bars and bool(exits[pair].get(d, False)):
                pend_out.append(pair)
        free = limits.max_open_positions - (len(pos) - len(pend_out))
        for pair in order:
            if pair in pos or pair not in bars or not bool(entries[pair].get(d, False)):
                continue
            a = float(atr[pair].get(d, np.nan))
            c = last_close[pair]
            base = dict(day=d, pair=pair, stage="signal", close=c, atr=a)
            if halted is not None or day_loss:
                note(**base, status="rejected", reason="drawdown halt" if halted is not None else "daily loss limit")
                continue
            if free <= 0:
                note(**base, status="rejected", reason="no free position slot (max open positions)")
                continue
            if not (np.isfinite(a) and a > 0):
                note(**base, status="rejected", reason="ATR unavailable")
                continue
            ok, why = rm.costs_acceptable(c, c - stop_atr * a, costs.fee, costs.slip, costs.stop_slip)
            if ok:
                pend_in.append((pair, a))
                free -= 1
                note(**base, status="accepted", reason="queued for next open", planned_stop=c - stop_atr * a)
            else:
                note(**base, status="rejected", reason=f"cost filter: {why}")
    if close_at_end and days:
        for pair in list(pos):
            close(pair, days[-1], last_close[pair], "end_of_period", costs.slip)
        eq_rows[-1] = (days[-1], cash)
    idx = [r[0] for r in eq_rows]
    return Result(pd.Series([r[1] for r in eq_rows], index=idx), pd.Series([r[1] for r in ex_rows], index=idx),
                  pd.Series([r[1] for r in no_rows], index=idx), pd.DataFrame(trades), halted)


def stats(res: Result, capital: float) -> dict:
    e = res.equity
    if e.empty:
        return {}
    years = (e.index[-1] - e.index[0]).days / 365.25
    ret = e.pct_change().dropna()
    t = res.trades
    gl = -t.loc[t.pnl <= 0, "pnl"].sum() if len(t) else 0
    return {"final_equity": float(e.iloc[-1]), "total_return_pct": float((e.iloc[-1] / capital - 1) * 100),
            "cagr_pct": float(((e.iloc[-1] / capital) ** (1 / years) - 1) * 100) if years > 0 else float("nan"),
            "max_dd_pct": float(-(e / e.cummax() - 1).min() * 100),
            "sharpe": float(ret.mean() / ret.std() * np.sqrt(365)) if ret.std() > 0 else float("nan"),
            "avg_exposure_pct": float(res.exposure.mean() * 100), "max_exposure_pct": float(res.exposure.max() * 100),
            "time_in_market_pct": float((res.n_open > 0).mean() * 100), "avg_open_positions": float(res.n_open.mean()),
            "trades": int(len(t)), "net_R": float(t.r.mean()) if len(t) else float("nan"),
            "pf": float(t.loc[t.pnl > 0, "pnl"].sum() / gl) if gl > 0 else float("nan"),
            "halted_on": str(res.halted_on.date()) if res.halted_on is not None else "-"}
