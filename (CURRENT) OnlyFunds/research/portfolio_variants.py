"""Single-account simulator with the pre-registered capital-preservation variants (ML_PROGRAMME.md section 7).

With both variants off this reproduces research/portfolio.py exactly (a test checks equity and trades are
identical). portfolio.py itself is frozen (it is part of the T50/RS3 shadow fingerprint), so it is not edited.

Variants
- trim_band=(hi, lo): if a position's value exceeds `hi` of equity at a close, sell down to `lo` of equity at the
  next open (slippage and fees apply). The sold part is recorded as a partial close ("trim").
- profit_lock=(k_r, n_atr): once the close is >= k_r initial R above the fill, the stop becomes
  max(stop, highest close since entry - n_atr x ATR(14) at that close). It is checked intrabar from the next bar.

Risk decomposition (recorded every close): initial stop risk of open positions, position value, unrealised profit,
open risk to the stops, largest single position share of equity, equity drawdown.
"""
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from core.risk_manager import RiskLimits, RiskManager
from portfolio import Costs, Result


def simulate(frames: Dict[str, pd.DataFrame], entries: Dict[str, pd.Series], exits: Dict[str, pd.Series],
             atr: Dict[str, pd.Series], start: str, end: str, limits: RiskLimits, costs: Costs = Costs(),
             stop_atr: float = 3.0, capital: float = 9000.0, vol_cap: float = 0.10, halt_permanent: bool = True,
             close_at_end: bool = True, trim_band: Optional[Tuple[float, float]] = None,
             profit_lock: Optional[Tuple[float, float]] = None) -> Tuple[Result, pd.DataFrame]:
    rm = RiskManager(limits)
    days = sorted(set().union(*[f.index for f in frames.values()]))
    days = [d for d in days if pd.Timestamp(start, tz="UTC") <= d < pd.Timestamp(end, tz="UTC")]
    order = list(frames)
    cash, peak, halted = capital, capital, None
    pos: Dict[str, dict] = {}
    pend_in: List[tuple] = []
    pend_out: List[str] = []
    pend_trim: Dict[str, float] = {}
    eq_rows, ex_rows, no_rows, trades, risk_rows = [], [], [], [], []
    last_close: Dict[str, float] = {}
    prev_equity = capital
    seq = 0

    def close(pair, d, raw, reason, slip, qty=None):
        nonlocal cash
        p = pos[pair]
        q = p["qty"] if qty is None else min(qty, p["qty"])
        frac = q / p["qty"]
        fill = raw * (1 - slip)
        fee = q * fill * costs.fee
        cost, risk = p["cost"] * frac, p["risk"] * frac
        cash += q * fill - fee
        pnl = q * fill - fee - cost
        trades.append({"pair": pair, "position_id": p["id"], "entry_day": p["day"], "exit_day": d, "entry": p["entry"],
                       "exit": fill, "exit_raw": raw, "exit_fee": fee, "stop": p["stop"], "initial_risk": risk,
                       "qty": q, "pnl": pnl, "r": pnl / risk if risk > 0 else np.nan, "reason": reason,
                       "notional_pct_at_entry": p["notional_pct"], "bars": p["bars"], "partial": qty is not None})
        if qty is None or q >= p["qty"] - 1e-12:
            pos.pop(pair)
        else:
            p["qty"] -= q
            p["cost"] -= cost
            p["risk"] -= risk

    for d in days:
        bars = {s: f.loc[d] for s, f in frames.items() if d in f.index}
        for pair in pend_out:
            if pair in pos and pair in bars:
                close(pair, d, float(bars[pair]["Open"]), "signal_exit", costs.slip)
        pend_out = []
        open_val = sum(p["qty"] * float(bars[s]["Open"]) if s in bars else p["qty"] * last_close[s]
                       for s, p in pos.items())
        equity_open = cash + open_val
        for pair, target_pct in pend_trim.items():                 # concentration band (variant R1)
            if pair in pos and pair in bars:
                o = float(bars[pair]["Open"])
                excess = pos[pair]["qty"] - target_pct * equity_open / o
                if excess > 0:
                    close(pair, d, o, "trim", costs.slip, qty=excess)
        pend_trim = {}
        open_val = sum(p["qty"] * float(bars[s]["Open"]) if s in bars else p["qty"] * last_close[s]
                       for s, p in pos.items())
        equity_open = cash + open_val
        for pair, a in pend_in:
            if pair in pos or pair not in bars or halted is not None or len(pos) >= limits.max_open_positions:
                continue
            o = float(bars[pair]["Open"])
            fill = o * (1 + costs.slip)
            stop = fill - stop_atr * a
            prev = frames[pair].loc[:d].iloc[-2] if len(frames[pair].loc[:d]) >= 2 else None
            max_qty = vol_cap * float(prev["Volume"]) if prev is not None else None
            size = rm.position_size(equity_open, cash, fill, stop, open_val, costs.fee,
                                    stop_slippage_pct=costs.stop_slip, max_qty=max_qty)
            if not size.ok:
                continue
            fee = size.qty * fill * costs.fee
            risk = size.qty * (fill * (1 + costs.fee) - stop * (1 - costs.stop_slip) * (1 - costs.fee))
            cash -= size.qty * fill + fee
            open_val += size.qty * fill
            seq += 1
            pos[pair] = {"id": seq, "qty": size.qty, "entry": fill, "stop": stop, "init_stop": stop,
                         "cost": size.qty * fill + fee, "risk": risk, "day": d, "bars": 0,
                         "notional_pct": size.qty * fill / equity_open * 100, "hi_close": fill}
        pend_in = []
        for pair in list(pos):
            if pair not in bars:
                continue
            b, p = bars[pair], pos[pair]
            locked = p["stop"] > p["init_stop"] + 1e-12
            if float(b["Open"]) <= p["stop"]:
                close(pair, d, float(b["Open"]), "profit_lock" if locked else "stop_loss", costs.stop_slip)
            elif float(b["Low"]) <= p["stop"]:
                close(pair, d, p["stop"], "profit_lock" if locked else "stop_loss", costs.stop_slip)
            else:
                p["bars"] += 1
        for s, b in bars.items():
            last_close[s] = float(b["Close"])
        val = sum(p["qty"] * last_close[s] for s, p in pos.items())
        equity = cash + val
        peak = max(peak, equity)
        eq_rows.append((d, equity))
        ex_rows.append((d, val / equity if equity > 0 else 0.0))
        no_rows.append((d, len(pos)))
        pv = {s: p["qty"] * last_close[s] for s, p in pos.items()}
        risk_rows.append({"day": d, "equity": equity, "drawdown_pct": (1 - equity / peak) * 100,
                          "initial_risk_pct": sum(p["risk"] for p in pos.values()) / equity * 100,
                          "position_value_pct": val / equity * 100,
                          "unrealised_pct": sum(pv[s] - p["cost"] for s, p in pos.items()) / equity * 100,
                          "open_risk_to_stop_pct": sum(max(pv[s] - p["qty"] * p["stop"], 0) for s, p in pos.items())
                          / equity * 100,
                          "max_position_pct": max(pv.values()) / equity * 100 if pv else 0.0})
        if halted is None and rm.drawdown_hit(equity, peak):
            halted = d
        if halted is not None and not halt_permanent and equity >= peak * (1 - limits.max_drawdown_pct / 2):
            halted = None
        day_loss = rm.daily_loss_hit(equity, prev_equity)
        prev_equity = equity
        for pair, p in pos.items():                                # variant hooks, decided at the close
            if pair not in bars:
                continue
            c = last_close[pair]
            p["hi_close"] = max(p["hi_close"], c)
            if profit_lock is not None:
                k_r, n_atr = profit_lock
                r_unit = p["entry"] - p["init_stop"]
                a = float(atr[pair].get(d, np.nan))
                if r_unit > 0 and c >= p["entry"] + k_r * r_unit and np.isfinite(a):
                    p["stop"] = max(p["stop"], p["hi_close"] - n_atr * a)
            if trim_band is not None and equity > 0 and pv.get(pair, 0) / equity > trim_band[0]:
                pend_trim[pair] = trim_band[1]
        for pair in list(pos):
            if pair in bars and bool(exits[pair].get(d, False)):
                pend_out.append(pair)
                pend_trim.pop(pair, None)
        free = limits.max_open_positions - (len(pos) - len(pend_out))
        for pair in order:
            if pair in pos or pair not in bars or not bool(entries[pair].get(d, False)):
                continue
            a = float(atr[pair].get(d, np.nan))
            c = last_close[pair]
            if halted is not None or day_loss or free <= 0 or not (np.isfinite(a) and a > 0):
                continue
            ok, _ = rm.costs_acceptable(c, c - stop_atr * a, costs.fee, costs.slip, costs.stop_slip)
            if ok:
                pend_in.append((pair, a))
                free -= 1
    if close_at_end and days:
        for pair in list(pos):
            close(pair, days[-1], last_close[pair], "end_of_period", costs.slip)
        eq_rows[-1] = (days[-1], cash)
    idx = [r[0] for r in eq_rows]
    res = Result(pd.Series([r[1] for r in eq_rows], index=idx), pd.Series([r[1] for r in ex_rows], index=idx),
                 pd.Series([r[1] for r in no_rows], index=idx), pd.DataFrame(trades), halted)
    return res, pd.DataFrame(risk_rows).set_index("day") if risk_rows else pd.DataFrame()
