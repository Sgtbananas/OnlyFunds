"""Single-account simulator with position-management rules for Round 3 (ROUND3_PM_PROGRAMME.md). Research only.

This is a separate copy of research/portfolio_variants.py. That file is part of the frozen X1/X2 experimental
fingerprint and of the frozen T50/RS3 shadow, so it is not edited. With only `trim_band` / `profit_lock` set,
this simulator reproduces portfolio_variants.simulate exactly; a test checks equity and trades.

Every rule only ever SELLS (or raises a stop). None adds to a position, changes entries, changes the 0.5% initial
trade risk or loosens a Conservative limit. Rules are evaluated at a daily close with information known at that
close, and executed at the next open with slippage and fees.

Rules (each optional; when several bind on one position the largest sale wins)
- trim_band=(hi, lo): position value > hi x equity -> sell to lo x equity (X1's concentration band).
- profit_lock=(k_r, n_atr): once close >= fill + k_r x initial R, stop = max(stop, highest close - n_atr x ATR14).
- growth_trim=(hi, lo): position value > hi x its entry notional -> sell to lo x entry notional.
- atr_cap=(hi, lo): volatility-adjusted size cap. qty x ATR14 (one typical daily move, in USDT) > hi x equity
  -> sell to lo x equity.
- open_risk_cap=(hi, lo): aggregate open risk to the stops, sum(max(value - qty x stop, 0)), > hi x equity ->
  sell from the largest contributors first until it is lo x equity. A held coin with no bar that day is
  counted at its last close (it cannot be sold until its next bar, so other positions are cut instead).
- gross_cap=(hi, lo): total position value > hi x equity -> scale every position down pro rata to lo.
- corr_cap=(hi, lo, lookback): correlation-aware exposure. Effective exposure E = sqrt(w' P w), w = position
  weights, P = correlation of daily log returns over the trailing `lookback` days (fewer than 20 shared days ->
  correlation 1). E > hi -> scale every position pro rata so E = lo. With perfectly correlated coins E equals
  gross exposure, with independent coins it is the root-sum-square.

Risk decomposition recorded at every close (percent of equity): drawdown, initial risk, position value (gross),
largest position, unrealised profit, open risk to stops (= loss if every stop filled at its stop price at once),
correlation-weighted exposure, and the instant loss from a uniform 30% gap down across every coin held.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from core.risk_manager import RiskLimits, RiskManager
from portfolio import Costs, Result


@dataclass(frozen=True)
class Rules:
    trim_band: Optional[Tuple[float, float]] = None
    profit_lock: Optional[Tuple[float, float]] = None
    growth_trim: Optional[Tuple[float, float]] = None
    atr_cap: Optional[Tuple[float, float]] = None
    open_risk_cap: Optional[Tuple[float, float]] = None
    gross_cap: Optional[Tuple[float, float]] = None
    corr_cap: Optional[Tuple[float, float, int]] = None

    def as_dict(self):
        return {k: (list(v) if v is not None else None) for k, v in self.__dict__.items()}


GAP = 0.30
MIN_CORR_DAYS = 20


def _corr(rets: pd.DataFrame, pairs: List[str], d, lookback: int) -> np.ndarray:
    n = len(pairs)
    if n < 2:
        return np.ones((n, n))
    win = rets.loc[:d, pairs].iloc[-lookback:].dropna()
    if len(win) < MIN_CORR_DAYS:
        return np.ones((n, n))
    c = np.corrcoef(win.to_numpy().T)
    c = np.where(np.isfinite(c), c, 1.0)
    np.fill_diagonal(c, 1.0)
    return c


def effective_exposure(w: np.ndarray, corr: np.ndarray) -> float:
    return float(np.sqrt(max(w @ corr @ w, 0.0)))


def simulate(frames: Dict[str, pd.DataFrame], entries: Dict[str, pd.Series], exits: Dict[str, pd.Series],
             atr: Dict[str, pd.Series], start: str, end: str, limits: RiskLimits, costs: Costs = Costs(),
             stop_atr: float = 3.0, capital: float = 9000.0, vol_cap: float = 0.10, halt_permanent: bool = True,
             close_at_end: bool = True, rules: Rules = Rules(), corr_lookback_report: int = 90
             ) -> Tuple[Result, pd.DataFrame]:
    rm = RiskManager(limits)
    days = sorted(set().union(*[f.index for f in frames.values()]))
    days = [d for d in days if pd.Timestamp(start, tz="UTC") <= d < pd.Timestamp(end, tz="UTC")]
    order = list(frames)
    rets = np.log(pd.DataFrame({s: f["Close"] for s, f in frames.items()})).diff()
    cash, peak, halted = capital, capital, None
    pos: Dict[str, dict] = {}
    pend_in: List[tuple] = []
    pend_out: List[str] = []
    pend_trim: Dict[str, float] = {}          # band: target share of equity at the next open (X1 semantics)
    pend_qty: Dict[str, Tuple[float, str]] = {}   # other rules: target quantity, rule name
    eq_rows, ex_rows, no_rows, trades, risk_rows = [], [], [], [], []
    last_close: Dict[str, float] = {}
    last_bar: Dict[str, pd.Timestamp] = {}       # day of each coin's latest bar (price age)
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

    def want(pair, q, rule):
        q = max(q, 0.0)
        if pair not in pend_qty or q < pend_qty[pair][0]:
            pend_qty[pair] = (q, rule)

    for d in days:
        bars = {s: f.loc[d] for s, f in frames.items() if d in f.index}
        for pair in pend_out:
            if pair in pos and pair in bars:
                close(pair, d, float(bars[pair]["Open"]), "signal_exit", costs.slip)
        pend_out = []
        open_val = sum(p["qty"] * float(bars[s]["Open"]) if s in bars else p["qty"] * last_close[s]
                       for s, p in pos.items())
        equity_open = cash + open_val
        for pair in list(pend_trim) + [x for x in pend_qty if x not in pend_trim]:
            if pair in pos and pair in bars:
                o = float(bars[pair]["Open"])
                ex_band = pos[pair]["qty"] - pend_trim[pair] * equity_open / o if pair in pend_trim else -np.inf
                ex_rule = pos[pair]["qty"] - pend_qty[pair][0] if pair in pend_qty else -np.inf
                excess = max(ex_band, ex_rule)
                if excess > 0:
                    reason = "trim" if ex_band >= ex_rule else f"trim_{pend_qty[pair][1]}"
                    close(pair, d, o, reason, costs.slip, qty=excess)
        pend_trim, pend_qty = {}, {}
        open_val = sum(p["qty"] * float(bars[s]["Open"]) if s in bars else p["qty"] * last_close[s]
                       for s, p in pos.items())
        equity_open = cash + open_val
        unpriced_open = sorted(s for s in pos if s not in bars)
        blocked_fills = len(pend_in) if unpriced_open else 0      # stale equity: approve nothing
        for pair, a in (pend_in if not unpriced_open else []):
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
                         "notional_pct": size.qty * fill / equity_open * 100, "hi_close": fill,
                         "init_notional": size.qty * fill}
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
            last_bar[s] = d
        val = sum(p["qty"] * last_close[s] for s, p in pos.items())
        equity = cash + val
        peak = max(peak, equity)
        eq_rows.append((d, equity))
        ex_rows.append((d, val / equity if equity > 0 else 0.0))
        no_rows.append((d, len(pos)))
        pv = {s: p["qty"] * last_close[s] for s, p in pos.items()}
        unpriced = sorted(s for s in pos if s not in bars)          # valued at a stale last close
        age = {s: (d - last_bar[s]).days for s in pos}
        held = list(pos)
        w = np.array([pv[s] / equity for s in held]) if equity > 0 else np.zeros(len(held))
        eff = effective_exposure(w, _corr(rets, held, d, corr_lookback_report)) if held else 0.0
        risk_rows.append({"day": d, "equity": equity, "drawdown_pct": (1 - equity / peak) * 100,
                          "initial_risk_pct": sum(p["risk"] for p in pos.values()) / equity * 100,
                          "position_value_pct": val / equity * 100,
                          "unrealised_pct": sum(pv[s] - p["cost"] for s, p in pos.items()) / equity * 100,
                          "open_risk_to_stop_pct": sum(max(pv[s] - p["qty"] * p["stop"], 0) for s, p in pos.items())
                          / equity * 100,
                          "max_position_pct": max(pv.values()) / equity * 100 if pv else 0.0,
                          "corr_exposure_pct": eff * 100,
                          "gap30_loss_pct": GAP * val / equity * 100,
                          "n_open": len(pos),
                          "unpriced_positions": len(unpriced),
                          "unpriced": ",".join(unpriced),
                          "max_price_age_days": max(age.values()) if age else 0,
                          "risk_reliable": not unpriced,
                          "entries_blocked_unreliable": blocked_fills})
        if halted is None and rm.drawdown_hit(equity, peak):
            halted = d
        if halted is not None and not halt_permanent and equity >= peak * (1 - limits.max_drawdown_pct / 2):
            halted = None
        day_loss = rm.daily_loss_hit(equity, prev_equity)
        prev_equity = equity
        for pair, p in pos.items():                                # per-position hooks, decided at the close
            if pair not in bars:
                continue
            c = last_close[pair]
            p["hi_close"] = max(p["hi_close"], c)
            a = float(atr[pair].get(d, np.nan))
            if rules.profit_lock is not None:
                k_r, n_atr = rules.profit_lock
                r_unit = p["entry"] - p["init_stop"]
                if r_unit > 0 and c >= p["entry"] + k_r * r_unit and np.isfinite(a):
                    p["stop"] = max(p["stop"], p["hi_close"] - n_atr * a)
            if rules.trim_band is not None and equity > 0 and pv.get(pair, 0) / equity > rules.trim_band[0]:
                pend_trim[pair] = rules.trim_band[1]
            if rules.growth_trim is not None and pv[pair] > rules.growth_trim[0] * p["init_notional"]:
                want(pair, rules.growth_trim[1] * p["init_notional"] / c, "growth")
            if rules.atr_cap is not None and np.isfinite(a) and a > 0 and p["qty"] * a > rules.atr_cap[0] * equity:
                want(pair, rules.atr_cap[1] * equity / a, "atr")
        for pair in list(pos):
            if pair in bars and bool(exits[pair].get(d, False)):
                pend_out.append(pair)
                pend_trim.pop(pair, None)
                pend_qty.pop(pair, None)
        # portfolio-level rules: positions already leaving at the next open are excluded
        # `stay` can be traded at the next open; a held coin with no bar today cannot, but its risk still counts
        held = [s for s in pos if s not in pend_out]
        stay = [s for s in held if s in bars]
        if stay and equity > 0:
            qty_now = {s: min(pos[s]["qty"], pend_qty[s][0]) if s in pend_qty else pos[s]["qty"] for s in stay}
            if rules.open_risk_cap is not None:
                hi, lo = rules.open_risk_cap
                r_i = {s: max(qty_now[s] * (last_close[s] - pos[s]["stop"]), 0.0) for s in stay}
                r_unbarred = sum(max(pos[s]["qty"] * (last_close[s] - pos[s]["stop"]), 0.0)
                                 for s in held if s not in bars)          # marked at the last known close
                total = sum(r_i.values()) + r_unbarred
                excess = total - lo * equity
                if total > hi * equity:
                    for s in sorted(stay, key=lambda x: -r_i[x]):
                        if excess <= 0 or r_i[s] <= 0:
                            break
                        cut = min(r_i[s], excess)
                        want(s, qty_now[s] * (1 - cut / r_i[s]), "open_risk")
                        qty_now[s] *= (1 - cut / r_i[s])
                        excess -= cut
            if rules.gross_cap is not None:
                hi, lo = rules.gross_cap
                g = sum(qty_now[s] * last_close[s] for s in stay) / equity
                g_unpriced = sum(pos[s]["qty"] * last_close[s] for s in held if s not in bars) / equity
                if g + g_unpriced > hi:
                    k = lo / g if not g_unpriced else max(lo - g_unpriced, 0.0) / g
                    for s in stay:
                        want(s, qty_now[s] * k, "gross")
                        qty_now[s] *= k
            if rules.corr_cap is not None:
                hi, lo, lb = rules.corr_cap
                cpairs = stay + [s for s in held if s not in bars]      # unpriced positions count too
                wv = np.array([(qty_now[s] if s in qty_now else pos[s]["qty"]) * last_close[s] / equity
                               for s in cpairs])
                e = effective_exposure(wv, _corr(rets, cpairs, d, int(lb)))
                if e > hi:
                    for s in stay:
                        want(s, qty_now[s] * lo / e, "corr")
        free = limits.max_open_positions - (len(pos) - len(pend_out))
        for pair in order:
            if pair in pos or pair not in bars or not bool(entries[pair].get(d, False)):
                continue
            a = float(atr[pair].get(d, np.nan))
            c = last_close[pair]
            if unpriced:                                   # risk not reliable: approve no new exposure
                risk_rows[-1]["entries_blocked_unreliable"] += 1
                continue
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
