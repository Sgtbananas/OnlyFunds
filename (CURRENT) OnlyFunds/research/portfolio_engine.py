"""Portfolio intelligence engine: one account, several strategies, account-level risk (ROUND4_PROGRAMME.md).

Research only: no orders, no keys, no capital. A separate file: research/portfolio.py, portfolio_variants.py and
portfolio_pm.py are not edited (they are part of frozen or published results). With every engine feature off this
reproduces portfolio_pm.simulate exactly (a test checks it), and so the frozen T50 book and X1.

Everything portfolio_pm does (position band, profit lock, growth/ATR/open-risk/gross/correlation caps, risk
decomposition), plus:
- strategies={name: (entries, exits)}: several strategies share the account. A position remembers the strategy
  that opened it and exits only on that strategy's exit signal (or a stop). A coin already held is not entered twice;
  if two strategies signal the same coin, the first strategy listed takes it.
- rank={pair: score series}: when more coins signal than there are free slots, candidates are taken in descending
  score (known at the close) instead of fixed universe order. NaN scores go last; ties keep universe order.
- replace=(min_bars, max_unrealised_R): when candidates are left over and the account is full, at most one open
  position per day that has been held >= min_bars and is below max_unrealised_R (in units of its initial risk per
  coin) is sold at the next open ("replaced") and its slot goes to the best remaining candidate.
- risk_off=bool series by day (known at that close): on risk-off days new entries are sized at
  risk_off_scale x the normal size (risk per trade falls below 0.5%, never above), and Rules.open_risk_cap_off,
  if set, replaces Rules.open_risk_cap.
- Rules.corr_risk_cap=(hi, lo, lookback): correlation-weighted open stop risk sqrt(r' P r) (r = each position's open
  risk to its stop as a share of equity) > hi -> every position is scaled pro rata so it is lo.
All rules only sell or skip; none adds to a position or raises the 0.5% initial risk.
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
    corr_risk_cap: Optional[Tuple[float, float, int]] = None
    open_risk_cap_off: Optional[Tuple[float, float]] = None

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
             close_at_end: bool = True, rules: Rules = Rules(), corr_lookback_report: int = 90,
             strategies: Optional[Dict[str, Tuple[Dict[str, pd.Series], Dict[str, pd.Series]]]] = None,
             rank: Optional[Dict[str, pd.Series]] = None, replace: Optional[Tuple[int, float]] = None,
             risk_off: Optional[pd.Series] = None, risk_off_scale: float = 0.5) -> Tuple[Result, pd.DataFrame]:
    if strategies is None:
        strategies = {"main": (entries, exits)}
    strat_names = list(strategies)
    rm = RiskManager(limits)
    days = sorted(set().union(*[f.index for f in frames.values()]))
    days = [d for d in days if pd.Timestamp(start, tz="UTC") <= d < pd.Timestamp(end, tz="UTC")]
    order = list(frames)
    rets = np.log(pd.DataFrame({s: f["Close"] for s, f in frames.items()})).diff()
    cash, peak, halted = capital, capital, None
    pos: Dict[str, dict] = {}
    pend_in: List[tuple] = []
    pend_out: List[str] = []
    pend_reason: Dict[str, str] = {}
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
                close(pair, d, float(bars[pair]["Open"]), pend_reason.get(pair, "signal_exit"), costs.slip)
        pend_out, pend_reason = [], {}
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
        for pair, a, strat, scale in (pend_in if not unpriced_open else []):
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
            q = size.qty * scale
            if q * fill < limits.min_notional:
                continue
            fee = q * fill * costs.fee
            risk = q * (fill * (1 + costs.fee) - stop * (1 - costs.stop_slip) * (1 - costs.fee))
            cash -= q * fill + fee
            open_val += q * fill
            seq += 1
            pos[pair] = {"id": seq, "qty": q, "entry": fill, "stop": stop, "init_stop": stop,
                         "cost": q * fill + fee, "risk": risk, "day": d, "bars": 0,
                         "notional_pct": q * fill / equity_open * 100, "hi_close": fill,
                         "init_notional": q * fill, "strat": strat}
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
            ex_s = strategies[pos[pair]["strat"]][1]
            if pair in bars and pair in ex_s and bool(ex_s[pair].get(d, False)):
                pend_out.append(pair)
                pend_trim.pop(pair, None)
                pend_qty.pop(pair, None)
        # portfolio-level rules: positions already leaving at the next open are excluded
        off = bool(risk_off.get(d, False)) if risk_off is not None else False
        # `stay` can be traded at the next open; a held coin with no bar today cannot, but its risk still counts
        held = [s for s in pos if s not in pend_out]
        stay = [s for s in held if s in bars]
        if stay and equity > 0:
            qty_now = {s: min(pos[s]["qty"], pend_qty[s][0]) if s in pend_qty else pos[s]["qty"] for s in stay}
            orc = rules.open_risk_cap_off if (off and rules.open_risk_cap_off is not None) else rules.open_risk_cap
            if orc is not None:
                hi, lo = orc
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
                        qty_now[s] *= lo / e
            if rules.corr_risk_cap is not None:
                hi, lo, lb = rules.corr_risk_cap
                rpairs = stay + [s for s in held if s not in bars]      # unpriced positions count too
                rv = np.array([max((qty_now[s] if s in qty_now else pos[s]["qty"]) *
                                   (last_close[s] - pos[s]["stop"]), 0.0) / equity for s in rpairs])
                e = effective_exposure(rv, _corr(rets, rpairs, d, int(lb)))
                if e > hi:
                    for s in stay:
                        want(s, qty_now[s] * lo / e, "corr_risk")
        free = limits.max_open_positions - (len(pos) - len(pend_out))
        cands = []
        for i, pair in enumerate(order):
            if pair in pos or pair not in bars:
                continue
            strat = next((n for n in strat_names if pair in strategies[n][0]
                          and bool(strategies[n][0][pair].get(d, False))), None)
            if strat is None:
                continue
            a = float(atr[pair].get(d, np.nan))
            c = last_close[pair]
            if unpriced:                                   # risk not reliable: approve no new exposure
                risk_rows[-1]["entries_blocked_unreliable"] += 1
                continue
            if halted is not None or day_loss or not (np.isfinite(a) and a > 0):
                continue
            ok, _ = rm.costs_acceptable(c, c - stop_atr * a, costs.fee, costs.slip, costs.stop_slip)
            if ok:
                score = float(rank[pair].get(d, np.nan)) if rank is not None and pair in rank else np.nan
                cands.append((-score if np.isfinite(score) else np.inf, i, pair, a, strat))
        if rank is not None:
            cands.sort(key=lambda x: (x[0], x[1]))
        scale = risk_off_scale if off else 1.0
        for _, _, pair, a, strat in cands[:max(free, 0)]:
            pend_in.append((pair, a, strat, scale))
        rest = cands[max(free, 0):]
        if replace is not None and rest and not unpriced:
            min_bars, max_r = replace
            weak = []
            for s_, p in pos.items():
                if s_ in pend_out or s_ not in bars or p["bars"] < min_bars:
                    continue
                r_unit = p["entry"] - p["init_stop"]
                ur = (last_close[s_] - p["entry"]) / r_unit if r_unit > 0 else np.inf
                if ur < max_r:
                    weak.append((ur, s_))
            if weak:
                _, victim = min(weak)
                pend_out.append(victim)
                pend_reason[victim] = "replaced"
                pend_trim.pop(victim, None)
                pend_qty.pop(victim, None)
                _, _, pair, a, strat = rest[0]
                pend_in.append((pair, a, strat, scale))
    if close_at_end and days:
        for pair in list(pos):
            close(pair, days[-1], last_close[pair], "end_of_period", costs.slip)
        eq_rows[-1] = (days[-1], cash)
    idx = [r[0] for r in eq_rows]
    res = Result(pd.Series([r[1] for r in eq_rows], index=idx), pd.Series([r[1] for r in ex_rows], index=idx),
                 pd.Series([r[1] for r in no_rows], index=idx), pd.DataFrame(trades), halted)
    return res, pd.DataFrame(risk_rows).set_index("day") if risk_rows else pd.DataFrame()
