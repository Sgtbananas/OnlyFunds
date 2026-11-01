# Round 3 pre-registration: position management and account-level risk

Written and committed **before any Round 3 result**. Research only: nothing is approved for trading, and live
trading stays locked. The frozen T50 and RS3 strategies, their shadow tracking and their pass criteria are unchanged.
X1 and X2 (`EXPERIMENTAL_FROZEN.json`) are also unchanged. The new simulator is a separate file,
`research/portfolio_pm.py`; the frozen `portfolio.py` and `portfolio_variants.py` are not edited. A test shows the
new simulator reproduces X1 and the frozen book exactly.

## Contamination statement
- **Data:** 2019–2024 only. The consumed 2025–2026 test window is never loaded (`round1.load` asserts this).
- **Already examined:** every split below was used in Rounds 1–2 for band and profit-lock rules. A historical pass
  here is therefore **"supported (historical)"**, never "proven".
- **Forward evidence:** the only clean evidence is forward data. A supported candidate can at most become another
  separate forward experiment, with its own fingerprint.

## Fixed set-up (identical for every book)
- **Signals:** frozen T50 entries and exits, unchanged.
- **Account:** a single account, 10,000 USDT, Conservative limits: 0.5% risk per trade, 10% position cap at entry,
  30% exposure at entry, 3 positions, 3% daily-loss stop, 10% drawdown kill switch.
- **Costs:** fee 0.20% per side, slippage 0.05%, stop slippage 0.20%. Stress costs: 0.40%, 0.20% and 0.50%.
- **Execution:** decisions at the daily close; fills and trims at the next open.
- **Rules can only sell, or raise a stop.** They never add to a position, change an entry or raise the 0.5% initial
  risk.

## Splits (reported separately, never pooled)
| Split | Coins | Period | Use |
|---|---|---|---|
| DEV | main 9 | 2019-01-01 → 2023-01-01 | selection |
| VAL | main 9 | 2023-01-01 → 2025-01-01 | confirmation |
| HOLDOUT | holdout 16 | 2019-01-01 → 2025-01-01 | confirmation |
| FORWARD | T50 universe, live Binance daily | from 2026-11-02 | the only clean evidence; X1/X2 already tracked |

**Modes.**
- Each split starts with fresh capital.
- **Primary mode: no halt** (`max_drawdown_pct = daily = 1.0`). A permanent kill switch makes a book's path
  degenerate after its first trip, so it cannot compare rules.
- **As configured:** reported for every book, including whether and when the 10% kill switch trips.

## Candidates (fixed here, not tuned)
| Id | Rules (`portfolio_pm.Rules`) | Question |
|---|---|---|
| R0 | none (frozen T50 book) | comparator |
| X1 | trim_band 15%→10%, profit_lock 2R / 6 ATR | the frozen experiment |
| X2 | X1 + no new entries while BTC ≤ its 100-day SMA | the frozen experiment |
| A-GROW | growth_trim: value > 3× entry notional → 2× | dynamic trim relative to the initial allocation |
| A-ATR | atr_cap: qty × ATR14 > 0.75% of equity → 0.50% | volatility-adjusted size cap |
| A-ORISK | open_risk_cap: open risk to stops > 10% of equity → 8% | aggregate stop-risk limit |
| A-GROSS | gross_cap: total value > 35% of equity → 30% | continuous exposure cap |
| A-CORR | corr_cap: effective exposure √(w′Pw) over 90 days > 30% → 25% | correlation-aware exposure |
| A-LOCK4 | profit_lock 4R / 4 ATR | protection only after substantial gains |
| K1 | X1 + A-ORISK | |
| K2 | X1 + A-CORR | |
| K3 | X1 + A-ATR + A-ORISK + A-CORR | the full account-level framework |

## Metrics (per split, per mode)
- **Performance:** return, CAGR, max drawdown, return/max DD, Sharpe, positions, trims by rule, and net R per
  position (position P&L ÷ initial risk).
- **Peak exposure:**
  - largest single position (% of equity);
  - gross exposure;
  - open risk to stops, i.e. the loss if every stop fills at its stop price on the same day (simultaneous
    stop-outs);
  - correlation-weighted exposure;
  - the instant loss from a uniform 30% gap down across every coin held (stops cannot protect against a gap).
- **Realised extremes:** the worst realised day and the worst realised 5 days.
- **Upside retention:** the share of R0's five most profitable positions (by P&L) that each book keeps.
- **Uncertainty:** paired block bootstrap of daily returns (20-day blocks, 1,000 draws, the same blocks for both
  books). It gives intervals for Δ CAGR and Δ max DD against R0. These intervals are descriptive; the sample is
  only tens of positions per split.

## Decision rules
**Selection (DEV, no halt).** A candidate is eligible if all of the following hold:
- return ≥ 80% of R0's DEV return;
- R0's top-5 positions keep ≥ 70% of their profit;
- max open risk to stops ≤ 15%;
- largest position ≤ 20%.

The eligible candidate with the highest return/max DD is selected (ties are broken by lower max DD). X1, X2 and
the A-/K- candidates all compete; R0 does not.

**H3-PM-SEL (primary).** The selected candidate is **supported (historical)** if, on both VAL and HOLDOUT
(no halt), it meets all of these:
1. return/max DD ≥ R0's;
2. max DD ≤ R0's;
3. return ≥ 80% of R0's (or ≥ R0's if R0's return ≤ 0);
4. top-5 kept ≥ 70%;
5. max open risk to stops ≤ 15%;
6. largest position ≤ 20%;
7. under stress costs, return/max DD ≥ R0's.

Any miss means **rejected**. If nothing is eligible on DEV, the result is **rejected (no candidate)**.

**H3-K3 (secondary, the framework candidate).** The same seven checks for K3 on VAL and HOLDOUT, whether or not
K3 was selected.

**H3-X2 (descriptive).** Does X2's return/max DD exceed X1's on VAL and HOLDOUT? The answer is reported only; it
does not alter X1 or X2.

**Not decision rules:** the kill-switch status as configured, the gap loss, the bootstrap intervals, and the
results for the 25-coin universe over 2021–24 are all reported but decide nothing.

## What a pass allows
A supported candidate may be proposed as a **new, separate forward experiment**, frozen with its own fingerprint.
It changes nothing in T50, RS3, X1, X2 or the engine. Any change to real risk controls needs the user's approval.

## Reconciliation (not a hypothesis)
`docs/research/07_reconciliation.md` explains, with sources, why the frozen T50 figures differ between the per-coin
signal level and the single-account book. It recomputes the published figures and changes none of them.
