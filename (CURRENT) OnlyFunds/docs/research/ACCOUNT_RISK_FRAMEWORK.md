# Account-level risk framework (proposal; nothing here is enabled)

**Status: design only.** No risk control in the engine, the frozen strategies or X1/X2 has been changed. Any change
to real risk controls needs explicit approval and a new frozen version with its own fingerprint. The 0.5% initial
trade risk stays as it is.

## 1. Seven quantities that must be kept apart
All are measured at every daily close, as a percentage of current equity (`research/portfolio_pm.py` records them
all).

| Quantity | Definition | What it protects against | Frozen T50 book, worst seen (no halt, 2019–24) |
|---|---|---|---|
| **Initial trade risk** | qty × (fill − initial stop), including costs, at entry | a single new trade failing | 0.5% per trade, 1.5% for 3 positions |
| **Current market value** | qty × close, per position and in total (gross) | concentration | largest 23.6%; gross 32.8% |
| **Open stop risk** | Σ max(value − qty × current stop, 0) | **every stop being hit on the same day** (simultaneous stop-outs) | **24.2%** (main 9), 22.8% (holdout) |
| **Unrealised profit** | value − cost basis | giving back open gains | up to 23.4% |
| **Correlated exposure** | √(w′Pw), with P the trailing 90-day correlation of the coins held | several positions that are really one bet | 26.1%; the coins held had a mean correlation of 0.52–0.70 |
| **Maximum potential loss** | max(open stop risk, gap loss); gap loss = 30% × gross | a crash that gaps through the stops | gap: 9.8%; stops: 24.2% |
| **Equity drawdown** | 1 − equity ÷ peak equity | the account | 17.9% (no halt); the 10% kill switch tripped in 2021 |

**The central point.** Initial risk is not portfolio risk. A 0.5%-risk trade whose stop never moves becomes a
10–24% open-stop risk as the position wins. In the frozen book, the drawdowns that tripped the kill switch were
open profit being given back, not new trades failing.

## 2. Proposed layered limits (in order of evidence)
Every limit only sells or raises a stop. None adds to a position. Each is checked at the close and executed at the
next open.

| Layer | Rule | Evidence (Round 3, `08_round3_pm.md`, no halt) | Status |
|---|---|---|---|
| L1 | **Aggregate open-stop-risk cap:** open stop risk > 10% → sell from the largest contributors until it is 8% | Better return/DD than R0 in DEV (4.89 vs 2.56), VAL (4.23 vs 1.71) and HOLDOUT (3.37 vs 2.81), and under stress costs in VAL and HOLDOUT. Never trips the kill switch. Top winners keep 95–160% of their profit. **Failed** its pre-registered peak checks on VAL (open risk 17.0%, position 20.1%), both from a single TRX day (+75% on 2024-12-03). | best-supported rule; **post hoc** (not the DEV pick); forward test first |
| L2 | **Continuous position band:** value > 15% → sell to 10% (X1) | Return/DD up in every split; worse than R0 under stress costs in VAL (1.24 vs 1.34) | under forward test as X1 |
| L3 | **Volatility-adjusted size cap:** qty × ATR14 > 0.75% of equity → 0.50% | Return/DD up in every split; equal to R0 under stress in VAL; better on HOLDOUT | supporting |
| L4 | **Profit lock** after 2R (6 × ATR trail) | Adds little on its own (Round 1 R2, Round 3 A-LOCK4) | only in combination |
| L5 | **Correlation-aware / gross exposure cap** | **Never triggered** at 30–35%: with 3 positions and a 30% entry cap the book never got that large | ineffective at these levels; correlation is better used to *count* exposure (§3) |
| — | 10% drawdown kill switch | Keep unchanged. L1 stopped it tripping in every split. | unchanged |

**What the evidence does not support:**
- **Selling winners merely because they are profitable.** Growth trims (A-GROW) cut holdout winners to 88% of their
  profit.
- **Exposure caps at levels the book never reaches.**
- **Gating entries on BTC.** X2 is worse than R0 on HOLDOUT.

## 3. How correlation should enter
- **Measured:** the coins T50 holds have a mean 90-day correlation of 0.52–0.70, so three positions behave like
  about 1.25–1.5 independent bets (n ÷ (1 + (n − 1)ρ)).
- **What correlation should do:** inflate *open stop risk* rather than cap gross exposure. A reasonable design is
  correlation-weighted open stop risk, √(r′Pr) with r the open-stop-risk vector, compared against the L1 limit.
  This is **untested**, and would need its own pre-registration.
- **Gaps:** the gap scenario (30% across all coins) assumes correlation 1. That is the right assumption for crash
  days.

## 4. What no close-based rule can prevent
- **A one-day move happens before any rule can act.** TRX's +75% close moved a position from 11% to 20% before any
  rule could act. Peak values are therefore an outcome, not a guarantee. Limits should be stated as "executed at the
  next open" and reported both at peak and after execution.
- **Gap risk is bounded only by gross exposure.** With a 30% entry cap the gap loss stays under 10%.

## 5. The larger profitability issue found while reconciling
The single account takes only a fraction of T50's signals.

- **Most signals are refused.** Most are turned away for "no free position slot". In the 25-coin book (2021–24) it
  took 47 of 289 signal trades and **none of the top 10**. It earned +41R against +251R at the signal level
  (`07_reconciliation.md`).
- **The cause is capacity.** Which signals get the three slots is decided by fixed universe order, not by quality.
- **What follows:** how slots are allocated, and when stagnant positions free their slot, may matter more for net
  profitability than any trim rule. This is a strategy-selection question, not a reason to loosen the 3-position
  limit.
