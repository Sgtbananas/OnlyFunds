# Round 4 pre-registration: portfolio intelligence, market regimes and strategy selection

Written and committed **before any Round 4 result**. Research only. Live trading is locked, there are no orders and
no capital. Unchanged:
- the frozen strategies T50 and RS3;
- X1 and X2;
- the shadow tracker and all earlier simulators.

New files:
- `research/portfolio_engine.py`: reproduces `portfolio_pm` exactly when its features are off, which is tested;
- `research/regime_engine.py`;
- `research/round4.py`.

Before writing this, I looked only at the class balance of the regime labels: up 41%, down 26%, high vol 32%,
stress 21%. I ran no model or book.

## Why these questions
`07_reconciliation.md` found that the single 3-slot account turns away most T50 signals because "no free position
slot" is available. Which signals get the slots is decided by fixed universe order, not by quality. On 25 coins over
2021–24 the account caught none of the top 10 signal trades. The cheapest possible profitability lever is therefore
**which candidates get the slots, and when a dead position should give its slot up**. This round tests that,
together with regime-aware risk, without loosening any limit:
- the 3-position limit, the 10% position cap at entry, 30% exposure and the 0.5% initial risk all stay;
- risk-off may only *reduce* new-trade risk.

## Contamination statement
2019–2024 has been examined in Rounds 1–3. Any pass is therefore **supported (historical)** at best. Proof can
only come from forward data, through a separately fingerprinted forward experiment. The 2025–26 window is never
loaded.

## Fixed set-up
As in Round 3:
- single 10,000 USDT account, Conservative limits;
- costs 0.20% / 0.05% / 0.20%; stress costs 0.40% / 0.20% / 0.50%;
- decisions at the close, fills at the next open;
- the same splits: DEV (main 9, 2019–22), VAL (main 9, 2023–24) and HOLDOUT (holdout 16, 2019–24);
- primary mode no halt; "as configured" is reported too;
- paired 20-day block bootstrap.

The RS3 signals are frozen `round1.signals("RS3")`, rotating within the split's own universe, with the same 3-ATR
stop. The market universe for regime features is fixed to the main 9 for every split.

## Part A: regime engine (`regime_engine.py`)
Five probability targets over the next 20 days:
- direction (bull / sideways / bear);
- vol_high and vol_low (against BTC's trailing-365-day terciles);
- vol_expand;
- stress (a 15% or larger drawdown).

The output is probabilities plus a confidence (1 − normalised entropy). The model is an L2 logistic regression with
fixed C = 0.1. It is walk-forward with test years 2021–24, and training ends 5 days before each test year (labels
fully known). Baselines: climatology, and a one-variable rule (BTC above its 100-day average, or the current
volatility tercile).

**H4-REG-<target>, one per target:**
- **Supported:** over 2021–24, mean log-loss gain against the better baseline > 0, with the lower end of its 90%
  month-block-bootstrap interval > 0, and gain > 0 in both 2021–22 and 2023–24.
- **Inconclusive:** gain > 0 but not significant.
- **Rejected:** otherwise.

Prediction accuracy is reported, but it does not count as profitability. Only Part B decides whether regime
information has any use.

## Part B: portfolio engine books
| Id | Definition | Compared with |
|---|---|---|
| R0 | frozen T50 book | — |
| B-ORISK | R0 + open-stop-risk cap 10% → 8% (Round 3 A-ORISK) | R0 (descriptive) |
| RS3B | frozen RS3 signals alone in the account | descriptive |
| E1 SLOT | T50; when signals exceed free slots, rank by 28-day momentum percentile across the split's universe (`ml_lab.rs_rank`; highest first) | R0 |
| E2 SLOT+REPL | E1 + replacement: when the account is full and candidates remain, sell the weakest position held ≥ 20 days below +0.5R (at most one a day) and take the best candidate | E1 |
| E3 RISKOFF-RULE | B-ORISK + risk-off while BTC close ≤ its 100-day average: new entries at half size (0.25% risk), open-risk cap 5% → 4% | B-ORISK |
| E3m | E3 with the rule active only on days that have a walk-forward model prediction (2021–24) | — (comparator for E4) |
| E4 RISKOFF-ML | B-ORISK + the same risk-off actions when the model's P(stress) ≥ 2 × that fold's training base rate (2021–24 only) | E3m |
| E5 CORR-RISK | B-ORISK + correlation-weighted open-stop-risk cap √(r′Pr) > 8% → 6.5% (90 days) | B-ORISK |
| PIE | T50 + RS3 in one account (T50 listed first), ranked as in E1, replacement as in E2, open-risk cap 10% → 8%, rule risk-off as in E3 | R0 |
| PIE-ML | PIE with the model risk-off of E4 instead of the rule | descriptive |

**"improves(X, B)":** for each of DEV, VAL and HOLDOUT (no halt, base costs), return/max DD of X > B **and** return
of X ≥ B, **and**, under stress costs on VAL and HOLDOUT, return/max DD of X ≥ B.
- **Supported:** all hold.
- **Inconclusive:** the base-cost conditions hold in at least 2 of the 3 splits including HOLDOUT, but not all
  conditions.
- **Rejected:** otherwise.

**Hypotheses:**
- **H4-SLOT:** improves(E1, R0).
- **H4-REPL:** improves(E2, E1).
- **H4-RISKOFF-RULE:** improves(E3, B-ORISK).
- **H4-RISKOFF-ML:** improves(E4, E3m). The model must beat the rule, not just the book without risk-off.
- **H4-CORR:** improves(E5, B-ORISK).
- **H4-PIE:** PIE against R0 on the seven Round 3 checks, on VAL and HOLDOUT, plus return/max DD ≥ R0 on DEV:
  1. return/DD ≥ R0;
  2. max DD ≤ R0;
  3. return ≥ 80% of R0;
  4. R0's top-5 positions keep ≥ 70% of their profit;
  5. peak open stop risk ≤ 15%;
  6. largest position ≤ 20%;
  7. stress-cost return/DD ≥ R0.

  The two peak checks are kept as in Round 3, even though a one-day jump can break them. They are not weakened.

## What a pass allows
At most a new, separately fingerprinted forward experiment. Nothing changes in T50, RS3, X1, X2, the engine or the
real risk controls without the user's approval.
