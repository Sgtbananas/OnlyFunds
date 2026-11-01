# Does machine learning have a useful role yet? (2026-11-02)

**Short answer: not yet.** No model has shown value beyond a simple baseline in any role that could change a
trade or a position size. All models stay experimental, and none influences T50, RS3, X1 or X2. No new ML
experiments were run for this assessment, as instructed. It uses the out-of-sample results already recorded in
`ml_round1_results.json` (Round 1) and `round2_results.json` (Round 2): walk-forward test years 2021–24, causal
features, purged and embargoed folds.

## The bar a model must clear
A model may influence a strategy only if all of the following hold:
1. It beats the **simple baseline for the same job** out of sample, with an interval that excludes zero, in both
   halves (2021–22 and 2023–24).
2. Its predictions actually change decisions.
3. The change improves the **account-level** result (return / max DD, not just total R) on DEV, VAL and HOLDOUT,
   under stress costs.
4. It then survives forward tracking as a separately fingerprinted experiment.

Every ML result so far fails at step 1 or step 3.

## Role by role
| Role | Simple baseline | Evidence so far | Verdict |
|---|---|---|---|
| **Entry filtering** (skip T50 signals) | random skipping of the same number | All 6 ML gates cut total R by 62–167R. The removed trades averaged +0.65 to +3.1R. | **Harmful.** T50 earns from rare large trends, and a filter that skips many signals skips them. Do not pursue. |
| **Volatility forecasting** (20-day realised vol) | persistence (next 20 days = last 20 days) | Ridge model: 30% lower error than climatology. Against persistence: +22% in 2021–22, **−1% in 2023–24**; the overall interval includes zero (−2% to +8%). | **The only real skill, and it is not better than persistence.** A position-management rule that needs a vol estimate should use ATR / persistence, which is what Round 3's A-ATR does. |
| **Market-wide stress** (a coin falls 20% within 20 days) | climatology (the base rate) | AUC 0.54–0.59, so it ranks somewhat. But log loss is worse than the base rate in 2021–24 (−0.06; interval −0.17 to +0.03), and it only matches the base rate in 2023–24. | **No usable probability.** The ranking is too weak to time risk reductions. |
| **Regime probability** (coin up more than costs over 20 days) | climatology, and a rule regime | AUC 0.47–0.50; the lowest-probability decile rose 66% of the time. | **No skill.** |
| **Adverse excursion / fast stop-out** (stopped within 5 days) | the base rate (about 8–10%) | AUC 0.63 / 0.62 in 2021–22, then 0.54 / 0.51 in 2023–24. Used as a gate it cost −86R. | **Decaying, weak.** It does not hold up in the second half. |
| **Trade success** (will this T50 trade win?) | the base rate | AUC 0.25–0.56 out of sample against 0.80–0.98 in training. | **Overfitting, no skill.** |
| **Correlation forecasting** | trailing 90-day correlation | Not modelled. In the T50 book, the coins held had a mean 90-day correlation of 0.52–0.70 (`07_reconciliation.md`). | **Untested.** Persistence is already a strong baseline, because crypto correlation is high and stable. |
| **When to reduce portfolio risk** | position-level rules (caps, trims, profit locks) | Round 3 (`08_round3_pm.md`) tests these simple rules first. | ML would have to beat the best simple rule, not R0. |

## Why ML struggles here
- **Too few decisive events.** T50's record rests on a handful of large trends: 2021, then 2023–24. Six years of daily
  data hold too few of them to learn from.
- **Relationships reverse between regimes.** Models fitted on 2019–20 learned patterns that reversed in 2021.
- **The useful decisions are about size, not entry.** Size decisions are already handled by simple, transparent
  rules, which have nothing to overfit.

## What would justify another ML attempt
Each would be pre-registered and kept small:
- **A volatility model, but only for sizing.** Test it only if it can beat persistence by a margin whose interval
  excludes zero in both halves. The current model misses this. The first test would be its effect on an ATR-capped
  book, against the same book using ATR.
- **A drawdown-probability model at portfolio level** (does the T50 book fall 5% or more in the next 10 days?).
  Climatology and "current open risk to stops" would be the baselines. This is the most natural ML target left. It
  is not run now, because the simple open-risk cap must first prove useful on its own.
- **Not worth trying again:** any entry gate.

Until one of these clears the bar above, **the honest role of ML in OnlyFunds is research tooling only**: the leak-safe
pipeline, model cards and the hypothesis register. It has no role in decision-making.
