# Audit of the existing ML and "AI" components (2026-11-02)

**Bottom line: before this round OnlyFunds had one real machine-learning model, an entry filter.**
- It is genuinely trained and its inference works end to end.
- It has **never shown out-of-sample value after costs**. It is correctly disabled (`ml.enabled: false`).
- Everything else that sounds adaptive is a fixed rule, a parameter search, or an unused placeholder.

## Inventory
| Component | What it actually is | Trained? | Used in decisions? | Evidence of value |
|---|---|---|---|---|
| `core/ml_filter.py` | Gradient-boosting classifier (sklearn HistGradientBoosting, depth 3, 200 iterations). Predicts whether a long entered at the next open hits a 3-ATR take-profit before a 1.5-ATR stop within 48 bars (triple-barrier label). | Yes (joblib model, versioned) | Optional entry gate in `core/backtester.py`, `core/paper_trader.py` and `core/validation.py`. Off by default. | **None.** See the evidence section. |
| `core/meta_learner.py` (`StrategyRotation`) | Fixed rule: bench a strategy for 24 h if its last 10 trades have profit factor < 0.8 | No | Paper trader | Never validated |
| `core/selection.py` | Evidence-gated selection: trades only strategies with an approved validation run | No | Selector; no approvals exist, so NO_TRADE | Correct behaviour; not ML |
| `core/regime.py` | Fixed-threshold regime labels (trend, volatility, volatility change), with a "confidence" equal to distance from the nearest threshold | No | Monitoring and research | A heuristic, not a calibrated probability |
| `core/autotune.py`, `scripts/auto_tuner.py`, `scripts/Optuna_tune.py` | Parameter grid / Optuna search with a 70/30 out-of-sample check | Searches, not models | Report only; nothing is applied automatically | Search increases overfitting risk |
| `scripts/continuous_learning.py` | Retrains the ML filter on the last 1,000 candles from the exchange and **overwrites** `state/ml_filter.pkl` | Yes | Feeds the filter | At 15 m, 1,000 candles is about 10 days, far too little |
| `data/models/models.py` | Unused save/load helper that prints | n/a | No | n/a |
| `legacy/sentiment.py` | **Returns `random.uniform(-0.5, 0.5)`**, a fake placeholder | n/a | Not imported anywhere (checked by a new test) | Fake. Must never be wired in. |

## Evidence on the ML filter (validation v1 on real data, `docs/validation/stage1_dev_val.md`)
- **Ranking:** internal DEV validation AUC was 0.513–0.530, close to chance, while training AUC was 0.75. That
  gap is overfitting.
- **Effect on trades in VAL:**
  - Trend following kept 23 of 380 trades, at −0.134R versus −0.082R without the filter: it made results worse.
  - Stacked kept 11 of 58, at −0.492R versus −0.194R.
- **Carry-over:** "mean reversion + ML" was carried to TEST because of **one** VAL trade, and made 0 TEST trades.
  This was a protocol flaw, fixed in v2 (P4 ML promotion rule).
- **Stored model** (`docs/validation/ml_mean_reversion.pkl`):
  - Its validation base win rate is 28%, but the gate threshold is a fixed 0.55. It **accepted 0 of 5,280 validation
    candidates**. As configured, the gate means "never trade".

## Defects found
1. **The label does not match the strategies.** The fixed take-profit/stop triple barrier is not how any
   validated candidate exits. T50 has no take-profit and exits on a 25-day low. The filter predicts an outcome
   that is never traded.
2. **The threshold ignores the base rate.** An absolute 0.55 against a 28% base rate rejects everything.
3. **The split leaks across coins.** Each pair is split at its own 70% point, so coins with different listing
   dates have training windows that overlap other coins' validation windows. Crypto moves together, so this leaks.
   No embargo is applied beyond the label horizon.
4. **The training population can change.** With fewer than 150 rule entries, training falls back to *all bars*
   without failing. This is recorded in the report but not surfaced as a warning.
5. **No context features:** no BTC or market state, no cross-sectional strength, no breadth, no correlation.
   Features are bar-level indicator values only.
6. **Models are not reproducible:** no model card, no data hashes, and retraining overwrites the only copy, so a
   past decision cannot be re-derived.
7. **Predictions are never monitored:** nothing compares paper or forward predictions with realised outcomes.
8. **The ML gate can disappear unnoticed.** `run_paper.py` with `ml.enabled: true` but no compatible model logs a
   warning and runs ungated.
9. **A fake placeholder exists** (`legacy/sentiment.py`). It is unused, and a test now fails if it is imported anywhere.

## What was built in ML Round 1 (`research/`)
- **`ml_lab.py`:** 24 causal coin and market features, forward labels, purged and embargoed walk-forward folds,
  fixed-hyperparameter models, and model cards with hashes. Inference refuses to run if the code, features, data
  or model file differ from the card.
- **`ml_round1.py`:** the pre-registered experiments (`ML_PROGRAMME.md`), each comparing:
  - ML against a non-ML rule baseline;
  - each gate against a random skip with the same skip count;
  - with Holm correction, month-block and week-cluster bootstraps, cost-stress runs, single-account economics, and
    a separate verdict for every hypothesis.
- **`portfolio_variants.py`:** a single-account simulator with:
  - the concentration band and profit-lock variants;
  - a daily risk breakdown: initial risk, position value, unrealised profit, open risk to stops, largest position,
    drawdown.
  - With both variants off it reproduces the frozen `portfolio.py` exactly.
- **`hypotheses.py`:** an append-only, hash-chained hypothesis register. Every AI- or human-proposed hypothesis is
  recorded before testing, and test data from 2025 onwards is refused.

The existing `core/ml_filter.py` was **not modified**.
- It is disabled by default and is superseded for research by `ml_lab.py`.
- Patching it would change the validated v1/v2 records that depend on it.
- Its defects are listed above so that it is redesigned or retired deliberately, not patched quietly.
- The core files it relies on (`core/backtester.py`, `core/validation.py` and the others) are part of the frozen
  shadow fingerprint. Editing them would stop the frozen T50 and RS3 forward tracking.

Results: `04_ml_round1.md`, with the summary in `ML_ROUND1_SUMMARY.md`.
