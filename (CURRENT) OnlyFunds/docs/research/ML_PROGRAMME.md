# ML Round 1: pre-registration (written and committed before any result was computed)

**Status: research only.**
- Nothing here is approved for trading, and the frozen T50 and RS3 shadow tracking is untouched: no ML code is
  imported by `research/forward_shadow.py`, and no ML code reads or writes `state/shadow/`.
- Live trading stays locked.

**Source of the hypotheses:** proposed by the AI research assistant (Claude, this session) after the Round 1 and
T50 scrutiny results. They are recorded with `source: ai` in `research/registry/hypotheses.jsonl` and judged by
the same rules as any human hypothesis.

## 1. Data and integrity
- **Coins:** 25, daily Binance spot candles, `2019-01-01 → 2024-12-31`. These are the 9 OF2 coins plus the 16-coin
  holdout fixed by rule in `RESEARCH_PROGRAMME.md`, loaded through `round1.load`, which asserts every candle is
  before 2025-01-01. Delisted coins (WAVES, XMR, OMG) are included up to delisting.
- **Never used:** data on or after 2025-01-01, the consumed TEST window and the forward shadow period. That
  covers development, model choice, feature choice and parameters.
- **No tuning:** every model, hyperparameter, feature, target, threshold and decision rule below is fixed now.
  There is no hyperparameter search, so there is no tuning set to leak through. Models that fail are reported as
  failed and are not adjusted.
- **Fingerprints:** every trained model gets a card (`docs/research/ml_models/<model>_<fold>.json`). It records
  the SHA-256 of the feature/label code, the feature list, the target, the training date range, the per-coin data
  hashes, the hyperparameters, the seed and the model file's hash. Inference refuses to run if the code, feature
  list or data hashes differ from the card.

## 2. Features (all causal: computed from data up to and including the decision day's close)
- **Coin features:**
  - `brk_atr`: close minus the highest close of the previous 50 days, in ATR(14) units.
  - `exit_dist_atr`: close minus the lowest close of the previous 25 days, in ATR units.
  - `ret_5`, `ret_20`, `ret_60`: log returns.
  - `ma200_dist`: log of close over SMA200.
  - `ma50_slope`: log of SMA50 over SMA50 ten days earlier.
  - `atr_pct`.
  - `vol_ratio`: standard deviation of daily returns over 20 days, divided by the same over 100 days.
  - `vol_expansion`: ATR over ATR 20 days earlier.
  - `volume_ratio`: log of 20-day over 100-day mean quote volume.
  - `trend_persist`: share of the last 50 closes above SMA50.
  - `rs_rank`: cross-sectional percentile of the 28-day return among coins with data that day.
  - `corr_btc_60`: 60-day correlation of daily returns with BTC.
- **Market features:**
  - BTC: `btc_ret_20`, `btc_ret_60`, `btc_ma200_dist`, `btc_above_ma100`, `btc_vol_20`, `btc_dd_90` (drawdown from
    the 90-day high), and `btc_trend_score` (the existing causal score in `core.regime`).
  - `breadth_ma50`: share of coins above SMA50.
  - `breadth_brk`: share of coins with a 50-day breakout in the last 5 days.
  - `mkt_corr`: cross-coin mean of `corr_btc_60`.
- **Trade-model extra:** `prior_r_shrunk`, the sum of this coin's earlier T50 trade results in R (trades closed
  before the decision) divided by (n + 5).

## 3. Models (fixed hyperparameters, seed 7)
| Id | Unit | Target | Model |
|---|---|---|---|
| M1 | coin-day | `y_up`: log(open[t+21] / open[t+1]) > 0.005 (round-trip cost) | L2 logistic regression, standardised features, C = 1.0 |
| M2 | coin-day | `y_up` | HistGradientBoosting: depth 3, learning rate 0.05, 200 iterations, min leaf 200, L2 1.0 |
| M1s / M2s | coin-day | `y_stress`: min(low[t+1..t+20]) / open[t+1] − 1 < −0.20 | as M1 / M2 |
| MV | coin-day | `y_vol`: log realised volatility of daily returns t+1..t+20 | ridge regression (alpha 1.0) |
| M3 | T50 trade | `y_win`: net R > 0, from the unchanged per-coin engine (costs included) | L2 logistic, C = 1.0 |
| M4 | T50 trade | `y_win` | HistGradientBoosting: depth 2, learning rate 0.05, 100 iterations, min leaf 20 |

## 4. Validation protocol
- **Walk-forward:** expanding-window folds with test years 2021, 2022, 2023 and 2024. Each model is trained once
  per fold on data before the test year.
  - Reported out-of-sample periods: all four years together, the DEV years (2021–22) and the VAL years (2023–24).
- **Purging and embargo:**
  - A training row is kept only if its label is fully known 5 days before the test year starts. For coin-days
    the label ends at t+21. For trades it ends at the exit date.
  - Features have no look-ahead. A test checks this by perturbing future prices and confirming nothing changes.
- **Correlated observations:**
  - Coin-day metrics use a block bootstrap by calendar month: all coins and days in a month are resampled together.
  - Trade metrics use a week-of-entry cluster bootstrap.
  - Both use 5,000 resamples.
- **Probabilistic skill (M1, M2, M1s, M2s):**
  - Out-of-sample log loss against two baselines: (a) climatology, the training-fold base rate; (b) the rule
    baseline, the training-fold frequency of the target within each existing `core.regime` BTC trend class.
  - Also reported: AUC, a reliability table and Brier score.
  - **MV** is compared with persistence (trailing 20-day realised volatility) by out-of-sample R².
- **Costs:** the engine's: 0.2% fee per side, 0.05% slippage, 0.2% stop slippage, 10% volume cap, 25%
  cost-to-risk filter. Sensitivity runs use 0.4% fee, 0.2% slippage and 0.5% stop slippage.

## 5. How predictions become trading decisions (gates on the frozen T50 entries; T50 itself is never edited)
| Gate | Rule (applied to an entry signal at the decision close) |
|---|---|
| G0 | none (frozen T50) |
| **ML gates** | |
| GM1 / GM2 | take the entry only if M1 / M2 gives P(y_up) ≥ 0.5 |
| GM3 / GM4 | take the entry only if M3 / M4 gives P(y_win) ≥ the 1/3 quantile of that fold's *training* predictions (skip the predicted-worst third) |
| **Non-ML baselines** | |
| G1 | BTC close > 100-day SMA |
| G2 | coin close > 200-day SMA |
| G3 | the existing rule regime: BTC trend class is not weak_down or strong_down |
| **Random benchmark** | for each gate, the same number of entry signals skipped at random (300 seeded draws) |

- **Re-running:** a gated signal set is run through the unchanged per-coin engine again. A skipped entry can let a
  later entry on the same coin happen.
- **Also run:** the single 10,000 USDT account with Conservative limits, as configured and in info-only no-halt
  mode.
- **RS3 (adaptive selection, descriptive only):** the GM1 gate is also applied to RS3 basket entries. The share of
  days in cash is reported for every gate. This shows whether one market-state model can switch strategies off.
  It is not a validated selector.

## 6. Decision rules
**Family A: ML (6 tests, Holm-corrected at 5%).**
- **H-ML1 (M1) and H-ML2 (M2):** supported if out-of-sample log loss beats both climatology and the rule baseline,
  with the month-block bootstrap 95% interval of the improvement above 0. This must hold over 2021–24 and in each of
  DEV-OOS and VAL-OOS.
- **H-ML3 to H-ML6 (gates GM1 to GM4):** supported only if all of the following hold.
  1. Total net R over 2021–24 on 25 coins exceeds G0.
  2. The improvement beats the random-skip benchmark, with Holm-adjusted p < 0.05.
  3. Total R improves in at least 3 of the 4 test years.
  4. The direction is the same on the main 9 and on the holdout 16.
  5. Single-account return / max DD is not worse than G0.
  6. The improvement survives the cost-sensitivity run.

**Family B: non-ML gates G1 to G3 (Holm within the family).** The same rules as the ML gates.

A gate with a positive effect that fails significance is reported as **inconclusive**, not as an improvement.

## 7. Capital preservation variants (deterministic, non-ML; P5)
**Setup:** single account, Conservative limits, frozen T50 entries and exits. Results are given as configured and in
info-only no-halt mode, for main 9 and holdout 16, 2019–2024.

**Risk decomposition:** for the baseline, the report shows per day:
- initial stop risk
- current position value
- unrealised profit
- open risk to the stops
- largest single-position share of equity
- equity drawdown

| Id | Rule |
|---|---|
| R0 | baseline |
| R1 | concentration band: if a position's value exceeds 15% of equity at a close, sell down to 10% (the configured `max_position_pct`) at the next open |
| R2 | profit lock: once the close is ≥ 2R above the entry, the stop becomes max(stop, highest close since entry − 6 × ATR(14)), checked intrabar |
| R3 | R1 + R2 |

**Supported** if all of these hold in both universes and both modes:
1. Max drawdown is at least 2 percentage points lower.
2. Return / max DD is not lower.
3. Total return is at least 80% of R0's.
4. The five largest R0 winners keep at least 70% of their R0 profit.

## 8. What will not be done
- Moving thresholds, features or hyperparameters after seeing results.
- Adding models beyond this list in this round.
- Using 2025–26 data.
- Feeding any result into the frozen T50 and RS3 shadow tracking.
- Calling any variant deployable. A variant that passes becomes a new candidate for its own frozen forward test.
