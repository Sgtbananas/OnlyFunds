# Real-market strategy evaluation (2026-11-01)

**Conclusion: no OF2 strategy shows a credible trading edge on real market data.** Every candidate fails the
pre-registered criteria on the validation period (2023–2024) and again on the untouched final test period
(2025-01 → 2026-10). All of them have negative net expectancy after realistic costs. The ML filter adds nothing.
This is the result the protocol was designed to be able to give. Nothing was re-tuned afterwards to change it.

> **Exchange caveat:** all data is **Binance** spot. It says nothing definitive about CoinEx prices, liquidity,
> fills or fees. CoinEx connectivity remains unverified (HTTP 503 from its gateway).

## What was done
| Step | Artefact |
|---|---|
| Real data: Binance spot 1h, 9 coins, 2019-01-01 → 2026-10-31, checksummed, gaps reported not filled | `scripts/fetch_history.py`, `data/historical/binance/*.manifest.json`, `docs/DATA_QUALITY_REPORT.md` |
| Rules fixed **before** any strategy ran on real data (commit `bc0033a`) | `docs/VALIDATION_PROTOCOL.md` |
| Stage 1: tuning on DEV 2019–2022, out-of-sample VAL 2023–2024, yearly walk-forward 2021–2024, ML train-on-DEV / judge-on-VAL, decisions frozen | `docs/validation/stage1_dev_val.md`, `docs/validation/frozen_decisions.json` |
| Stage 2: one evaluation of the frozen candidates on TEST 2025-01 → 2026-10 (protocol fingerprint verified, re-runs refused) | `docs/validation/stage2_test.md` |
| Reproduce | `python scripts/fetch_history.py`, then `python scripts/real_market_validation.py --stage devval`, then `--stage test` (refuses to overwrite) |

**Data quality:**
- 68,604 candles per coin (EOS: 56,031, delisted 2025-05-26 and kept to avoid survivorship bias).
- 60 missing candles per coin, all exchange-wide Binance outages; never filled. Backtests block entries for 100 bars after a gap.
- No duplicates, misaligned timestamps or inconsistent OHLC.
- BTC/ETH closes vs OKX (2023): median deviation 0.007%.

**Costs:**
- 0.2% fee per side (CoinEx default taker)
- 0.05% slippage; 0.2% stop slippage
- cost-to-risk filter on
- Conservative risk, 1,000 USDT per coin, long-only spot

## Results: pooled across coins, net of all costs
Expectancy is in R (1R = the planned loss if the stop is hit, including fees and stop slippage). "CI low" is the
95% bootstrap lower bound of the expectancy.

### Validation, 2023–2024 (out-of-sample for tuning and ML)
| Candidate | Trades | Expectancy R | CI low R | Profit factor | Win % | Avg win R | Avg loss R | Coins positive | Mean net % | Mean buy & hold % | Worst DD % |
|---|---|---|---|---|---|---|---|---|---|---|---|
| auto / fixed | 255 | −0.065 | −0.190 | 0.91 | 35.7 | +1.19 | −0.76 | 40% of 5 | −0.5 | +260.5 | 4.7 |
| auto / tuned | 428 | −0.073 | −0.167 | 0.88 | 34.8 | +1.15 | −0.73 | 50% of 6 | −1.1 | +260.5 | 7.6 |
| auto / fixed + ML | 0 | – | – | – | – | – | – | – | 0.0 | +260.5 | 0.0 |
| trend following / fixed | 380 | −0.082 | −0.184 | 0.87 | 32.4 | +1.27 | −0.73 | 29% of 7 | −1.1 | +260.5 | 8.1 |
| trend following / tuned | 380 | −0.082 | −0.184 | 0.87 | 32.4 | +1.27 | −0.73 | 29% of 7 | −1.1 | +260.5 | 8.1 |
| trend following / fixed + ML | 23 | −0.134 | −0.544 | 0.81 | 26.1 | +1.59 | −0.74 | – | −0.1 | +260.5 | 1.1 |
| mean reversion / fixed | 283 | −0.084 | −0.197 | 0.86 | 37.1 | +1.06 | −0.76 | 25% of 8 | −0.9 | +260.5 | 6.0 |
| mean reversion / tuned | 137 | −0.176 | −0.360 | 0.79 | 31.4 | +1.45 | −0.92 | 20% of 5 | −0.8 | +260.5 | 6.4 |
| mean reversion / fixed + ML | 1 | +1.232 | – | – | 100 | +1.23 | – | – | +0.1 | +260.5 | 0.5 |
| stacked / fixed | 58 | −0.194 | −0.415 | 0.65 | 34.5 | +0.86 | −0.75 | 0% of 3 | −0.4 | +260.5 | 3.0 |
| stacked / tuned | 790 | −0.172 | −0.224 | 0.61 | 28.7 | +0.88 | −0.60 | 0% of 9 | −5.5 | +260.5 | 10.2 |
| stacked / fixed + ML | 11 | −0.492 | −0.882 | 0.30 | 18.2 | +0.95 | −0.81 | – | −0.2 | +260.5 | 1.4 |

### Final test, 2025-01 → 2026-10 (run once, frozen candidates)
| Candidate | Trades | Expectancy R | CI low R | Profit factor | Win % | Avg win R | Avg loss R | Coins positive | Mean net % | Mean buy & hold % | Worst DD % |
|---|---|---|---|---|---|---|---|---|---|---|---|
| auto / fixed | 144 | −0.295 | −0.434 | 0.47 | 23.6 | +1.03 | −0.70 | 0% of 5 | −1.7 | −27.2 | 4.1 |
| auto / tuned | 18 | −0.134 | −0.650 | 0.86 | 27.8 | +1.60 | −0.80 | – | −0.1 | −27.2 | 1.4 |
| trend following / fixed | 266 | −0.200 | −0.311 | 0.64 | 28.6 | +1.12 | −0.73 | 0% of 6 | −2.0 | −27.2 | 8.1 |
| trend following / tuned | 266 | −0.200 | −0.311 | 0.64 | 28.6 | +1.12 | −0.73 | 0% of 6 | −2.0 | −27.2 | 8.1 |
| mean reversion / fixed | 343 | −0.135 | −0.233 | 0.77 | 34.7 | +1.08 | −0.78 | 14% of 7 | −1.7 | −27.2 | 7.8 |
| mean reversion / tuned | 109 | −0.034 | −0.245 | 0.96 | 38.5 | +1.37 | −0.92 | 33% of 6 | −0.1 | −27.2 | 2.3 |
| mean reversion / fixed + ML | 0 | – | – | – | – | – | – | – | 0.0 | −27.2 | 0.0 |
| stacked / fixed | 43 | −0.122 | −0.385 | 0.79 | 23.3 | +1.44 | −0.60 | 0% of 1 | −0.2 | −27.2 | 2.4 |
| stacked / tuned | 618 | −0.187 | −0.245 | 0.58 | 27.0 | +0.90 | −0.59 | 0% of 7 | −4.7 | −27.2 | 10.2 |

At Binance-level 0.1% fees (information only, never used for selection) every TEST candidate is still negative:
−0.11R to −0.37R. Lower fees let more trades past the cost filter but don't create an edge.

### Walk-forward (tuned on the two previous years, scored on the year, pooled R)
| Strategy | 2021 | 2022 | 2023 | 2024 | Positive years |
|---|---|---|---|---|---|
| auto | −0.120 (1,054 trades) | −0.609 (62) | −0.178 (480) | −0.060 (614) | 0/4 |
| trend following | −0.119 (1,193) | −0.288 (209) | −0.216 (480) | −0.095 (726) | 0/4 |
| mean reversion | −0.152 (448) | −0.160 (788) | +0.540 (9) | −0.111 (126) | 1/4 |
| stacked | −0.264 (166) | −0.719 (36) | −0.217 (414) | −0.108 (610) | 0/4 |

### ML enabled vs disabled
- **Ranking ability:** on its internal DEV validation the filter barely beats chance (AUC 0.513–0.530), and none of the four models qualified as "useful".
- **Effect on trades:** on VAL it removed most trades without improving expectancy (trend following: −0.134R with ML vs −0.082R without).
- **The one ML carry-over:** "mean reversion + ML" moved to TEST because of a **single** VAL trade (+1.23R). It then made 0 TEST trades.

**Protocol flaw (reported, not patched after the fact):** the ML carry-over rule should have needed a minimum trade
count. It couldn't create a false pass, because the TEST criteria need at least 100 trades.

### By market regime (regime at entry, fixed variants, pooled R)
- **Trending-up and high-volatility regimes** carry almost all trades, and they're negative in both VAL and TEST for every strategy.
- **Ranging regimes produce almost no trades** (3–11 per period). Calm markets have small ATR, so the cost filter
  blocks them. At 1h with CoinEx-level fees, `auto`'s "range → mean reversion" path is effectively inactive.
- **Mean reversion in down-trends:** +0.13R on TEST (89 trades), but −0.14R on VAL (59 trades). Inconsistent between
  periods, and found by slicing after the fact, so **not** evidence of an edge.

### By calendar year (fixed variants, pooled R)
- Every strategy-year is negative except mean reversion 2023 (+0.29R, 59 trades) and stacked 2026 (+0.07R, 7 trades).
- Neither repeats in the adjacent year.

### By coin
Per-coin tables are in the stage files. No coin is consistently positive for any strategy. On TEST, the share of coins
with positive expectancy is 0–33% for every candidate (the criterion is ≥ 60%).

## Context: capital preservation is not an edge
- **TEST (2025-01 → 2026-10):** buy-and-hold lost 27% on average; the strategies lost 0.1–4.7%, with worst drawdowns ≤ 10.2%.
- **VAL (2023–2024):** buy-and-hold gained 260% on average; the strategies lost up to 5.5%.

This is the risk limits doing their job (small positions, capped exposure), not a source of profit. A strategy with
negative expectancy loses slowly in every regime.

## Answers to the brief
- **Credible edge?** **No.** 0 of 12 VAL candidates and 0 of 9 TEST candidates pass. Not one has positive pooled
  expectancy on both periods; none has a 95% lower bound above zero; none passes walk-forward.
- **Best strategy?** None qualifies, so OF2 should **not** select dynamically among these strategies for real trading.
  The least-bad TEST result (mean reversion tuned, −0.03R, profit factor 0.96) failed VAL and walk-forward.
- **ML?** No measurable benefit on real data.
- **CoinEx?** Not established. Binance results are an optimistic proxy at most: CoinEx liquidity is thinner and its default fees are higher.

## Recommendation
- **Don't paper trade these strategies hoping for profit.** Paper trading remains useful only for testing the software.
- **Any new strategy idea must go through the same pipeline.** That means new periods, or a fresh untouched test window (e.g. data after
  2026-10), because this TEST period has now been used once.
