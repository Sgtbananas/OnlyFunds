# Real-market validation protocol (pre-registered 2026-11-01, before any real-data strategy run)

This protocol is committed **before** any strategy was run on the real data. Results are judged only
against the rules below. If a rule turns out to be wrong, that will be stated as a deviation in the
results, not silently changed.

## Data
- Binance spot 1h candles, 2019-01-01 → 2026-10-31, from `scripts/fetch_history.py` (checksummed
  manifests, `docs/DATA_QUALITY_REPORT.md`). Missing candles are not filled. Backtests block entries
  within 100 bars after a gap, same as paper trading.
- Universe (fixed in advance, chosen by Jan-2019 size to limit survivorship bias): BTC, ETH, BNB, XRP,
  LTC, ADA, EOS (delisted 2025-05-26; included until its data ends), XLM, TRX, all vs USDT.
- **Binance data cannot establish CoinEx profitability, liquidity, fills or fees.**

## Periods (chronological, no overlap in trading)
| Period | Trading window | Purpose |
|---|---|---|
| Development (DEV) | 2019-01-01 → 2022-12-31 | tuning, ML training |
| Validation (VAL) | 2023-01-01 → 2024-12-31 | first out-of-sample check, ML on/off decision |
| Final test (TEST) | 2025-01-01 → 2026-10-31 | **one evaluation**, after everything else is frozen |

Each period may use earlier candles as indicator history (`trade_from`), but trades and metrics start
at the period start. The TEST period is not looked at until the DEV/VAL results are written.

## Costs and risk (OF2 defaults, not tuned)
Fee 0.2% per side (CoinEx default taker), slippage 0.05%, stop slippage 0.2%, the cost-to-risk filter
(costs ≤ 25% of the stop distance), Conservative profile (0.5% risk per trade, ≤ 10% per position),
1,000 USDT per coin, one position per coin, long-only spot. As information only (never used for
selection), TEST is also reported at 0.1% fees.

## Candidates
For each strategy in {auto, trend_following, mean_reversion, stacked}:
1. **Fixed**: shipped defaults (no tuning).
2. **Tuned**: grid threshold ∈ {0.25, 0.35, 0.45} × stop ATR ∈ {1.0, 1.5, 2.0} × take-profit ATR ∈ {2, 3}.
   The objective is pooled expectancy in R across all coins on the training window, with ≥ 100 trades. Tuned on DEV,
   then applied to VAL. Re-tuned on DEV+VAL, then applied once to TEST.
3. **ML-gated**: the ML filter trained on DEV candles only, applied to the Fixed variant's entries. It is only carried
   to TEST if, on VAL, it beats the same strategy without ML on pooled expectancy in R.

## Walk-forward
Yearly folds 2021, 2022, 2023, 2024. Each is tuned (same grid and objective) on the two previous calendar years
and scored on the year. The TEST period is not part of the walk-forward.

## Metrics reported
Per coin, per period, pooled: net return after costs, profit factor, expectancy in R, max drawdown,
trades, win rate, average winner vs average loser (R), buy-and-hold over the same window (net of the same
costs), results by regime at entry and by calendar year.

## Credible-edge criteria (all must hold, on VAL **and** on TEST separately)
1. ≥ 100 pooled trades.
2. Pooled expectancy > 0 R, **and** the 95% bootstrap lower bound > 0 (10,000 trade resamples, seed 7).
3. Pooled profit factor ≥ 1.10.
4. Breadth: expectancy > 0 R on ≥ 60% of coins that have ≥ 10 trades in the period.
5. Not one-coin dependent: pooled expectancy stays > 0 R after removing the single best coin.
6. Walk-forward (Fixed/Tuned only): pooled OOS expectancy > 0 R in ≥ 3 of the 4 years.

Buy-and-hold and drawdown are reported for context but are not pass/fail. A capped-exposure,
long-only system is not expected to beat buy-and-hold in bull years.
Win rate is never a selection criterion.

## Interpretation
- If no candidate meets every criterion, the conclusion is **"no credible edge"**. Nothing is re-tuned
  afterwards to change that.
- Passing would justify CoinEx paper trading of that candidate, not live trading.
