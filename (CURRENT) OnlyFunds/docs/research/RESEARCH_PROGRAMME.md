# OF2 research programme: profitability first (pre-registered 2026-11-01)

This file is committed **before** any Round 1 candidate is run. Rules, variants, selection and pass criteria below
are fixed. Results go in `docs/research/02_round1_results.md`. Nothing here changes `docs/VALIDATION_PROTOCOL*.md`,
the frozen v1 results, the risk limits or the consumed TEST window (2025-01-01 → 2026-11-01), which is never loaded.

## 1. What the diagnosis says (`docs/research/01_diagnosis.md`, DEV+VAL 2019–2024, unchanged engine)
| Finding | Evidence |
|---|---|
| **At 1h there is no edge even before costs.** | Mean gross −0.008R over 4,110 trades. Costs (0.15R/trade) only deepen the loss. More 1h indicator combinations are not worth testing. |
| **Costs are a function of stop distance, so of timeframe.** | Median stop 3.3% (1h), 4.1% (4h), 9.1% (1d). Cost per trade 0.148R, 0.127R, 0.068R. |
| **4h has a small gross edge that costs erase.** | Gross +0.043R, net −0.084R. |
| **1d is the only timeframe net positive.** | Existing trend following at 1d: +0.085R net, 756 trades (gross +0.155R). Never part of v1 (which tested 1h only). |
| **The biggest single loss source is noise stop-outs.** | Trades closed within 0–2 bars: −0.44R (1h), −0.40R (4h), −0.20R (1d), roughly a third of all trades. Trades held ≥ 7 bars are positive on every timeframe. The 2-ATR trailing stop is the largest drain (−0.38 to −0.45R average, ~45% of trades). |
| **Regime, hour, weekday, liquidity:** no robust pattern. | Every 1h/4h slice is negative. Weekday/hour differences are within noise and have no rationale; they are not pursued. |
| **2022 (bear market) is negative on every timeframe.** | 1d: −0.22R. A long-only system needs a way to stand aside. |

**Contamination disclosure.** The diagnosis included 2023–2024 (VAL), so VAL has now been *seen* for the daily trend
family. VAL is therefore not a clean confirmation for daily trend candidates. Two independent checks are added:
1. A **holdout universe** of 16 coins never used by OF2 (fixed by rule, §4).
2. **Forward paper/shadow evidence** collected from 2026-11 onward, which is the only fully unseen data. The 2025–26
   window stays consumed.

## 2. Ranked hypotheses
| Rank | Hypothesis | Economic rationale | Data | Decision |
|---|---|---|---|---|
| 1 | **Daily time-series trend (breakout) with wide stops** | Under-reaction and herding produce persistent trends in crypto (documented time-series momentum); low turnover keeps costs ≈ 0.07R | Binance 1d OHLCV (have) | **Round 1** |
| 2 | **Cross-sectional relative strength rotation with a BTC regime filter** | Capital rotates into recent winners; the BTC filter targets the 2022-type bear market | Binance 1d (have) | **Round 1** |
| 3 | Short-term reversal after extreme daily drops (event study first) | Liquidation cascades overshoot. But v1 mean reversion failed and costs are high relative to a 1–5 day move. | Binance 1d (have) | Round 1 **information test only** |
| 4 | BTC → altcoin lead-lag at 1h (information test first) | Price discovery happens in BTC first. At hourly resolution it is probably arbitraged away. | Binance 1h (have) | Round 1 **information test only** |
| 5 | Funding-rate extremes as a contrarian spot signal | Crowded leverage unwinds | Binance futures funding (public; not yet downloaded; starts 2019-09) | Round 2 if Round 1 warrants |
| 6 | Volatility-contraction breakout at 4h | Volatility clusters | have | Deprioritised: 4h costs (0.13R) exceed its likely gross edge |
| – | Calendar effects, more 1h indicator stacks, ML filters, order-flow | No rationale or no evidence. Order-flow needs depth history we don't have. | – | **Not pursued** |

## 3. Round 1 candidates (exact rules)
**Common rules:**
- **Bars and execution:** daily bars (UTC). Decisions at the daily close; fills at the next open.
- **Engine:** the unchanged OF2 engine (`core/backtester.py`) with its costs: fee 0.20%/side, slippage 0.05%, stop
  slippage 0.20%, 10% volume cap, cost-to-risk filter 25%.
- **Risk:** Conservative limits unchanged (0.5% risk/trade, ≤ 10% notional/position, 10% drawdown halt). Long-only spot,
  1,000 USDT per coin.
- **ATR:** ATR(14) on daily bars.
- **Stops:** initial protective stop **3 × ATR**. No take-profit (`tp_atr_mult = 1000`, i.e. never reached). No trailing
  stop. The stop distance is a strategy rule, not a risk limit: the risk per trade stays 0.5%.
- **Indicators:** all causal; signals use only bars up to the decision close.

| Id | Entry | Exit |
|---|---|---|
| **T50** | close > highest close of the previous 50 days | close < lowest close of the previous 25 days, or stop |
| **T20** | close > highest close of the previous 20 days | close < lowest close of the previous 10 days, or stop |
| **T50B** | T50 entry **and** BTC close > BTC 100-day simple average | as T50 |
| **RS3** | weekly, at Monday's close: coin is in the top 3 of the universe by 28-day return, its 28-day return > 0, and BTC close > BTC 100-day average | at Monday's close if any entry condition fails; or stop |
| REF | existing `trend_following` signals (fixed parameters) at 1d, with the standard exits (1.5/3/2 ATR, trailing) | reference only: its VAL result is already known |

**Selection, fixed in advance:**
1. Trend family {T50, T20, T50B}: the variant with the highest pooled net expectancy on **DEV (2019–2022)** with
   ≥ 100 DEV trades is the family pick. RS3 has no variants.
2. **Walk-forward:** folds 2021, 2022, 2023 and 2024. For each fold, the trend-family variant is re-selected on the two
   prior years. A variant needs ≥ 50 trades in that window to be selectable; otherwise the fold counts as failed.
   RS3 has no selection, so its folds are the fixed rule's yearly results.
3. Each pick is evaluated **once** on VAL (2023–2024), and **once** on the holdout universe (2019–2024, with
   rotation within the holdout universe for RS3).

## 4. Holdout universe (fixed by rule before fetching)
**Rule:** Binance USDT spot pairs listed before 2019-01-01, excluding stablecoins and the 9 OF2 coins.
- **Coins:** ETC, LINK, NEO, QTUM, IOTA, ONT, ICX, WAVES, BAT, VET, ZIL, NULS, ZEC, DASH, XMR, OMG.
- **Data:** 1d only, 2019-01-01 → 2025-01-01. No TEST-period data is downloaded.
- **Delisted coins:** WAVES, XMR and OMG were delisted later. Their data up to delisting is kept, which avoids
  survivorship bias.
- **Quality:** `docs/data_quality/holdout_binance_1d.md`.

## 5. Pass criteria (unchanged protocol P3, plus stricter additions)
A candidate is **research-eligible** only if all of these hold:
1. **On VAL:** ≥ 100 pooled trades, expectancy > 0R with the 95% bootstrap lower bound > 0, profit factor ≥ 1.10,
   positive on ≥ 60% of coins with ≥ 10 trades, and positive without the best coin.
2. **Walk-forward:** positive in ≥ 3 of 4 folds.
3. **Holdout universe (added):** pooled expectancy > 0R, profit factor ≥ 1.10, positive on ≥ 50% of coins with ≥ 10
   trades.
4. **Random-entry benchmark (added):** the pick's pooled net expectancy on DEV+VAL is above the 95th percentile of
   200 random-entry runs. Those runs use the same exit rules and stops and the same per-bar entry probability, and
   show how much of the result is market beta.
5. **Cost stress (information only):** results at 0.30%/side are reported.

**Eligible ≠ deployable.**
- An eligible candidate becomes a forward paper/shadow candidate. The consumed TEST window cannot confirm it.
- Activation still needs forward evidence under the existing rules.
- `LIVE_TRADING_LOCKED` stays `True`.

## 6. Information tests (no strategy is built unless the test is positive after costs)
- **H3, reversal:** DEV only. Event: a daily log return below the coin's trailing-365-day 5th percentile. Measure
  forward 1-, 3- and 5-day returns from the next open, against the 0.5% round-trip cost.
- **H4, lead-lag:** DEV only, 1h. Measure an altcoin's next-hour return after BTC hourly moves above +1% or below −1%,
  against its unconditional next-hour return and the 0.5% round-trip cost.
