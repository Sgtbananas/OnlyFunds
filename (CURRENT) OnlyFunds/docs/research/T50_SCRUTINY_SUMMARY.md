# T50 scrutiny: verdict (2026-11-01)

Full measurements: `docs/research/03_t50_scrutiny.md` (reproduce with `python research/t50_scrutiny.py`, ~13 min;
figures in `docs/research/fig/`). Rules frozen as pre-registered. No data ≥ 2025-01-01 was loaded. Nothing
was tuned after Round 1.

## The question
> Does T50 possess a repeatable trading edge, or is its apparent profitability primarily explained by a few
> exceptional bull-market trades?

**Answer: both, and the second part dominates how it behaves.**
- T50 captures trends better than chance, repeatably across coins, years and an unseen universe.
- That edge exists **only through rare, very large winners in rising markets**. Bear and sideways markets show none.
- It is **not** a strategy that wins often or in every regime.
- It has **not passed** validation (62 < 100 VAL trades) and is **not deployable**.

## Evidence for a repeatable component
| Test | Result |
|---|---|
| Matched random entries (1,000 runs, trade count matched per coin, same exits/stops/sizing/costs, seeded, no price access) | Main 2019–24: T50 +1.36R vs random median +0.55R, 95th pct +0.95R, max +1.44R → **99.9th percentile, p ≈ 0.002**. Holdout: +0.74R vs median +0.21R, 95th pct +0.40R, max +0.61R → **100th percentile, p ≈ 0.001**. |
| Unseen coins (16-coin holdout, delisted coins included) | 277 trades, +0.74R (95% lower bound +0.36R), profit factor 2.6, positive on 13 of 15 eligible coins |
| Leave one coin out (25 coins) | Expectancy stays +0.87R to +1.03R whichever coin is removed. Without BNB: +0.87R. 22 of 25 coins positive individually. |
| Outside 2020–21 (all 25 coins) | 262 trades, +0.43R, 95% lower bound **+0.10R**, profit factor 1.87 |
| Big winners spread out | The top trades come from 2019, 2020, 2021, 2023 and 2024 and from many coins (BNB, ZIL, XLM, ETC, VET, ADA, BTC, ETH) |
| Costs | Robust: 0.40% fee + 0.20% slippage + 0.50% stop slippage changes the single-account return from +50.4% to +47.1% |

## Evidence that the profit is concentrated in rare bull-market trades
| Test | Result |
|---|---|
| Median trade | **−0.28R** (main 2019–24). Most trades lose. |
| Top 5 trades' share of net profit | 88% (DEV), 69% (VAL), 58% (main 2019–24), 49% (holdout), 37% (all 25 coins) |
| Excluding the top 5 winners | DEV +0.16R (lower bound **−0.15R**); main 2019–24 +0.57R; holdout +0.39R |
| Excluding the top 10% of trades | **Negative in every sample**: −0.17R (DEV), −0.03R (main), −0.10R (holdout), −0.08R (25 coins) |
| By BTC market state at the decision (OF2 regime detector) | Bull +1.23R (341 trades); sideways +0.21R (68, lower bound −0.18R); **bear −0.13R** (27) |
| 2022 | **−0.66R to −0.80R per trade** on every universe (58 trades on 25 coins). The March–April rally entries were stopped out within two weeks. The August entries mostly lost 0.6–0.9R. The November entries all stopped out within 3 days of the FTX collapse. |
| Holdout outside 2020–21 | +0.14R, lower bound −0.11R: **not significant** |
| Exposure | T50 is in the market more than its matched random entries (34% vs 26% of days). Per unit of exposure it still earns ~1.8× random. Part of its advantage is holding winners longer in up-markets. |

## Drawdown (the 22.5% BNB figure)
- **Not a bug.** Each per-coin backtest account sizes for 0.5% loss at the stop.
- The BNB position started at 3.1% of its account (risk 0.50%) and rose about 10×, until the open position was 38% of
  the account.
- Giving back part of that open profit before the 25-day-low exit fired is a 22.5% intraday drawdown of that coin's
  account. It is **concentration of open profit**, not lost capital. The 0.5% limit only bounds the loss at entry.
- Round 1's per-coin accounts let up to 9 coins be in position at once. A single Conservative account allows 3.

## Economics: one 10,000 USDT account, Conservative limits across coins
| Portfolio (2019–2024) | Return | CAGR | Max DD | Sharpe | Avg exposure | Trades | Note |
|---|---|---|---|---|---|---|---|
| Main 9, as configured | +50.4% | 7.0% | 14.0% | 0.83 | 3.5% | 31 | **Kill switch (10% DD, permanent) trips 2021-05-19.** No trades after that. |
| Main 9, info only: no halt | +70.4% | 9.3% | 17.9% | 0.86 | 8.3% | 83 | Passive basket at the same 8.3% exposure: +186%, max DD 51.5% |
| Holdout 16, as configured | +29.0% | 4.3% | 10.0% | 0.64 | 3.9% | 68 | Kill switch trips 2023-08-16 |
| Holdout 16, info only: no halt | +32.0% | 4.7% | 11.4% | 0.64 | 5.2% | 91 | Passive at the same exposure: +17.2%, max DD 45.9% |
| Buy & hold, 100% (main / holdout) | +2,250% / +331% | | 73% / 91% | | 100% | | |

**What the economics say:**
1. **The current limits and T50 are incompatible.** T50 profits from letting open profit run, so its marked-to-market
   drawdowns exceed 10%. The kill switch is permanent, so it stops the strategy after its first big winner gives back
   gains. The limits were not changed.
2. **Capital use is very low.** At 0.5% risk and a 3-ATR stop, a position is about 3% of capital. Average exposure is
   4–8%. The returns, about 4–9% a year, are small in absolute terms.
3. **Against passive exposure it is mixed.**
   - On the main coins, a passive basket at the same average exposure earned more (+186% vs +70%), with about 3× the drawdown.
   - On the holdout, T50 earned more (+32% vs +17%), with a quarter of the drawdown.
   - Risk-adjusted, T50's main advantage is **drawdown control**, not higher return.

## Holdout data availability
- All 16 holdout coins have data from their Binance listing (2019-01-01 to 2019-04-03) to 2024-12-31.
- WAVES (2024-06-17), XMR (2024-02-20) and OMG (2024-06-17) are included up to delisting.
- Counting positions still open at the end or at delisting changed the holdout from +0.737R (276 trades) to +0.744R (277).

## Decision
- **T50:** not promoted, not eligible. It goes to frozen forward shadow tracking as the leading hypothesis.
  The question that matters most, behaviour in a bear or sideways market, can only be answered prospectively.
  The 2025–26 window is consumed.
- **Not done:** no rule changes and no "bear filter" bolted on after seeing 2022. A regime guard would be a new,
  separately pre-registered candidate, tested on data not used to motivate it.
