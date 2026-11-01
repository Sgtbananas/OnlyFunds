# Overnight research report (2026-11-02)

**No new strategy has demonstrated a trading edge. Nothing is approved for trading. Live trading is still locked.**

The most useful findings were these:
1. **ML filtering does not help T50.** Every model tried made it worse.
2. **Position management does help, robustly.** It is now being tracked prospectively as a separate experiment.
3. **RS3's record is not one or two lucky trades.** It survives removing its best trades and beats random baskets,
   so RS3 research continues.

Evidence labels used below:
- **Proven:** a pre-registered test passed on data not used to design it. Nothing qualifies yet.
- **Supported (historical):** a pre-registered rule passed on 2019–2024 data. That data has been examined repeatedly,
  so this is not proof.
- **Preliminary:** descriptive or narrowly missed.

## Shadow tracking
- **Status:** the daily task `\OnlyFunds\OF2 Shadow Daily Update` is registered and verified (`SHADOW_TRACKING.md`).
  - It runs at 00:30 UTC, launched by Task Scheduler (parent process `svchost`).
  - Overlapping runs are blocked twice over: by the scheduler setting and by an OS file lock.
  - Each run writes a log and a journal entry, and a crash is recorded with its traceback.
  - Exchange-key environment variables are removed before the tracker starts.
- **Tested today:** three scheduler runs, all exit 0 and status "waiting". 20 of 25 T50 coins have live Binance data:
  ICX is stale, and EOS, NULS, OMG, WAVES and XMR are unavailable and recorded as such.
- **First evaluated day:** the 2026-11-03 00:30 UTC run, which is the first one after the 2026-11-02 candle completes.
- **Frozen fingerprints are unchanged:** T50 `29139c64…`, RS3 `1ffc1d23…`. A test verifies them.
- **New: two experimental variants tracked alongside**, with their own fingerprints, frozen at 14:23 UTC before the
  first forward decision:
  - X1 = T50 + position management
  - X2 = X1 + BTC trend gate

## Machine learning
- **What existed:** one real model, a disabled gradient-boosting entry filter.
  - Validation AUC was 0.51–0.53 against a training AUC of 0.75.
  - It never improved net results, and its stored copy accepted 0 of 5,280 trades.
  - Nine defects are listed in `ML_AUDIT.md`. Among them: the label doesn't match any strategy's exits; the threshold
    ignores the base rate; the split leaks across coins; and there is a fake random "sentiment" placeholder (unused;
    a test now guards it).
- **What was built:**
  - A leak-safe research engine: 24 causal features, purged and embargoed walk-forward folds, fixed hyperparameters,
    and model cards with hashes that refuse inference if anything changed.
  - Its test protocol: random-skip benchmarks, Holm correction, and month-block and week-cluster bootstraps.
  - A hash-chained register of hypotheses, recorded before testing: 20 hypotheses and 18 results. The two without a
    result are the forward experiments.
  - An experimental forward tracker.
  - 28 new tests.
- **What the models can actually predict, out of sample 2021–24:**
  - The 20-day direction of a coin: **no** (AUC 0.47–0.50, worse than the historical average).
  - Whether a T50 trade wins: **no** (AUC 0.25–0.56).
  - 20-day volatility: **yes, partly** (30% better than the historical average; not clearly better than "same as
    last month").
  - Fast stop-outs: **weakly** (AUC 0.51–0.63).
  - Stress (a 20% drop): **weakly ranked** (AUC 0.54–0.59), but no better than the historical average.
- **Does ML improve net profitability?** **No.** All 6 ML gates on T50 were rejected, with changes of −62R to −167R
  against +251R. Each removed trades that averaged **+0.65R to +3.1R**: the gates filter out T50's large winners.

## Profitability research (2021–24 walk-forward unless noted; all after fees, slippage and stop slippage)
| Idea | Result | Evidence |
|---|---|---|
| Frozen T50 | 289 trades, +251R, +0.87R per trade (95% interval +0.06 to +1.89), PF 2.83; **−0.12R per trade without the top 10% of trades** | baseline |
| BTC > 100-day SMA gate (cash in BTC downtrends) | +10.6R (+4%), beats random skipping (Holm p 0.010), survives costs, same sign on both coin sets; helped in only 2 of 4 years | inconclusive |
| Rule-based regime gate | +2.9R, beats random (p 0.013); wrong sign on the holdout | inconclusive |
| Coin > 200-day SMA, ML regime gates, ML trade scoring, sparse ML, fast-stop model | all reduce total R | rejected |
| **Position management** (trim positions above 15% of equity back to 10%; profit lock after +2R) | Return/DD up in all 8 cells (2019–24). Main 9: DD 17.9 → 10.7%, return +70 → +88%. Holdout: DD 11.4 → 9.7%. Top winners keep ≥ 100%. 9 of 12 parameter neighbours improve all cells. Survives higher costs. | Pre-registered test **rejected** (holdout DD fell < 2 points); robustness and cost checks **supported**; now under prospective test as X1 |
| RS3 relative-strength rotation | Not driven by one or two trades: without its top 2 trades +0.35R (lower bound +0.11R). Beats random baskets on main 9 and holdout 16 (p 0.002). **Fragile:** negative without its top 10% of trades; holdout lost money in 2023–24. | supported (historical) → keep researching |

**Failure conditions shared by T50 and RS3:**
- Both make money in bull markets only (2022: −0.7R per trade).
- Both depend on a few large trends.
- Neither works if those trends do not recur.

## Risk
**The 38% BNB position explained.** T50's stop is fixed at 3 ATR below entry, and the 10% position cap applies only
at entry. A winning position therefore grows without limit, and its open profit is unprotected until the
25-day-low exit fires.

**In one shared account under Conservative limits (main 9, 2019–24, as configured):**
- Initial risk is 0.5% per trade, but:
  - the largest position reached **23.6%** of equity;
  - total position value reached **29.5%**;
  - **open risk to the stops reached 24.2% of equity**.
- Giving back that profit caused a 14% drawdown. The permanent 10% kill switch tripped on 2021-05-19 and the account
  stopped trading for good.
- **After a sudden reversal:**
  - If every open position had hit its stop at the peak, the loss would have been about 24% of equity.
  - A 30% overnight gap across all coins would have cost about 9%. Crypto moves together, so three positions
    behave almost like one.

**With position management (R3), on the same coins:**
- The largest position stays at 15.7% (20.1% without the halt).
- Open risk to stops drops to 15–16%.
- Drawdown drops from 14 to 10% as configured, and from 17.9 to 10.7% without the halt.

**Safeguards recommended (not enabled):**
- Enforce the position cap continuously, not only at entry.
- Cap open risk to stops, measured at current prices.
- Leave the kill switch as it is.

## Next three priorities
1. **Let the forward record build.** It is the only uncontaminated evidence. Review T50, RS3, X1 and X2 monthly,
   descriptively, without changing anything until the pre-registered sample sizes are reached (T50 about 16 months;
   X1 and X2 24 months).
2. **Propose continuous position caps and an open-risk cap as a risk-control change**, for your decision. They are
   supported historically and are generic risk reduction: they improved 70% of random-entry books too, not only T50.
   The change would go into the engine only with your approval, as a new frozen version.
3. **Get new, independent data rather than re-testing 2019–24.** That data has been examined too often to establish
   anything. Two routes:
   - extend the holdout to more coins chosen by rule;
   - pre-register a small number of RS3 and T50 refinements and test them on the forward period only.
