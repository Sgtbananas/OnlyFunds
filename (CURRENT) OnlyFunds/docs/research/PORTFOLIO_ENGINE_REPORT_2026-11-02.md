# Portfolio intelligence, regimes and strategy selection: Round 4 report (2026-11-02)

**No new rule, model or engine demonstrated a repeatable edge.** Nothing is approved for trading, and live trading
is still locked. T50, RS3, X1, X2 and the shadow tracker are unchanged. Full tables: `09_round4.md`.
Pre-registration: `ROUND4_PROGRAMME.md` (commit `08a283a`, before any result).

Evidence labels:
- **proven:** forward-confirmed. Nothing qualifies.
- **supported (historical):** passed pre-registered checks on 2019–24. Nothing new qualifies this round.
- **preliminary / post hoc:** descriptive only.

## 1. Previous phase verified independently
- **Branch and commit:** `of2-completion` at `960c6d8` when this phase began, working tree clean; 322 tests passed.
- **Freezes:** match. T50 `29139c64…`, RS3 `1ffc1d23…`, X1 `3de65f01…`, X2 `c2edc9f7…`.
- **Accounting:** all 72 Round 3 books were recomputed independently. In every one, final equity = capital + Σ trade
  P&L to within 2e-11 USDT, R × initial risk = P&L for every trade, and every published return reproduces.
- **Reconciliation:** the earlier figures are reconciled (`07_reconciliation.md`), and the +251.2R baseline
  reproduces exactly.
- **Smoke-run caveat:** before the official run, a smoke test computed the regime verdicts on the full main-9 data.
  The regime part is deterministic and its rules were fixed and committed, so the official run reproduced them.
  This is noted for completeness.

## 2. Regime engine (`research/regime_engine.py`)
**What it is.** Daily probabilities with a confidence value (1 − normalised entropy) for:
- direction: bull / sideways / bear;
- high and low volatility;
- volatility expansion;
- stress (a 15% or larger BTC drawdown within 20 days).

Its inputs are BTC trend, momentum and volatility, volume trend, breadth and cross-asset correlation, all causal.
It is a walk-forward logistic model against climatology and a one-variable rule.

| Target | 2021–24 gain vs best baseline (90% CI) | 2021–22 | 2023–24 | Verdict |
|---|---|---|---|---|
| direction | −0.153 (−0.243 to −0.067) | −0.121 | −0.185 | rejected: worse than the base rate |
| vol_high | +0.024 (−0.037 to +0.088) | +0.053 | −0.028 | inconclusive |
| vol_low | −0.043 (−0.108 to +0.019) | −0.099 | −0.001 | rejected |
| vol_expand | +0.019 (−0.036 to +0.076) | +0.052 | −0.034 | inconclusive |
| stress | −0.051 (−0.088 to −0.013) | −0.020 | −0.083 | rejected: **calibration inverted** (top quintile 0.45 predicted → 0.15 observed) |

**The pattern matches Rounds 1–2.**
- Volatility states are slightly predictable, but only in the first half, and not beyond simple baselines.
- Direction and stress are not predictable from these features.
- The engine's confidence is low (mean 0.07–0.41). That is the honest output: it does not know.

## 3. Portfolio engine (`research/portfolio_engine.py`) and strategy selection
**What was added.** A single account that can run several strategies, with:
- ranking of competing signals when slots are full;
- replacement of stale positions;
- regime-driven risk-off (smaller new positions and a tighter open-risk cap; it never raises risk);
- a correlation-weighted open-stop-risk cap.

It includes all the Round 3 controls, and reproduces the earlier simulators exactly when its features are off.

Results (no halt; return % / max DD % / return-to-DD; R0 = frozen T50 book):

| Book | DEV | VAL | HOLDOUT | Verdict |
|---|---|---|---|---|
| R0 frozen T50 | +44.5 / 17.4 / 2.56 | +17.9 / 10.5 / 1.71 | +32.0 / 11.4 / 2.81 | baseline |
| B-ORISK (open-risk cap) | +42.4 / 8.7 / 4.89 | +27.7 / 6.5 / 4.23 | +32.8 / 9.8 / 3.37 | Round 3 post-hoc lead; unchanged |
| E1 momentum-ranked slots | +45.8 / 17.1 / 2.68 | +18.4 / 10.5 / 1.76 | +28.0 / 14.7 / 1.90 | **rejected** (worse on HOLDOUT) |
| E2 E1 + stale replacement | +53.5 / 17.7 / 3.02 | +17.3 / 9.4 / 1.85 | +43.8 / 16.7 / 2.62 | **inconclusive**: more return on DEV and HOLDOUT, but drawdown is higher and VAL return lower |
| E3 rule risk-off (BTC ≤ SMA100) | +43.3 / 6.7 / 6.46 | +26.4 / 6.2 / 4.23 | +30.0 / 9.2 / 3.25 | **rejected**: lower drawdown, but costs return in VAL and HOLDOUT |
| E4 ML risk-off (P(stress)) | +35.9 / 7.2 / 5.00 | +27.7 / 6.5 / 4.23 | +24.6 / 9.8 / 2.52 | **rejected**: worse than the rule (E3m) |
| E5 correlation-weighted risk cap | +40.1 / 8.7 / 4.62 | +27.3 / 6.7 / 4.10 | +31.0 / 9.8 / 3.18 | **rejected** |
| RS3B RS3 alone | +29.9 / 5.6 / 5.30 | +14.8 / 5.9 / 2.51 | +36.9 / 9.8 / 3.77 | descriptive |
| PIE (T50 + RS3, all features) | +48.3 / 6.5 / 7.42 | +13.2 / 6.9 / 1.92 | +47.7 / 13.9 / 3.43 | **rejected**: RS3 positions take the slots T50's big trends need (R0's top-5 profit kept: 6–47%) |

**Uncertainty.** In the paired bootstrap, the CAGR intervals of every book include zero in at least one split. The
only reliable effect is lower drawdown from the open-risk cap: P(ΔDD > 0) = 0.02 in DEV and VAL.

## 4. What was learned
1. **The best account-level control is still the aggregate open-stop-risk cap.**
   - On the Round 3 numbers it improved return/DD in all three splits and under stress costs. It never tripped the
     kill switch, and it kept top winners.
   - It is still **post hoc**, and it failed the pre-registered VAL peak checks because of one TRX day.
2. **Regime information did not help.**
   - Neither the rule nor the model risk-off improved results once return is counted.
   - The model was worse than the rule, so ML adds nothing here.
3. **Strategy selection by momentum rank did not help.**
   - Ranking candidates did not beat fixed order.
   - Mixing in RS3 crowded out T50's rare large trends: higher activity, lower capture.
4. **Replacing stale positions is the only new lever with a plausible upside**, with large gains on HOLDOUT.
   - Its cost is higher drawdown and a weaker VAL.
   - If pursued, it would need to be combined with the open-risk cap and tested forward.
   - It is inconclusive, and not a result.
5. **Net profitability is modest everywhere.**
   - The best books compound at about 4–13% a year in single-account simulation, with drawdowns of 6–17%.
   - Only in bull years do T50 and RS3 earn their returns.
   - Nothing here changes that.

## 5. Recommendation
- **Do not change real risk controls or strategies.**
- **One forward experiment is worth freezing next, if you approve:** B-ORISK (frozen T50 + open-stop-risk cap
  10% → 8%), optionally with the stale-replacement rule as a second variant. Each would get its own fingerprint,
  labelled post hoc, and be tracked beside X1 and X2. This needs one change to the scheduled wrapper, which is not
  fingerprinted.
- **Stop pursuing:** regime-model risk-off, momentum slot ranking, entry ML, correlation caps at these levels.
