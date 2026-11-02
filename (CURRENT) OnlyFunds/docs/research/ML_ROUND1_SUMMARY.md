# ML Round 1: verdict (2026-11-02)

Full tables: `04_ml_round1.md`. Raw numbers: `ml_round1_results.json`. Pre-registration: `ML_PROGRAMME.md`
(commit `c0854be`, made before any result). Data: 2019–2024, 25 coins, daily. No tuning was done.

## Answer
**No model improved net trading results.** Out of sample, every ML model was worse than simply predicting the
historical average. Every ML gate on T50 cut total net R by a large margin. Two simple non-ML gates helped slightly
but were inconclusive. Position management was the only consistently positive lever, and it narrowly failed its
pre-registered threshold.

## The models: what they predict and how well (walk-forward test years 2021–24)
| Model | Predicts | Out-of-sample result |
|---|---|---|
| M1 logistic, M2 boosting | Coin up more than costs over the next 20 days | **Worse than climatology** in every span (log-loss gain −0.15 / −0.26). AUC 0.50 / 0.47. Badly miscalibrated: the lowest-probability decile rose 66% of the time. Training AUC 0.69 / 0.90 is overfitting. |
| M1s, M2s | 20% drawdown within 20 days (stress) | Some ranking (AUC 0.54–0.59), but worse than climatology overall. In VAL 2023–24 alone, M1s matched climatology (+0.004, interval includes 0). |
| MV ridge | 20-day realised volatility | **The only real predictive skill:** 30% lower error than climatology. Against plain persistence: +11% overall, +22% in 2021–22, −1% in 2023–24. Not significant. |
| M3 logistic, M4 boosting | Whether a T50 trade wins | AUC 0.25–0.56 out of sample against 0.80–0.98 in training. **No skill, severe overfitting.** |

**Why:** T50's results come from a few huge trend trades (2021, 2023–24), and their conditions do not repeat in a
learnable way within six years of daily data. Models fitted to 2019–20 learned relationships that reversed in 2021.

## How predictions became decisions, and what happened to T50 (2021–24, 25 coins)
**Baseline:** frozen T50 made 289 trades, total **+251.2R**, mean +0.87R (95% interval +0.06 to +1.89), profit factor
2.83. **Without its top 10% of trades it is −0.12R per trade.**

| Gate | Signal days blocked | Change in total R | Trades it removed (mean) | Against random skipping of the same count | Verdict |
|---|---|---|---|---|---|
| GM1 (M1 P(up) < 0.5) | 50% | **−136R** | +1.39R | worse (p 1.00) | rejected |
| GM2 (M2) | 60% | −62R | +0.65R | no better (p 0.57) | rejected |
| GM3 (M3, skip worst third) | 58% | **−146R** | +1.05R | worse | rejected |
| GM4 (M4) | 43% | −90R | +1.63R | worse | rejected |
| G1 BTC > 100-day SMA | 8% | **+10.6R** | −0.16R | better (p 0.003; Holm 0.010) | **inconclusive**: helped in only 2 of 4 years |
| G2 coin > 200-day SMA | 19% | −13R | +0.24R | no better | rejected |
| G3 rule regime not bearish | 3% | +2.9R | −0.25R | better (p 0.007) | **inconclusive**: wrong sign on the holdout |

**The lesson:** any filter that removes many signals removes T50's rare large winners. The only filters that help
are narrow ones that skip entries in clearly bearish BTC conditions. Their effect is small: about +4% of total R.

## Capital preservation (single account, Conservative limits, 2019–24)
| | Max drawdown | Return | Return / DD | Largest position | Max open risk to stops |
|---|---|---|---|---|---|
| R0 frozen T50, main 9, no halt | 17.9% | +70.4% | 3.9 | 23.6% | 24.2% of equity |
| **R3** (band + profit lock), main 9, no halt | **10.7%** | **+88.3%** | **8.2** | 20.1% | 16.4% |
| R0, holdout 16, no halt | 11.4% | +32.0% | 2.8 | 16.0% | 22.8% |
| **R3**, holdout 16, no halt | 9.7% | +35.0% | 3.6 | 16.0% | 15.3% |

**Results:**
- R1 and R3 improved return, drawdown, Sharpe and return/DD in all 8 comparisons.
- The five largest winners kept 100–136% of their profit. Trimming freed capital for later trades.
- **Rejected anyway.** On the holdout, drawdown fell by 0.3–1.7 points, short of the pre-registered 2 points.
  The rule is kept as written.
- **R2** (profit lock alone) cut open risk but barely changed drawdown, and was rejected.

## The 38% BNB position, explained
- **Where it happened.** In the per-coin engine each coin has its own account. A 3.1% BNB position rose about 10×,
  to 38% of that account.
- **In one shared account.** The same mechanism produced:
  - a largest position of 23.6% of equity;
  - **open risk to the stop of up to 24.2% of equity**, against 0.5% initial risk.
- **Why:** T50's 3-ATR stop is fixed at entry and never moves, and `max_position_pct` (10%) is checked only at
  entry. A winner's open profit is therefore unprotected until the 25-day-low exit fires.
- **The consequence:** giving back that profit caused the 14% drawdown that permanently tripped the 10% kill switch
  on 2021-05-19. **Initial risk is not portfolio risk.**
- **Mitigation:** R1 and R3 bound concentration (15% band) and open risk (about 15%).

## What was decided
- **Nothing changes in frozen T50 or RS3.** Their shadow tracking is untouched; its fingerprints are verified by a test.
- **ML gating of T50 is not pursued further** in this form.
- **R3 goes forward, prospectively.** It and R3 plus G1 are frozen as separate experimental strategies (X1, X2;
  `ROUND2_PROGRAMME.md`) and tracked forward alongside T50. They are not promoted.
