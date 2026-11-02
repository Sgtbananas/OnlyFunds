# ML Round 1 results (pre-registered in `ML_PROGRAMME.md`, commit c0854be)

Research only. Data 2019-01-01 → 2024-12-31 (25 coins, daily); nothing ≥ 2025 loaded; no tuning; the frozen T50/RS3 shadow tracking was not touched. Reproduce: `python research/ml_round1.py` (random-skip draws: 300; runtime 5 min).

## Verdicts

| Hypothesis | Verdict |
|---|---|
| H-ML1 | **rejected** |
| H-ML2 | **rejected** |
| H-ML3 | **rejected** |
| H-ML4 | **rejected** |
| H-ML5 | **rejected** |
| H-ML6 | **rejected** |
| H-G1 | **inconclusive** |
| H-G2 | **rejected** |
| H-G3 | **inconclusive** |
| H-R1 | **rejected** |
| H-R2 | **rejected** |
| H-R3 | **rejected** |

## A. Market-state models: probabilistic skill out of sample (walk-forward 2021–24)

Gain = mean log-loss improvement per coin-day (positive = model better); 95% interval from a month-block bootstrap (all coins in a month resampled together).

| model | target | span | rows | base_rate | logloss_model | gain_vs_clim | gain_vs_clim_lo | gain_vs_rule | gain_vs_rule_lo | auc | auc_rule |
|---|---|---|---|---|---|---|---|---|---|---|---|
| M1 | y_up | 2021-24 | 35291 | +0.466 | +0.851 | -0.152 | -0.320 | -0.155 | -0.323 | +0.502 | +0.512 |
| M1 | y_up | DEV-OOS 2021-22 | 18250 | +0.442 | +0.971 | -0.266 | -0.574 | -0.276 | -0.583 | +0.492 | +0.510 |
| M1 | y_up | VAL-OOS 2023-24 | 17041 | +0.491 | +0.723 | -0.030 | -0.073 | -0.026 | -0.071 | +0.523 | +0.511 |
| M2 | y_up | 2021-24 | 35291 | +0.466 | +0.957 | -0.258 | -0.386 | -0.261 | -0.393 | +0.469 | +0.512 |
| M2 | y_up | DEV-OOS 2021-22 | 18250 | +0.442 | +1.147 | -0.443 | -0.661 | -0.452 | -0.671 | +0.486 | +0.510 |
| M2 | y_up | VAL-OOS 2023-24 | 17041 | +0.491 | +0.754 | -0.061 | -0.107 | -0.056 | -0.110 | +0.495 | +0.511 |
| M1s | y_stress | 2021-24 | 35291 | +0.304 | +0.683 | -0.057 | -0.166 | -0.056 | -0.161 | +0.594 | +0.462 |
| M1s | y_stress | DEV-OOS 2021-22 | 18250 | +0.394 | +0.813 | -0.114 | -0.317 | -0.116 | -0.315 | +0.577 | +0.473 |
| M1s | y_stress | VAL-OOS 2023-24 | 17041 | +0.208 | +0.544 | +0.004 | -0.041 | +0.009 | -0.039 | +0.567 | +0.434 |
| M2s | y_stress | 2021-24 | 35291 | +0.304 | +0.728 | -0.103 | -0.184 | -0.101 | -0.185 | +0.538 | +0.462 |
| M2s | y_stress | DEV-OOS 2021-22 | 18250 | +0.394 | +0.854 | -0.155 | -0.288 | -0.157 | -0.290 | +0.586 | +0.473 |
| M2s | y_stress | VAL-OOS 2023-24 | 17041 | +0.208 | +0.594 | -0.046 | -0.131 | -0.041 | -0.127 | +0.527 | +0.434 |

**Reliability (M1, OOS 2021–24, deciles of predicted probability):**

| bin | predicted | observed | rows |
|---|---|---|---|
| 0 | +0.070 | +0.658 | 3530 |
| 1 | +0.259 | +0.423 | 3529 |
| 2 | +0.345 | +0.383 | 3529 |
| 3 | +0.400 | +0.385 | 3529 |
| 4 | +0.444 | +0.413 | 3529 |
| 5 | +0.483 | +0.455 | 3529 |
| 6 | +0.521 | +0.454 | 3529 |
| 7 | +0.567 | +0.432 | 3529 |
| 8 | +0.634 | +0.478 | 3529 |
| 9 | +0.788 | +0.574 | 3529 |

**Volatility forecast (MV, log 20-day realised volatility):**

| span | rows | skill_vs_persistence | skill_vs_climatology | mse_gain_lo | mse_gain_hi |
|---|---|---|---|---|---|
| 2021-24 | +35291.0 | +0.109 | +0.299 | -0.023 | +0.076 |
| DEV-OOS 2021-22 | +18250.0 | +0.216 | +0.245 | -0.023 | +0.129 |
| VAL-OOS 2023-24 | +17041.0 | -0.014 | +0.342 | -0.062 | +0.054 |

## B. T50 trade-scoring models (M3 logistic, M4 boosting)

| model | test_year | train_trades | test_trades | threshold_q33 | auc_oos | auc_train |
|---|---|---|---|---|---|---|
| M3 | 2021 | 109 | 80 | +0.297 | +0.440 | +0.913 |
| M4 | 2021 | 109 | 80 | +0.348 | +0.327 | +0.979 |
| M3 | 2022 | 197 | 58 | +0.342 | +0.561 | +0.842 |
| M4 | 2022 | 197 | 58 | +0.382 | +0.561 | +0.956 |
| M3 | 2023 | 255 | 74 | +0.236 | +0.289 | +0.833 |
| M4 | 2023 | 255 | 74 | +0.252 | +0.472 | +0.948 |
| M3 | 2024 | 312 | 68 | +0.270 | +0.412 | +0.796 |
| M4 | 2024 | 312 | 68 | +0.274 | +0.248 | +0.926 |

## C/D. Gates on the frozen T50 entries (2021–24, 25 coins, unchanged engine)

G0 frozen T50: 289 trades, total +251.2R, mean +0.869R (95% +0.059 to +1.894), PF 2.83, without top 10% -0.117R.

| gate | blocked_signal_days | trades | total_R | mean_R | pf | dTotal_R | random_skip_mean_d | random_p95_d | p_random | p_holm | years_up | d_main9 | d_holdout16 | d_cost_stress |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| GM1 | 1249/2507 | 223 | +115.1 | +0.516 | +2.228 | -136.1 | -42.9 | -21.6 | +1.000 | +1.000 | 1 | -47.6 | -88.5 | -129.2 |
| GM2 | 1515/2507 | 223 | +188.9 | +0.847 | +2.707 | -62.3 | -61.0 | -35.0 | +0.568 | +1.000 | 0 | -16.1 | -46.1 | -54.5 |
| GM3 | 1453/2507 | 239 | +104.7 | +0.438 | +1.833 | -146.5 | -55.9 | -30.0 | +1.000 | +1.000 | 1 | -33.8 | -112.7 | -137.6 |
| GM4 | 1066/2507 | 265 | +161.5 | +0.610 | +2.153 | -89.7 | -31.4 | -14.3 | +1.000 | +1.000 | 0 | -22.2 | -67.4 | -87.1 |
| G1 | 202/2507 | 260 | +261.8 | +1.007 | +3.216 | +10.6 | -3.930 | +1.528 | +0.003 | +0.010 | 2 | +6.067 | +4.489 | +13.8 |
| G2 | 482/2507 | 234 | +238.1 | +1.018 | +3.233 | -13.1 | -10.2 | -0.024 | +0.691 | +0.691 | 2 | -2.509 | -10.6 | -8.461 |
| G3 | 71/2507 | 280 | +254.1 | +0.907 | +2.911 | +2.876 | -1.241 | +1.468 | +0.007 | +0.013 | 3 | +3.214 | -0.338 | +3.084 |

**Trades each gate removed** (frozen-T50 trades whose decision day the gate rejects; a gate that helps should remove trades with negative mean R):

| gate | removed | removed_total_R | removed_mean_R | ci_lo | ci_hi | removed_top5_R |
|---|---|---|---|---|---|---|
| GM1 | 110 | +153.2 | +1.393 | -0.148 | +3.559 | +108.9 |
| GM2 | 117 | +76.0 | +0.650 | -0.032 | +1.312 | +37.4 |
| GM3 | 129 | +134.9 | +1.046 | +0.178 | +1.847 | +65.2 |
| GM4 | 86 | +140.2 | +1.630 | +0.508 | +2.874 | +74.6 |
| G1 | 45 | -7.327 | -0.163 | -0.560 | +0.537 | +13.2 |
| G2 | 128 | +30.1 | +0.235 | -0.379 | +0.878 | +34.0 |
| G3 | 17 | -4.206 | -0.247 | -0.590 | +0.218 | +4.017 |

**Single account, Conservative limits, 25 coins, 2021–24:**

| gate | mode | return_pct | max_dd_pct | ret_over_dd | sharpe | trades | cash_days_pct | halted_on |
|---|---|---|---|---|---|---|---|---|
| G0 | as_configured | +18.0 | +14.2 | +1.268 | +0.538 | 8 | +90.3 | 2021-05-19 |
| G0 | no_halt | +21.0 | +19.2 | +1.092 | +0.533 | 78 | +20.0 | - |
| GM1 | as_configured | +11.7 | +7.781 | +1.499 | +0.507 | 51 | +35.2 | - |
| GM1 | no_halt | +11.7 | +7.781 | +1.499 | +0.507 | 51 | +35.2 | - |
| GM2 | as_configured | +15.1 | +10.5 | +1.448 | +0.439 | 63 | +31.0 | 2024-12-19 |
| GM2 | no_halt | +15.1 | +10.5 | +1.448 | +0.439 | 63 | +31.0 | - |
| GM3 | as_configured | +6.198 | +8.585 | +0.722 | +0.283 | 77 | +21.2 | - |
| GM3 | no_halt | +6.198 | +8.585 | +0.722 | +0.283 | 77 | +21.2 | - |
| GM4 | as_configured | +18.0 | +14.2 | +1.268 | +0.538 | 8 | +90.3 | 2021-05-19 |
| GM4 | no_halt | +20.9 | +19.2 | +1.089 | +0.534 | 77 | +20.0 | - |
| G1 | as_configured | +18.0 | +14.2 | +1.268 | +0.538 | 8 | +90.3 | 2021-05-19 |
| G1 | no_halt | +21.0 | +18.3 | +1.150 | +0.541 | 66 | +37.4 | - |
| G2 | as_configured | +18.0 | +14.2 | +1.268 | +0.538 | 8 | +90.3 | 2021-05-19 |
| G2 | no_halt | +22.7 | +19.0 | +1.192 | +0.572 | 74 | +24.4 | - |
| G3 | as_configured | +18.0 | +14.2 | +1.268 | +0.538 | 8 | +90.3 | 2021-05-19 |
| G3 | no_halt | +21.9 | +18.3 | +1.197 | +0.555 | 70 | +30.6 | - |

**Per-year change in total R:**

| index | 2021 | 2022 | 2023 | 2024 |
|---|---|---|---|---|
| GM1 | -140.3 | -1.829 | -3.425 | +9.478 |
| GM2 | -17.8 | -0.000 | -37.9 | -6.526 |
| GM3 | -76.4 | +9.810 | -54.8 | -25.1 |
| GM4 | -49.7 | -0.700 | -5.390 | -33.9 |
| G1 | -0.286 | +15.0 | -4.305 | +0.189 |
| G2 | -20.1 | +31.8 | -31.9 | +7.101 |
| G3 | +0.000 | +0.578 | +2.109 | +0.189 |

**RS3 with the GM1 market-state gate (descriptive selection check, main 9, 2021–24):**

| index | trades | total_R | net_R | pf | win_pct | ci_lo | ci_hi | net_R_ex_top10pct |
|---|---|---|---|---|---|---|---|---|
| RS3 | +130.0 | +50.5 | +0.388 | +2.325 | +44.6 | +0.089 | +0.721 | -0.103 |
| RS3+GM1 | +83.0 | +22.8 | +0.275 | +1.941 | +45.8 | -0.042 | +0.633 | -0.144 |

## P5. Capital preservation variants (single account, 2019–24)

| universe | mode | variant | return_pct | cagr_pct | max_dd_pct | sharpe | ret_over_dd | positions | trims | profit_lock_exits | max_position_pct | max_open_risk_pct | top5_R0_profit_kept | halted_on |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| main9 | as_configured | R0 | +50.4 | +7.036 | +14.0 | +0.827 | +3.587 | 31 | 0 | 0 | +23.6 | +24.2 | +1.000 | 2021-05-19 |
| main9 | as_configured | R1 | +50.8 | +7.094 | +10.8 | +0.909 | +4.723 | 49 | 2 | 0 | +15.7 | +22.5 | +1.146 | 2022-11-09 |
| main9 | as_configured | R2 | +53.4 | +7.392 | +12.4 | +0.887 | +4.304 | 31 | 0 | 3 | +23.6 | +15.2 | +1.050 | 2021-05-19 |
| main9 | as_configured | R3 | +52.3 | +7.260 | +10.0 | +0.937 | +5.212 | 50 | 2 | 4 | +15.7 | +15.2 | +1.167 | 2022-12-17 |
| main9 | no_halt | R0 | +70.4 | +9.287 | +17.9 | +0.857 | +3.930 | 83 | 0 | 0 | +23.6 | +24.2 | +1.000 | - |
| main9 | no_halt | R1 | +85.7 | +10.9 | +11.5 | +1.081 | +7.467 | 83 | 3 | 0 | +20.1 | +22.5 | +1.319 | - |
| main9 | no_halt | R2 | +75.5 | +9.833 | +16.5 | +0.924 | +4.584 | 83 | 0 | 7 | +23.6 | +16.4 | +1.094 | - |
| main9 | no_halt | R3 | +88.3 | +11.1 | +10.7 | +1.114 | +8.235 | 83 | 3 | 7 | +20.1 | +16.4 | +1.360 | - |
| holdout16 | as_configured | R0 | +29.0 | +4.341 | +10.0 | +0.644 | +2.898 | 68 | 0 | 0 | +16.0 | +22.8 | +1.000 | 2023-08-16 |
| holdout16 | as_configured | R1 | +34.9 | +5.123 | +9.751 | +0.716 | +3.584 | 91 | 1 | 0 | +16.0 | +22.8 | +1.107 | - |
| holdout16 | as_configured | R2 | +32.6 | +4.816 | +10.2 | +0.703 | +3.183 | 69 | 0 | 2 | +16.0 | +15.3 | +1.000 | 2023-10-02 |
| holdout16 | as_configured | R3 | +35.0 | +5.128 | +9.706 | +0.717 | +3.605 | 91 | 1 | 2 | +16.0 | +15.3 | +1.107 | - |
| holdout16 | no_halt | R0 | +32.0 | +4.743 | +11.4 | +0.642 | +2.809 | 91 | 0 | 0 | +16.0 | +22.8 | +1.000 | - |
| holdout16 | no_halt | R1 | +34.9 | +5.123 | +9.751 | +0.716 | +3.584 | 91 | 1 | 0 | +16.0 | +22.8 | +1.103 | - |
| holdout16 | no_halt | R2 | +32.1 | +4.748 | +11.4 | +0.643 | +2.820 | 91 | 0 | 2 | +16.0 | +15.3 | +1.000 | - |
| holdout16 | no_halt | R3 | +35.0 | +5.128 | +9.706 | +0.717 | +3.605 | 91 | 1 | 2 | +16.0 | +15.3 | +1.103 | - |

**Risk decomposition, R0 main 9 as configured (percent of equity):**

| index | mean | 50% | max |
|---|---|---|---|
| drawdown_pct | +9.411 | +14.0 | +14.0 |
| initial_risk_pct | +0.333 | +0.000 | +1.516 |
| position_value_pct | +3.535 | +0.000 | +29.5 |
| unrealised_pct | +1.550 | +0.000 | +22.9 |
| open_risk_to_stop_pct | +1.876 | +0.000 | +24.2 |
| max_position_pct | +2.004 | +0.000 | +23.6 |

## Automated anomaly checks

- M1: mean train AUC 0.69 vs OOS 0.50 -> overfitting
- M2: mean train AUC 0.90 vs OOS 0.47 -> overfitting
- M1s: mean train AUC 0.74 vs OOS 0.59 -> overfitting
- M2s: mean train AUC 0.93 vs OOS 0.54 -> overfitting
- M3 2021: train AUC 0.91 vs OOS 0.44 -> overfitting
- M4 2021: train AUC 0.98 vs OOS 0.33 -> overfitting
- M3 2022: train AUC 0.84 vs OOS 0.56 -> overfitting
- M4 2022: train AUC 0.96 vs OOS 0.56 -> overfitting
- M3 2023: train AUC 0.83 vs OOS 0.29 -> overfitting
- M4 2023: train AUC 0.95 vs OOS 0.47 -> overfitting
- M3 2024: train AUC 0.80 vs OOS 0.41 -> overfitting
- M4 2024: train AUC 0.93 vs OOS 0.25 -> overfitting

## Run log

```
data: 25 coins, last candle 2024-12-31
panel: 53618 coin-days
  coin-day models fold 2021: train rows 17152, test rows 9125
  coin-day models fold 2022: train rows 26277, test rows 9125
  coin-day models fold 2023: train rows 35402, test rows 9125
  coin-day models fold 2024: train rows 44527, test rows 8441
  trade models fold 2021: train trades 157, signals scored 860
  trade models fold 2022: train trades 245, signals scored 243
  trade models fold 2023: train trades 303, signals scored 779
  trade models fold 2024: train trades 360, signals scored 625
G0 frozen T50 2021-24: 289 trades, total +251.2R, mean +0.869R
  gate GM1: blocked 1249/2507 signal days, dTotalR -136.10, random p 1.000 (36s random)
  gate GM2: blocked 1515/2507 signal days, dTotalR -62.26, random p 0.568 (35s random)
  gate GM3: blocked 1453/2507 signal days, dTotalR -146.49, random p 1.000 (33s random)
  gate GM4: blocked 1066/2507 signal days, dTotalR -89.67, random p 1.000 (34s random)
  gate G1: blocked 202/2507 signal days, dTotalR +10.56, random p 0.003 (34s random)
  gate G2: blocked 482/2507 signal days, dTotalR -13.10, random p 0.691 (33s random)
  gate G3: blocked 71/2507 signal days, dTotalR +2.88, random p 0.007 (33s random)
  risk variants main9 done
  risk variants holdout16 done
```
