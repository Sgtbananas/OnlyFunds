# Round 2 historical checks (pre-registered in `ROUND2_PROGRAMME.md`)

Data 2019–2024 only, **already examined in Round 1**: these results can reject or flag fragility; they cannot establish an edge. Reproduce: `python research/round2.py` (runtime 10 min).

## Verdicts

| Hypothesis | Verdict | Detail |
|---|---|---|
| H2-PM-ROBUST | **supported** | {"share_of_neighbours_improving_all_4_cells": 0.75, "per_neighbour_cells_improved": {"band0.12/0.08+lock1.5R/5.0ATR": 4, "band0.12/0.08+lock2.0R/6.0ATR": 4, "band0.12/0.08+lock3.0R/8.0ATR": 4, "band0.12/0.08+lock-off": 4, "band0.15/0.10+lock1.5R/5.0ATR": 4, "band0.15/0.10+lock2.0R/6.0ATR": 4, "band0 |
| H2-PM-COST | **supported** | {"cells_improved": 4} |
| H2-PM-GENERIC | **descriptive** | {"as_configured": 0.7, "no_halt": 0.71} |
| H2-ML7 | **rejected** | {"checks": {"total_R_up": false, "beats_random_holm": false, "years_3_of_4": false, "same_sign_main_holdout": false, "portfolio_ret_dd_not_worse": false, "survives_cost_stress": false}, "holm_p": 1.0} |
| H2-ML8 | **rejected** | {"checks": {"total_R_up": false, "beats_random_holm": false, "years_3_of_4": false, "same_sign_main_holdout": false, "portfolio_ret_dd_not_worse": true, "survives_cost_stress": false}, "holm_p": 1.0} |

## Position management: return / max DD for 12 pre-declared neighbours (2019–24)

| ('config', '') | ('holdout16', 'as_configured') | ('holdout16', 'no_halt') | ('main9', 'as_configured') | ('main9', 'no_halt') |
|---|---|---|---|---|
| R0 | +2.898 | +2.809 | +3.587 | +3.930 |
| band0.12/0.08+lock-off | +3.570 | +3.570 | +4.470 | +7.308 |
| band0.12/0.08+lock1.5R/5.0ATR | +3.358 | +3.358 | +4.441 | +7.826 |
| band0.12/0.08+lock2.0R/6.0ATR | +3.591 | +3.591 | +4.941 | +8.043 |
| band0.12/0.08+lock3.0R/8.0ATR | +3.570 | +3.570 | +4.470 | +7.235 |
| band0.15/0.10+lock-off | +3.584 | +3.584 | +4.723 | +7.467 |
| band0.15/0.10+lock1.5R/5.0ATR | +3.321 | +3.321 | +3.944 | +6.074 |
| band0.15/0.10+lock2.0R/6.0ATR | +3.605 | +3.605 | +5.212 | +8.235 |
| band0.15/0.10+lock3.0R/8.0ATR | +3.584 | +3.584 | +4.723 | +7.375 |
| band0.20/0.15+lock-off | +2.898 | +2.809 | +5.187 | +5.452 |
| band0.20/0.15+lock1.5R/5.0ATR | +2.841 | +2.291 | +3.944 | +5.854 |
| band0.20/0.15+lock2.0R/6.0ATR | +3.183 | +2.820 | +5.264 | +6.179 |
| band0.20/0.15+lock3.0R/8.0ATR | +2.898 | +2.809 | +5.187 | +5.345 |

## Cost stress (0.4% fee, 0.2% slippage, 0.5% stop slippage)

| universe | mode | config | return_pct | max_dd_pct | ret_over_dd | sharpe |
|---|---|---|---|---|---|---|
| main9 | as_configured | R0 | +47.1 | +13.8 | +3.412 | +0.806 |
| main9 | as_configured | R3 | +48.3 | +10.3 | +4.691 | +0.906 |
| main9 | no_halt | R0 | +54.3 | +18.3 | +2.965 | +0.815 |
| main9 | no_halt | R3 | +61.8 | +11.4 | +5.408 | +0.990 |
| holdout16 | as_configured | R0 | +27.5 | +10.0 | +2.749 | +0.627 |
| holdout16 | as_configured | R3 | +29.0 | +10.2 | +2.841 | +0.648 |
| holdout16 | no_halt | R0 | +29.2 | +12.1 | +2.412 | +0.607 |
| holdout16 | no_halt | R3 | +32.0 | +10.2 | +3.123 | +0.679 |

## Random-entry books (main 9): share of draws where R3 raises return/DD

| mode | draws | share_R3_better | median_R0 | median_R3 |
|---|---|---|---|---|
| as_configured | +100.0 | +0.700 | +2.889 | +4.412 |
| no_halt | +100.0 | +0.710 | +4.268 | +5.604 |

## Sparse ML gates on frozen T50 (2021–24, 25 coins)

G0: 289 trades, total +251.2R.

| gate | blocked | trades | total_R | dTotal_R | p_random | p_holm | years_up | d_main9 | d_holdout16 | removed_mean_R | d_cost_stress |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ML7 | 519/2507 | 284 | +84.2 | -167.0 | +1.000 | +1.000 | 1 | -62.1 | -104.9 | +3.128 | -159.4 |
| ML8 | 1408/2507 | 244 | +165.4 | -85.8 | +0.977 | +1.000 | 1 | -55.1 | -30.7 | +1.072 | -67.4 |

**Model evaluation per fold:**

| model | test_year | train_rows | base_rate | oos_logloss | oos_logloss_clim | oos_auc |
|---|---|---|---|---|---|---|
| ML7 | 2021 | 12650 | +0.530 | +0.832 | +0.693 | +0.373 |
| ML7 | 2022 | 21775 | +0.537 | +0.715 | +0.716 | +0.505 |
| ML7 | 2023 | 30900 | +0.487 | +0.700 | +0.693 | +0.556 |
| ML7 | 2024 | 40025 | +0.492 | +0.714 | +0.693 | +0.374 |

| model | test_year | train_trades | train_rate | threshold_p80 | oos_auc |
|---|---|---|---|---|---|
| ML8 | 2021 | 157 | +0.076 | +0.099 | +0.628 |
| ML8 | 2022 | 245 | +0.073 | +0.099 | +0.625 |
| ML8 | 2023 | 303 | +0.096 | +0.137 | +0.543 |
| ML8 | 2024 | 360 | +0.092 | +0.117 | +0.509 |

## Run log

```
  robustness: 75% of 12 neighbours improve ret/DD in all 4 cells
  generic random-entry books: 100 draws
  ML7 trained
  ML8 trained
  gate ML7: blocked 519/2507 signal days, dTotalR -167.00, random p 1.000 (38s random)
  gate ML8: blocked 1408/2507 signal days, dTotalR -85.77, random p 0.977 (33s random)
```
