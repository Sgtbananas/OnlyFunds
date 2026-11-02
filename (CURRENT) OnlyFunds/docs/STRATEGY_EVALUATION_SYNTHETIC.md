# Strategy evaluation

Datasets: synthetic:trend_up, synthetic:trend_down, synthetic:range, synthetic:high_vol, synthetic:random_walk, synthetic:switching; interval 1h; profile config default; runtime 304s.

> Synthetic regimes test that the logic behaves as designed. They are **not** evidence about real markets.

## Fixed default parameters (full period, after fees and slippage)

| dataset | strategy | trades | trades_per_day | total_return_pct | buy_and_hold_return_pct | max_drawdown_pct | profit_factor | expectancy | win_rate_pct | sharpe | sortino | fees_paid |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| synthetic:trend_up | auto | 103 | 0.626 | -3.549 | 138.550 | 5.765 | 0.801 | -0.345 | 33.010 | -1.373 | -0.588 | 40.112 |
| synthetic:trend_up | trend_following | 105 | 0.638 | -2.865 | 138.550 | 6.079 | 0.839 | -0.273 | 34.286 | -1.080 | -0.473 | 40.932 |
| synthetic:trend_up | mean_reversion | 50 | 0.304 | -2.277 | 138.550 | 3.194 | 0.750 | -0.455 | 38.000 | -1.261 | -0.396 | 19.720 |
| synthetic:trend_up | stacked | 21 | 0.128 | 2.613 | 138.550 | 2.111 | 1.745 | 1.244 | 47.619 | 2.033 | 0.445 | 8.727 |
| synthetic:trend_down | auto | 68 | 0.413 | -4.423 | -91.044 | 4.821 | 0.651 | -0.650 | 32.353 | -2.124 | -0.704 | 26.512 |
| synthetic:trend_down | trend_following | 70 | 0.426 | -5.067 | -91.044 | 5.631 | 0.621 | -0.724 | 31.429 | -2.424 | -0.814 | 27.250 |
| synthetic:trend_down | mean_reversion | 90 | 0.547 | -9.986 | -91.044 | 10.041 | 0.483 | -1.110 | 30.000 | -4.366 | -1.798 | 33.495 |
| synthetic:trend_down | stacked | 7 | 0.043 | -1.134 | -91.044 | 1.393 | 0.359 | -1.620 | 14.286 | -1.658 | -0.199 | 2.771 |
| synthetic:range | auto | 82 | 0.499 | -9.295 | 12.839 | 10.077 | 0.404 | -1.134 | 24.390 | -4.481 | -1.591 | 31.450 |
| synthetic:range | trend_following | 80 | 0.486 | -9.374 | 12.839 | 10.155 | 0.390 | -1.172 | 22.500 | -4.489 | -1.566 | 30.641 |
| synthetic:range | mean_reversion | 35 | 0.213 | 1.990 | 12.839 | 1.991 | 1.340 | 0.569 | 45.714 | 1.189 | 0.346 | 14.330 |
| synthetic:range | stacked | 12 | 0.073 | -0.354 | 12.839 | 0.983 | 0.797 | -0.295 | 41.667 | -0.355 | -0.059 | 4.798 |
| synthetic:high_vol | auto | 100 | 0.608 | -6.361 | -90.026 | 8.236 | 0.744 | -0.636 | 29.000 | -1.810 | -0.688 | 20.859 |
| synthetic:high_vol | trend_following | 102 | 0.620 | -6.347 | -90.026 | 9.175 | 0.754 | -0.622 | 29.412 | -1.795 | -0.683 | 21.438 |
| synthetic:high_vol | mean_reversion | 51 | 0.310 | -4.553 | -90.026 | 5.394 | 0.596 | -0.893 | 35.294 | -2.008 | -0.745 | 7.489 |
| synthetic:high_vol | stacked | 11 | 0.067 | 0.605 | -90.026 | 2.037 | 1.249 | 0.550 | 36.364 | 0.426 | 0.077 | 2.213 |
| synthetic:random_walk | auto | 86 | 0.523 | -5.585 | -53.779 | 6.676 | 0.675 | -0.649 | 30.233 | -2.485 | -0.913 | 33.104 |
| synthetic:random_walk | trend_following | 90 | 0.547 | -5.304 | -53.779 | 6.542 | 0.708 | -0.589 | 30.000 | -2.297 | -0.865 | 34.893 |
| synthetic:random_walk | mean_reversion | 76 | 0.462 | -5.913 | -53.779 | 6.510 | 0.613 | -0.778 | 34.211 | -2.737 | -1.022 | 29.015 |
| synthetic:random_walk | stacked | 13 | 0.079 | 0.471 | -53.779 | 1.819 | 1.219 | 0.362 | 38.462 | 0.443 | 0.084 | 5.282 |
| synthetic:switching | auto | 93 | 0.566 | -5.761 | 71.342 | 7.180 | 0.686 | -0.620 | 34.409 | -2.090 | -0.850 | 34.363 |
| synthetic:switching | trend_following | 98 | 0.597 | -6.507 | 71.342 | 8.351 | 0.671 | -0.664 | 35.714 | -2.263 | -0.946 | 35.827 |
| synthetic:switching | mean_reversion | 37 | 0.225 | 2.681 | 71.342 | 3.364 | 1.463 | 0.724 | 45.946 | 1.491 | 0.458 | 14.179 |
| synthetic:switching | stacked | 9 | 0.055 | -0.496 | 71.342 | 1.387 | 0.714 | -0.551 | 22.222 | -0.690 | -0.077 | 3.599 |

## Walk-forward (tuned on each training window, scored on the next unseen window)

| dataset | strategy | oos_compounded_return_pct | oos_buy_hold_compounded_pct | profitable_folds | folds | oos_trades |
|---|---|---|---|---|---|---|
| synthetic:trend_up | auto | -6.605 | 365.398 | 1 | 4 | 108 |
| synthetic:trend_up | trend_following | -6.798 | 365.398 | 0 | 4 | 114 |
| synthetic:trend_up | mean_reversion | -0.387 | 365.398 | 1 | 4 | 24 |
| synthetic:trend_up | stacked | -0.803 | 365.398 | 2 | 4 | 62 |
| synthetic:trend_down | auto | -6.011 | -67.352 | 0 | 4 | 66 |
| synthetic:trend_down | trend_following | -8.112 | -67.352 | 0 | 4 | 75 |
| synthetic:trend_down | mean_reversion | -3.698 | -67.352 | 1 | 4 | 35 |
| synthetic:trend_down | stacked | -4.143 | -67.352 | 1 | 4 | 65 |
| synthetic:range | auto | -8.783 | 12.911 | 0 | 4 | 65 |
| synthetic:range | trend_following | -8.249 | 12.911 | 0 | 4 | 69 |
| synthetic:range | mean_reversion | 0.214 | 26.214 | 1 | 4 | 18 |
| synthetic:range | stacked | -11.407 | 12.911 | 0 | 4 | 84 |
| synthetic:high_vol | auto | -8.136 | 94.938 | 1 | 4 | 140 |
| synthetic:high_vol | trend_following | -8.537 | 94.938 | 0 | 4 | 113 |
| synthetic:high_vol | mean_reversion | -1.255 | 94.938 | 1 | 4 | 27 |
| synthetic:high_vol | stacked | -3.967 | 94.938 | 2 | 4 | 76 |
| synthetic:random_walk | auto | -11.360 | 23.265 | 0 | 4 | 116 |
| synthetic:random_walk | trend_following | -10.432 | 23.265 | 0 | 4 | 107 |
| synthetic:random_walk | mean_reversion | -0.987 | 23.265 | 1 | 4 | 24 |
| synthetic:random_walk | stacked | -1.814 | 23.265 | 2 | 4 | 58 |
| synthetic:switching | auto | -0.284 | 12.171 | 2 | 4 | 98 |
| synthetic:switching | trend_following | -2.143 | 12.171 | 2 | 4 | 121 |
| synthetic:switching | mean_reversion | 2.562 | 61.964 | 2 | 4 | 9 |
| synthetic:switching | stacked | -5.425 | 12.171 | 0 | 4 | 39 |

## Verdict

| dataset | strategy | credible_edge | reasons |
|---|---|---|---|
| synthetic:high_vol | auto | False | profit factor 0.74; walk-forward OOS not profitable; 1/4 profitable folds; walk-forward OOS does not beat buy-and-hold |
| synthetic:high_vol | mean_reversion | False | profit factor 0.60; walk-forward OOS not profitable; 1/4 profitable folds; walk-forward OOS does not beat buy-and-hold |
| synthetic:high_vol | stacked | False | only 11 trades; walk-forward OOS not profitable; 2/4 profitable folds; walk-forward OOS does not beat buy-and-hold |
| synthetic:high_vol | trend_following | False | profit factor 0.75; walk-forward OOS not profitable; 0/4 profitable folds; walk-forward OOS does not beat buy-and-hold |
| synthetic:random_walk | auto | False | profit factor 0.68; walk-forward OOS not profitable; 0/4 profitable folds; walk-forward OOS does not beat buy-and-hold |
| synthetic:random_walk | mean_reversion | False | profit factor 0.61; walk-forward OOS not profitable; 1/4 profitable folds; walk-forward OOS does not beat buy-and-hold |
| synthetic:random_walk | stacked | False | only 13 trades; walk-forward OOS not profitable; 2/4 profitable folds; walk-forward OOS does not beat buy-and-hold |
| synthetic:random_walk | trend_following | False | profit factor 0.71; walk-forward OOS not profitable; 0/4 profitable folds; walk-forward OOS does not beat buy-and-hold |
| synthetic:range | auto | False | profit factor 0.40; does not beat buy-and-hold; walk-forward OOS not profitable; 0/4 profitable folds; walk-forward OOS does not beat buy-and-hold |
| synthetic:range | mean_reversion | False | does not beat buy-and-hold; 1/4 profitable folds; walk-forward OOS does not beat buy-and-hold |
| synthetic:range | stacked | False | only 12 trades; profit factor 0.80; does not beat buy-and-hold; walk-forward OOS not profitable; 0/4 profitable folds; walk-forward OOS does not beat buy-and-hold |
| synthetic:range | trend_following | False | profit factor 0.39; does not beat buy-and-hold; walk-forward OOS not profitable; 0/4 profitable folds; walk-forward OOS does not beat buy-and-hold |
| synthetic:switching | auto | False | profit factor 0.69; does not beat buy-and-hold; walk-forward OOS not profitable; 2/4 profitable folds; walk-forward OOS does not beat buy-and-hold |
| synthetic:switching | mean_reversion | False | does not beat buy-and-hold; 2/4 profitable folds; walk-forward OOS does not beat buy-and-hold |
| synthetic:switching | stacked | False | only 9 trades; profit factor 0.71; does not beat buy-and-hold; walk-forward OOS not profitable; 0/4 profitable folds; walk-forward OOS does not beat buy-and-hold |
| synthetic:switching | trend_following | False | profit factor 0.67; does not beat buy-and-hold; walk-forward OOS not profitable; 2/4 profitable folds; walk-forward OOS does not beat buy-and-hold |
| synthetic:trend_down | auto | False | profit factor 0.65; walk-forward OOS not profitable; 0/4 profitable folds |
| synthetic:trend_down | mean_reversion | False | profit factor 0.48; walk-forward OOS not profitable; 1/4 profitable folds |
| synthetic:trend_down | stacked | False | only 7 trades; profit factor 0.36; walk-forward OOS not profitable; 1/4 profitable folds |
| synthetic:trend_down | trend_following | False | profit factor 0.62; walk-forward OOS not profitable; 0/4 profitable folds |
| synthetic:trend_up | auto | False | profit factor 0.80; does not beat buy-and-hold; walk-forward OOS not profitable; 1/4 profitable folds; walk-forward OOS does not beat buy-and-hold |
| synthetic:trend_up | mean_reversion | False | profit factor 0.75; does not beat buy-and-hold; walk-forward OOS not profitable; 1/4 profitable folds; walk-forward OOS does not beat buy-and-hold |
| synthetic:trend_up | stacked | False | only 21 trades; does not beat buy-and-hold; walk-forward OOS not profitable; 2/4 profitable folds; walk-forward OOS does not beat buy-and-hold |
| synthetic:trend_up | trend_following | False | profit factor 0.84; does not beat buy-and-hold; walk-forward OOS not profitable; 0/4 profitable folds; walk-forward OOS does not beat buy-and-hold |

## ML filter: rule-only vs ML-gated on the unseen last 30%

| dataset | useful | val_auc | exp_r_all | exp_r_accepted | rule_return_pct | ml_return_pct | rule_pf | ml_pf | rule_trades | ml_trades |
|---|---|---|---|---|---|---|---|---|---|---|
| synthetic:trend_up | False | 0.493 | -0.322 | -0.388 | 0.675 | -1.940 | 1.143 | 0.019 | 31 | 9 |
| synthetic:trend_down | False | 0.437 | -0.303 | -0.983 | -1.338 | -0.343 | 0.715 | 0.000 | 24 | 1 |
| synthetic:range | False | 0.625 | -0.815 | -0.777 | -3.372 | -2.947 | 0.247 | 0.000 | 23 | 11 |
| synthetic:high_vol | False | 0.358 | -0.181 | -0.370 | -0.309 | 0.119 | 0.957 | 1.058 | 32 | 9 |
| synthetic:random_walk | False | 0.379 | -0.330 | -0.900 | -0.532 | -0.812 | 0.897 | 0.000 | 27 | 4 |
| synthetic:switching | False | 0.506 | -0.286 | -0.606 | 0.052 | -0.049 | 1.009 | 0.980 | 32 | 13 |

## Overfitting check: tuned on first 70%, scored on last 30%

| dataset | best_params | in_sample_return_pct | oos_return_pct | oos_buy_hold_pct | confirmed |
|---|---|---|---|---|---|
| synthetic:trend_up | {'threshold': 0.25, 'exit_rules.stop_atr_mult': 1.0, 'exit_rules.tp_atr_mult': 2.0, 'strategy': 'mean_reversion'} | 2.055 | 0.000 | 72.981 | False |
| synthetic:trend_down | {'threshold': 0.25, 'exit_rules.stop_atr_mult': 1.0, 'exit_rules.tp_atr_mult': 2.0, 'strategy': 'mean_reversion'} | 0.165 | -0.282 | -36.162 | False |
| synthetic:range | {'threshold': 0.25, 'exit_rules.stop_atr_mult': 1.0, 'exit_rules.tp_atr_mult': 3.0, 'strategy': 'mean_reversion'} | 3.560 | 0.000 | 2.616 | False |
| synthetic:high_vol | {'threshold': 0.45, 'exit_rules.stop_atr_mult': 1.5, 'exit_rules.tp_atr_mult': 2.0, 'strategy': 'trend_following'} | -2.778 | 0.409 | 17.209 | False |
| synthetic:random_walk | {'threshold': 0.25, 'exit_rules.stop_atr_mult': 1.0, 'exit_rules.tp_atr_mult': 2.0, 'strategy': 'mean_reversion'} | 0.954 | 0.000 | 5.085 | False |
| synthetic:switching | {'threshold': 0.25, 'exit_rules.stop_atr_mult': 1.0, 'exit_rules.tp_atr_mult': 3.0, 'strategy': 'mean_reversion'} | 1.297 | 1.782 | 81.479 | False |
