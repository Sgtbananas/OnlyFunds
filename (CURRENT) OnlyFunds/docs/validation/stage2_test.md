## Stage 2: final test, run once (2026-11-01T04:18:46+00:00)

Frozen decisions from 2026-11-01T03:58:31+00:00; protocol fingerprint verified.

| costs | candidate | period | trades | expectancy_r | ci95_low_r | profit_factor | win_rate_pct | avg_win_r | avg_loss_r | breadth | exp_r_wo_best | mean_net_ret_pct | mean_buy_hold_pct | worst_dd_pct | criteria |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CoinEx-level costs | auto/fixed | TEST | 144 | -0.295 | -0.434 | 0.466 | 23.611 | 1.028 | -0.704 | 0.000 | -0.302 | -1.680 | -27.194 | 4.119 | expectancy -0.295R ≤ 0; 95% CI lower bound -0.434R ≤ 0; profit factor 0.47 < 1.10; positive on 0% of 5 coins < 60%; without best coin -0.302R ≤ 0; 0/4 walk-forward years positive |
| CoinEx-level costs | auto/tuned | TEST | 18 | -0.134 | -0.650 | 0.862 | 27.778 | 1.599 | -0.801 | 0.000 | -0.687 | -0.061 | -27.194 | 1.439 | 18 trades < 100; expectancy -0.134R ≤ 0; 95% CI lower bound -0.650R ≤ 0; profit factor 0.86 < 1.10; positive on 0% of 0 coins < 60%; without best coin -0.687R ≤ 0; 0/4 walk-forward years positive |
| CoinEx-level costs | trend_following/fixed | TEST | 266 | -0.200 | -0.311 | 0.641 | 28.571 | 1.117 | -0.726 | 0.000 | -0.206 | -2.043 | -27.194 | 8.056 | expectancy -0.200R ≤ 0; 95% CI lower bound -0.311R ≤ 0; profit factor 0.64 < 1.10; positive on 0% of 6 coins < 60%; without best coin -0.206R ≤ 0; 0/4 walk-forward years positive |
| CoinEx-level costs | trend_following/tuned | TEST | 266 | -0.200 | -0.311 | 0.641 | 28.571 | 1.117 | -0.726 | 0.000 | -0.206 | -2.043 | -27.194 | 8.056 | expectancy -0.200R ≤ 0; 95% CI lower bound -0.311R ≤ 0; profit factor 0.64 < 1.10; positive on 0% of 6 coins < 60%; without best coin -0.206R ≤ 0; 0/4 walk-forward years positive |
| CoinEx-level costs | mean_reversion/fixed | TEST | 343 | -0.135 | -0.233 | 0.771 | 34.694 | 1.081 | -0.781 | 0.143 | -0.150 | -1.726 | -27.194 | 7.793 | expectancy -0.135R ≤ 0; 95% CI lower bound -0.233R ≤ 0; profit factor 0.77 < 1.10; positive on 14% of 7 coins < 60%; without best coin -0.150R ≤ 0; 1/4 walk-forward years positive |
| CoinEx-level costs | mean_reversion/tuned | TEST | 109 | -0.034 | -0.245 | 0.964 | 38.532 | 1.372 | -0.916 | 0.333 | -0.047 | -0.093 | -27.194 | 2.346 | expectancy -0.034R ≤ 0; 95% CI lower bound -0.245R ≤ 0; profit factor 0.96 < 1.10; positive on 33% of 6 coins < 60%; without best coin -0.047R ≤ 0; 1/4 walk-forward years positive |
| CoinEx-level costs | mean_reversion/fixed+ML | TEST | 0 | nan | nan | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | nan | 0.000 | -27.194 | 0.000 | 0 trades < 100; expectancy +nanR ≤ 0; 95% CI lower bound +nanR ≤ 0; profit factor 0.00 < 1.10; positive on 0% of 0 coins < 60%; without best coin +nanR ≤ 0 |
| CoinEx-level costs | stacked/fixed | TEST | 43 | -0.122 | -0.385 | 0.785 | 23.256 | 1.439 | -0.595 | 0.000 | -0.196 | -0.176 | -27.194 | 2.401 | 43 trades < 100; expectancy -0.122R ≤ 0; 95% CI lower bound -0.385R ≤ 0; profit factor 0.78 < 1.10; positive on 0% of 1 coins < 60%; without best coin -0.196R ≤ 0; 0/4 walk-forward years positive |
| CoinEx-level costs | stacked/tuned | TEST | 618 | -0.187 | -0.245 | 0.580 | 27.023 | 0.899 | -0.589 | 0.000 | -0.187 | -4.749 | -27.194 | 10.230 | expectancy -0.187R ≤ 0; 95% CI lower bound -0.245R ≤ 0; profit factor 0.58 < 1.10; positive on 0% of 7 coins < 60%; without best coin -0.187R ≤ 0; 0/4 walk-forward years positive |
| info only: 0.1% fee | auto/fixed | TEST | 566 | -0.163 | -0.240 | 0.663 | 29.859 | 1.132 | -0.714 | 0.000 | -0.163 | -2.857 | -27.049 | 7.078 | expectancy -0.163R ≤ 0; 95% CI lower bound -0.240R ≤ 0; profit factor 0.66 < 1.10; positive on 0% of 7 coins < 60%; without best coin -0.163R ≤ 0; 0/4 walk-forward years positive |
| info only: 0.1% fee | auto/tuned | TEST | 144 | -0.374 | -0.548 | 0.491 | 20.833 | 1.533 | -0.876 | 0.000 | -0.385 | -1.426 | -27.049 | 4.283 | expectancy -0.374R ≤ 0; 95% CI lower bound -0.548R ≤ 0; profit factor 0.49 < 1.10; positive on 0% of 5 coins < 60%; without best coin -0.385R ≤ 0; 0/4 walk-forward years positive |
| info only: 0.1% fee | trend_following/fixed | TEST | 822 | -0.113 | -0.181 | 0.773 | 31.509 | 1.218 | -0.725 | 0.000 | -0.112 | -2.866 | -27.049 | 10.165 | expectancy -0.113R ≤ 0; 95% CI lower bound -0.181R ≤ 0; profit factor 0.77 < 1.10; positive on 0% of 8 coins < 60%; without best coin -0.112R ≤ 0; 0/4 walk-forward years positive |
| info only: 0.1% fee | trend_following/tuned | TEST | 822 | -0.113 | -0.181 | 0.773 | 31.509 | 1.218 | -0.725 | 0.000 | -0.112 | -2.866 | -27.049 | 10.165 | expectancy -0.113R ≤ 0; 95% CI lower bound -0.181R ≤ 0; profit factor 0.77 < 1.10; positive on 0% of 8 coins < 60%; without best coin -0.112R ≤ 0; 0/4 walk-forward years positive |
| info only: 0.1% fee | mean_reversion/fixed | TEST | 804 | -0.126 | -0.191 | 0.781 | 34.204 | 1.065 | -0.745 | 0.111 | -0.152 | -2.826 | -27.049 | 10.083 | expectancy -0.126R ≤ 0; 95% CI lower bound -0.191R ≤ 0; profit factor 0.78 < 1.10; positive on 11% of 9 coins < 60%; without best coin -0.152R ≤ 0; 1/4 walk-forward years positive |
| info only: 0.1% fee | mean_reversion/tuned | TEST | 365 | -0.210 | -0.325 | 0.724 | 29.863 | 1.457 | -0.920 | 0.143 | -0.231 | -1.973 | -27.049 | 6.377 | expectancy -0.210R ≤ 0; 95% CI lower bound -0.325R ≤ 0; profit factor 0.72 < 1.10; positive on 14% of 7 coins < 60%; without best coin -0.231R ≤ 0; 1/4 walk-forward years positive |
| info only: 0.1% fee | mean_reversion/fixed+ML | TEST | 0 | nan | nan | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | nan | 0.000 | -27.049 | 0.000 | 0 trades < 100; expectancy +nanR ≤ 0; 95% CI lower bound +nanR ≤ 0; profit factor 0.00 < 1.10; positive on 0% of 0 coins < 60%; without best coin +nanR ≤ 0 |
| info only: 0.1% fee | stacked/fixed | TEST | 123 | -0.115 | -0.284 | 0.804 | 27.642 | 1.359 | -0.678 | 0.500 | -0.150 | -0.373 | -27.049 | 2.927 | expectancy -0.115R ≤ 0; 95% CI lower bound -0.284R ≤ 0; profit factor 0.80 < 1.10; positive on 50% of 6 coins < 60%; without best coin -0.150R ≤ 0; 0/4 walk-forward years positive |
| info only: 0.1% fee | stacked/tuned | TEST | 1251 | -0.165 | -0.208 | 0.639 | 28.937 | 0.928 | -0.610 | 0.000 | -0.163 | -6.499 | -27.049 | 10.251 | expectancy -0.165R ≤ 0; 95% CI lower bound -0.208R ≤ 0; profit factor 0.64 < 1.10; positive on 0% of 9 coins < 60%; without best coin -0.163R ≤ 0; 0/4 walk-forward years positive |

### TEST detail (CoinEx-level costs)

#### auto/fixed, TEST

Per coin:

| pair | trades | net_return_pct | buy_hold_pct | max_dd_pct | profit_factor | expectancy_r | win_rate_pct | avg_win_r | avg_loss_r | fees |
|---|---|---|---|---|---|---|---|---|---|---|
| BTCUSDT | 0 | 0.000 | -17.505 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| ETHUSDT | 6 | -0.477 | -32.757 | 0.988 | 0.611 | -0.269 | 33.333 | 0.900 | -0.853 | 2.388 |
| BNBUSDT | 6 | -0.258 | -0.266 | 0.818 | 0.701 | -0.133 | 16.667 | 1.574 | -0.474 | 2.399 |
| XRPUSDT | 18 | -3.396 | -40.216 | 3.490 | 0.122 | -0.526 | 5.556 | 1.497 | -0.645 | 7.011 |
| LTCUSDT | 29 | -3.842 | -46.261 | 4.119 | 0.378 | -0.369 | 27.586 | 0.820 | -0.822 | 11.401 |
| ADAUSDT | 28 | -2.543 | -77.274 | 3.741 | 0.502 | -0.224 | 21.429 | 1.275 | -0.633 | 10.592 |
| EOSUSDT | 16 | -1.971 | 0.156 | 2.584 | 0.372 | -0.324 | 18.750 | 1.067 | -0.645 | 6.269 |
| XLMUSDT | 41 | -2.636 | -49.307 | 3.378 | 0.668 | -0.206 | 31.707 | 0.975 | -0.754 | 15.670 |
| TRXUSDT | 0 | 0.000 | 18.683 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

By regime at entry:

| regime | trades | expectancy_r | total_r |
|---|---|---|---|
| high_volatility | 21 | -0.160 | -3.367 |
| ranging | 5 | -0.227 | -1.137 |
| trending_up | 118 | -0.322 | -37.980 |

By year of entry:

| entry_time | trades | expectancy_r | total_r |
|---|---|---|---|
| 2025 | 131 | -0.303 | -39.638 |
| 2026 | 13 | -0.219 | -2.847 |

#### auto/tuned, TEST

Per coin:

| pair | trades | net_return_pct | buy_hold_pct | max_dd_pct | profit_factor | expectancy_r | win_rate_pct | avg_win_r | avg_loss_r | fees |
|---|---|---|---|---|---|---|---|---|---|---|
| BTCUSDT | 0 | 0.000 | -17.505 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| ETHUSDT | 0 | 0.000 | -32.757 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| BNBUSDT | 0 | 0.000 | -0.266 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| XRPUSDT | 2 | -0.703 | -40.216 | 0.869 | 0.000 | -1.000 | 0.000 | 0.000 | -1.000 | 0.786 |
| LTCUSDT | 0 | 0.000 | -46.261 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| ADAUSDT | 3 | -0.736 | -77.274 | 1.402 | 0.070 | -0.491 | 33.333 | 0.111 | -0.793 | 1.051 |
| EOSUSDT | 4 | -0.929 | 0.156 | 1.145 | 0.000 | -0.677 | 0.000 | 0.000 | -0.677 | 1.578 |
| XLMUSDT | 9 | 1.819 | -49.307 | 1.439 | 2.177 | 0.419 | 44.444 | 1.971 | -0.823 | 3.671 |
| TRXUSDT | 0 | 0.000 | 18.683 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

By regime at entry:

| regime | trades | expectancy_r | total_r |
|---|---|---|---|
| high_volatility | 5 | -0.096 | -0.479 |
| ranging | 1 | -1.000 | -1.000 |
| trending_up | 12 | -0.078 | -0.935 |

By year of entry:

| entry_time | trades | expectancy_r | total_r |
|---|---|---|---|
| 2025 | 16 | -0.246 | -3.935 |
| 2026 | 2 | 0.761 | 1.522 |

#### trend_following/fixed, TEST

Per coin:

| pair | trades | net_return_pct | buy_hold_pct | max_dd_pct | profit_factor | expectancy_r | win_rate_pct | avg_win_r | avg_loss_r | fees |
|---|---|---|---|---|---|---|---|---|---|---|
| BTCUSDT | 1 | -0.350 | -17.505 | 0.350 | 0.000 | -1.000 | 0.000 | 0.000 | -1.000 | 0.394 |
| ETHUSDT | 10 | -0.461 | -32.757 | 0.900 | 0.731 | -0.125 | 30.000 | 1.140 | -0.667 | 3.993 |
| BNBUSDT | 5 | 0.175 | -0.266 | 0.812 | 1.405 | 0.098 | 40.000 | 0.831 | -0.390 | 2.011 |
| XRPUSDT | 43 | -3.767 | -40.216 | 5.214 | 0.552 | -0.246 | 20.930 | 1.284 | -0.651 | 16.387 |
| LTCUSDT | 52 | -7.655 | -46.261 | 8.056 | 0.346 | -0.411 | 25.000 | 0.886 | -0.843 | 20.028 |
| ADAUSDT | 49 | -4.900 | -77.274 | 6.881 | 0.481 | -0.277 | 26.531 | 1.000 | -0.738 | 18.594 |
| EOSUSDT | 33 | -0.986 | 0.156 | 2.668 | 0.821 | -0.087 | 27.273 | 1.428 | -0.655 | 13.350 |
| XLMUSDT | 73 | -0.446 | -49.307 | 4.844 | 0.966 | -0.041 | 36.986 | 1.143 | -0.736 | 27.908 |
| TRXUSDT | 0 | 0.000 | 18.683 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

By regime at entry:

| regime | trades | expectancy_r | total_r |
|---|---|---|---|
| high_volatility | 157 | -0.097 | -15.169 |
| ranging | 11 | -0.544 | -5.979 |
| trending_up | 98 | -0.327 | -31.998 |

By year of entry:

| entry_time | trades | expectancy_r | total_r |
|---|---|---|---|
| 2025 | 198 | -0.234 | -46.311 |
| 2026 | 68 | -0.101 | -6.836 |

#### trend_following/tuned, TEST

Per coin:

| pair | trades | net_return_pct | buy_hold_pct | max_dd_pct | profit_factor | expectancy_r | win_rate_pct | avg_win_r | avg_loss_r | fees |
|---|---|---|---|---|---|---|---|---|---|---|
| BTCUSDT | 1 | -0.350 | -17.505 | 0.350 | 0.000 | -1.000 | 0.000 | 0.000 | -1.000 | 0.394 |
| ETHUSDT | 10 | -0.461 | -32.757 | 0.900 | 0.731 | -0.125 | 30.000 | 1.140 | -0.667 | 3.993 |
| BNBUSDT | 5 | 0.175 | -0.266 | 0.812 | 1.405 | 0.098 | 40.000 | 0.831 | -0.390 | 2.011 |
| XRPUSDT | 43 | -3.767 | -40.216 | 5.214 | 0.552 | -0.246 | 20.930 | 1.284 | -0.651 | 16.387 |
| LTCUSDT | 52 | -7.655 | -46.261 | 8.056 | 0.346 | -0.411 | 25.000 | 0.886 | -0.843 | 20.028 |
| ADAUSDT | 49 | -4.900 | -77.274 | 6.881 | 0.481 | -0.277 | 26.531 | 1.000 | -0.738 | 18.594 |
| EOSUSDT | 33 | -0.986 | 0.156 | 2.668 | 0.821 | -0.087 | 27.273 | 1.428 | -0.655 | 13.350 |
| XLMUSDT | 73 | -0.446 | -49.307 | 4.844 | 0.966 | -0.041 | 36.986 | 1.143 | -0.736 | 27.908 |
| TRXUSDT | 0 | 0.000 | 18.683 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

By regime at entry:

| regime | trades | expectancy_r | total_r |
|---|---|---|---|
| high_volatility | 157 | -0.097 | -15.169 |
| ranging | 11 | -0.544 | -5.979 |
| trending_up | 98 | -0.327 | -31.998 |

By year of entry:

| entry_time | trades | expectancy_r | total_r |
|---|---|---|---|
| 2025 | 198 | -0.234 | -46.311 |
| 2026 | 68 | -0.101 | -6.836 |

#### mean_reversion/fixed, TEST

Per coin:

| pair | trades | net_return_pct | buy_hold_pct | max_dd_pct | profit_factor | expectancy_r | win_rate_pct | avg_win_r | avg_loss_r | fees |
|---|---|---|---|---|---|---|---|---|---|---|
| BTCUSDT | 6 | -1.061 | -17.505 | 1.600 | 0.290 | -0.561 | 16.667 | 0.948 | -0.863 | 2.366 |
| ETHUSDT | 48 | -3.492 | -32.757 | 3.872 | 0.625 | -0.214 | 31.250 | 0.984 | -0.758 | 18.553 |
| BNBUSDT | 19 | 1.071 | -0.266 | 1.139 | 1.383 | 0.122 | 52.632 | 0.974 | -0.824 | 7.510 |
| XRPUSDT | 52 | -2.813 | -40.216 | 4.045 | 0.751 | -0.173 | 30.769 | 1.217 | -0.790 | 19.524 |
| LTCUSDT | 48 | -0.334 | -46.261 | 2.559 | 0.962 | -0.016 | 39.583 | 1.093 | -0.743 | 18.336 |
| ADAUSDT | 76 | -6.874 | -77.274 | 7.793 | 0.568 | -0.234 | 30.263 | 1.021 | -0.778 | 28.186 |
| EOSUSDT | 36 | -1.550 | 0.156 | 3.052 | 0.784 | -0.130 | 36.111 | 1.068 | -0.806 | 13.898 |
| XLMUSDT | 56 | -0.732 | -49.307 | 2.596 | 0.932 | -0.062 | 37.500 | 1.142 | -0.784 | 21.818 |
| TRXUSDT | 2 | 0.245 | 18.683 | 0.394 | 1.682 | 0.287 | 50.000 | 1.575 | -1.000 | 0.805 |

By regime at entry:

| regime | trades | expectancy_r | total_r |
|---|---|---|---|
| high_volatility | 249 | -0.224 | -55.694 |
| ranging | 5 | -0.442 | -2.209 |
| trending_down | 89 | 0.130 | 11.556 |

By year of entry:

| entry_time | trades | expectancy_r | total_r |
|---|---|---|---|
| 2025 | 265 | -0.084 | -22.275 |
| 2026 | 78 | -0.309 | -24.072 |

#### mean_reversion/tuned, TEST

Per coin:

| pair | trades | net_return_pct | buy_hold_pct | max_dd_pct | profit_factor | expectancy_r | win_rate_pct | avg_win_r | avg_loss_r | fees |
|---|---|---|---|---|---|---|---|---|---|---|
| BTCUSDT | 1 | 0.440 | -17.505 | 0.305 | inf | 1.336 | 100.000 | 1.336 | 0.000 | 0.410 |
| ETHUSDT | 11 | -0.751 | -32.757 | 1.405 | 0.715 | -0.229 | 27.273 | 1.576 | -0.905 | 4.381 |
| BNBUSDT | 6 | 0.060 | -0.266 | 0.823 | 1.054 | -0.014 | 50.000 | 0.964 | -0.991 | 2.404 |
| XRPUSDT | 19 | 0.054 | -40.216 | 1.723 | 1.013 | -0.025 | 36.842 | 1.479 | -0.901 | 7.148 |
| LTCUSDT | 19 | 0.337 | -46.261 | 1.809 | 1.095 | 0.067 | 42.105 | 1.299 | -0.829 | 7.173 |
| ADAUSDT | 22 | -0.493 | -77.274 | 1.645 | 0.900 | -0.086 | 36.364 | 1.415 | -0.944 | 8.228 |
| EOSUSDT | 13 | -0.876 | 0.156 | 2.346 | 0.728 | -0.182 | 30.769 | 1.568 | -0.960 | 4.973 |
| XLMUSDT | 18 | 0.387 | -49.307 | 1.494 | 1.109 | 0.054 | 44.444 | 1.293 | -0.937 | 6.818 |
| TRXUSDT | 0 | 0.000 | 18.683 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

By regime at entry:

| regime | trades | expectancy_r | total_r |
|---|---|---|---|
| high_volatility | 100 | -0.058 | -5.762 |
| ranging | 1 | -1.000 | -1.000 |
| trending_down | 8 | 0.377 | 3.019 |

By year of entry:

| entry_time | trades | expectancy_r | total_r |
|---|---|---|---|
| 2025 | 87 | -0.008 | -0.710 |
| 2026 | 22 | -0.138 | -3.034 |

#### mean_reversion/fixed+ML, TEST

Per coin:

| pair | trades | net_return_pct | buy_hold_pct | max_dd_pct | profit_factor | expectancy_r | win_rate_pct | avg_win_r | avg_loss_r | fees |
|---|---|---|---|---|---|---|---|---|---|---|
| BTCUSDT | 0 | 0.000 | -17.505 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| ETHUSDT | 0 | 0.000 | -32.757 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| BNBUSDT | 0 | 0.000 | -0.266 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| XRPUSDT | 0 | 0.000 | -40.216 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| LTCUSDT | 0 | 0.000 | -46.261 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| ADAUSDT | 0 | 0.000 | -77.274 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| EOSUSDT | 0 | 0.000 | 0.156 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| XLMUSDT | 0 | 0.000 | -49.307 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| TRXUSDT | 0 | 0.000 | 18.683 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

#### stacked/fixed, TEST

Per coin:

| pair | trades | net_return_pct | buy_hold_pct | max_dd_pct | profit_factor | expectancy_r | win_rate_pct | avg_win_r | avg_loss_r | fees |
|---|---|---|---|---|---|---|---|---|---|---|
| BTCUSDT | 0 | 0.000 | -17.505 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| ETHUSDT | 3 | 0.979 | -32.757 | 0.494 | 6.641 | 0.857 | 66.667 | 1.547 | -0.525 | 1.227 |
| BNBUSDT | 3 | -0.586 | -0.266 | 0.699 | 0.000 | -0.513 | 0.000 | 0.000 | -0.513 | 1.189 |
| XRPUSDT | 2 | -0.575 | -40.216 | 0.650 | 0.000 | -0.774 | 0.000 | 0.000 | -0.774 | 0.789 |
| LTCUSDT | 9 | -0.329 | -46.261 | 1.913 | 0.764 | -0.095 | 22.222 | 1.534 | -0.561 | 3.573 |
| ADAUSDT | 10 | -1.679 | -77.274 | 2.401 | 0.232 | -0.414 | 10.000 | 1.520 | -0.629 | 3.617 |
| EOSUSDT | 7 | 0.386 | 0.156 | 0.760 | 1.342 | 0.042 | 42.857 | 1.129 | -0.773 | 2.711 |
| XLMUSDT | 9 | 0.222 | -49.307 | 1.342 | 1.170 | -0.004 | 22.222 | 1.661 | -0.479 | 3.505 |
| TRXUSDT | 0 | 0.000 | 18.683 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

By regime at entry:

| regime | trades | expectancy_r | total_r |
|---|---|---|---|
| high_volatility | 22 | -0.207 | -4.546 |
| trending_up | 21 | -0.034 | -0.706 |

By year of entry:

| entry_time | trades | expectancy_r | total_r |
|---|---|---|---|
| 2025 | 36 | -0.159 | -5.735 |
| 2026 | 7 | 0.069 | 0.482 |

#### stacked/tuned, TEST

Per coin:

| pair | trades | net_return_pct | buy_hold_pct | max_dd_pct | profit_factor | expectancy_r | win_rate_pct | avg_win_r | avg_loss_r | fees |
|---|---|---|---|---|---|---|---|---|---|---|
| BTCUSDT | 5 | -0.508 | -17.505 | 0.773 | 0.412 | -0.264 | 20.000 | 1.090 | -0.602 | 1.992 |
| ETHUSDT | 60 | -4.661 | -32.757 | 5.085 | 0.535 | -0.217 | 26.667 | 0.941 | -0.639 | 23.254 |
| BNBUSDT | 19 | -1.527 | -0.266 | 1.527 | 0.476 | -0.216 | 21.053 | 0.891 | -0.512 | 7.464 |
| XRPUSDT | 109 | -7.699 | -40.216 | 10.230 | 0.571 | -0.223 | 21.101 | 1.049 | -0.564 | 42.145 |
| LTCUSDT | 103 | -5.002 | -46.261 | 7.127 | 0.704 | -0.107 | 34.951 | 0.853 | -0.623 | 39.492 |
| ADAUSDT | 128 | -8.856 | -77.274 | 10.067 | 0.549 | -0.177 | 25.781 | 0.871 | -0.540 | 48.131 |
| EOSUSDT | 66 | -5.025 | 0.156 | 6.475 | 0.586 | -0.201 | 25.758 | 0.993 | -0.615 | 24.727 |
| XLMUSDT | 126 | -9.286 | -49.307 | 10.082 | 0.556 | -0.200 | 28.571 | 0.824 | -0.610 | 46.594 |
| TRXUSDT | 2 | -0.182 | 18.683 | 0.418 | 0.396 | -0.304 | 50.000 | 0.345 | -0.953 | 0.797 |

By regime at entry:

| regime | trades | expectancy_r | total_r |
|---|---|---|---|
| high_volatility | 224 | -0.101 | -22.564 |
| ranging | 23 | -0.283 | -6.514 |
| trending_up | 371 | -0.233 | -86.527 |

By year of entry:

| entry_time | trades | expectancy_r | total_r |
|---|---|---|---|
| 2025 | 542 | -0.169 | -91.393 |
| 2026 | 76 | -0.319 | -24.211 |

