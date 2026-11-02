## Stage 1: development and validation (run 2026-11-01T03:58:31+00:00, 1158s)

Coins: BTCUSDT, ETHUSDT, BNBUSDT, XRPUSDT, LTCUSDT, ADAUSDT, EOSUSDT, XLMUSDT, TRXUSDT. Costs: fee 0.20%/side, slippage 0.05%, stop slippage 0.20%; Conservative risk; 1h; 1,000 USDT per coin.

### Summary (criteria judged on VAL; DEV is in-sample)

| candidate | period | trades | expectancy_r | ci95_low_r | profit_factor | win_rate_pct | avg_win_r | avg_loss_r | breadth | exp_r_wo_best | mean_net_ret_pct | mean_buy_hold_pct | worst_dd_pct | criteria |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| auto/fixed | DEV | 1092 | -0.147 | -0.205 | 0.724 | 30.403 | 1.155 | -0.716 | 0.111 | -0.173 | -6.493 | 626.150 | 10.145 | nan |
| auto/fixed | VAL | 255 | -0.065 | -0.190 | 0.911 | 35.686 | 1.194 | -0.763 | 0.400 | -0.120 | -0.501 | 260.537 | 4.728 | expectancy -0.065R ≤ 0; 95% CI lower bound -0.190R ≤ 0; profit factor 0.91 < 1.10; positive on 40% of 5 coins < 60%; without best coin -0.120R ≤ 0 |
| auto/tuned{'threshold': 0.35, 'exit_rules.stop_atr_mult': 1.5, 'exit_rules.tp_atr_mult': 3.0} | VAL | 428 | -0.073 | -0.167 | 0.881 | 34.813 | 1.147 | -0.725 | 0.500 | -0.119 | -1.056 | 260.537 | 7.555 | expectancy -0.073R ≤ 0; 95% CI lower bound -0.167R ≤ 0; profit factor 0.88 < 1.10; positive on 50% of 6 coins < 60%; without best coin -0.119R ≤ 0 |
| auto/fixed+ML | VAL | 0 | nan | nan | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | nan | 0.000 | 260.537 | 0.000 | 0 trades < 100; expectancy +nanR ≤ 0; 95% CI lower bound +nanR ≤ 0; profit factor 0.00 < 1.10; positive on 0% of 0 coins < 60%; without best coin +nanR ≤ 0 |
| trend_following/fixed | DEV | 1497 | -0.116 | -0.166 | 0.786 | 31.931 | 1.194 | -0.731 | 0.000 | -0.131 | -6.938 | 626.150 | 10.170 | nan |
| trend_following/fixed | VAL | 380 | -0.082 | -0.184 | 0.867 | 32.368 | 1.265 | -0.726 | 0.286 | -0.124 | -1.105 | 260.537 | 8.052 | expectancy -0.082R ≤ 0; 95% CI lower bound -0.184R ≤ 0; profit factor 0.87 < 1.10; positive on 29% of 7 coins < 60%; without best coin -0.124R ≤ 0 |
| trend_following/tuned{'threshold': 0.45, 'exit_rules.stop_atr_mult': 1.5, 'exit_rules.tp_atr_mult': 3.0} | VAL | 380 | -0.082 | -0.184 | 0.867 | 32.368 | 1.265 | -0.726 | 0.286 | -0.124 | -1.105 | 260.537 | 8.052 | expectancy -0.082R ≤ 0; 95% CI lower bound -0.184R ≤ 0; profit factor 0.87 < 1.10; positive on 29% of 7 coins < 60%; without best coin -0.124R ≤ 0 |
| trend_following/fixed+ML | VAL | 23 | -0.134 | -0.544 | 0.805 | 26.087 | 1.587 | -0.742 | 0.000 | -0.288 | -0.105 | 260.537 | 1.065 | 23 trades < 100; expectancy -0.134R ≤ 0; 95% CI lower bound -0.544R ≤ 0; profit factor 0.80 < 1.10; positive on 0% of 0 coins < 60%; without best coin -0.288R ≤ 0 |
| mean_reversion/fixed | DEV | 1084 | -0.214 | -0.265 | 0.605 | 31.273 | 0.940 | -0.739 | 0.000 | -0.234 | -9.520 | 626.150 | 10.284 | nan |
| mean_reversion/fixed | VAL | 283 | -0.084 | -0.197 | 0.858 | 37.102 | 1.062 | -0.760 | 0.250 | -0.104 | -0.882 | 260.537 | 6.034 | expectancy -0.084R ≤ 0; 95% CI lower bound -0.197R ≤ 0; profit factor 0.86 < 1.10; positive on 25% of 8 coins < 60%; without best coin -0.104R ≤ 0 |
| mean_reversion/tuned{'threshold': 0.25, 'exit_rules.stop_atr_mult': 1.0, 'exit_rules.tp_atr_mult': 2.0} | VAL | 137 | -0.176 | -0.360 | 0.785 | 31.387 | 1.448 | -0.918 | 0.200 | -0.226 | -0.787 | 260.537 | 6.444 | expectancy -0.176R ≤ 0; 95% CI lower bound -0.360R ≤ 0; profit factor 0.78 < 1.10; positive on 20% of 5 coins < 60%; without best coin -0.226R ≤ 0 |
| mean_reversion/fixed+ML | VAL | 1 | 1.232 | nan | inf | 100.000 | 1.232 | 0.000 | 0.000 | nan | 0.068 | 260.537 | 0.483 | 1 trades < 100; 95% CI lower bound +nanR ≤ 0; positive on 0% of 0 coins < 60%; without best coin +nanR ≤ 0 |
| stacked/fixed | DEV | 274 | -0.228 | -0.342 | 0.618 | 26.277 | 1.253 | -0.755 | 0.111 | -0.259 | -2.528 | 626.150 | 7.679 | nan |
| stacked/fixed | VAL | 58 | -0.194 | -0.415 | 0.653 | 34.483 | 0.864 | -0.751 | 0.000 | -0.283 | -0.449 | 260.537 | 3.007 | 58 trades < 100; expectancy -0.194R ≤ 0; 95% CI lower bound -0.415R ≤ 0; profit factor 0.65 < 1.10; positive on 0% of 3 coins < 60%; without best coin -0.283R ≤ 0 |
| stacked/tuned{'threshold': 0.25, 'exit_rules.stop_atr_mult': 2.0, 'exit_rules.tp_atr_mult': 3.0} | VAL | 790 | -0.172 | -0.224 | 0.606 | 28.734 | 0.880 | -0.596 | 0.000 | -0.175 | -5.521 | 260.537 | 10.233 | expectancy -0.172R ≤ 0; 95% CI lower bound -0.224R ≤ 0; profit factor 0.61 < 1.10; positive on 0% of 9 coins < 60%; without best coin -0.175R ≤ 0 |
| stacked/fixed+ML | VAL | 11 | -0.492 | -0.882 | 0.303 | 18.182 | 0.953 | -0.813 | 0.000 | -0.693 | -0.243 | 260.537 | 1.439 | 11 trades < 100; expectancy -0.492R ≤ 0; 95% CI lower bound -0.882R ≤ 0; profit factor 0.30 < 1.10; positive on 0% of 0 coins < 60%; without best coin -0.693R ≤ 0 |

### Walk-forward (tuned on the 2 prior years, scored on the year)

**auto**: 0/4 walk-forward years positive

| year | params | trades | expectancy_r | profit_factor |
|---|---|---|---|---|
| 2021 | {'threshold': 0.35, 'exit_rules.stop_atr_mult': 1.5, 'exit_rules.tp_atr_mult': 3.0} | 1054 | -0.120 | 0.775 |
| 2022 | {'threshold': 0.35, 'exit_rules.stop_atr_mult': 1.0, 'exit_rules.tp_atr_mult': 3.0} | 62 | -0.609 | 0.200 |
| 2023 | {'threshold': 0.25, 'exit_rules.stop_atr_mult': 2.0, 'exit_rules.tp_atr_mult': 3.0} | 480 | -0.178 | 0.566 |
| 2024 | {'threshold': 0.35, 'exit_rules.stop_atr_mult': 2.0, 'exit_rules.tp_atr_mult': 2.0} | 614 | -0.060 | 0.865 |

**trend_following**: 0/4 walk-forward years positive

| year | params | trades | expectancy_r | profit_factor |
|---|---|---|---|---|
| 2021 | {'threshold': 0.45, 'exit_rules.stop_atr_mult': 2.0, 'exit_rules.tp_atr_mult': 3.0} | 1193 | -0.119 | 0.729 |
| 2022 | {'threshold': 0.45, 'exit_rules.stop_atr_mult': 1.5, 'exit_rules.tp_atr_mult': 3.0} | 209 | -0.288 | 0.466 |
| 2023 | {'threshold': 0.25, 'exit_rules.stop_atr_mult': 2.0, 'exit_rules.tp_atr_mult': 3.0} | 480 | -0.216 | 0.531 |
| 2024 | {'threshold': 0.35, 'exit_rules.stop_atr_mult': 2.0, 'exit_rules.tp_atr_mult': 2.0} | 726 | -0.095 | 0.776 |

**mean_reversion**: 1/4 walk-forward years positive

| year | params | trades | expectancy_r | profit_factor |
|---|---|---|---|---|
| 2021 | {'threshold': 0.25, 'exit_rules.stop_atr_mult': 1.0, 'exit_rules.tp_atr_mult': 2.0} | 448 | -0.152 | 0.772 |
| 2022 | {'threshold': 0.25, 'exit_rules.stop_atr_mult': 2.0, 'exit_rules.tp_atr_mult': 2.0} | 788 | -0.160 | 0.636 |
| 2023 | {'threshold': 0.25, 'exit_rules.stop_atr_mult': 1.0, 'exit_rules.tp_atr_mult': 3.0} | 9 | 0.540 | 4.526 |
| 2024 | {'threshold': 0.25, 'exit_rules.stop_atr_mult': 1.0, 'exit_rules.tp_atr_mult': 3.0} | 126 | -0.111 | 0.912 |

**stacked**: 0/4 walk-forward years positive

| year | params | trades | expectancy_r | profit_factor |
|---|---|---|---|---|
| 2021 | {'threshold': 0.45, 'exit_rules.stop_atr_mult': 2.0, 'exit_rules.tp_atr_mult': 3.0} | 166 | -0.264 | 0.484 |
| 2022 | {'threshold': 0.35, 'exit_rules.stop_atr_mult': 1.0, 'exit_rules.tp_atr_mult': 3.0} | 36 | -0.719 | 0.113 |
| 2023 | {'threshold': 0.25, 'exit_rules.stop_atr_mult': 2.0, 'exit_rules.tp_atr_mult': 3.0} | 414 | -0.217 | 0.504 |
| 2024 | {'threshold': 0.25, 'exit_rules.stop_atr_mult': 2.0, 'exit_rules.tp_atr_mult': 2.0} | 610 | -0.108 | 0.747 |

### ML

- **auto**: internal DEV validation: AUC 0.529, useful=False; VAL expectancy with ML +nanR (0 trades) vs without -0.065R (255 trades) -> carried to TEST: False
- **trend_following**: internal DEV validation: AUC 0.530, useful=False; VAL expectancy with ML -0.134R (23 trades) vs without -0.082R (380 trades) -> carried to TEST: False
- **mean_reversion**: internal DEV validation: AUC 0.525, useful=False; VAL expectancy with ML +1.232R (1 trades) vs without -0.084R (283 trades) -> carried to TEST: True
- **stacked**: internal DEV validation: AUC 0.513, useful=False; VAL expectancy with ML -0.492R (11 trades) vs without -0.194R (58 trades) -> carried to TEST: False

### Frozen for TEST

```json
{
 "auto": {
  "fixed": null,
  "tuned_dev_val": {
   "threshold": 0.45,
   "exit_rules.stop_atr_mult": 1.0,
   "exit_rules.tp_atr_mult": 3.0
  },
  "walk_forward_pass": false,
  "walk_forward": "0/4 walk-forward years positive",
  "ml_carry": false,
  "ml_note": "internal DEV validation: AUC 0.529, useful=False; VAL expectancy with ML +nanR (0 trades) vs without -0.065R (255 trades) -> carried to TEST: False"
 },
 "trend_following": {
  "fixed": null,
  "tuned_dev_val": {
   "threshold": 0.45,
   "exit_rules.stop_atr_mult": 1.5,
   "exit_rules.tp_atr_mult": 3.0
  },
  "walk_forward_pass": false,
  "walk_forward": "0/4 walk-forward years positive",
  "ml_carry": false,
  "ml_note": "internal DEV validation: AUC 0.530, useful=False; VAL expectancy with ML -0.134R (23 trades) vs without -0.082R (380 trades) -> carried to TEST: False"
 },
 "mean_reversion": {
  "fixed": null,
  "tuned_dev_val": {
   "threshold": 0.25,
   "exit_rules.stop_atr_mult": 1.0,
   "exit_rules.tp_atr_mult": 2.0
  },
  "walk_forward_pass": false,
  "walk_forward": "1/4 walk-forward years positive",
  "ml_carry": true,
  "ml_note": "internal DEV validation: AUC 0.525, useful=False; VAL expectancy with ML +1.232R (1 trades) vs without -0.084R (283 trades) -> carried to TEST: True"
 },
 "stacked": {
  "fixed": null,
  "tuned_dev_val": {
   "threshold": 0.25,
   "exit_rules.stop_atr_mult": 2.0,
   "exit_rules.tp_atr_mult": 3.0
  },
  "walk_forward_pass": false,
  "walk_forward": "0/4 walk-forward years positive",
  "ml_carry": false,
  "ml_note": "internal DEV validation: AUC 0.513, useful=False; VAL expectancy with ML -0.492R (11 trades) vs without -0.194R (58 trades) -> carried to TEST: False"
 }
}
```

### VAL detail

#### auto/fixed, VAL

Per coin:

| pair | trades | net_return_pct | buy_hold_pct | max_dd_pct | profit_factor | expectancy_r | win_rate_pct | avg_win_r | avg_loss_r | fees |
|---|---|---|---|---|---|---|---|---|---|---|
| BTCUSDT | 0 | 0.000 | 462.874 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| ETHUSDT | 3 | -0.118 | 177.657 | 0.451 | 0.806 | -0.099 | 33.333 | 1.497 | -0.898 | 1.200 |
| BNBUSDT | 4 | -1.354 | 183.718 | 1.354 | 0.000 | -0.945 | 0.000 | 0.000 | -0.945 | 1.568 |
| XRPUSDT | 60 | 2.909 | 511.746 | 2.552 | 1.258 | 0.114 | 43.333 | 1.280 | -0.778 | 23.095 |
| LTCUSDT | 28 | -1.199 | 46.514 | 2.232 | 0.716 | -0.108 | 35.714 | 0.860 | -0.645 | 11.004 |
| ADAUSDT | 57 | -2.313 | 242.101 | 4.728 | 0.797 | -0.138 | 33.333 | 1.105 | -0.760 | 21.550 |
| EOSUSDT | 44 | -3.605 | -10.564 | 4.007 | 0.618 | -0.210 | 29.545 | 1.242 | -0.819 | 17.224 |
| XLMUSDT | 55 | 1.822 | 365.599 | 3.102 | 1.159 | 0.054 | 38.182 | 1.334 | -0.737 | 19.479 |
| TRXUSDT | 4 | -0.653 | 365.191 | 0.781 | 0.063 | -0.519 | 25.000 | 0.088 | -0.721 | 1.370 |

By regime at entry:

| regime | trades | expectancy_r | total_r |
|---|---|---|---|
| high_volatility | 36 | -0.288 | -10.386 |
| ranging | 3 | 0.029 | 0.086 |
| trending_up | 216 | -0.029 | -6.226 |

By year of entry:

| entry_time | trades | expectancy_r | total_r |
|---|---|---|---|
| 2023 | 57 | -0.271 | -15.467 |
| 2024 | 198 | -0.005 | -1.059 |

#### trend_following/fixed, VAL

Per coin:

| pair | trades | net_return_pct | buy_hold_pct | max_dd_pct | profit_factor | expectancy_r | win_rate_pct | avg_win_r | avg_loss_r | fees |
|---|---|---|---|---|---|---|---|---|---|---|
| BTCUSDT | 4 | -0.995 | 462.874 | 0.995 | 0.000 | -0.768 | 0.000 | 0.000 | -0.768 | 1.577 |
| ETHUSDT | 5 | -0.067 | 177.657 | 0.784 | 0.895 | -0.017 | 40.000 | 0.875 | -0.612 | 2.001 |
| BNBUSDT | 10 | -1.373 | 183.718 | 1.667 | 0.324 | -0.407 | 30.000 | 0.666 | -0.867 | 3.950 |
| XRPUSDT | 75 | 2.644 | 511.746 | 3.219 | 1.187 | 0.090 | 38.667 | 1.361 | -0.711 | 28.290 |
| LTCUSDT | 47 | -2.933 | 46.514 | 4.190 | 0.666 | -0.130 | 31.915 | 1.095 | -0.703 | 18.388 |
| ADAUSDT | 79 | -2.406 | 242.101 | 4.569 | 0.842 | -0.097 | 29.114 | 1.360 | -0.695 | 30.448 |
| EOSUSDT | 68 | -7.311 | -10.564 | 8.052 | 0.507 | -0.269 | 27.941 | 1.123 | -0.808 | 25.717 |
| XLMUSDT | 77 | 0.347 | 365.599 | 4.067 | 1.022 | -0.027 | 33.766 | 1.332 | -0.720 | 26.600 |
| TRXUSDT | 15 | 2.148 | 365.191 | 1.328 | 2.013 | 0.236 | 40.000 | 1.448 | -0.571 | 5.092 |

By regime at entry:

| regime | trades | expectancy_r | total_r |
|---|---|---|---|
| high_volatility | 179 | -0.118 | -21.053 |
| ranging | 4 | 0.188 | 0.752 |
| trending_up | 197 | -0.054 | -10.726 |

By year of entry:

| entry_time | trades | expectancy_r | total_r |
|---|---|---|---|
| 2023 | 104 | -0.148 | -15.341 |
| 2024 | 276 | -0.057 | -15.686 |

#### mean_reversion/fixed, VAL

Per coin:

| pair | trades | net_return_pct | buy_hold_pct | max_dd_pct | profit_factor | expectancy_r | win_rate_pct | avg_win_r | avg_loss_r | fees |
|---|---|---|---|---|---|---|---|---|---|---|
| BTCUSDT | 6 | 0.202 | 462.874 | 1.393 | 1.155 | -0.049 | 33.333 | 1.642 | -0.894 | 2.410 |
| ETHUSDT | 19 | -0.918 | 177.657 | 2.583 | 0.731 | -0.145 | 36.842 | 0.924 | -0.768 | 7.448 |
| BNBUSDT | 24 | 0.117 | 183.718 | 2.555 | 1.027 | -0.019 | 41.667 | 1.118 | -0.831 | 9.488 |
| XRPUSDT | 36 | 1.106 | 511.746 | 2.070 | 1.169 | 0.032 | 36.111 | 1.257 | -0.660 | 13.624 |
| LTCUSDT | 46 | -3.812 | 46.514 | 5.907 | 0.600 | -0.233 | 34.783 | 0.801 | -0.785 | 17.660 |
| ADAUSDT | 46 | -2.384 | 242.101 | 5.271 | 0.775 | -0.141 | 34.783 | 1.162 | -0.837 | 17.779 |
| EOSUSDT | 61 | -2.693 | -10.564 | 6.034 | 0.773 | -0.085 | 39.344 | 0.929 | -0.744 | 22.875 |
| XLMUSDT | 32 | -1.243 | 365.599 | 3.096 | 0.794 | -0.105 | 31.250 | 1.055 | -0.632 | 11.872 |
| TRXUSDT | 13 | 1.685 | 365.191 | 1.438 | 1.717 | 0.340 | 53.846 | 1.430 | -0.932 | 5.193 |

By regime at entry:

| regime | trades | expectancy_r | total_r |
|---|---|---|---|
| high_volatility | 221 | -0.075 | -16.624 |
| ranging | 3 | 0.373 | 1.120 |
| trending_down | 59 | -0.139 | -8.189 |

By year of entry:

| entry_time | trades | expectancy_r | total_r |
|---|---|---|---|
| 2023 | 59 | 0.294 | 17.358 |
| 2024 | 224 | -0.183 | -41.051 |

#### stacked/fixed, VAL

Per coin:

| pair | trades | net_return_pct | buy_hold_pct | max_dd_pct | profit_factor | expectancy_r | win_rate_pct | avg_win_r | avg_loss_r | fees |
|---|---|---|---|---|---|---|---|---|---|---|
| BTCUSDT | 0 | 0.000 | 462.874 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| ETHUSDT | 0 | 0.000 | 177.657 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| BNBUSDT | 3 | -1.009 | 183.718 | 1.009 | 0.000 | -0.927 | 0.000 | 0.000 | -0.927 | 1.178 |
| XRPUSDT | 15 | -1.621 | 511.746 | 3.007 | 0.531 | -0.283 | 26.667 | 0.985 | -0.744 | 5.264 |
| LTCUSDT | 7 | -1.188 | 46.514 | 1.579 | 0.122 | -0.537 | 28.571 | 0.166 | -0.818 | 2.671 |
| ADAUSDT | 14 | -0.585 | 242.101 | 2.105 | 0.748 | -0.093 | 42.857 | 0.659 | -0.658 | 5.000 |
| EOSUSDT | 10 | -0.229 | -10.564 | 1.572 | 0.867 | -0.021 | 40.000 | 0.956 | -0.673 | 3.985 |
| XLMUSDT | 6 | 1.470 | 365.599 | 1.337 | 2.651 | 0.573 | 66.667 | 1.305 | -0.893 | 1.842 |
| TRXUSDT | 3 | -0.875 | 365.191 | 0.940 | 0.000 | -0.797 | 0.000 | 0.000 | -0.797 | 0.967 |

By regime at entry:

| regime | trades | expectancy_r | total_r |
|---|---|---|---|
| high_volatility | 34 | -0.283 | -9.628 |
| trending_up | 24 | -0.068 | -1.630 |

By year of entry:

| entry_time | trades | expectancy_r | total_r |
|---|---|---|---|
| 2023 | 22 | -0.447 | -9.837 |
| 2024 | 36 | -0.039 | -1.421 |

