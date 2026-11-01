# T50 scrutiny (2026-11-01 12:05 UTC)

T50 rules frozen as pre-registered: daily close > previous 50-day highest close -> buy next open; exit at next open after a close < previous 25-day lowest close; 3-ATR(14) protective stop; no take-profit; no trailing stop. Unchanged engine, costs (0.20%/side fee, 0.05% slippage, 0.20% stop slippage) and Conservative risk. **Data >= 2025-01-01 never loaded.**

**Change in measurement vs Round 1 (not in the rules):** positions still open at the end of a period or at a coin's delisting are now closed at the last available close and counted. Round 1 dropped them (engine default `close_at_end=False`), which can hide losses. Both versions are shown.

## 0. Effect of counting positions open at period end / delisting

| sample | round1_trades | round1_net_R | now_trades | now_net_R | now_ci_low | now_pf |
|---|---|---|---|---|---|---|
| main DEV | 112 | +1.236 | 112 | +1.236 | +0.274 | +3.665 |
| main VAL | 62 | +1.436 | 64 | +1.565 | +0.565 | +4.998 |
| holdout 2019-24 | 276 | +0.737 | 277 | +0.744 | +0.365 | +2.631 |


## 1. Outlier sensitivity

| sample | variant | trades | net_R | ci95_low | median_R | pf | total_R | total_pnl_usdt |
|---|---|---|---|---|---|---|---|---|
| main DEV | all trades | 112 | +1.236 | +0.286 | -0.334 | +3.665 | +138.4 | +706.0 |
| main DEV | excl. largest winner | 111 | +0.764 | +0.150 | -0.357 | +2.646 | +84.8 | +436.0 |
| main DEV | excl. 5 largest winners | 107 | +0.161 | -0.146 | -0.412 | +1.319 | +17.2 | +84.5 |
| main DEV | excl. top 5% of trades | 106 | +0.074 | -0.198 | -0.431 | +1.142 | +7.865 | +37.5 |
| main DEV | excl. top 10% of trades | 101 | -0.165 | -0.355 | -0.643 | +0.668 | -16.7 | -87.8 |
| main VAL | all trades | 64 | +1.565 | +0.575 | -0.268 | +4.998 | +100.1 | +503.0 |
| main VAL | excl. largest winner | 63 | +1.130 | +0.426 | -0.279 | +3.849 | +71.2 | +358.4 |
| main VAL | excl. 5 largest winners | 59 | +0.525 | +0.040 | -0.339 | +2.251 | +31.0 | +157.3 |
| main VAL | excl. top 5% of trades | 61 | +0.784 | +0.211 | -0.283 | +2.932 | +47.8 | +243.0 |
| main VAL | excl. top 10% of trades | 58 | +0.407 | -0.030 | -0.376 | +1.958 | +23.6 | +120.5 |
| main 2019-24 | all trades | 169 | +1.358 | +0.598 | -0.279 | +4.113 | +229.6 | +1191.7 |
| main 2019-24 | excl. largest winner | 168 | +1.047 | +0.480 | -0.281 | +3.407 | +176.0 | +921.7 |
| main 2019-24 | excl. 5 largest winners | 164 | +0.571 | +0.197 | -0.334 | +2.322 | +93.6 | +506.3 |
| main 2019-24 | excl. top 5% of trades | 161 | +0.348 | +0.052 | -0.383 | +1.812 | +56.0 | +310.9 |
| main 2019-24 | excl. top 10% of trades | 152 | -0.034 | -0.219 | -0.477 | +0.936 | -5.152 | -24.5 |
| holdout 2019-24 | all trades | 277 | +0.744 | +0.361 | -0.279 | +2.631 | +206.0 | +1045.2 |
| holdout 2019-24 | excl. largest winner | 276 | +0.635 | +0.302 | -0.287 | +2.386 | +175.3 | +887.7 |
| holdout 2019-24 | excl. 5 largest winners | 272 | +0.390 | +0.157 | -0.311 | +1.832 | +106.0 | +532.9 |
| holdout 2019-24 | excl. top 5% of trades | 263 | +0.139 | -0.038 | -0.447 | +1.289 | +36.6 | +184.9 |
| holdout 2019-24 | excl. top 10% of trades | 249 | -0.103 | -0.236 | -0.521 | +0.791 | -25.6 | -134.0 |
| all 25 coins 2019-24 | all trades | 446 | +0.977 | +0.591 | -0.279 | +3.185 | +435.6 | +2236.9 |
| all 25 coins 2019-24 | excl. largest winner | 445 | +0.858 | +0.527 | -0.279 | +2.922 | +382.0 | +1966.9 |
| all 25 coins 2019-24 | excl. 5 largest winners | 441 | +0.622 | +0.371 | -0.296 | +2.387 | +274.1 | +1420.0 |
| all 25 coins 2019-24 | excl. top 5% of trades | 424 | +0.197 | +0.043 | -0.412 | +1.426 | +83.7 | +435.6 |
| all 25 coins 2019-24 | excl. top 10% of trades | 401 | -0.077 | -0.187 | -0.491 | +0.844 | -30.9 | -159.5 |


Concentration of profit in the 5 largest trades:

| sample | top5_share_of_net_R_% | top5_share_of_net_pnl_% | top5_share_of_gross_profit_% | top5_trades |
|---|---|---|---|---|
| main DEV | +87.6 | +88.0 | +64.0 | BNB 2020-12 +53.6R; ADA 2020-12 +18.9R; BTC 2019-04 +18.5R; ETH 2020-11 +15.9R; BTC 2020-10 +14.2R |
| main VAL | +69.1 | +68.7 | +55.0 | XLM 2024-11 +29.0R; XRP 2024-11 +13.4R; TRX 2024-10 +10.0R; ADA 2023-10 +9.5R; ADA 2024-11 +7.4R |
| main 2019-24 | +59.2 | +57.5 | +43.5 | BNB 2020-12 +53.6R; XLM 2024-11 +29.0R; ADA 2020-12 +18.9R; BTC 2019-04 +18.5R; ETH 2020-11 +15.9R |
| holdout 2019-24 | +48.6 | +49.0 | +30.4 | ZIL 2020-11 +30.7R; ETC 2021-01 +25.0R; VET 2020-12 +23.1R; ICX 2020-12 +10.7R; LINK 2019-05 +10.5R |
| all 25 coins 2019-24 | +37.1 | +36.5 | +25.1 | BNB 2020-12 +53.6R; ZIL 2020-11 +30.7R; XLM 2024-11 +29.0R; ETC 2021-01 +25.0R; VET 2020-12 +23.1R |


## 2. Coin independence

### Main universe 2019-2024: leave one coin out (and that coin alone)

| left_out | remaining_trades | remaining_net_R | remaining_ci_low | remaining_pf | coin_trades | coin_net_R | coin_pf | coin_total_R |
|---|---|---|---|---|---|---|---|---|
| ADAUSDT | 152 | +1.202 | +0.410 | +3.715 | 17 | +2.753 | +7.754 | +46.8 |
| BNBUSDT | 159 | +1.097 | +0.512 | +3.506 | 10 | +5.511 | +16.4 | +55.1 |
| BTCUSDT | 151 | +1.204 | +0.407 | +3.586 | 18 | +2.655 | +12.4 | +47.8 |
| EOSUSDT | 152 | +1.538 | +0.696 | +4.605 | 17 | -0.247 | +0.548 | -4.196 |
| ETHUSDT | 147 | +1.387 | +0.531 | +4.168 | 22 | +1.167 | +3.733 | +25.7 |
| LTCUSDT | 150 | +1.482 | +0.651 | +4.401 | 19 | +0.381 | +1.833 | +7.240 |
| TRXUSDT | 146 | +1.492 | +0.627 | +4.454 | 23 | +0.510 | +2.068 | +11.7 |
| XLMUSDT | 147 | +1.393 | +0.610 | +4.267 | 22 | +1.128 | +3.196 | +24.8 |
| XRPUSDT | 148 | +1.452 | +0.605 | +4.430 | 21 | +0.696 | +2.285 | +14.6 |


### All 25 coins 2019-2024: leave one coin out

| left_out | remaining_trades | remaining_net_R | remaining_ci_low | remaining_pf | coin_trades | coin_net_R | coin_pf | coin_total_R |
|---|---|---|---|---|---|---|---|---|
| ADAUSDT | 429 | +0.906 | +0.529 | +3.011 | 17 | +2.753 | +7.754 | +46.8 |
| BATUSDT | 426 | +1.017 | +0.620 | +3.296 | 20 | +0.107 | +1.180 | +2.146 |
| BNBUSDT | 436 | +0.873 | +0.540 | +2.949 | 10 | +5.511 | +16.4 | +55.1 |
| BTCUSDT | 428 | +0.906 | +0.514 | +2.975 | 18 | +2.655 | +12.4 | +47.8 |
| DASHUSDT | 429 | +1.011 | +0.614 | +3.261 | 17 | +0.117 | +1.251 | +1.993 |
| EOSUSDT | 429 | +1.025 | +0.631 | +3.311 | 17 | -0.247 | +0.548 | -4.196 |
| ETCUSDT | 439 | +0.926 | +0.554 | +3.079 | 7 | +4.120 | +9.069 | +28.8 |
| ETHUSDT | 424 | +0.967 | +0.576 | +3.158 | 22 | +1.167 | +3.733 | +25.7 |
| ICXUSDT | 428 | +0.981 | +0.591 | +3.189 | 18 | +0.880 | +3.095 | +15.8 |
| IOTAUSDT | 429 | +0.994 | +0.606 | +3.215 | 17 | +0.529 | +2.335 | +8.986 |
| LINKUSDT | 426 | +0.973 | +0.583 | +3.190 | 20 | +1.047 | +3.092 | +20.9 |
| LTCUSDT | 427 | +1.003 | +0.609 | +3.245 | 19 | +0.381 | +1.833 | +7.240 |
| NEOUSDT | 428 | +0.989 | +0.592 | +3.217 | 18 | +0.692 | +2.470 | +12.5 |
| NULSUSDT | 428 | +0.989 | +0.596 | +3.196 | 18 | +0.685 | +2.862 | +12.3 |
| OMGUSDT | 428 | +0.998 | +0.608 | +3.255 | 18 | +0.475 | +1.845 | +8.553 |
| ONTUSDT | 425 | +1.027 | +0.632 | +3.324 | 21 | -0.048 | +0.906 | -1.010 |
| QTUMUSDT | 428 | +1.001 | +0.605 | +3.232 | 18 | +0.395 | +1.950 | +7.116 |
| TRXUSDT | 423 | +1.002 | +0.601 | +3.249 | 23 | +0.510 | +2.068 | +11.7 |
| VETUSDT | 426 | +0.946 | +0.564 | +3.137 | 20 | +1.636 | +4.027 | +32.7 |
| WAVESUSDT | 432 | +0.974 | +0.579 | +3.154 | 14 | +1.059 | +4.743 | +14.8 |
| XLMUSDT | 424 | +0.969 | +0.594 | +3.185 | 22 | +1.128 | +3.196 | +24.8 |
| XMRUSDT | 427 | +1.021 | +0.625 | +3.271 | 19 | -0.023 | +0.934 | -0.443 |
| XRPUSDT | 425 | +0.990 | +0.597 | +3.238 | 21 | +0.696 | +2.285 | +14.6 |
| ZECUSDT | 429 | +1.006 | +0.612 | +3.251 | 17 | +0.229 | +1.503 | +3.898 |
| ZILUSDT | 431 | +0.925 | +0.557 | +3.069 | 15 | +2.453 | +6.678 | +36.8 |


## 3. Time-period robustness

### By entry year

Main universe:

| year | trades | net_R | median_R | ci95_low | pf | win_pct | total_R |
|---|---|---|---|---|---|---|---|
| 2019 | 24 | +0.726 | -0.687 | -0.440 | +2.358 | +33.3 | +17.4 |
| 2020 | 39 | +3.454 | +0.415 | +1.043 | +15.7 | +64.1 | +134.7 |
| 2021 | 28 | +0.112 | -0.183 | -0.372 | +1.303 | +46.4 | +3.139 |
| 2022 | 21 | -0.803 | -1.000 | -0.922 | +0.015 | +4.762 | -16.9 |
| 2023 | 32 | +0.820 | -0.172 | +0.066 | +3.676 | +46.9 | +26.2 |
| 2024 | 25 | +2.597 | -0.553 | +0.446 | +6.408 | +48.0 | +64.9 |

Holdout:

| year | trades | net_R | median_R | ci95_low | pf | win_pct | total_R |
|---|---|---|---|---|---|---|---|
| 2019 | 38 | +0.095 | -0.587 | -0.431 | +1.183 | +31.6 | +3.602 |
| 2020 | 65 | +1.808 | +0.006 | +0.679 | +6.264 | +50.8 | +117.5 |
| 2021 | 52 | +1.285 | -0.099 | +0.294 | +3.852 | +50.0 | +66.8 |
| 2022 | 37 | -0.659 | -1.000 | -0.849 | +0.145 | +13.5 | -24.4 |
| 2023 | 42 | +0.762 | +0.185 | +0.258 | +3.827 | +54.8 | +32.0 |
| 2024 | 43 | +0.242 | -0.143 | -0.186 | +1.511 | +46.5 | +10.4 |

All 25 coins:

| year | trades | net_R | median_R | ci95_low | pf | win_pct | total_R |
|---|---|---|---|---|---|---|---|
| 2019 | 62 | +0.339 | -0.594 | -0.278 | +1.647 | +32.3 | +21.0 |
| 2020 | 104 | +2.425 | +0.263 | +1.211 | +9.029 | +55.8 | +252.2 |
| 2021 | 80 | +0.875 | -0.136 | +0.206 | +2.964 | +48.8 | +70.0 |
| 2022 | 58 | -0.711 | -1.000 | -0.845 | +0.096 | +10.3 | -41.3 |
| 2023 | 74 | +0.787 | +0.088 | +0.334 | +3.754 | +51.4 | +58.3 |
| 2024 | 68 | +1.108 | -0.249 | +0.247 | +3.352 | +47.1 | +75.3 |


### By BTC market state at the decision close (OF2 regime detector `core/regime.detect` on BTC daily: strong/weak up = bull, range = sideways, weak/strong down = bear)

| btc_market | trades | net_R | median_R | ci95_low | pf | win_pct | total_R |
|---|---|---|---|---|---|---|---|
| bear | 27 | -0.130 | -0.383 | -0.488 | +0.720 | +37.0 | -3.500 |
| bull | 341 | +1.228 | -0.250 | +0.746 | +3.792 | +43.7 | +418.7 |
| n/a | 10 | +0.616 | +0.793 | -0.263 | +2.539 | +60.0 | +6.156 |
| sideways | 68 | +0.209 | -0.611 | -0.177 | +1.422 | +41.2 | +14.2 |


### By the coin's own engine regime label at entry (`classify_regime`)

| coin_regime | trades | net_R | median_R | ci95_low | pf | win_pct | total_R |
|---|---|---|---|---|---|---|---|
| None | 14 | +0.416 | +0.580 | -0.270 | +2.033 | +57.1 | +5.823 |
| high_volatility | 6 | -0.468 | -0.725 | -0.908 | +0.202 | +16.7 | -2.808 |
| ranging | 160 | +0.446 | -0.165 | +0.161 | +2.131 | +46.2 | +71.4 |
| trending_up | 266 | +1.358 | -0.448 | +0.749 | +3.870 | +41.4 | +361.1 |


### Outside the 2020-21 bull market

| sample | trades | net_R | ci95_low | median_R | pf | total_R |
|---|---|---|---|---|---|---|
| all 25 coins, excluding 2020 and 2021 entries | 262 | +0.433 | +0.102 | -0.544 | +1.871 | +113.4 |
| main only, excl. 2020-21 | 102 | +0.899 | +0.184 | -0.566 | +2.766 | +91.7 |
| holdout only, excl. 2020-21 | 160 | +0.135 | -0.108 | -0.533 | +1.275 | +21.6 |


### Every 2022 trade (all 25 coins)

| pair | entry_time | exit_time | bars_held | exit_reason | r_multiple | btc_market |
|---|---|---|---|---|---|---|
| WAVESUSDT | 2022-03-01 00:00:00+00:00 | 2022-04-09 00:00:00+00:00 | 39 | signal_exit | +2.019 | bear |
| ZECUSDT | 2022-03-10 00:00:00+00:00 | 2022-04-12 00:00:00+00:00 | 33 | signal_exit | +0.013 | bear |
| DASHUSDT | 2022-03-22 00:00:00+00:00 | 2022-04-25 00:00:00+00:00 | 34 | signal_exit | -0.916 | sideways |
| QTUMUSDT | 2022-03-24 00:00:00+00:00 | 2022-04-11 00:00:00+00:00 | 18 | stop_loss | -1.000 | sideways |
| XMRUSDT | 2022-03-25 00:00:00+00:00 | 2022-05-02 00:00:00+00:00 | 38 | signal_exit | +0.162 | sideways |
| ZILUSDT | 2022-03-27 00:00:00+00:00 | 2022-04-25 00:00:00+00:00 | 29 | signal_exit | +0.535 | bull |
| BTCUSDT | 2022-03-28 00:00:00+00:00 | 2022-04-11 00:00:00+00:00 | 14 | stop_loss | -1.000 | bull |
| ETHUSDT | 2022-03-28 00:00:00+00:00 | 2022-04-23 00:00:00+00:00 | 26 | signal_exit | -0.738 | bull |
| VETUSDT | 2022-03-28 00:00:00+00:00 | 2022-04-11 00:00:00+00:00 | 14 | stop_loss | -1.000 | bull |
| ICXUSDT | 2022-03-28 00:00:00+00:00 | 2022-04-11 00:00:00+00:00 | 14 | stop_loss | -1.000 | bull |
| ONTUSDT | 2022-03-28 00:00:00+00:00 | 2022-04-10 00:00:00+00:00 | 13 | stop_loss | -1.000 | bull |
| OMGUSDT | 2022-03-28 00:00:00+00:00 | 2022-04-11 00:00:00+00:00 | 14 | stop_loss | -1.000 | bull |
| EOSUSDT | 2022-03-29 00:00:00+00:00 | 2022-04-08 00:00:00+00:00 | 10 | stop_loss | -1.000 | bull |
| NEOUSDT | 2022-03-30 00:00:00+00:00 | 2022-04-08 00:00:00+00:00 | 9 | stop_loss | -1.000 | bull |
| TRXUSDT | 2022-03-31 00:00:00+00:00 | 2022-04-06 00:00:00+00:00 | 6 | stop_loss | -1.000 | bull |
| LINKUSDT | 2022-04-03 00:00:00+00:00 | 2022-04-11 00:00:00+00:00 | 8 | stop_loss | -1.000 | bull |
| BATUSDT | 2022-04-03 00:00:00+00:00 | 2022-04-08 00:00:00+00:00 | 5 | stop_loss | -1.000 | bull |
| NULSUSDT | 2022-04-04 00:00:00+00:00 | 2022-04-25 00:00:00+00:00 | 21 | signal_exit | -0.430 | bull |
| IOTAUSDT | 2022-04-04 00:00:00+00:00 | 2022-04-06 00:00:00+00:00 | 2 | stop_loss | -1.000 | bull |
| XLMUSDT | 2022-04-04 00:00:00+00:00 | 2022-04-06 00:00:00+00:00 | 2 | stop_loss | -1.000 | bull |
| ADAUSDT | 2022-04-05 00:00:00+00:00 | 2022-04-11 00:00:00+00:00 | 6 | stop_loss | -1.000 | bull |
| TRXUSDT | 2022-05-05 00:00:00+00:00 | 2022-05-11 00:00:00+00:00 | 6 | stop_loss | -1.000 | bear |
| BATUSDT | 2022-07-06 00:00:00+00:00 | 2022-07-26 00:00:00+00:00 | 20 | signal_exit | -0.700 | bear |
| NULSUSDT | 2022-07-09 00:00:00+00:00 | 2022-08-21 00:00:00+00:00 | 43 | signal_exit | -0.447 | bear |
| QTUMUSDT | 2022-07-28 00:00:00+00:00 | 2022-08-26 00:00:00+00:00 | 29 | stop_loss | -1.000 | sideways |
| EOSUSDT | 2022-07-29 00:00:00+00:00 | 2022-09-19 00:00:00+00:00 | 52 | signal_exit | -0.179 | sideways |
| LTCUSDT | 2022-07-29 00:00:00+00:00 | 2022-08-27 00:00:00+00:00 | 29 | signal_exit | -0.876 | sideways |
| XRPUSDT | 2022-07-31 00:00:00+00:00 | 2022-08-20 00:00:00+00:00 | 20 | signal_exit | -0.770 | sideways |
| IOTAUSDT | 2022-08-01 00:00:00+00:00 | 2022-08-27 00:00:00+00:00 | 26 | signal_exit | -0.576 | sideways |
| VETUSDT | 2022-08-02 00:00:00+00:00 | 2022-08-27 00:00:00+00:00 | 25 | signal_exit | -0.640 | sideways |
| ONTUSDT | 2022-08-02 00:00:00+00:00 | 2022-08-27 00:00:00+00:00 | 25 | signal_exit | -0.749 | sideways |
| ICXUSDT | 2022-08-02 00:00:00+00:00 | 2022-08-23 00:00:00+00:00 | 21 | signal_exit | -0.796 | sideways |
| ETHUSDT | 2022-08-06 00:00:00+00:00 | 2022-08-27 00:00:00+00:00 | 21 | signal_exit | -0.643 | sideways |
| OMGUSDT | 2022-08-06 00:00:00+00:00 | 2022-08-20 00:00:00+00:00 | 14 | stop_loss | -1.000 | sideways |
| NEOUSDT | 2022-08-06 00:00:00+00:00 | 2022-08-22 00:00:00+00:00 | 16 | stop_loss | -1.000 | sideways |
| TRXUSDT | 2022-08-06 00:00:00+00:00 | 2022-08-26 00:00:00+00:00 | 20 | stop_loss | -1.000 | sideways |
| DASHUSDT | 2022-08-07 00:00:00+00:00 | 2022-08-27 00:00:00+00:00 | 20 | signal_exit | -0.796 | sideways |
| XMRUSDT | 2022-08-08 00:00:00+00:00 | 2022-08-20 00:00:00+00:00 | 12 | signal_exit | -0.848 | sideways |
| ADAUSDT | 2022-08-08 00:00:00+00:00 | 2022-08-20 00:00:00+00:00 | 12 | signal_exit | -0.821 | sideways |
| ZECUSDT | 2022-08-08 00:00:00+00:00 | 2022-08-27 00:00:00+00:00 | 19 | signal_exit | -0.626 | sideways |
| LINKUSDT | 2022-08-08 00:00:00+00:00 | 2022-08-20 00:00:00+00:00 | 12 | stop_loss | -1.000 | sideways |
| XLMUSDT | 2022-08-09 00:00:00+00:00 | 2022-08-19 00:00:00+00:00 | 10 | stop_loss | -1.000 | sideways |
| BTCUSDT | 2022-08-11 00:00:00+00:00 | 2022-08-20 00:00:00+00:00 | 9 | signal_exit | -0.956 | sideways |
| BATUSDT | 2022-08-11 00:00:00+00:00 | 2022-08-19 00:00:00+00:00 | 8 | stop_loss | -1.000 | sideways |
| XRPUSDT | 2022-09-20 00:00:00+00:00 | 2022-11-09 00:00:00+00:00 | 50 | signal_exit | +0.269 | bear |
| XLMUSDT | 2022-10-09 00:00:00+00:00 | 2022-11-02 00:00:00+00:00 | 24 | signal_exit | -0.773 | bear |
| NULSUSDT | 2022-10-24 00:00:00+00:00 | 2022-11-09 00:00:00+00:00 | 16 | stop_loss | -1.000 | bear |
| LTCUSDT | 2022-11-03 00:00:00+00:00 | 2022-11-09 00:00:00+00:00 | 6 | stop_loss | -1.000 | bull |
| ETHUSDT | 2022-11-05 00:00:00+00:00 | 2022-11-08 00:00:00+00:00 | 3 | stop_loss | -1.000 | bull |
| LINKUSDT | 2022-11-05 00:00:00+00:00 | 2022-11-08 00:00:00+00:00 | 3 | stop_loss | -1.000 | bull |
| BTCUSDT | 2022-11-05 00:00:00+00:00 | 2022-11-08 00:00:00+00:00 | 3 | stop_loss | -1.000 | bull |
| BATUSDT | 2022-11-05 00:00:00+00:00 | 2022-11-08 00:00:00+00:00 | 3 | stop_loss | -1.000 | bull |
| DASHUSDT | 2022-11-05 00:00:00+00:00 | 2022-11-08 00:00:00+00:00 | 3 | stop_loss | -1.000 | bull |
| XMRUSDT | 2022-11-05 00:00:00+00:00 | 2022-11-08 00:00:00+00:00 | 3 | stop_loss | -1.000 | bull |
| VETUSDT | 2022-11-05 00:00:00+00:00 | 2022-11-08 00:00:00+00:00 | 3 | stop_loss | -1.000 | bull |
| ONTUSDT | 2022-11-05 00:00:00+00:00 | 2022-11-08 00:00:00+00:00 | 3 | stop_loss | -1.000 | bull |
| LTCUSDT | 2022-11-23 00:00:00+00:00 | 2022-12-18 00:00:00+00:00 | 25 | signal_exit | -0.383 | bear |
| DASHUSDT | 2022-12-03 00:00:00+00:00 | 2023-03-09 00:00:00+00:00 | 96 | signal_exit | +1.404 | bear |


## 4. Drawdown investigation

Round 1 ran **one independent 1,000 USDT account per coin** (the engine is single-asset), compounding within each account. The 0.5% limit applies to the loss *at the stop at entry*. It does not cap later drawdowns of open profit.

| pair | max_dd_pct (engine metric) | worst_close_to_close_dd_pct | max_position_notional_%_of_account | worst_realised_trade_R |
|---|---|---|---|---|
| BTCUSDT | +5.736 | +5.709 | +20.0 | -1.000 |
| ETHUSDT | +4.697 | +4.235 | +13.9 | -1.000 |
| BNBUSDT | +22.5 | +19.4 | +38.4 | -1.000 |
| XRPUSDT | +4.204 | +3.782 | +9.996 | -1.000 |
| LTCUSDT | +4.722 | +4.655 | +9.562 | -1.000 |
| ADAUSDT | +3.573 | +3.512 | +12.9 | -1.000 |
| EOSUSDT | +5.412 | +5.364 | +5.408 | -1.000 |
| XLMUSDT | +6.052 | +5.988 | +7.321 | -1.000 |
| TRXUSDT | +5.518 | +5.518 | +7.781 | -1.000 |


BNB's largest trade: entered 2020-12-21 at 34.32 with 0.901 BNB = 30.9 USDT, i.e. 3.1% of the 1005 USDT account. Stop 28.92 -> planned risk 5.04 USDT (0.50% of the account, the 0.5% limit). Exited 2021-05-20 at 334.63 (+53.6R, 270 USDT). At the account peak (2021-05-03) the open position was worth 609 USDT = 38% of account equity 1586. The drawdown low (2021-05-19, intraday) was 1229 USDT.

**Reconciliation:** the 22.5% figure is real arithmetic, not a bug. The stop protects only the entry risk. As BNB rose about 10x, a position that was ~3% of the account became the majority of it. Giving back part of that open profit (the 25-day-low exit sits far below the peak) is a drawdown of the account, not a loss of capital. The engine's metric uses intraday lows (`equity_low`). It is a **concentration effect inside a single-coin account.** A single-account portfolio (section 7) enforces the 10% per-position cap only at entry, so the same effect can appear there. See its max drawdown.

### Position overlap across the 9 per-coin accounts (2019-2024)

| index | share_of_days_% |
|---|---|
| 0 | +31.1 |
| 1 | +16.6 |
| 2 | +8.759 |
| 3 | +5.109 |
| 4 | +6.752 |
| 5 | +5.018 |
| 6 | +5.566 |
| 7 | +5.292 |
| 8 | +10.5 |
| 9 | +5.338 |


Aggregate open notional / total equity of the 9 accounts: mean 1.8%, max 9.7%. Up to 9 coins in position at once. A single Conservative account allows at most 3 positions and 30% exposure, so Round 1's per-coin figures overstate what one account could hold. Section 7 is the realistic figure.

## 6. Holdout (16 coins never used before; rules unchanged)

| pair | first_day | last_day | delisted_before_2025 | buy_hold_% | trades | net_R | pf | total_R | closed_at_data_end |
|---|---|---|---|---|---|---|---|---|---|
| ETCUSDT | 2019-01-01 | 2024-12-31 | False | +377.2 | 7 | +4.120 | +9.069 | +28.8 | 0 |
| LINKUSDT | 2019-01-16 | 2024-12-31 | False | +3987.8 | 20 | +1.047 | +3.092 | +20.9 | 0 |
| NEOUSDT | 2019-01-01 | 2024-12-31 | False | +74.7 | 18 | +0.692 | +2.470 | +12.5 | 0 |
| QTUMUSDT | 2019-01-01 | 2024-12-31 | False | +35.0 | 18 | +0.395 | +1.950 | +7.116 | 0 |
| IOTAUSDT | 2019-01-01 | 2024-12-31 | False | -22.2 | 17 | +0.529 | +2.335 | +8.986 | 0 |
| ONTUSDT | 2019-01-01 | 2024-12-31 | False | -61.3 | 21 | -0.048 | +0.906 | -1.010 | 0 |
| ICXUSDT | 2019-01-01 | 2024-12-31 | False | -25.2 | 18 | +0.880 | +3.095 | +15.8 | 0 |
| WAVESUSDT | 2019-01-18 | 2024-06-17 | True | -57.4 | 14 | +1.059 | +4.743 | +14.8 | 0 |
| BATUSDT | 2019-03-04 | 2024-12-31 | False | +39.1 | 20 | +0.107 | +1.180 | +2.146 | 0 |
| VETUSDT | 2019-01-01 | 2024-12-31 | False | +956.8 | 20 | +1.636 | +4.027 | +32.7 | 0 |
| ZILUSDT | 2019-02-19 | 2024-12-31 | False | +7.709 | 15 | +2.453 | +6.678 | +36.8 | 0 |
| NULSUSDT | 2019-01-01 | 2024-12-31 | False | -1.101 | 18 | +0.685 | +2.862 | +12.3 | 0 |
| ZECUSDT | 2019-03-21 | 2024-12-31 | False | +2.049 | 17 | +0.229 | +1.503 | +3.898 | 1 |
| DASHUSDT | 2019-03-28 | 2024-12-31 | False | -60.4 | 17 | +0.117 | +1.251 | +1.993 | 0 |
| XMRUSDT | 2019-03-15 | 2024-02-20 | True | +127.8 | 19 | -0.023 | +0.934 | -0.443 | 0 |
| OMGUSDT | 2019-04-03 | 2024-06-17 | True | -80.7 | 18 | +0.475 | +1.845 | +8.553 | 0 |


Collective: 277 trades, net +0.744R (95% lower bound +0.365R), PF 2.63, median -0.279R; positive on 13 of 15 coins with >= 10 trades. Delisted coins are included up to their last trading day; a position open at delisting is closed at the last close (real exits around delisting can be worse).

## 5. Matched random-entry benchmark

1000 independent runs per sample (seed 2026). Per coin, the random entry probability is calibrated (3 x 30 pilot runs, separate seeds) so the expected number of random trades matches T50's trades on that coin. Everything else is identical: next-open fills, the 25-day-low exit, the 3-ATR stop, sizing, costs and the cost filter. Entries are drawn from a seeded generator that never sees prices, so random entries cannot use future information.

| sample | T50_net_R | T50_trades | T50_exposure_pct | random_runs | random_mean_R | random_p5 | random_median | random_p95 | random_p99 | random_max | random_mean_trades | random_mean_exposure_pct | T50_percentile | empirical_p_value | T50_pnl | random_mean_pnl | pnl_percentile |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| main 2019-24 | +1.358 | 169 | +33.8 | 1000 | +0.569 | +0.264 | +0.549 | +0.951 | +1.165 | +1.437 | +172.7 | +25.5 | +99.9 | +0.002 | +1191.7 | +494.4 | +100.0 |
| holdout 2019-24 | +0.744 | 277 | +31.5 | 1000 | +0.213 | +0.045 | +0.211 | +0.395 | +0.487 | +0.608 | +280.3 | +21.2 | +100.0 | +0.001 | +1045.2 | +297.1 | +100.0 |


Figures: `docs/research/fig/t50_random_main.png`, `docs/research/fig/t50_random_holdout.png`.

## 7. Economic viability: one account, Conservative limits enforced across coins

**The configured 10% drawdown halt is permanent (kill switch); when it triggers, no new trades are taken for the rest of the period.** The `INFO ONLY` rows switch it off to show the rule's own economics; they are not a proposal to change the limit.

10,000 USDT, at most 3 positions, 30% total exposure, 10% per position, 0.5% risk per trade, 3% daily loss limit, 10% drawdown halt (permanent, as the kill switch). Signals taken in fixed universe order when slots are short. `passive_same_exposure` = equal-weight buy-and-hold basket holding T50's average exposure, rest in cash.

| portfolio | costs | final_equity | total_return_pct | cagr_pct | max_dd_pct | sharpe | avg_exposure_pct | max_exposure_pct | time_in_market_pct | avg_open_positions | trades | net_R | pf | halted_on | passive_same_exposure_return_% | passive_same_exposure_max_dd_% | buy_hold_return_% | buy_hold_max_dd_% |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| main 9 coins | base 0.20% fee, 0.05% slip, 0.20% stop slip | +15036.1 | +50.4 | +7.036 | +14.0 | +0.827 | +3.535 | +29.5 | +27.7 | +0.698 | 31 | +2.819 | +9.246 | 2021-05-19 | +79.5 | +35.9 | +2249.9 | +73.3 |
| main 9 coins | Binance-like 0.10% fee | +15138.7 | +51.4 | +7.157 | +14.1 | +0.833 | +3.573 | +29.9 | +27.7 | +0.698 | 31 | +2.868 | +9.416 | 2021-05-19 | +80.4 | +36.1 | +2249.9 | +73.3 |
| main 9 coins | 0.30% fee, 0.10% slip, 0.30% stop slip | +14890.8 | +48.9 | +6.863 | +13.9 | +0.818 | +3.481 | +29.0 | +27.7 | +0.698 | 31 | +2.750 | +9.001 | 2021-05-19 | +78.3 | +35.6 | +2249.9 | +73.3 |
| main 9 coins | 0.40% fee, 0.20% slip, 0.50% stop slip | +14708.4 | +47.1 | +6.643 | +13.8 | +0.806 | +3.413 | +28.3 | +27.7 | +0.698 | 31 | +2.661 | +8.692 | 2021-05-19 | +76.8 | +35.2 | +2249.9 | +73.3 |
| main 9 coins | base costs, INFO ONLY: no 10% drawdown halt | +17035.4 | +70.4 | +9.287 | +17.9 | +0.857 | +8.267 | +32.8 | +68.4 | +1.616 | 83 | +1.372 | +4.106 | - | +186.0 | +51.5 | +2249.9 | +73.3 |
| main 9 coins DEV | base 0.20% fee, 0.05% slip, 0.20% stop slip | +15036.1 | +50.4 | +10.7 | +14.0 | +1.014 | +5.304 | +29.5 | +41.6 | +1.047 | 31 | +2.819 | +9.246 | 2021-05-19 | +34.1 | +43.6 | +643.0 | +73.3 |
| main 9 coins DEV | base costs, INFO ONLY: no 10% drawdown halt | +14448.2 | +44.5 | +9.643 | +17.4 | +0.899 | +6.533 | +29.5 | +60.7 | +1.433 | 50 | +1.590 | +4.309 | - | +42.0 | +47.4 | +643.0 | +73.3 |
| main 9 coins VAL | base 0.20% fee, 0.05% slip, 0.20% stop slip | +11790.7 | +17.9 | +8.591 | +10.5 | +0.777 | +11.7 | +32.8 | +83.9 | +1.982 | 33 | +1.041 | +3.810 | 2024-12-19 | +30.5 | +9.006 | +259.9 | +35.5 |
| main 9 coins VAL | base costs, INFO ONLY: no 10% drawdown halt | +11790.7 | +17.9 | +8.591 | +10.5 | +0.777 | +11.7 | +32.8 | +83.9 | +1.982 | 33 | +1.041 | +3.810 | - | +30.5 | +9.006 | +259.9 | +35.5 |
| holdout 16 coins | base 0.20% fee, 0.05% slip, 0.20% stop slip | +12903.3 | +29.0 | +4.341 | +10.0 | +0.644 | +3.930 | +25.2 | +52.3 | +1.236 | 68 | +0.797 | +2.744 | 2023-08-16 | +13.0 | +39.3 | +331.2 | +90.9 |
| holdout 16 coins | Binance-like 0.10% fee | +13326.7 | +33.3 | +4.904 | +10.2 | +0.711 | +4.155 | +25.3 | +55.9 | +1.273 | 69 | +0.880 | +3.000 | 2023-10-02 | +13.8 | +40.6 | +331.2 | +90.9 |
| holdout 16 coins | 0.30% fee, 0.10% slip, 0.30% stop slip | +12812.4 | +28.1 | +4.218 | +10.2 | +0.632 | +3.879 | +25.1 | +52.3 | +1.234 | 68 | +0.775 | +2.687 | 2023-08-15 | +12.8 | +39.0 | +331.2 | +90.9 |
| holdout 16 coins | 0.40% fee, 0.20% slip, 0.50% stop slip | +12750.3 | +27.5 | +4.134 | +10.0 | +0.627 | +3.811 | +24.9 | +51.9 | +1.230 | 67 | +0.771 | +2.699 | 2023-07-31 | +12.6 | +38.6 | +331.2 | +90.9 |
| holdout 16 coins | base costs, INFO ONLY: no 10% drawdown halt | +13204.1 | +32.0 | +4.743 | +11.4 | +0.642 | +5.206 | +25.2 | +70.0 | +1.602 | 91 | +0.649 | +2.352 | - | +17.2 | +45.9 | +331.2 | +90.9 |


Consistency check (BTC, DEV, single coin, halts off): portfolio simulator 12 trades, mean +2.568R, final equity 1159.0; engine 12 trades, mean +2.568R, final equity 1159.0.

Figure: `docs/research/fig/t50_portfolio.png`.
