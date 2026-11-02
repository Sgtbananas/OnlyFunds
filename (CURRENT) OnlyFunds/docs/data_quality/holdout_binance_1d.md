# Historical data quality report

Generated 2026-11-01T11:20:28+00:00 by `scripts/fetch_history.py`.

- Source: **Binance spot** public klines API (`https://api.binance.com/api/v3/klines`), interval `1d`, requested 2019-01-01 to 2025-01-01 (exclusive).
- Candles are stored exactly as returned. **Missing candles are reported, never filled.**
- Each CSV has a manifest with its SHA-256; `core.history.load_dataset` refuses a CSV that doesn't match.
- Universe = large USDT pairs as of Jan 2019 (start of data), including later laggards and one delisted coin (EOS), to limit survivorship bias. BCH was excluded: its Binance history breaks at the 2019 BCHABC→BCH symbol change.
- **This is Binance data. It does not establish CoinEx prices, liquidity, fees or execution.**

| symbol | rows | first candle | last candle | expected | missing | gaps | largest gap (h) | zero-vol | moves >25%/h | max |log ret| | hard problems | sha256 (12) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ETCUSDT | 2192 | 2019-01-01 00:00:00+00:00 | 2024-12-31 00:00:00+00:00 | 2192 | 0 | 0 | 0 | 0 | 17 | 0.573 | none | 71c96f6bb5b3 |
| LINKUSDT | 2177 | 2019-01-16 00:00:00+00:00 | 2024-12-31 00:00:00+00:00 | 2177 | 0 | 0 | 0 | 0 | 10 | 0.650 | none | 2a19715f2f6b |
| NEOUSDT | 2192 | 2019-01-01 00:00:00+00:00 | 2024-12-31 00:00:00+00:00 | 2192 | 0 | 0 | 0 | 0 | 13 | 0.502 | none | 8585bc5f171f |
| QTUMUSDT | 2192 | 2019-01-01 00:00:00+00:00 | 2024-12-31 00:00:00+00:00 | 2192 | 0 | 0 | 0 | 0 | 19 | 0.626 | none | f7d6062cd9c0 |
| IOTAUSDT | 2192 | 2019-01-01 00:00:00+00:00 | 2024-12-31 00:00:00+00:00 | 2192 | 0 | 0 | 0 | 0 | 11 | 0.586 | none | 38d30446c670 |
| ONTUSDT | 2192 | 2019-01-01 00:00:00+00:00 | 2024-12-31 00:00:00+00:00 | 2192 | 0 | 0 | 0 | 0 | 14 | 0.635 | none | c9ce70b9c1cb |
| ICXUSDT | 2192 | 2019-01-01 00:00:00+00:00 | 2024-12-31 00:00:00+00:00 | 2192 | 0 | 0 | 0 | 0 | 11 | 0.601 | none | f21633118fad |
| WAVESUSDT | 1978 | 2019-01-18 00:00:00+00:00 | 2024-06-17 00:00:00+00:00 | 1978 | 0 | 0 | 0 | 0 | 24 | 0.533 | none | 41a03ce7ebb3 |
| BATUSDT | 2130 | 2019-03-04 00:00:00+00:00 | 2024-12-31 00:00:00+00:00 | 2130 | 0 | 0 | 0 | 0 | 13 | 0.596 | none | b6f048921332 |
| VETUSDT | 2192 | 2019-01-01 00:00:00+00:00 | 2024-12-31 00:00:00+00:00 | 2192 | 0 | 0 | 0 | 0 | 14 | 0.670 | none | a4f508e7e086 |
| ZILUSDT | 2143 | 2019-02-19 00:00:00+00:00 | 2024-12-31 00:00:00+00:00 | 2143 | 0 | 0 | 0 | 0 | 21 | 0.691 | none | 682936684f1a |
| NULSUSDT | 2192 | 2019-01-01 00:00:00+00:00 | 2024-12-31 00:00:00+00:00 | 2192 | 0 | 0 | 0 | 0 | 23 | 0.748 | none | 8260361d0eac |
| ZECUSDT | 2113 | 2019-03-21 00:00:00+00:00 | 2024-12-31 00:00:00+00:00 | 2113 | 0 | 0 | 0 | 0 | 10 | 0.556 | none | 155cf6e1df6b |
| DASHUSDT | 2106 | 2019-03-28 00:00:00+00:00 | 2024-12-31 00:00:00+00:00 | 2106 | 0 | 0 | 0 | 0 | 11 | 0.487 | none | 8084a75f7caf |
| XMRUSDT | 1804 | 2019-03-15 00:00:00+00:00 | 2024-02-20 00:00:00+00:00 | 1804 | 0 | 0 | 0 | 0 | 8 | 0.547 | none | 8486cd91d69d |
| OMGUSDT | 1903 | 2019-04-03 00:00:00+00:00 | 2024-06-17 00:00:00+00:00 | 1903 | 0 | 0 | 0 | 0 | 23 | 0.614 | none | b3858bc5d740 |

## Largest gaps per symbol

- **ETCUSDT**: none
- **LINKUSDT**: none
- **NEOUSDT**: none
- **QTUMUSDT**: none
- **IOTAUSDT**: none
- **ONTUSDT**: none
- **ICXUSDT**: none
- **WAVESUSDT**: none
- **BATUSDT**: none
- **VETUSDT**: none
- **ZILUSDT**: none
- **NULSUSDT**: none
- **ZECUSDT**: none
- **DASHUSDT**: none
- **XMRUSDT**: none
- **OMGUSDT**: none
