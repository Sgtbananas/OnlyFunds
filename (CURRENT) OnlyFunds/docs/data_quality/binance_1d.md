# Historical data quality report

Generated 2026-11-01T06:28:44+00:00 by `scripts/fetch_history.py`.

- Source: **Binance spot** public klines API (`https://api.binance.com/api/v3/klines`), interval `1d`, requested 2019-01-01 to 2026-11-01 (exclusive).
- Candles are stored exactly as returned. **Missing candles are reported, never filled.**
- Each CSV has a manifest with its SHA-256; `core.history.load_dataset` refuses a CSV that doesn't match.
- Universe = large USDT pairs as of Jan 2019 (start of data), including later laggards and one delisted coin (EOS), to limit survivorship bias. BCH was excluded: its Binance history breaks at the 2019 BCHABC→BCH symbol change.
- **This is Binance data. It does not establish CoinEx prices, liquidity, fees or execution.**

| symbol | rows | first candle | last candle | expected | missing | gaps | largest gap (h) | zero-vol | moves >25%/h | max |log ret| | hard problems | sha256 (12) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| BTCUSDT | 2861 | 2019-01-01 00:00:00+00:00 | 2026-10-31 00:00:00+00:00 | 2861 | 0 | 0 | 0 | 0 | 1 | 0.503 | none | b1abd2349d9f |
| ETHUSDT | 2861 | 2019-01-01 00:00:00+00:00 | 2026-10-31 00:00:00+00:00 | 2861 | 0 | 0 | 0 | 0 | 4 | 0.591 | none | 3425917968cf |
| BNBUSDT | 2861 | 2019-01-01 00:00:00+00:00 | 2026-10-31 00:00:00+00:00 | 2861 | 0 | 0 | 0 | 0 | 8 | 0.582 | none | 4e4a532df555 |
| XRPUSDT | 2861 | 2019-01-01 00:00:00+00:00 | 2026-10-31 00:00:00+00:00 | 2861 | 0 | 0 | 0 | 0 | 19 | 0.549 | none | a68cbc01808e |
| LTCUSDT | 2861 | 2019-01-01 00:00:00+00:00 | 2026-10-31 00:00:00+00:00 | 2861 | 0 | 0 | 0 | 0 | 8 | 0.487 | none | 84ef00d87a64 |
| ADAUSDT | 2861 | 2019-01-01 00:00:00+00:00 | 2026-10-31 00:00:00+00:00 | 2861 | 0 | 0 | 0 | 0 | 7 | 0.543 | none | 07768ee72724 |
| EOSUSDT | 2338 | 2019-01-01 00:00:00+00:00 | 2025-05-26 00:00:00+00:00 | 2338 | 0 | 0 | 0 | 0 | 14 | 0.552 | none | 2a0e601c10bb |
| XLMUSDT | 2861 | 2019-01-01 00:00:00+00:00 | 2026-10-31 00:00:00+00:00 | 2861 | 0 | 0 | 0 | 0 | 16 | 0.565 | none | 4b69c15d46ee |
| TRXUSDT | 2861 | 2019-01-01 00:00:00+00:00 | 2026-10-31 00:00:00+00:00 | 2861 | 0 | 0 | 0 | 0 | 8 | 0.673 | none | ecc38252494f |

## Largest gaps per symbol

- **BTCUSDT**: none
- **ETHUSDT**: none
- **BNBUSDT**: none
- **XRPUSDT**: none
- **LTCUSDT**: none
- **ADAUSDT**: none
- **EOSUSDT**: none
- **XLMUSDT**: none
- **TRXUSDT**: none
