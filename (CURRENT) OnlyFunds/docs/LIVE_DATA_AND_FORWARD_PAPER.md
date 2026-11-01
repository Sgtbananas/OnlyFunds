# Live market data and forward paper trading

Research and paper capability only. **No real orders, no API keys, live trading stays locked**
(`core/trade_execution.py: LIVE_TRADING_LOCKED`). Data comes from **Binance public endpoints**, the same
source as the historical datasets. CoinEx remains the deferred final integration.

## Run
```bat
python run_forward_paper.py --symbols BTCUSDT ETHUSDT --interval 1h --every 60 --websocket
python run_forward_paper.py --interval 1h --report
```
State is kept per interval in `state/forward/<interval>/` (git-ignored):

| File | Content |
|---|---|
| `observations/YYYY-MM-DD.jsonl` | one line per symbol per **completed** candle: the candle, bid/ask/spread/depth snapshot (exchange source, UTC times), regime and confidence, every strategy's signal, and the selector's decision and reason |
| `paper/paper_ledger.json` (+ `.bak`) | simulated trades, positions and decisions (accepted **and** rejected, with reasons) |
| `shadow.json` | every entry opportunity OF2 did **not** take, with the reason and its hypothetical outcome |
| `feed_health.jsonl` | feed reliability per step: WS reconnects, malformed messages, stale quotes, REST retries and rate limits, candle gaps and backfills |
| `engine_state.json` | last observed candle per symbol, so a restart neither duplicates nor skips anything |

## Live data (`core/live_data.py`)
- **REST:** klines (completed only), best bid/ask, order book (1,000 levels), 24h stats.
  - Weight-aware throttling using `X-MBX-USED-WEIGHT-1M`.
  - `Retry-After` honoured on 429/418; retries on 5xx and network errors.
- **WebSocket:** kline (`x == true` only) and bookTicker streams, with silence detection (30 s) and exponential back-off reconnect.
- **Quotes:** every quote carries its receive time. `LiveMarket.quote()` raises `StaleDataError` past `max_quote_age_s`, so a stale price is never treated as current.
- **Candles:** malformed, crossed, misaligned or out-of-order candles are rejected. Gaps are detected and backfilled **only** with real exchange candles.
- **Depth:** depth within 10 and 50 bps of mid is reported, flagged as a lower bound when the fetched book doesn't reach the band.
- **Consistency:** a completed candle produces the same indicators and signals whether it came from the historical files or the live buffer (`tests/test_live_data.py::test_same_completed_candles_give_same_signals_in_both_pipelines`).

**Verified live on 2026-11-01:**
- **WebSocket:** BTC/ETH 1m, 4,825 messages in 75 s, 0 reconnects, 0 malformed, new completed candles accepted.
- **Forward smoke run:** 4 coins × 13 steps with 0 errors.

## Forward paper rules
- **Fills:** buys at the **ask** and sells at the **bid**, plus slippage (stops get wider slippage).
- **Costs:** fees at the configured rate; the cost-to-risk filter applies.
- **Gaps:** gap-through-stop fills at the open, and replayed exits never fill better than the current price.
- **Inherited protections:** risk limits, kill switch, persistent ledger with backup, restart recovery, duplicate-order protection, and the stale-data and missing-candle guards.
- **Entries need a strategy approved by a v2 validation run** (`core/selection.py`). **Today none is approved, so
  OF2 observes and records but does not trade.** If the regime is uncertain, or no approval covers it, the answer is
  also no trade.
- **Rejected opportunities** (any strategy, any reason) go to the shadow tracker. It resolves them with the same
  exit and cost rules on later completed candles, which shows whether OF2 is declining poor trades or good ones.

## Monitoring (`core/monitoring.py`)
The monitor reports these as **separate** sections, never one combined number:
- v2 validation runs
- forward paper trades (R expectancy, profit factor, drawdown, win rate, average win/loss, fees, frequency; by pair, regime and strategy)
- rejection counts and reasons
- shadow outcomes (by strategy, regime, pair and rejection reason)
- execution quality (fill vs ask)
- feed reliability

The historical backtest stays in `docs/REAL_MARKET_EVALUATION.md` and is never mixed in.

## Limitations
- **Binance ≠ CoinEx.** Spreads, depth, fees and fills on CoinEx are unverified.
- **Simulated fills.** Paper fills are top-of-book plus a fixed slippage, which isn't a depth-walking simulation. At 1,000 USDT sizes the book depth (millions of USD within 10 bps on BTC) makes this a small effect.
- **Too early to judge.** Forward and shadow samples are tiny so far. Minimum sample sizes apply before any conclusion: the validation rules require ≥ 100 trades.
- **The regime detector has no predictive value** for future direction (`docs/REGIME_STUDY.md`). It's used only as a no-trade condition and for monitoring.
