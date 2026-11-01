# OF2 audit and completion report (2026-11-01, branch `of2-completion`)

## 1. Starting state (audit of `main` @ 5a19aaa)

The build plan marked Phase 1 complete and Phase 2 in progress. The code did not support that.
Every finding below was verified by reading or running the code.

### Critical: the trading pipeline did not work
| # | Finding | Where |
|---|---|---|
| C1 | `add_indicators` computed **no** EMA/RSI/MACD/Bollinger. It set `ATR` = rolling std of close, and invented a fake `Close` column if missing. | `core/core_data.py` |
| C2 | `fetch_klines` never fetched market data; it read `data/{pair}_{interval}_{limit}.csv`. The only file was a 15-row hand-made series (prices 100→114). | `core/core_data.py`, `data/BTCUSDT_15m_1000.csv` |
| C3 | Signals depended on `rsi`/`macd` columns that never existed, so the reversion strategy was always 0 and backtests made **zero trades**. | `core/core_signals.py` |
| C4 | Backtester called `risk_manager.position_size(...)` on the *module*. The function only exists as a method, so any entry would raise `AttributeError`. | `core/backtester.py` |
| C5 | `main.py`: `main_loop()` was never called. It used undefined `place_order`, `signal`, `atomic_save_json`, `BACKTEST_RESULTS_FILE` and `RETRAIN_TRADE.interval`. The backtest block was duplicated, and the second copy used an undefined `threshold_used`. `st.stop()` sat inside the except handler. | `main.py` |
| C6 | "Live" orders were `random.choice([True, False])` with a fake order id. Cancel always returned `True`. | `core/trade_execution.py` |
| C7 | `run_live.py` built a real ccxt CoinEx client, and its `place_limit` placed **real limit orders**. It also referenced undefined `risk_manager` and `get_current_price`. | `run_live.py` |
| C8 | `scripts/backtest_suite.py` wrote a hard-coded `DUMMY` result (+10 profit) "so the UI always displays something". | `scripts/backtest_suite.py` |
| C9 | The meta-model was a `DummyClassifier` trained on `np.random.rand`, written to disk when no model existed. | `main.py` |

### Major
| # | Finding |
|---|---|
| M1 | All CoinEx calls used the **v1** API (`/v1/market/list`, `/v1/market/ticker/all`). No authenticated code existed (no signing, balance, order, cancel or status). |
| M2 | No look-ahead protection: entries filled at the signal bar's own close, and the stop and take-profit were derived from the fill price *plus fee*. |
| M3 | Strategy selection (`meta_learner.select_strategy`) ranked strategies by historical PnL/win-rate features, i.e. selection by past profitability. The backtester ignored the result anyway. |
| M4 | ML training labels came from the trade log only (SL vs non-SL), with no validation split and features normalised on the full dataset. `ml_confidence` returned `1.0` (always pass) when no model existed. |
| M5 | `RiskManager.check_daily_loss` compounded `return_pct` *forward* from current equity, so the logic was inverted. There was no exposure cap, no kill switch, and no duplicate-order protection. |
| M6 | `core/core__init__.py` was misnamed and imported non-existent modules (`trade_executor`, `track_trade_result`). |
| M7 | `core/atr_tuner.py` crashed on import under Python 3.9 (`RandomForestRegressor | None`). |
| M8 | `scripts/auto_tuner.py`, `Optuna_tune.py`, `walk_forward_backtest.py` and `core/walkforward_optuna.py` called `run_backtest`/`add_indicators` with signatures that never existed, or imported missing functions. |
| M9 | The config held `api_key`/`api_secret` placeholders. `run_live.py` read `COINEX_API_SECRET` while `utils/config.py` read `COINEX_SECRET_KEY`. |
| M10 | Prometheus server started on every Streamlit rerun, and the background thread wrote `st.session_state` from outside the script thread. |

### Minor / dead code
`sentiment.py` returned `random.uniform`, `strategy_arb.py` was a stub, and `market_maker.py`, `scalper.py`, `arbitrage.py`, grid modules and `ws_feed.py` (v1 websocket) were never wired in. There was a large amount of `st.write('DEBUG …')` output and duplicate imports. Trading windows existed in the config but were never enforced.

## 2. What was done

### Created
- `core/indicators.py`: causal EMA, Wilder RSI/ATR, MACD, Bollinger (+%B, width), trend score, volatility.
- `core/coinex_client.py`: CoinEx **API v2** from the official docs. Covers signing (HMAC-SHA256 over method+path+query+body+timestamp), markets, tickers, klines (unclosed candle dropped), depth, balance, order, cancel and status. It has a retry/backoff for 3008/4001/4002/4213/5xx, maps errors to classes, and handles precision and `min_amount`. Orders are locked, never auto-retried, and a network failure raises `OrderStateUnknown`. Secrets never reach logs or `repr`.
- `core/positions.py`: shared exit rules. The stop is checked before TP, gaps fill at the open, and the trailing stop ratchets only after a bar survives.
- `core/paper_trader.py`: persistent atomic ledger with one decision per completed bar. It handles failed-order recording and exit retry, emergency stop, reconciliation, a cross-process lock, a stale-data guard, trading window and strategy rotation.
- `run_paper.py`: headless paper loop with optional Prometheus metrics.
- `tests/`: unit, integration and UI tests (see §4). `tests/synthetic.py` provides seeded data.
- `docs/AUDIT_REPORT.md`, `docs/LIVE_TRADING_CHECKLIST.md`, `.env.example`, `requirements-dev.txt`, `pytest.ini`, `legacy/README.md`.

### Rewritten / repaired
- `core/core_data.py`: real CoinEx candles, a cache that accumulates history, a cache fallback labelled as such, and a CSV importer.
- `core/core_signals.py`: five components (EMA cross, RSI reversion, MACD momentum, Bollinger reversion, trend score), regime detection (trending up/down, ranging, high-volatility) and three stacks. `auto` selects by regime and stands aside in down-trends. Entries need N confirming components. Confidence is |signal| × agreement. Entry and exit are mutually exclusive.
- `core/backtester.py`: next-bar-open execution, fees + adverse slippage on every fill, and risk-based sizing. It reports realised and unrealised PnL, win rate, profit factor, drawdown, Sharpe, exposure, fees and a buy-and-hold benchmark, plus an out-of-sample split.
- `core/risk_manager.py`: risk-per-trade sizing capped by max position %, remaining exposure and cash. It floors to exchange precision and rejects orders below `min_notional`/`min_amount`, never rounding them up. It enforces daily loss, max drawdown, max positions, duplicate-pair checks and the kill switch.
- `core/trade_execution.py`: `LIVE_TRADING_LOCKED = True` and `PaperExecutor`, which applies slippage, fees, precision checks, rejections, optional simulated failures and duplicate `client_id` protection.
- `core/meta_learner.py`: regime-based selection, plus `StrategyRotation` that only *benches* a strategy whose recent trades have a profit factor below 0.8.
- `core/ml_filter.py`: triple-barrier labels using the same ATR exits, causal scale-free features, and a chronological per-pair split with a purge gap of `horizon` bars. The report includes an explicit `useful` flag (validation AUC ≥ 0.55 **and** precision above base rate). The filter is versioned and saved atomically.
- `core/autotune.py`: grid optimisation on the training segment only, a single out-of-sample evaluation, a `confirmed` flag (OOS profitable, beats buy-and-hold, enough trades), walk-forward, and an ML on/off OOS comparison.
- `main.py`: new Streamlit app with Market, Backtest, Paper trading, ML & optimisation, and Risk & status tabs.
- `utils/config.py`: profiles, settings builder, atomic save that refuses secrets and non-paper mode. `utils/helpers.py` uses v2 tickers.
- `config/config.yaml`: restructured with profiles, risk, exits, signals, indicators, ml and trading windows. No secrets.
- `scripts/*`: rebuilt on the new engine; no dummy output.
- `run_live.py`: refuses to run. `START.bat`, supervisor config, `requirements.txt` (unused heavy dependencies removed), `.gitignore` (adds `.env`, cache).

### Moved to `legacy/` (kept, not imported)
`arbitrage, market_maker, scalper, strategy_arb, sentiment, grid_trader, strategy_grid, ws_feed, walkforward_optuna, atr_tuner, autotune_checker` (reasons in `legacy/README.md`).

### Deleted
`data/BTCUSDT_15m_1000.csv` (fake 15-row data). `state/heartbeat_A.json` was untracked (runtime state is git-ignored).

## 3. Design decisions
- **Spot long-only.** A negative signal means exit or stay flat. In `auto`, the trending-down regime stands aside.
- **Build plan "1 USDT or 5% of profit" sizing** is implemented as risk-based sizing with a 1 USDT minimum notional and percentage caps. "5% of profit" is ambiguous and was not implemented literally.
- **Stacked strategy** = trend filter + pullback (trend 0.4, EMA 0.2, RSI 0.2, Bollinger 0.2). An equal-weight blend of all five never fired, because the trend and reversion components cancel out.
- **Every exit applies adverse slippage**, including take-profit, so backtest and paper results match.
- **The ML filter is off by default.** On random-walk data it correctly reports no edge (tested). Enable it only when its report on real data says `useful: true`.

## 4. Verification
See the PR description for the exact test run. Commands:

```bat
python -m pytest                     # full suite
python -m pytest tests/test_app.py   # Streamlit UI (AppTest, exchange faked)
streamlit run main.py                # manual
```

Tests cover:
- indicator maths against independent reference implementations
- causality: tampering with future bars leaves past indicators, signals and trades unchanged
- backtest accounting identity and a hand-computed single trade
- fee/slippage monotonicity and reproducibility
- risk sizing caps and halts
- CoinEx signing, URL/body shape, error mapping, retries, no order retry, locked trading, precision and decimal formatting
- paper trading over thousands of simulated bars with reconciliation, failed orders, restart persistence, corrupt-ledger quarantine, emergency stop, the lock and stale data
- ML labels, purge gap, no false edge on random data
- optimisation train/test separation, config safety
- every UI tab and button, and the CLIs

## 5. Known limitations
- **CoinEx was unreachable from the development machine.** Every `api.coinex.com` request, including `/v2/time`, returned HTTP 503 from the gateway, while other sites worked. The client is therefore verified against the official documentation with mocked HTTP, **not** against live responses. Run `Risk & status → Test CoinEx connection` or `python scripts/backtest_suite.py` where CoinEx is reachable.
- No strategy has demonstrated profitability. On synthetic random-walk data every strategy loses roughly its fees, which is the correct result. Real-data walk-forward results are still needed.
- Max history per request is 1000 candles; longer history builds up in `data/cache/` over repeated fetches.
- Paper fills are simulated against the last price with fixed slippage. Order-book depth is not modelled.
- Signal computation takes about 60 ms per pair per step. That is fine for minute-level paper trading, but a 200-pair loop takes about 12 s per step.
