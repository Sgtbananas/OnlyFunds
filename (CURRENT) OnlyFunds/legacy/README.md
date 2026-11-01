# Legacy modules (not part of the OF2 execution path)

These files were in `core/` / `scripts/` before the OF2 completion work. None of
them was imported by the app, and several were placeholders. They are kept
for reference instead of deleted. Nothing in `main.py`, `run_paper.py` or
`core/` imports them.

| File | Why it is here |
|---|---|
| `sentiment.py` | Returned `random.uniform(-0.5, 0.5)`: a fake signal. |
| `strategy_arb.py` | Stub that always returned an empty DataFrame. |
| `arbitrage.py` | Single hard-coded USDT→BTC→ETH cycle, never wired in. |
| `market_maker.py` | Blocking loop with `time.sleep`; would place real orders when `dry_run=False`. |
| `scalper.py` | Async loop calling functions that don't exist in the app. |
| `grid_trader.py`, `strategy_grid.py` | Grid strategies, never integrated with risk or execution. |
| `ws_feed.py` | CoinEx **v1** websocket protocol (`depth.subscribe`); v2 uses a different endpoint and auth. |
| `walkforward_optuna.py` | Called `run_backtest` / `add_indicators` with signatures that never existed. Replaced by `core/autotune.py`. |
| `atr_tuner.py` | Crashed on import on Python 3.9 (`X | None` annotations); no labelled training data source existed. Import fixed, still unintegrated. |
| `autotune_checker.py` | Parsed log lines ("Entered LONG at …") that no code ever wrote. |

To integrate any of these properly, route orders through `core.trade_execution`
and sizing through `core.risk_manager`, and add tests.
