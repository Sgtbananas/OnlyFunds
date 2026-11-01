# OF2 development checklist (revised priorities, 2026-11-01)

CoinEx connectivity is deliberately **last**. Exchange outages (HTTP 503) must not block anything above
it. Live trading stays disabled throughout. ✅ = done and covered by tests; ⏳ = open.

## 1. Trading engine
- ✅ Causal indicators (EMA, Wilder RSI/ATR, MACD, Bollinger, trend score) checked against reference formulas
- ✅ Regime detection, strategy stacks, confirmation, confidence; entry/exit mutually exclusive
- ✅ Look-ahead tests: changing future bars never changes past signals or trades
- ✅ Signal assembly vectorised; identical output to the previous version on 64 dataset/strategy cases

## 2. Backtesting
- ✅ Next-bar-open execution; gap fills at the open; stop checked before take-profit in the same candle
- ✅ Fees on both legs; normal slippage; wider stop slippage (default 0.2%)
- ✅ Liquidity cap: at most 10% of the signal candle's volume
- ✅ Drawdown measured at intrabar lows; buy-and-hold over the same window net of the same costs
- ✅ Expectancy, Sortino, trades/day, per-regime results, missing-candle count
- ✅ Out-of-sample split and walk-forward
- ⏳ Partial fills against real order-book depth (not modelled; only the volume cap)

## 3. Risk management
- ✅ Sizing budgets fees and stop slippage, so a stop-out loses at most `risk_per_trade` (gaps excepted)
- ✅ Position / exposure / cash / liquidity / exchange-minimum caps; verified with 3 simultaneous positions
- ✅ Daily loss limit, drawdown halt, kill switch, duplicate-position and duplicate-order protection
- ✅ Conservative defaults (Conservative profile, 0.5% risk, 10% per position, 30% exposure)

## 4. ML and optimisation
- ✅ Triple-barrier labels and purged chronological split
- ✅ Usefulness judged on **expectancy after costs**, not win rate
- ✅ Train-only optimisation with OOS confirmation; walk-forward; overfitting check
- ✅ Strategy evaluation harness (`scripts/evaluate_strategies.py`, UI "Compare strategies")
- ✅ Real-market validation on Binance 1h data, 9 coins, 2019–2026, under a pre-registered protocol
  (`docs/REAL_MARKET_EVALUATION.md`). **Result: no credible edge for any strategy, with or without ML.**
- ⏳ New strategy research, judged by the same pipeline on a fresh untouched test window

## 5. Paper trading and ledger
- ✅ Atomic, fsync'd ledger with a last-good backup; a corrupt ledger is never silently replaced
- ✅ Reconciliation every step; any inconsistency blocks new entries
- ✅ Candle replay includes the entry candle; replayed exits never fill better than the current price
- ✅ Stale-data and missing-candle guards; network interruption recovery
- ✅ Cross-process lock with a heartbeat (app + `run_paper.py` together)
- ✅ Emergency stop persists across restarts

## 6. Streamlit interface
- ✅ All tabs and buttons exercised by `AppTest`; a corrupt ledger doesn't take down the other tabs
- ✅ Backtest shows the expectancy / Sortino / regime / missing-candle information

## 7. Integration and security
- ✅ Static guards: no app code can enable `allow_trading`, no ccxt, no secrets in config
- ✅ Secrets only from environment / `.env`; never logged or shown

## 8. CoinEx (final stage)
- ✅ v2 client per official docs: signing, server-time sync, markets/ticker/kline/depth, balance, order-status,
  pending/finished-order queries (for reconciliation), maintenance info; mocked tests only
- ⏳ Live public-data verification. Open questions the docs don't answer:
  - is kline `created_at` the open or the close time, and in what sort order?
  - is `min_amount` in base or quote currency?
  - which currency is ticker `value` in?
  - is `/v2` part of the signed path?
- ⏳ Read-only key: balance, then a signed pending-order query
- ⏳ Everything in `LIVE_TRADING_CHECKLIST.md` before any unlock
