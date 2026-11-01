# OnlyFunds 2 (OF2)

Crypto spot trading research app for CoinEx: market data → indicators → regime-aware strategy
signals → risk-managed backtesting and **paper trading**, with an optional ML entry filter.

> **Live trading is locked in code** (`core/trade_execution.py: LIVE_TRADING_LOCKED = True`).
> The app never places real orders. See `(CURRENT) OnlyFunds/docs/LIVE_TRADING_CHECKLIST.md`.

The application lives in **`(CURRENT) OnlyFunds/`**. `(Original) OnlyFunds.zip` is the untouched
original backup.

## Quick start (Windows)

```bat
cd "(CURRENT) OnlyFunds"
pip install -r requirements.txt
streamlit run main.py
```

Or double-click `(CURRENT) OnlyFunds/START.bat`. The app opens at http://localhost:8501.

Headless paper trading (same ledger as the app):

```bat
python run_paper.py --pairs BTCUSDT ETHUSDT --interval 1h
```

Tests:

```bat
pip install -r requirements-dev.txt
python -m pytest
```

## What's inside

| Area | Module |
|---|---|
| CoinEx API v2 client (public data, signing, precision, rate limits; orders locked) | `core/coinex_client.py` |
| Candle fetch + local cache, CSV import | `core/core_data.py` |
| EMA, RSI, MACD, Bollinger, ATR, trend score, volatility | `core/indicators.py` |
| Regime detection, strategy stacks, confirmation, confidence | `core/core_signals.py` |
| Strategy selection by regime, rotation of losing strategies | `core/meta_learner.py` |
| Position sizing, exposure caps, daily loss, drawdown halt, kill switch | `core/risk_manager.py` |
| Stop-loss / take-profit / trailing stop rules (shared) | `core/positions.py` |
| Backtester (next-bar execution, fees, slippage, out-of-sample) | `core/backtester.py` |
| Paper trader (persistent ledger, reconciliation, emergency stop) | `core/paper_trader.py` |
| ML entry filter (triple-barrier labels, purged validation) | `core/ml_filter.py` |
| Parameter optimisation, walk-forward | `core/autotune.py` |
| Streamlit UI | `main.py` |

Configuration: `(CURRENT) OnlyFunds/config/config.yaml`. API keys are read only from environment
variables / `.env` (see `.env.example`) and are needed only for the optional balance check.

Full audit, fixes and known limitations: `(CURRENT) OnlyFunds/docs/AUDIT_REPORT.md`.
