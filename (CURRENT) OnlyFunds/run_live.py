"""Live trading entry point — LOCKED.

The previous version of this file created a ccxt CoinEx client and exposed
functions that placed real limit orders. Live trading is disabled in
OnlyFunds 2 until the order path has been independently verified (see
docs/LIVE_TRADING_CHECKLIST.md). Use ``python run_paper.py``.
"""
import sys

from core.trade_execution import LIVE_TRADING_LOCKED

if __name__ == "__main__":
    if LIVE_TRADING_LOCKED:
        print("Live trading is locked in OnlyFunds 2. Use: python run_paper.py")
        sys.exit(2)
