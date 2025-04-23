# core/__init__.py

"""
Core module for CryptoTrader AI.
Includes data fetching, indicators, signals, trading logic, backtesting, and dashboards.
"""

from .core_data import fetch_klines, validate_df, add_indicators, TRADING_PAIRS
from .core_signals import generate_signal, smooth_signal, track_trade_result, adaptive_threshold
from .trade_executor import execute_trade
from .autotune import run_autotune
from .backtester import run_backtest
from .dashboard import render_dashboard

__all__ = [
    "fetch_klines", "validate_df", "add_indicators", "TRADING_PAIRS",
    "generate_signal", "smooth_signal", "track_trade_result", "adaptive_threshold",
    "execute_trade", "run_autotune", "run_backtest", "render_dashboard"
]
