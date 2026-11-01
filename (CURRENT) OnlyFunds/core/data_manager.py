"""Pair universe and historical data helpers (CoinEx API v2)."""
from __future__ import annotations

import logging
from typing import Iterable, List

import pandas as pd

from core.coinex_client import top_usdt_markets
from core.core_data import fetch_klines, get_client
from core.indicators import add_indicators

logger = logging.getLogger(__name__)


def get_top_coinex_symbols_marketcap(limit: int = 200) -> List[str]:
    """Top tradable USDT pairs by 24h quote volume (CoinEx has no market-cap field; volume is the proxy)."""
    return top_usdt_markets(get_client(), limit=limit)


def update_historical_data(symbols: Iterable[str], interval: str = "5m", lookback: int = 1000) -> List[str]:
    """Refresh the local candle cache for ``symbols``; returns the symbols that succeeded."""
    if isinstance(symbols, str):
        symbols = [symbols]
    ok = []
    for pair in symbols:
        try:
            df = fetch_klines(pair, interval, lookback)
            if not df.empty:
                ok.append(pair)
        except Exception as exc:
            logger.error("Failed to update %s: %s", pair, exc)
    return ok


def load_data_for_backtest(pair: str, interval: str = "5m", lookback: int = 1000) -> pd.DataFrame:
    return add_indicators(fetch_klines(pair, interval, lookback))
