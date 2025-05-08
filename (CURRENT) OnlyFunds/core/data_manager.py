import os
import logging
import requests
from core.core_data import fetch_klines, load_data, save_data, add_indicators

logger = logging.getLogger(__name__)

def get_top_coinex_symbols_marketcap(limit=250):
    """
    Return top USDT symbols by volume * % change (proxy for market cap impact).
    """
    try:
        url = "https://api.coinex.com/v1/market/ticker/all"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        tickers = response.json()["data"]["ticker"]
    except Exception as e:
        logger.error(f"Failed to fetch tickers: {e}")
        return ["BTCUSDT", "ETHUSDT"]

    candidates = []
    for pair, stats in tickers.items():
        if pair.endswith("USDT") and not any(x in pair for x in ["UP", "DOWN", "BEAR", "BULL"]):
            try:
                volume = float(stats.get("vol", 0))
                change_pct = abs(float(stats.get("percent_change_24h", 0)))
                score = volume * change_pct
                candidates.append((pair, score))
            except Exception:
                continue

    sorted_pairs = sorted(candidates, key=lambda x: x[1], reverse=True)
    return [p[0] for p in sorted_pairs[:limit]]

def update_historical_data(symbols, interval="5m", lookback=1000):
    updated_symbols = []
    for pair in symbols:
        try:
            df = load_data(pair, interval=interval, limit=lookback)
            if df.empty:
                logger.warning(f"No data for {pair}, fetching fresh.")
                df = fetch_klines(pair, interval=interval, limit=lookback)
                if df.empty:
                    continue
                df = add_indicators(df)
                save_data(df, pair, interval, lookback)
            else:
                logger.info(f"Data loaded for {pair}. {len(df)} rows.")
            updated_symbols.append(pair)
        except Exception as e:
            logger.error(f"Failed to update {pair}: {e}")
            continue
    return updated_symbols

def load_data_for_backtest(pair, interval="5m", lookback=1000):
    df = load_data(pair, interval=interval, limit=lookback)
    df = add_indicators(df)
    return df
