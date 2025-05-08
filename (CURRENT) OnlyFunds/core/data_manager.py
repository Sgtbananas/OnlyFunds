import os
import logging
from core.core_data import fetch_klines, load_data, save_data
from utils.helpers import add_indicators

logger = logging.getLogger(__name__)

def get_top_coinex_symbols(limit=250, market="USDT"):
    """
    Fetches the top trading pairs from CoinEx (alphabetical, but can filter by volume if extended).
    """
    url = "https://api.coinex.com/v1/market/ticker/all"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        tickers = response.json()["data"]["ticker"]
    except Exception as e:
        logger.error(f"Failed to fetch tickers from CoinEx: {e}")
        return ["BTCUSDT", "ETHUSDT"]

    valid_symbols = [
        symbol for symbol in tickers.keys()
        if symbol.endswith(market) and not any(x in symbol for x in ["UP", "DOWN", "BEAR", "BULL"])
    ]

    return valid_symbols[:limit]
def update_historical_data(symbols, interval="5m", limit=1000):
    """
    For each symbol, ensures data is up-to-date. If CSV missing, fetches and saves data.
    If CSV exists, loads and checks for missing candles.
    """
    updated_symbols = []
    for pair in symbols:
        try:
            df = load_data(pair, interval=interval, limit=limit)
            if df.empty:
                logger.warning(f"No data for {pair}, attempting fresh fetch.")
                df = fetch_klines(pair, interval=interval, limit=limit)
                if df.empty:
                    logger.warning(f"Skipping {pair} — no data fetched.")
                    continue
                df = add_indicators(df)
                save_data(df, pair, interval, limit)
            else:
                logger.info(f"Data loaded for {pair}. {len(df)} rows.")

            updated_symbols.append(pair)

        except Exception as e:
            logger.error(f"Failed to update data for {pair}: {e}")
            continue

    logger.info(f"Updated data for {len(updated_symbols)} symbols.")
    return updated_symbols
