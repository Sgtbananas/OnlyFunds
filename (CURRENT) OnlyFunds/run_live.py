import os
from dotenv import load_dotenv
import ccxt
import logging
from core.arbitrage import find_triangular_arbitrage
from core.market_maker import market_make

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load API keys from .env
load_dotenv()
api_key = os.getenv("COINEX_API_KEY")
api_secret = os.getenv("COINEX_API_SECRET")

exchange = ccxt.coinex({
    'apiKey': api_key,
    'secret': api_secret,
    'enableRateLimit': True,
})

def fetch_depths():
    # Map to your arbitrage function's expected format
    orderbooks = {
        "BTCUSDT": exchange.fetch_order_book("BTC/USDT"),
        "ETHBTC": exchange.fetch_order_book("ETH/BTC"),
        "ETHUSDT": exchange.fetch_order_book("ETH/USDT"),
    }
    depths = {
        key: {
            "bid": ob["bids"][0][0],
            "ask": ob["asks"][0][0],
        } for key, ob in orderbooks.items()
    }
    return depths

def get_order_book(pair):
    # CCXT expects 'BTC/USDT' format, adapt as needed
    return exchange.fetch_order_book(pair)

def place_limit(pair, side, size, price):
    # Only implemented for live trading, not dry-run
    if side == "buy":
        order = exchange.create_limit_buy_order(pair, size, price)
    else:
        order = exchange.create_limit_sell_order(pair, size, price)
    logging.info(f"Placed {side} order for {pair}: size={size}, price={price}")
    return order['id']

def cancel_all(pair):
    # Cancels all open orders for the pair
    open_orders = exchange.fetch_open_orders(pair)
    for order in open_orders:
        exchange.cancel_order(order['id'], pair)
    logging.info(f"Cancelled all open orders for {pair}")

if __name__ == "__main__":
    # ---- Arbitrage Demo ----
    print("Running triangular arbitrage check (dry run)...")
    depths = fetch_depths()
    arbs = find_triangular_arbitrage(depths, dry_run=True)
    print("Arbitrage opportunities:", arbs)

    # ---- Market Maker Demo ----
    print("\nRunning market maker bot (dry run, one loop)...")
    market_make(
        "BTC/USDT",
        get_order_book=get_order_book,
        place_limit=place_limit,
        cancel_all=cancel_all,
        spread=0.002,
        size=0.001,
        refresh_interval=5,
        dry_run=True,
    )
    print("Market making dry run complete.")