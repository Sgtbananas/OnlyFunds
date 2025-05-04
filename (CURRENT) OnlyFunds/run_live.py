import os
from dotenv import load_dotenv
import ccxt
import logging
from core.arbitrage import find_triangular_arbitrage
from core.market_maker import market_make
from core.meta_learner import select_strategy
from core.trade_execution import place_live_order

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load API keys from .env
load_dotenv()
api_key = os.getenv("COINEX_API_KEY")
api_secret = os.getenv("COINEX_API_SECRET")

# Initialize the exchange with the API credentials
exchange = ccxt.coinex({
    'apiKey': api_key,
    'secret': api_secret,
    'enableRateLimit': True,
})

# Live trade execution using Spot market logic
def execute_live_trade(pair, action, amount, price, performance_dict, meta_model=None):
    # Select the strategy using the AI/ML model
    strategy, _ = select_strategy(performance_dict, meta_model)

    # Calculate position size based on available equity and risk management
    equity = 1000  # Example capital (use actual balance in production)
    position_size = risk_manager.position_size(equity, price, risk_pct=0.01, volatility=price['ATR'], v_adj=True)

    # Apply ATR-based stop-loss and take-profit levels
    stop_loss_price = price - (2 * price['ATR'][-1])  # Example ATR-based stop loss
    take_profit_price = price + (2 * price['ATR'][-1])  # Example ATR-based take profit

    # Place order (simulated or real)
    if action == "BUY":
        place_live_order(pair, action, position_size, price)
    elif action == "SELL":
        place_live_order(pair, action, position_size, price)

    # Risk management during live trading
    while True:
        current_price = get_current_price(pair)  # Placeholder function to get live price
        if current_price <= stop_loss_price:
            logging.info(f"Stop loss triggered at {current_price}")
            break
        elif current_price >= take_profit_price:
            logging.info(f"Take profit triggered at {current_price}")
            break

# Fetch market depths for arbitrage opportunities
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

# Fetch the order book for a given pair
def get_order_book(pair):
    # CCXT expects 'BTC/USDT' format, adapt as needed
    return exchange.fetch_order_book(pair)

# Place a limit order (Spot market only, no margin/futures)
def place_limit(pair, side, size, price):
    """Place a limit order in the Spot market."""
    try:
        if side == "buy":
            order = exchange.create_limit_buy_order(pair, size, price)
        else:
            order = exchange.create_limit_sell_order(pair, size, price)
        logging.info(f"Placed {side} order for {pair}: size={size}, price={price}")
        return order['id']
    except Exception as e:
        logging.error(f"Error placing {side} limit order for {pair}: {e}")
        return None

# Cancel all open orders for the given pair
def cancel_all(pair):
    """Cancel all open orders for the pair in the Spot market."""
    try:
        open_orders = exchange.fetch_open_orders(pair)
        for order in open_orders:
            exchange.cancel_order(order['id'], pair)
        logging.info(f"Cancelled all open orders for {pair}")
    except Exception as e:
        logging.error(f"Error cancelling orders for {pair}: {e}")

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
