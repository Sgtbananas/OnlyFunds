import logging
import random

def place_order(
    pair: str,
    action: str,
    amount: float,
    price: float,
    is_dry_run: bool = True
) -> dict:
    """
    Place a live or simulated order.

    Parameters:
    - pair (str): The trading pair (e.g., BTC/USD).
    - action (str): "BUY" or "SELL".
    - amount (float): The amount of the asset to trade.
    - price (float): The price at which to place the order.
    - is_dry_run (bool): Whether to simulate the order (dry run).

    Returns:
    - dict: A dictionary containing order details:
        {
          "order_id": str,
          "amount": float,
          "order_price": float,
          "filled": bool,
        }
    """
    try:
        if is_dry_run:
            # Simulated order
            order_id = f"SIM-{random.randint(10000, 99999)}"
            filled = True  # For dry runs, assume the order is always filled
            logging.info(f"Simulated {action} order for {pair}: {amount} at {price:.2f}")
        else:
            # Live order logic here (stubbed for now)
            # For example, using an exchange API like Binance or Coinbase
            # response = exchange_api.place_order(pair, action, amount, price)
            # order_id = response["order_id"]
            # filled = response["filled"]
            # For simplicity, we'll simulate live orders here
            order_id = f"LIVE-{random.randint(10000, 99999)}"
            filled = random.choice([True, False])  # Simulate partial fills randomly
            logging.info(f"Live {action} order placed for {pair}: {amount} at {price:.2f}")

        # Return consistent fields
        return {
            "order_id": order_id,
            "amount": amount,
            "order_price": price,
            "filled": filled,
        }

    except Exception as e:
        logging.error(f"Error placing {action} order for {pair}: {e}")
        return {
            "order_id": None,
            "amount": amount,
            "order_price": price,
            "filled": False,
            "error": str(e),  # Include the error message for debugging
        }