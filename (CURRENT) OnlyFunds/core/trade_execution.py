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
    """
    try:
        if is_dry_run:
            order_id = f"SIM-{random.randint(10000, 99999)}"
            filled = True
            logging.info(f"Simulated {action} order for {pair}: {amount} at {price:.2f}")
        else:
            order_id = f"LIVE-{random.randint(10000, 99999)}"
            filled = random.choice([True, False])
            logging.info(f"Live {action} order placed for {pair}: {amount} at {price:.2f}")
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
            "error": str(e),
        }

def place_limit_order(pair, side, amount, price, is_dry_run=True):
    """
    Place a limit order (stub for grid trading). Returns order_id.
    """
    try:
        if is_dry_run:
            order_id = f"SIMGRID-{random.randint(10000, 99999)}"
            logging.info(f"Simulated {side.upper()} limit order for {pair}: {amount} @ {price:.2f}")
        else:
            order_id = f"GRID-{random.randint(10000, 99999)}"
            logging.info(f"Live {side.upper()} limit order for {pair}: {amount} @ {price:.2f}")
        return order_id
    except Exception as e:
        logging.error(f"Error placing grid limit order for {pair}: {e}")
        return None

def cancel_order(order_id, is_dry_run=True):
    """
    Cancel an order (stub for grid trading). Returns True if cancelled.
    """
    try:
        if is_dry_run:
            logging.info(f"Simulated cancel for order {order_id}")
            return True
        else:
            logging.info(f"Live cancel for order {order_id}")
            return True
    except Exception as e:
        logging.error(f"Error cancelling order {order_id}: {e}")
        return False