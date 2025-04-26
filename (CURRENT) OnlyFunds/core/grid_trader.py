import logging
from typing import List, Dict, Callable

class GridTrader:
    """
    GridTrader manages a set of grid limit orders around a base price.
    """

    def __init__(
        self,
        pair: str,
        base_price: float,
        grid_levels: int = 6,
        grid_spacing_pct: float = 0.004,
        size: float = 0.001,
        direction: str = "both",  # "buy", "sell", or "both"
    ):
        self.pair = pair
        self.base = base_price
        self.levels = grid_levels
        self.grid_pct = grid_spacing_pct
        self.size = size
        self.direction = direction.lower()
        self.orders: List[Dict] = []  # [{side, price, size, order_id, active}, ...]

    def generate_grid_prices(self) -> List[float]:
        """
        Generate grid prices centered on the base price.
        For n levels, returns 2n+1 prices (n below, center, n above).
        """
        return [
            self.base * (1 + i * self.grid_pct)
            for i in range(-self.levels, self.levels + 1)
        ]

    def build_orders(self):
        grid_prices = self.generate_grid_prices()
        self.orders.clear()
        for i, price in enumerate(grid_prices):
            if i < self.levels:
                side = "buy"
            elif i > self.levels:
                side = "sell"
            else:
                continue  # Skip center price (optional)
            if self.direction in [side, "both"]:
                self.orders.append({
                    "pair": self.pair,
                    "side": side,
                    "price": round(price, 6),
                    "size": self.size,
                    "active": True,
                    "order_id": None,
                })

    def place_orders(self, place_limit_order_func: Callable):
        """
        Call this to actually place the grid orders.
        place_limit_order_func(pair, side, amount, price) should return order_id on success.
        """
        for order in self.orders:
            if order["active"] and order["order_id"] is None:
                try:
                    order_id = place_limit_order_func(
                        order["pair"], order["side"], order["size"], order["price"]
                    )
                    order["order_id"] = order_id
                    logging.info(
                        f"Placed {order['side']} grid order {order_id} at {order['price']} ({order['size']})"
                    )
                except Exception as e:
                    logging.error(f"Failed to place grid order: {e}")

    def refresh_orders(self, open_orders: List[Dict], cancel_order_func: Callable = None):
        """
        Refresh/cancel filled or stale orders.
        open_orders: list of currently open orders, e.g. from exchange/account.
        cancel_order_func(order_id) to cancel (optional).
        """
        for order in self.orders:
            if not order.get("order_id"):
                continue
            # If order is not found in open_orders, mark as inactive
            if not any(o.get("order_id") == order["order_id"] for o in open_orders):
                order["active"] = False
                logging.info(f"Grid order {order['order_id']} completed/removed")
            # Optionally, add logic to cancel/refresh if too far from mid-price
            # Example: if abs(order["price"]/self.base-1) > threshold and cancel_order_func:
            #     cancel_order_func(order["order_id"])
            #     order["active"] = False

    def active_grid_summary(self) -> List[Dict]:
        return [o for o in self.orders if o["active"]]

    def reset_grid(self, new_base: float):
        self.base = new_base
        self.build_orders()

    def clear_grid(self, cancel_order_func: Callable = None):
        """
        Cancel all grid orders (if live), or mark as inactive (if backtest/sim).
        """
        for order in self.orders:
            if cancel_order_func and order.get("order_id"):
                cancel_order_func(order["order_id"])
            order["active"] = False