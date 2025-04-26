import logging
from typing import List, Dict, Callable
import numpy as np

class GridOrder:
    """Represents a single grid order and its fill state."""
    def __init__(self, pair, side, price, size):
        self.pair = pair
        self.side = side
        self.price = price
        self.size = size
        self.active = True
        self.order_id = None
        self.filled = False
        self.fill_price = None
        self.fill_time = None

    def to_dict(self):
        return {
            "pair": self.pair,
            "side": self.side,
            "price": self.price,
            "size": self.size,
            "active": self.active,
            "order_id": self.order_id,
            "filled": self.filled,
            "fill_price": self.fill_price,
            "fill_time": self.fill_time,
        }

class GridTrader:
    """
    GridTrader manages a set of grid limit orders around a base price.
    Now supports simulated fills, trailing grid, adaptive spacing, grid backtest, multi-pair.
    """

    def __init__(
        self,
        pair: str,
        base_price: float,
        grid_levels: int = 6,
        grid_spacing_pct: float = 0.004,
        size: float = 0.001,
        direction: str = "both",  # "buy", "sell", or "both"
        trailing: bool = False,
        adaptive: bool = False,
        min_spacing: float = 0.001,
        max_spacing: float = 0.03,
    ):
        self.pair = pair
        self.base = base_price
        self.levels = grid_levels
        self.grid_pct = grid_spacing_pct
        self.size = size
        self.direction = direction.lower()
        self.trailing = trailing
        self.adaptive = adaptive
        self.min_spacing = min_spacing
        self.max_spacing = max_spacing
        self.orders: List[GridOrder] = []
        self.last_price = base_price

    def generate_grid_prices(self, base=None, spacing=None) -> List[float]:
        """
        Generate grid prices centered on the base price.
        For n levels, returns 2n+1 prices (n below, center, n above).
        """
        base = base if base is not None else self.base
        spacing = spacing if spacing is not None else self.grid_pct
        return [
            base * (1 + i * spacing)
            for i in range(-self.levels, self.levels + 1)
        ]

    def build_orders(self, base=None, spacing=None):
        """Builds grid orders at current (or provided) base/spacing."""
        grid_prices = self.generate_grid_prices(base, spacing)
        self.orders.clear()
        for i, price in enumerate(grid_prices):
            if i < self.levels:
                side = "buy"
            elif i > self.levels:
                side = "sell"
            else:
                continue  # Skip center price
            if self.direction in [side, "both"]:
                self.orders.append(GridOrder(self.pair, side, round(price, 6), self.size))

    def place_orders(self, place_limit_order_func: Callable):
        """
        Place all grid orders using the given function.
        place_limit_order_func(pair, side, amount, price) should return order_id.
        """
        for order in self.orders:
            if order.active and order.order_id is None:
                try:
                    order_id = place_limit_order_func(
                        order.pair, order.side, order.size, order.price
                    )
                    order.order_id = order_id
                    logging.info(
                        f"Placed {order.side} grid order {order_id} at {order.price} ({order.size})"
                    )
                except Exception as e:
                    logging.error(f"Failed to place grid order: {e}")

    def check_and_fill_orders(self, price, timestamp=None, log_func=None):
        """
        In simulation/backtest: check which orders should be filled at given price.
        If filled, set fill status and optionally log.
        """
        for order in self.orders:
            if order.active and not order.filled:
                if (order.side == "buy" and price <= order.price) or (order.side == "sell" and price >= order.price):
                    order.filled = True
                    order.active = False
                    order.fill_price = price
                    order.fill_time = timestamp
                    if log_func:
                        log_func(order.to_dict())
                    logging.info(f"Sim fill: {order.side} {order.pair} @{order.price:.4f} (fill price {price:.4f})")

    def refresh_orders(self, open_orders: List[Dict], cancel_order_func: Callable = None):
        """
        Refresh/cancel filled or stale orders (for live).
        open_orders: list of currently open orders.
        cancel_order_func(order_id) to cancel.
        """
        for order in self.orders:
            if order.order_id is None:
                continue
            if not any(o.get("order_id") == order.order_id for o in open_orders):
                order.active = False
                logging.info(f"Grid order {order.order_id} completed/removed")
            # Optionally cancel orders too far from price
            # [Advanced: implement as needed]

    def active_grid_summary(self) -> List[Dict]:
        return [o.to_dict() for o in self.orders if o.active]

    def reset_grid(self, new_base: float):
        self.base = new_base
        self.build_orders()

    def clear_grid(self, cancel_order_func: Callable = None):
        for order in self.orders:
            if cancel_order_func and order.order_id:
                cancel_order_func(order.order_id)
            order.active = False

    def trail_grid(self, new_price: float, threshold_pct=0.003):
        """
        If the price moves more than threshold_pct from base, rebuild grid at new price.
        """
        pct_move = abs(new_price / self.base - 1)
        if pct_move >= threshold_pct:
            logging.info(f"Trailing grid: shifting base from {self.base:.4f} to {new_price:.4f}")
            self.base = new_price
            self.build_orders()
            self.last_price = new_price

    def adapt_spacing(self, volatility: float):
        """
        Adjust grid spacing according to volatility (stddev of returns).
        """
        new_spacing = np.clip(volatility, self.min_spacing, self.max_spacing)
        if abs(new_spacing - self.grid_pct) > 1e-6:
            logging.info(f"Adaptive grid: changing spacing from {self.grid_pct:.4f} to {new_spacing:.4f}")
            self.grid_pct = new_spacing
            self.build_orders()