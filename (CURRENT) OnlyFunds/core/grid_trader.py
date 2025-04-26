import numpy as np

class GridTrader:
    """Grid trading engine for a single pair."""
    def __init__(self, pair, grid_levels=10, grid_spacing=0.005, grid_size=0.001, base_price=None):
        self.pair = pair
        self.grid_levels = grid_levels
        self.grid_spacing = grid_spacing  # e.g. 0.005 = 0.5%
        self.grid_size = grid_size
        self.base_price = base_price  # Last mid-price

    def generate_orders(self, mid_price):
        """Return list of grid order dicts around mid_price."""
        self.base_price = mid_price
        orders = []
        for lvl in range(1, self.grid_levels + 1):
            up = mid_price * (1 + self.grid_spacing * lvl)
            down = mid_price * (1 - self.grid_spacing * lvl)
            orders.append({"pair": self.pair, "side": "sell", "price": up, "size": self.grid_size})
            orders.append({"pair": self.pair, "side": "buy", "price": down, "size": self.grid_size})
        return orders

    def backtest_grid(self, price_series):
        """Vectorized grid simulation: returns pnl, fills count, etc."""
        # TODO: Implement a fast simulation for historical price_series and this grid spec
        pass