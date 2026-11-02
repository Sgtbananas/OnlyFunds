import pandas as pd
import numpy as np

class GridStrategy:
    def __init__(self, lower, upper, grid_size, qty_per_level):
        self.lower = lower
        self.upper = upper
        self.grid_size = grid_size
        self.qty_per_level = qty_per_level
        self.levels = np.linspace(lower, upper, grid_size)
        self.positions = {level: 0 for level in self.levels}
        self.trades = []

    def on_price(self, price, timestamp=None):
        for level in self.levels:
            if price <= level and self.positions[level] == 0:
                self.positions[level] = self.qty_per_level
                self.trades.append({
                    "timestamp": timestamp,
                    "side": "buy",
                    "price": price,
                    "level": level,
                    "qty": self.qty_per_level,
                })
            elif price >= level and self.positions[level] > 0:
                self.positions[level] = 0
                self.trades.append({
                    "timestamp": timestamp,
                    "side": "sell",
                    "price": price,
                    "level": level,
                    "qty": self.qty_per_level,
                })

    def get_trades(self):
        return pd.DataFrame(self.trades)

def run_grid_strategy(prices: pd.Series, params: dict) -> pd.DataFrame:
    lower = params.get("lower", prices.min())
    upper = params.get("upper", prices.max())
    grid_size = params.get("grid_size", 5)
    qty_per_level = params.get("qty_per_level", 1)
    strat = GridStrategy(lower, upper, grid_size, qty_per_level)
    for i, price in enumerate(prices):
        strat.on_price(price, timestamp=prices.index[i] if prices.index is not None else i)
    trades = strat.get_trades()
    return trades