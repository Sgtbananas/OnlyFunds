class GridTrader:
    def __init__(self, pair, base_price, grid_levels=10, grid_pct=0.005, size=0.001):
        self.pair = pair
        self.base = base_price
        self.levels = grid_levels
        self.grid_pct = grid_pct
        self.size = size

    def generate_orders(self):
        # symmetric grid above/below base
        levels = []
        for i in range(1, self.levels + 1):
            levels.append(self.base * (1 + self.grid_pct * i))
            levels.append(self.base * (1 - self.grid_pct * i))
        orders = [{"pair": self.pair, "price": p, "size": self.size} for p in levels]
        return orders