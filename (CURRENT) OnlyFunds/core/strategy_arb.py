strategy_arb.pyimport pandas as pd

def run_triangular_arb(prices_dict: dict, params: dict) -> pd.DataFrame:
    """
    prices_dict: {'BTCUSDT': pd.Series, 'ETHBTC': pd.Series, ...}
    params: dict of params for arb logic
    Returns: DataFrame of trades/arbs
    """
    # Stub implementation: just return empty DataFrame for now
    # Replace with real logic later!
    columns = ['timestamp', 'leg1', 'leg2', 'leg3', 'pnl']
    return pd.DataFrame([], columns=columns)