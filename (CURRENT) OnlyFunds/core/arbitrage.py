import ccxt

def fetch_depths(exchange, symbols):
    """Fetch orderbook depths for each symbol using ccxt."""
    depths = {}
    for sym in symbols:
        orderbook = exchange.fetch_order_book(sym)
        depths[sym] = {
            "bid": orderbook['bids'][0][0] if orderbook['bids'] else None,
            "ask": orderbook['asks'][0][0] if orderbook['asks'] else None
        }
    return depths

def find_triangular_arbitrage(depths, fee_threshold=0.001):
    """
    Returns list of arbitrage ops (cycle, net_rate) if found.
    Example: USDT -> BTC -> ETH -> USDT
    """
    ops = []
    try:
        rate = (1/depths["BTC/USDT"]["ask"]) * depths["ETH/BTC"]["bid"] * depths["ETH/USDT"]["bid"]
        if rate > 1 + fee_threshold:
            ops.append((["USDT","BTC","ETH","USDT"], rate))
    except Exception:
        pass
    return ops

def execute_arbitrage(exchange, op, amount):
    """Atomically execute the arbitrage leg-by-leg. Abort if slippage or fill fails."""
    # TODO: Fill logic and risk checks
    pass