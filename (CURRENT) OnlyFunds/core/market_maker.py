import time

def market_make(pair, get_mid_price, place_limit, cancel_all, spread=0.002, size=0.01, interval=10):
    """
    Places tight bid/ask on both sides of the book, refreshes every interval seconds.
    """
    while True:
        mid = get_mid_price(pair)
        bids = place_limit(pair, "buy", size, mid*(1-spread))
        asks = place_limit(pair, "sell", size, mid*(1+spread))
        time.sleep(interval)
        cancel_all(pair)