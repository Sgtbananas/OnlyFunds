import time
def market_make(pair, get_mid_price, place_limit, cancel_all, spread=0.002, size=0.01):
    mid = get_mid_price(pair)
    bids = place_limit(pair, "buy", size, mid*(1-spread))
    asks = place_limit(pair, "sell", size, mid*(1+spread))
    time.sleep(10)
    cancel_all(pair)