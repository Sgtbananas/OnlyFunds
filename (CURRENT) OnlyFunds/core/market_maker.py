import time
import logging

def market_make(
    pair, 
    get_order_book, 
    place_limit, 
    cancel_all, 
    spread=0.002, 
    size=0.01, 
    refresh_interval=10
):
    """
    Market-making loop: places symmetric bid/ask quotes around mid-price.
    pair: trading symbol (e.g., 'BTC/USDT')
    get_order_book: function(pair) -> {'bids': [[price, qty]], 'asks': [[price, qty]]}
    place_limit: function(pair, side, size, price) -> order_id
    cancel_all: function(pair)
    spread: total spread between bid and ask as a fraction (e.g., 0.002 = 0.2%)
    size: order size
    refresh_interval: seconds before refreshing quotes
    """
    try:
        order_book = get_order_book(pair)
        best_bid = order_book['bids'][0][0]
        best_ask = order_book['asks'][0][0]
        mid_price = (best_bid + best_ask) / 2

        buy_price = round(mid_price * (1 - spread / 2), 8)
        sell_price = round(mid_price * (1 + spread / 2), 8)

        logging.info(f"Placing market-making quotes for {pair}: BUY@{buy_price}, SELL@{sell_price}, size={size}")
        bid_order = place_limit(pair, "buy", size, buy_price)
        ask_order = place_limit(pair, "sell", size, sell_price)
        time.sleep(refresh_interval)
        cancel_all(pair)
        logging.info(f"Refreshed market-maker quotes for {pair}")
    except Exception as e:
        logging.error(f"Market-making error for {pair}: {e}")
