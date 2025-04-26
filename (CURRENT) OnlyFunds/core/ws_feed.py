import asyncio
import json
import threading
import websockets
import time

# Supported CoinEx websocket endpoint
COINEX_WS_URL = "wss://socket.coinex.com/"

class CoinExWsFeed:
    def __init__(self, symbols, depth_levels=5, reconnect_delay=5):
        self.symbols = symbols
        self.depth_levels = depth_levels
        self.mid_prices = {s: None for s in symbols}
        self.orderbooks = {s: None for s in symbols}
        self._lock = threading.Lock()
        self._running = False
        self._reconnect_delay = reconnect_delay

    def start(self):
        self._running = True
        thread = threading.Thread(target=lambda: asyncio.run(self._run()))
        thread.daemon = True
        thread.start()

    def stop(self):
        self._running = False

    def get_mid_price(self, symbol):
        with self._lock:
            return self.mid_prices.get(symbol)

    def get_orderbook(self, symbol):
        with self._lock:
            return self.orderbooks.get(symbol)

    async def _connect_and_subscribe(self, ws):
        # Subscribe to depth for each symbol
        for sym in self.symbols:
            msg = {
                "method": "depth.subscribe",
                "params": [sym, self.depth_levels, "0"],
                "id": sym
            }
            await ws.send(json.dumps(msg))

    async def _run(self):
        while self._running:
            try:
                async with websockets.connect(COINEX_WS_URL, ping_interval=15, ping_timeout=10) as ws:
                    await self._connect_and_subscribe(ws)
                    while self._running:
                        try:
                            msg = await ws.recv()
                            data = json.loads(msg)
                            # CoinEx depth update format
                            if data.get("method") == "depth.update":
                                params = data.get("params", [])
                                if len(params) >= 2:
                                    book = params[1]
                                    symbol = data.get("id")
                                    # PATCH: fallback to params[2] if id is None
                                    if not symbol and len(params) > 2:
                                        symbol = params[2]
                                    if symbol in self.symbols:
                                        bids = book.get("bids", [])
                                        asks = book.get("asks", [])
                                        if bids and asks:
                                            best_bid = float(bids[0][0])
                                            best_ask = float(asks[0][0])
                                            mid = (best_bid + best_ask) / 2
                                            with self._lock:
                                                self.mid_prices[symbol] = mid
                                                self.orderbooks[symbol] = {
                                                    "bids": bids,
                                                    "asks": asks,
                                                    "best_bid": best_bid,
                                                    "best_ask": best_ask,
                                                    "mid": mid,
                                                }
                        except Exception as e:
                            print(f"CoinEx WS feed inner error: {e}")
                            # If it's a recoverable error, try to continue. If it's a connection error, break out.
                            break
            except Exception as e:
                print(f"CoinEx WS error: {e}")
            # Wait before attempting to reconnect
            if self._running:
                print(f"CoinEx WS reconnecting in {self._reconnect_delay}s...")
                await asyncio.sleep(self._reconnect_delay)

# --- Singleton for your app ---

ws_feed = None

def start_ws_feed(symbols):
    global ws_feed
    if ws_feed is None:
        ws_feed = CoinExWsFeed(symbols)
        ws_feed.start()
    return ws_feed

def get_mid_price(symbol):
    if ws_feed:
        return ws_feed.get_mid_price(symbol)
    return None

def get_orderbook(symbol):
    if ws_feed:
        return ws_feed.get_orderbook(symbol)
    return None