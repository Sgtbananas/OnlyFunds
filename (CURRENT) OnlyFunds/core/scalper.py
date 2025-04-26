import asyncio

async def scalp_pair(pair, fetch_ws_klines, generate_signal, place_limit, threshold=0.7, qty=0.001):
    """Async scalper for a single pair, using WebSocket bars and smart fused signals."""
    while True:
        df = await fetch_ws_klines(pair)
        sig = generate_signal(df).iloc[-1]
        price = df["Close"].iloc[-1]
        if sig > threshold:
            await place_limit(pair, "buy", qty, price)
        elif sig < -threshold:
            await place_limit(pair, "sell", qty, price)
        await asyncio.sleep(0.1)