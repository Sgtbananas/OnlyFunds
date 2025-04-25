import asyncio

async def scalp_pair(pair, fetch_ws_klines, place_limit_buy, thresh, qty):
    while True:
        df = await fetch_ws_klines(pair)
        signal = generate_signal(df).iloc[-1]
        if signal > thresh:
            await place_limit_buy(pair, qty, df["Close"].iloc[-1])
        await asyncio.sleep(0.1)