1| # core/core_data.py
2| 
3| import logging
4| import os
5| import requests
6| import pandas as pd
7| import ta
8| from dotenv import load_dotenv
9| 
10| load_dotenv()
11| 
12| # Interval mapping for CoinEx API
13| _INTERVAL_MAP = {
14|     "1m":  "1min",  "3m":  "3min", "5m":  "5min", "15m": "15min",
15|     "30m": "30min", "1h":  "1hour","2h":  "2hour","4h":  "4hour",
16|     "6h":  "6hour","12h": "12hour","1d":  "1day","3d":  "3day",
17|     "1w":  "1week"
18| }
19| 
20| # Default trading pairs
21| TRADING_PAIRS = os.getenv("TRADING_PAIRS", "BTCUSDT,ETHUSDT,LTCUSDT").split(",")
22| 
23| COINEX_BASE = os.getenv("API_BASE_URL", "https://api.coinex.com/v1")
24| 
25| def fetch_klines(pair: str, interval: str = "5m", limit: int = 500) -> pd.DataFrame:
26|     """
27|     Fetch OHLCV data from CoinEx. Returns DataFrame indexed by Timestamp
28|     with Open, High, Low, Close, Volume columns.
29|     """
30|     resolution = _INTERVAL_MAP.get(interval)
31|     if not resolution:
32|         logging.error(f"Invalid interval: {interval}")
33|         return pd.DataFrame()
34| 
35|     try:
36|         resp = requests.get(
37|             f"{COINEX_BASE}/market/kline",
38|             params={"market": pair, "type": resolution, "limit": limit},
39|             timeout=15
40|         )
41|         resp.raise_for_status()
42|         data = resp.json().get("data", [])
43|         if not data:
44|             logging.warning(f"No kline data for {pair}@{interval}")
45|             return pd.DataFrame()
46| 
47|         # CoinEx: [timestamp, open, high, low, close, volume, turnover]
48|         df = pd.DataFrame(data, columns=[
49|             "timestamp", "open", "high", "low", "close", "volume", "turnover"
50|         ]).drop(columns=["turnover"])
51|         df.rename(columns={
52|             "timestamp": "Timestamp",
53|             "open":      "Open",
54|             "high":      "High",
55|             "low":       "Low",
56|             "close":     "Close",
57|             "volume":    "Volume"
58|         }, inplace=True)
59|         df["Timestamp"] = pd.to_datetime(df["Timestamp"], unit="s")
60|         df.set_index("Timestamp", inplace=True)
61|         return df[["Open","High","Low","Close","Volume"]].astype(float)
62|     except requests.exceptions.RequestException as e:
63|         logging.error(f"RequestException in fetch_klines for {pair}: {e}")
64|     except ValueError as e:
65|         logging.error(f"ValueError in fetch_klines for {pair}, possibly malformed data: {e}")
66|     return pd.DataFrame()
67| 
68| def validate_df(df: pd.DataFrame) -> bool:
69|     """Ensure DataFrame has OHLCV & no NaNs."""
70|     required = {"Open","High","Low","Close","Volume"}
71|     missing_columns = required - set(df.columns)
72|     if missing_columns:
73|         logging.error(f"validate_df missing columns: {missing_columns}")
74|         return False
75|     if df[list(required)].isnull().any().any():
76|         logging.error("validate_df found NaNs in OHLCV")
77|         return False
78|     return True
79| 
80| def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
81|     """
82|     Append RSI, MACD, Bollinger Bands, EMA20, EMA_diff, Volatility.
83|     Fully forward‐fills to avoid NaNs.
84|     """
85|     df2 = df.copy()
86|     close = df2["Close"]
87| 
88|     try:
89|         # RSI
90|         df2["rsi"] = (
91|             ta.momentum.RSIIndicator(close, window=14)
92|             .rsi()
93|             .fillna(50)
94|             .ffill()
95|         )
96| 
97|         # MACD
98|         macd = ta.trend.MACD(close)
99|         df2["macd"]        = macd.macd_diff().fillna(0).ffill()
100|         df2["macd_signal"] = macd.macd_signal().fillna(0).ffill()
101| 
102|         # Bollinger Bands
103|         mid = close.rolling(20, min_periods=1).mean()
104|         std = close.rolling(20, min_periods=1).std().fillna(0)
105|         df2["bollinger_mid"]   = mid.ffill()
106|         df2["bollinger_upper"] = (mid + 2 * std).ffill()
107|         df2["bollinger_lower"] = (mid - 2 * std).ffill()
108| 
109|         # EMA 20
110|         ema20 = close.ewm(span=20, adjust=False).mean()
111|         df2["ema20"]    = ema20.ffill()
112|         df2["ema_diff"] = (close - ema20).fillna(0)
113| 
114|         # Volatility
115|         df2["volatility"] = close.rolling(10, min_periods=1).std().fillna(0)
116| 
117|     except Exception as e:
118|         logging.error(f"Error in add_indicators: {e}")
119|         return pd.DataFrame()
120| 
121|     return df2
