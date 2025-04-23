# main.py

import os
import logging
import time
from datetime import datetime

import streamlit as st
import pandas as pd
from dotenv import load_dotenv

from core.core_data      import fetch_klines, validate_df, add_indicators, TRADING_PAIRS
from core.core_signals   import generate_signal, smooth_signal, adaptive_threshold
from core.trade_execution import place_order
from core.backtester     import run_backtest
from utils.helpers       import compute_trade_metrics, suggest_tuning

# ─── Load env & defaults ──────────────────────────────────────────────────────
load_dotenv()
DEFAULT_DRY_RUN = os.getenv("USE_DRY_RUN", "True").lower() == "true"
DEFAULT_CAPITAL = float(os.getenv("DEFAULT_CAPITAL", 1000))
RISK_PER_TRADE  = float(os.getenv("RISK_PER_TRADE", 0.01))

# ─── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ─── Streamlit UI ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="CryptoTrader AI", layout="wide")
st.title("🧠 CryptoTrader AI Bot")
st.sidebar.header("⚙️ Configuration")

dry_run       = st.sidebar.checkbox("Dry Run Mode (Simulated)", value=DEFAULT_DRY_RUN)
autotune      = st.sidebar.checkbox("Enable Adaptive-Threshold Autotune", value=False)
backtest_mode = st.sidebar.checkbox("Enable Backtesting", value=False)
mode          = st.sidebar.selectbox("Trading Mode", ["Conservative", "Normal", "Aggressive"], index=1)
interval      = st.sidebar.selectbox("Candle Interval", ["5m","15m","30m","1h","4h","1d"], index=0)
lookback      = st.sidebar.slider("Historical Lookback", 100, 1000, 300)
max_positions = st.sidebar.number_input("Max Open Positions", 1, 5, 2)

# Strategy risk params
if mode == "Conservative":
    risk_pct, trailing_stop_pct, scale_in = RISK_PER_TRADE * 0.5, 0.01, False
elif mode == "Aggressive":
    risk_pct, trailing_stop_pct, scale_in = RISK_PER_TRADE * 1.5, 0.05, True
else:  # Normal
    risk_pct, trailing_stop_pct, scale_in = RISK_PER_TRADE, 0.03, True

# Runtime store
open_positions = {}   # pair -> buy result dict
trade_log      = []   # list of uniform trade-records

# ─── Core Trade Logic ─────────────────────────────────────────────────────────
def trade_logic(pair: str):
    logger.info(f"🔍 Analyzing {pair}")
    df = fetch_klines(pair=pair, interval=interval, limit=lookback)
    if df.empty or not validate_df(df):
        logger.warning(f"⚠️ Invalid/empty data for {pair}")
        return

    df = add_indicators(df)

    # Determine threshold
    threshold = 0.5
    if autotune:
        threshold = adaptive_threshold(df)
        logger.debug(f"Adaptive threshold for {pair}: {threshold}")

    # Signal
    raw = generate_signal(df)
    smoothed = smooth_signal(raw)
    latest = smoothed.iloc[-1]

    # Backtest UI
    if backtest_mode:
        bt = run_backtest(smoothed, df["Close"], threshold)
        st.subheader(f"📊 Backtest: {pair}")
        st.dataframe(bt)
        return

    # Decide action
    action = None
    if latest >  threshold:
        action = "buy"
    elif latest < -threshold:
        action = "sell"
    if not action:
        return

    # Enforce position limits
    if action == "buy":
        if pair in open_positions:
            logger.info(f"⏸ Already long {pair} → skipping BUY")
            return
        if len(open_positions) >= max_positions:
            logger.info("🚫 Max positions reached → skipping BUY")
            return

    # Compute amount & price
    price = df["Close"].iloc[-1]
    amount = (DEFAULT_CAPITAL * risk_pct) if action == "buy" else open_positions[pair]["amount"]

    # Place the order (dry-run or real)
    response = place_order(pair=pair, action=action, amount=amount, price=price, is_dry_run=dry_run)

    # Build a uniform trade-record for metrics
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    record = {
        "timestamp":   now,
        "pair":        pair,
        "action":      action.upper(),
        "amount":      amount,
    }
    if action == "buy":
        record["entry_price"] = price
        open_positions[pair] = {"amount": amount, "entry_price": price, **response}
    else:
        record["exit_price"] = price
        open_positions.pop(pair, None)

    # Log
    trade_log.append(record)
    track_trade_result(response, pair, action.upper())

    # Attach trailing‐stop if provided by API
    if "order_price" in response:
        response["trailing_stop"] = response["order_price"] * (1 - trailing_stop_pct)

# ─── Dashboard & Metrics ─────────────────────────────────────────────────────
def display_dashboard():
    st.subheader("📈 Live Dashboard")
    perf = compute_trade_metrics(trade_log, DEFAULT_CAPITAL)
    st.metric("Total Return", f"{perf['total_return']:.2%}")
    st.metric("Win Rate",     f"{perf['win_rate']:.2%}")

    suggestions = suggest_tuning(perf)
    for s in suggestions:
        st.info(f"🔧 {s}")

    if open_positions:
        st.write("🟢 Open Positions")
        st.dataframe(pd.DataFrame(open_positions).T)
    else:
        st.info("No active trades.")

    if trade_log:
        st.write("📘 Trade History")
        st.dataframe(pd.DataFrame(trade_log))
    else:
        st.info("No trade history yet.")

# ─── Main Loop ────────────────────────────────────────────────────────────────
def main_loop():
    while True:
        for pair in TRADING_PAIRS:
            try:
                trade_logic(pair)
            except Exception as e:
                logger.exception(f"❌ Error in trade cycle for {pair}: {e}")
        display_dashboard()
        time.sleep(10)

# ─── Entry Point ─────────────────────────────────────────────────────────────
if st.sidebar.button("🚀 Start Trading Bot"):
    st.success("Bot started! 🚀")
    main_loop()
else:
    st.info("Ready. Configure & click Start.")

