import os
import logging
import time
from datetime import datetime

import streamlit as st
import pandas as pd
from dotenv import load_dotenv

from core.core_data import fetch_klines, validate_df, add_indicators, TRADING_PAIRS
from core.core_signals import (
    generate_signal,
    smooth_signal,
    adaptive_threshold,
    track_trade_result,
)
from core.trade_execution import place_order
from core.backtester import run_backtest
from utils.helpers import compute_trade_metrics, suggest_tuning

# ─── Load env & defaults ──────────────────────────────────────────────────────
load_dotenv()
DEFAULT_DRY_RUN = os.getenv("USE_DRY_RUN", "True").lower() == "true"
DEFAULT_CAPITAL = float(os.getenv("DEFAULT_CAPITAL", 1000))
RISK_PER_TRADE = float(os.getenv("RISK_PER_TRADE", 0.01))

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ─── Streamlit UI ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="CryptoTrader AI", layout="wide")
st.title("🧠 CryptoTrader AI Bot (SPOT Market Only)")
st.sidebar.header("⚙️ Configuration")

dry_run = st.sidebar.checkbox("Dry Run Mode (Simulated)", value=DEFAULT_DRY_RUN)
autotune = st.sidebar.checkbox("Enable Adaptive-Threshold Autotune", value=False)
backtest_mode = st.sidebar.checkbox("Enable Backtesting", value=False)
mode = st.sidebar.selectbox("Trading Mode",
                             ["Conservative", "Normal", "Aggressive"],
                             index=1)
interval = st.sidebar.selectbox("Candle Interval",
                                 ["5m", "15m", "30m", "1h", "4h", "1d"],
                                 index=0)
lookback = st.sidebar.slider("Historical Lookback", 100, 1000, 300)
max_positions = st.sidebar.number_input("Max Open Positions", 1, 5, 2)

# Rename the start button for clarity
start_btn = st.sidebar.button("🚀 Start Trading Bot (Spot Only)")
if start_btn:
    st.success("Bot started! (Spot market only)")
else:
    st.info("Ready. Configure & click Start.")

# ─── Strategy risk params ─────────────────────────────────────────────────────
if mode == "Conservative":
    risk_pct, trailing_stop_pct, scale_in = RISK_PER_TRADE * 0.5, 0.01, False
elif mode == "Aggressive":
    risk_pct, trailing_stop_pct, scale_in = RISK_PER_TRADE * 1.5, 0.05, True
else:  # Normal
    risk_pct, trailing_stop_pct, scale_in = RISK_PER_TRADE, 0.03, True

# ─── Runtime store ────────────────────────────────────────────────────────────
open_positions = {}  # pair -> { amount, entry_price, ...API response }
trade_log = []  # list of trade-record dicts for metrics

# ─── Core Trade Logic ─────────────────────────────────────────────────────────
def trade_logic(pair: str):
    logger.info(f"🔍 Analyzing {pair}")
    df = fetch_klines(pair=pair, interval=interval, limit=lookback)
    if df.empty or not validate_df(df):
        logger.warning(f"⚠️ Invalid/empty data for {pair}")
        return

    df = add_indicators(df)

    # Adaptive threshold or default
    threshold = 0.5
    if autotune:
        threshold = adaptive_threshold(df)
        logger.debug(f"Adaptive threshold for {pair}: {threshold}")

    raw_signal = generate_signal(df)
    smoothed = smooth_signal(raw_signal)
    latest_signal = smoothed.iloc[-1]

    # Backtest mode override
    if backtest_mode:
        bt = run_backtest(smoothed, df["Close"], threshold)
        st.subheader(f"📊 Backtest: {pair}")
        st.dataframe(bt)
        return

    action = None
    # BUY only if signal strong positive AND you’re not already long
    if latest_signal > threshold and pair not in open_positions:
        action = "buy"
    # SELL (exit) only if signal strong negative AND you’re currently long
    elif latest_signal < -threshold and pair in open_positions:
        action = "sell"
    else:
        return  # either flat no‐trade, or negative signal while flat

    # Enforce position limits on BUY
    if action == "buy":
        if len(open_positions) >= max_positions:
            logger.info("🚫 Max open positions reached → skipping BUY")
            return

    price = df["Close"].iloc[-1]
    amount = (DEFAULT_CAPITAL * risk_pct) / price if action == "buy" else open_positions[pair]["amount"]

    result = place_order(
        pair=pair,
        action=action,
        amount=amount,
        price=price,
        is_dry_run=dry_run
    )

    # Record/exit position
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    record = {
        "timestamp": now,
        "pair": pair,
        "action": action.upper(),
        "amount": amount
    }

    if action == "buy":
        record["entry_price"] = price
        open_positions[pair] = {"amount": amount, "entry_price": price, **result}
    else:  # sell/exit
        record["exit_price"] = price
        open_positions.pop(pair, None)

    trade_log.append(record)
    track_trade_result(result, pair, action.upper())

    # Add trailing stop, if provided
    if "order_price" in result:
        result["trailing_stop"] = result["order_price"] * (1 - trailing_stop_pct)

# ─── Dashboard & Metrics ─────────────────────────────────────────────────────
def display_dashboard():
    st.subheader("📈 Live Dashboard")

    # PnL metrics
    perf = compute_trade_metrics(trade_log, DEFAULT_CAPITAL)
    st.metric("Total Return", f"{perf['total_return']:.2%}")
    st.metric("Win Rate", f"{perf['win_rate']:.2%}")

    # Tuning hints
    for hint in suggest_tuning(perf):
        st.info(f"🔧 {hint}")

    # Open positions table
    if open_positions:
        df_open = pd.DataFrame(open_positions).T
        st.write("🟢 Open Positions")
        st.dataframe(df_open[[
            "amount", "entry_price", "trailing_stop"
        ]])
    else:
        st.info("No active trades.")

    # History table
    if trade_log:
        df_hist = pd.DataFrame(trade_log)
        st.write("📘 Trade History")
        st.dataframe(df_hist[[
            "timestamp", "pair", "action", "amount", "entry_price", "exit_price"
        ]])
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
if start_btn:
    main_loop()
