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

# ─── Streamlit Configuration ──────────────────────────────────────────────────
# This must be the first Streamlit command in the script
st.set_page_config(page_title="CryptoTrader AI", layout="wide")

# ─── Load env & defaults ──────────────────────────────────────────────────────
load_dotenv()
DEFAULT_DRY_RUN = os.getenv("USE_DRY_RUN", "True").lower() == "true"
DEFAULT_CAPITAL = float(os.getenv("DEFAULT_CAPITAL", 1000))
RISK_PER_TRADE = float(os.getenv("RISK_PER_TRADE", 0.01))
DEFAULT_THRESHOLD = float(os.getenv("DEFAULT_THRESHOLD", 0.05))  # Lower default threshold for better trading

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ─── Streamlit UI ─────────────────────────────────────────────────────────────
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
lookback = st.sidebar.slider("Historical Lookback", 300, 2000, 1000)  # Increased default to 1000
max_positions = st.sidebar.number_input("Max Open Positions", 1, 5, 2)

# Manual override for entry threshold
threshold_slider = st.sidebar.slider(
    "Entry Threshold", 
    min_value=0.0, max_value=1.0, 
    value=DEFAULT_THRESHOLD, step=0.01,
    help="How strong must the signal be before we BUY/SELL?"
)

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
    raw_signal = generate_signal(df)
    df["signal"] = smooth_signal(raw_signal)  # Store smoothed signal for thresholding

    # Determine threshold (manual or AI-tuned)
    if autotune:
        threshold = adaptive_threshold(df, signal_col="signal")
    else:
        threshold = DEFAULT_THRESHOLD

    logger.debug(f"Threshold for {pair}: {threshold}")

    latest_signal = df["signal"].iloc[-1]

    # Decide BUY or EXIT
    action = None
    if latest_signal > threshold and pair not in open_positions:
        action = "buy"
    elif latest_signal < 0 and pair in open_positions:
        action = "sell"  # Close existing long position
    else:
        return  # No action

    # Enforce position limits on BUY
    if action == "buy":
        if len(open_positions) >= max_positions:
            logger.info("🚫 Max open positions reached → skipping BUY")
            return

    price = df["Close"].iloc[-1]
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    # Handle BUY action
    if action == "buy":
        amount = (DEFAULT_CAPITAL * risk_pct) / price
        open_positions[pair] = {
            "timestamp": now,
            "action": "BUY",
            "entry_price": price,
            "amount": amount,
        }
        logger.info(f"📥 BUY {pair} at {price:.2f}")
        return

    # Handle SELL action
    if action == "sell":
        position = open_positions.pop(pair)
        exit_price = price
        return_pct = (exit_price - position["entry_price"]) / position["entry_price"]

        record = {
            "timestamp": now,
            "pair": pair,
            "action": "SELL",
            "entry_price": position["entry_price"],
            "exit_price": exit_price,
            "amount": position["amount"],
            "return_pct": return_pct,
        }
        trade_log.append(record)
        logger.info(f"📤 SELL {pair} at {exit_price:.2f} → Return: {return_pct:.2%}")

        if not backtest_mode:
            result = place_order(
                pair=pair,
                action=action,
                amount=position["amount"],
                price=exit_price,
                is_dry_run=dry_run,
            )
            track_trade_result(result, pair, action.upper())

# ─── Dashboard & Metrics ─────────────────────────────────────────────────────
def display_dashboard():
    st.subheader("📈 Live Dashboard")

    # Performance metrics
    perf = compute_trade_metrics(trade_log, DEFAULT_CAPITAL)
    st.metric("Total Return", f"{perf['total_return']:.2%}")
    st.metric("Win Rate",     f"{perf['win_rate']:.2%}")

    # Open positions
    if open_positions:
        st.write("🟢 Open Positions")
        df_open = pd.DataFrame(open_positions).T.reset_index(drop=True)
        desired_cols = ["amount", "entry_price"]
        cols = [c for c in desired_cols if c in df_open.columns]
        st.dataframe(df_open[cols])
    else:
        st.info("No active trades.")

    # Trade History
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
if start_btn:
    main_loop()
