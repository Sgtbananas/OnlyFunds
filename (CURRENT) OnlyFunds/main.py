https://github.com/Sgtbananas/OnlyFunds/tree/Sgtbananas-patch-1/(CURRENT)%20OnlyFundsmain.py

import os
import logging
import time

import streamlit as st
import pandas as pd
from dotenv import load_dotenv

from core.core_data import fetch_klines, validate_df, add_indicators, TRADING_PAIRS
from core.core_signals import generate_signal, smooth_signal, track_trade_result
from core.trade_execution import place_order
from core.autotune import tune_threshold
from core.backtester import run_backtest

# ← updated imports here:
from utils.helpers import calculate_performance, suggest_tuning_parameters

# Load environment variables
load_dotenv()

# Default settings from .env
DEFAULT_DRY_RUN  = os.getenv("USE_DRY_RUN", "True").lower() == "true"
DEFAULT_CAPITAL = float(os.getenv("DEFAULT_CAPITAL", 1000))
RISK_PER_TRADE  = float(os.getenv("RISK_PER_TRADE", 0.01))

# Logging config
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Streamlit UI setup
st.set_page_config(page_title="CryptoTrader AI", layout="wide")
st.title("🧠 CryptoTrader AI Bot")
st.sidebar.header("⚙️ Configuration")

# UI Toggles and Inputs
dry_run       = st.sidebar.checkbox("Dry Run Mode (Simulated Trades)", value=DEFAULT_DRY_RUN)
autotune      = st.sidebar.checkbox("Enable AI Autotuning", value=False)
backtest_mode = st.sidebar.checkbox("Enable Backtesting", value=False)
mode          = st.sidebar.selectbox(
    "Trading Mode", ["Conservative", "Normal", "Aggressive"], index=1
)
interval      = st.sidebar.selectbox(
    "Candle Interval", ["5m", "15m", "30m", "1h", "4h", "1d"], index=0
)
lookback      = st.sidebar.slider("Historical Lookback", min_value=100, max_value=1000, value=300)
max_positions = st.sidebar.number_input("Max Open Positions", min_value=1, max_value=5, value=2)

# Runtime state
open_positions = {}
trade_log      = []

# Strategy parameters based on mode
if mode == "Conservative":
    risk_pct          = RISK_PER_TRADE * 0.5
    trailing_stop_pct = 0.01
    scale_in          = False
elif mode == "Aggressive":
    risk_pct          = RISK_PER_TRADE * 2.0
    trailing_stop_pct = 0.05
    scale_in          = True
else:  # Normal
    risk_pct          = RISK_PER_TRADE
    trailing_stop_pct = 0.03
    scale_in          = True

def trade_logic(pair: str):
    logger.info(f"🔍 Analyzing {pair}")
    df = fetch_klines(pair=pair, interval=interval, limit=lookback)
    if df.empty or not validate_df(df):
        logger.warning(f"⚠️ Invalid or empty data for {pair}")
        return

    df = add_indicators(df)

    # Determine threshold
    threshold = 0.5
    if autotune:
        threshold = tune_threshold(df, pair)

    # Generate and smooth signal
    raw_signal    = generate_signal(df)
    smoothed      = smooth_signal(raw_signal)
    latest_signal = smoothed.iloc[-1]

    # Backtesting mode
    if backtest_mode:
        bt = run_backtest(smoothed, df["Close"], threshold)
        st.subheader(f"📊 Backtest: {pair}")
        st.dataframe(bt)
        return

    # Determine action
    action = None
    if latest_signal > threshold:
        action = "buy"
    elif latest_signal < -threshold:
        action = "sell"

    # Execute trade if action identified
    if action:
        # Enforce position limits on BUY
        if action == "buy":
            if pair in open_positions:
                logger.info(f"⏸ Already in {pair}, skipping buy")
                return
            if len(open_positions) >= max_positions:
                logger.info("🚫 Max positions reached")
                return

        # Calculate trade amount
        if action == "buy":
            trade_amount = DEFAULT_CAPITAL * risk_pct
        else:
            # for SELL, close using the same qty we bought
            trade_amount = open_positions.get(pair, {}).get("amount", DEFAULT_CAPITAL * risk_pct)

        # Place order
        result = place_order(
            pair=pair,
            action=action,
            amount=trade_amount,
            price=df["Close"].iloc[-1],
            is_dry_run=dry_run
        )

        # Record position and log
        if action == "buy":
            open_positions[pair] = result
        else:
            open_positions.pop(pair, None)

        trade_log.append(result)
        track_trade_result(result, pair, action.upper())

        # Attach trailing stop if returned
        if "order_price" in result:
            result["trailing_stop"] = result["order_price"] * (1 - trailing_stop_pct)

def display_dashboard():
    st.subheader("📈 Live Dashboard")
    # Performance metrics
    perf = calculate_performance(trade_log, initial_capital=DEFAULT_CAPITAL)
    st.metric("Total Return", f"{perf['total_return']:.2%}")
    st.metric("Win Rate",    f"{perf['win_rate']:.2%}")

    # Tuning suggestions
    suggestions = suggest_tuning_parameters(perf)
    if suggestions:
        st.info("🔧 " + "\n".join(suggestions))

    # Open positions & trade history
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

def main_loop():
    while True:
        for pair in TRADING_PAIRS:
            try:
                trade_logic(pair)
            except Exception as e:
                logger.exception(f"❌ Error in cycle for {pair}: {e}")
        display_dashboard()
        time.sleep(10)

# Entry point
if st.sidebar.button("🚀 Start Trading Bot"):
    st.success("Bot started!")
    main_loop()
else:
    st.info("Ready to start. Configure and click launch.")
