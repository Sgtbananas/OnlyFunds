# main.py

import os
import logging
import time

import streamlit as st
import pandas as pd
from dotenv import load_dotenv

from core.core_data      import fetch_klines, validate_df, add_indicators, TRADING_PAIRS
from core.core_signals   import (
    generate_signal,
    smooth_signal,
    track_trade_result,
    adaptive_threshold
)
from core.trade_execution import place_order
from core.backtester     import run_backtest
from utils.helpers       import compute_trade_metrics, suggest_tuning

# Load environment variables
load_dotenv()

# Defaults from .env
DEFAULT_DRY_RUN  = os.getenv("USE_DRY_RUN", "True").lower() == "true"
DEFAULT_CAPITAL  = float(os.getenv("DEFAULT_CAPITAL", 1000))
RISK_PER_TRADE   = float(os.getenv("RISK_PER_TRADE", 0.01))

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Streamlit setup
st.set_page_config(page_title="CryptoTrader AI", layout="wide")
st.title("🧠 CryptoTrader AI Bot")
st.sidebar.header("⚙️ Configuration")

# ─── Sidebar controls ─────────────────────────────────────────────────────────
dry_run        = st.sidebar.checkbox("Dry Run Mode (Simulated)", value=DEFAULT_DRY_RUN)
autotune       = st.sidebar.checkbox("Enable Adaptive-Threshold Autotune", value=False)
backtest_mode  = st.sidebar.checkbox("Enable Backtesting", value=False)
mode           = st.sidebar.selectbox(
    "Trading Mode",
    ["Conservative", "Normal", "Aggressive"],
    index=1
)
interval       = st.sidebar.selectbox(
    "Candle Interval",
    ["5m", "15m", "30m", "1h", "4h", "1d"],
    index=0
)
lookback       = st.sidebar.slider("Historical Lookback", 100, 1000, 300)
max_positions  = st.sidebar.number_input("Max Open Positions", 1, 5, 2)

# ─── Strategy parameters by mode ─────────────────────────────────────────────
if mode == "Conservative":
    risk_pct          = RISK_PER_TRADE * 0.5
    trailing_stop_pct = 0.01
    scale_in          = False
elif mode == "Aggressive":
    risk_pct          = RISK_PER_TRADE * 1.5
    trailing_stop_pct = 0.05
    scale_in          = True
else:  # Normal
    risk_pct          = RISK_PER_TRADE
    trailing_stop_pct = 0.03
    scale_in          = True

# ─── Runtime state ───────────────────────────────────────────────────────────
open_positions = {}   # pair → order info
trade_log      = []   # list of dicts for compute_trade_metrics

# ─── Core trade logic ────────────────────────────────────────────────────────
def trade_logic(pair: str):
    logger.info(f"🔍 Analyzing {pair}")
    df = fetch_klines(pair=pair, interval=interval, limit=lookback)
    if df.empty or not validate_df(df):
        logger.warning(f"⚠️ Invalid or empty data for {pair}")
        return

    # Add indicators
    df = add_indicators(df)

    # Decide threshold
    threshold = 0.5
    if autotune:
        threshold = adaptive_threshold(df)
        logger.debug(f"Adaptive threshold for {pair}: {threshold}")

    # Compute and smooth signal
    raw_signal    = generate_signal(df)
    smoothed      = smooth_signal(raw_signal)
    latest_signal = smoothed.iloc[-1]

    # If backtesting, show results and skip live
    if backtest_mode:
        bt = run_backtest(smoothed, df["Close"], threshold)
        st.subheader(f"📊 Backtest: {pair}")
        st.dataframe(bt)
        return

    # Determine action
    action = None
    if latest_signal >  threshold:
        action = "buy"
    elif latest_signal < -threshold:
        action = "sell"

    if not action:
        return

    # Skip buying if already in or at max positions
    if action == "buy":
        if pair in open_positions:
            logger.info(f"⏸ Already in {pair} → skip buy")
            return
        if len(open_positions) >= max_positions:
            logger.info("🚫 Max open positions reached → skip new buy")
            return

    # Compute trade amount
    if action == "buy":
        amount = DEFAULT_CAPITAL * risk_pct
    else:  # selling
        amount = open_positions.get(pair, {}).get("amount", DEFAULT_CAPITAL * risk_pct)

    price = df["Close"].iloc[-1]
    result = place_order(
        pair      = pair,
        action    = action,
        amount    = amount,
        price     = price,
        is_dry_run= dry_run
    )

    # Record position / close
    if action == "buy":
        open_positions[pair] = result
    else:
        open_positions.pop(pair, None)

    # Log and track
    trade_log.append(result)
    track_trade_result(result, pair, action.upper())

    # Attach trailing-stop if available
    if "order_price" in result:
        result["trailing_stop"] = result["order_price"] * (1 - trailing_stop_pct)

# ─── Dashboard / Metrics ────────────────────────────────────────────────────
def display_dashboard():
    st.subheader("📈 Live Dashboard")

    # Performance metrics
    perf = compute_trade_metrics(trade_log, DEFAULT_CAPITAL)
    st.metric("Total Return", f"{perf['total_return']:.2%}")
    st.metric("Win Rate",     f"{perf['win_rate']:.2%}")

    # Tuning suggestions
    suggestions = suggest_tuning(perf)
    for s in suggestions:
        st.info(f"🔧 {s}")

    # Open Positions
    if open_positions:
        st.write("🟢 Open Positions")
        st.dataframe(pd.DataFrame(open_positions).T)
    else:
        st.info("No active trades.")

    # Trade History
    if trade_log:
        st.write("📘 Trade History")
        st.dataframe(pd.DataFrame(trade_log))
    else:
        st.info("No trade history yet.")

# ─── Main loop & entry point ────────────────────────────────────────────────
def main_loop():
    while True:
        for pair in TRADING_PAIRS:
            try:
                trade_logic(pair)
            except Exception as e:
                logger.exception(f"❌ Error in cycle for {pair}: {e}")
        display_dashboard()
        time.sleep(10)  # Polling delay

if st.sidebar.button("🚀 Start Trading Bot"):
    st.success("Bot started! 🚀")
    main_loop()
else:
    st.info("Ready. Configure settings & click ‘Start Trading Bot’ to begin.")
