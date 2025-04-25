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
from utils.helpers import (
    compute_trade_metrics,
    suggest_tuning,
    save_json,
    load_json,
    validate_pair,
)

# ─── Streamlit Configuration ──────────────────────────────────────────────────
st.set_page_config(page_title="CryptoTrader AI", layout="wide")

# ─── Load env & defaults ──────────────────────────────────────────────────────
load_dotenv()
DEFAULT_DRY_RUN = os.getenv("USE_DRY_RUN", "True").lower() == "true"
DEFAULT_CAPITAL = float(os.getenv("DEFAULT_CAPITAL", 10))
RISK_PER_TRADE = float(os.getenv("RISK_PER_TRADE", 0.01))
DEFAULT_THRESHOLD = float(os.getenv("DEFAULT_THRESHOLD", 0.1))  # Adjusted for spot scalping

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ─── File paths for persistence ───────────────────────────────────────────────
POSITIONS_FILE = "state/open_positions.json"
TRADE_LOG_FILE = "state/trade_log.json"

# ─── State Loading ────────────────────────────────────────────────────────────
os.makedirs("state", exist_ok=True)
try:
    if os.path.exists(POSITIONS_FILE):
        open_positions = load_json(POSITIONS_FILE)
    else:
        open_positions = {}  # pair -> { amount, entry_price, ...API response }
except Exception as e:
    logger.warning(f"Failed to load {POSITIONS_FILE}: {e}")
    open_positions = {}

try:
    if os.path.exists(TRADE_LOG_FILE):
        trade_log = load_json(TRADE_LOG_FILE)
    else:
        trade_log = []  # list of trade-record dicts for metrics
except Exception as e:
    logger.warning(f"Failed to load {TRADE_LOG_FILE}: {e}")
    trade_log = []

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

# ─── Core Trade Logic ─────────────────────────────────────────────────────────
def trade_logic(pair: str):
    try:
        base, quote = validate_pair(pair)
    except ValueError as ve:
        logger.error(f"❌ Invalid trading pair '{pair}': {ve}")
        return

    logger.info(f"🔍 Analyzing {pair}")
    df = fetch_klines(pair=pair, interval=interval, limit=lookback)
    if df.empty or not validate_df(df):
        logger.warning(f"⚠️ Invalid/empty data for {pair}")
        return

    df = add_indicators(df)
    raw_signal = generate_signal(df)
    smoothed = smooth_signal(raw_signal)  # Store smoothed signal for thresholding

    # Determine threshold (manual or AI-tuned)
    if autotune:
        threshold = adaptive_threshold(df, target_profit=0.01)
    else:
        threshold = DEFAULT_THRESHOLD

    logger.debug(f"Threshold for {pair}: {threshold}")

    latest_signal = smoothed.iloc[-1]

    # Handle backtesting mode
    if backtest_mode:
        # run_backtest now returns a single DataFrame with type='summary' and type='trade'
        combined_df = run_backtest(smoothed, df["Close"], threshold)
        # split out summary vs. trade details
        summary_df = (
            combined_df
            .loc[combined_df["type"] == "summary"]
            .drop(columns=["type"])
        )
        trades_df = (
            combined_df
            .loc[combined_df["type"] == "trade"]
            .drop(columns=["type"])
        )

        st.write(f"📊 Backtest Summary for {pair}:")
        st.dataframe(summary_df)
        st.write(f"📘 Trade Details for {pair}:")
        st.dataframe(trades_df)
        # Do not return here; allow main_loop to run all pairs in backtest mode

    # Decide BUY or SELL
    action = None
    if latest_signal > threshold and pair not in open_positions:  # Buy signal
        action = "buy"
    elif pair in open_positions:  # Sell signal if already long
        action = "sell"
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
        record = {
            "timestamp": now,
            "pair": pair,
            "action": "BUY",
            "amount": amount,
            "entry_price": price,
        }
        trade_log.append(record)  # Record BUY action
        open_positions[pair] = {"amount": amount, "entry_price": price}
        logger.info(f"📥 BUY {pair} at {price:.2f}")
        # Persist state
        save_json(open_positions, POSITIONS_FILE, indent=2)
        save_json(trade_log, TRADE_LOG_FILE, indent=2)
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
            "amount": position["amount"],
            "entry_price": position["entry_price"],  # Include entry price for performance tracking
            "exit_price": exit_price,
            "return_pct": return_pct,
        }
        trade_log.append(record)  # Record SELL action
        logger.info(f"📤 SELL {pair} at {exit_price:.2f} → Return: {return_pct:.2%}")
        # Persist state
        save_json(open_positions, POSITIONS_FILE, indent=2)
        save_json(trade_log, TRADE_LOG_FILE, indent=2)

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

# ─── Main Loop ───────────────────────────────────────────────────────────────
def main_loop():
    # If user asked for backtest, run it once for each pair, display, then exit.
    if backtest_mode:
        for pair in TRADING_PAIRS:
            try:
                trade_logic(pair)           # this will render backtest tables
            except Exception as e:
                logger.exception(f"❌ Error in backtest for {pair}: {e}")
        display_dashboard()               # show combined metrics (from trade_log)
        return                           # exit main_loop immediately

    # Otherwise, live-trading mode: repeat every 10s
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
