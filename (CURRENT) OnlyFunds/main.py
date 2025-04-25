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
DEFAULT_STOP_LOSS = 0.005  # 0.5%
DEFAULT_TAKE_PROFIT = 0.01  # 1%
DEFAULT_FEE = 0.0004  # 0.04%

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ─── File paths for persistence ───────────────────────────────────────────────
POSITIONS_FILE = "state/open_positions.json"
TRADE_LOG_FILE = "state/trade_log.json"
CAPITAL_FILE = "state/current_capital.json"

# ─── State Loading ────────────────────────────────────────────────────────────
os.makedirs("state", exist_ok=True)
try:
    if os.path.exists(POSITIONS_FILE):
        open_positions = load_json(POSITIONS_FILE)
    else:
        open_positions = {}
except Exception as e:
    logger.warning(f"Failed to load {POSITIONS_FILE}: {e}")
    open_positions = {}

try:
    if os.path.exists(TRADE_LOG_FILE):
        trade_log = load_json(TRADE_LOG_FILE)
    else:
        trade_log = []
except Exception as e:
    logger.warning(f"Failed to load {TRADE_LOG_FILE}: {e}")
    trade_log = []

try:
    if os.path.exists(CAPITAL_FILE):
        current_capital = load_json(CAPITAL_FILE)
        if not isinstance(current_capital, (float, int)):
            current_capital = DEFAULT_CAPITAL
    else:
        current_capital = DEFAULT_CAPITAL
except Exception as e:
    logger.warning(f"Failed to load {CAPITAL_FILE}: {e}")
    current_capital = DEFAULT_CAPITAL

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
lookback = st.sidebar.slider("Historical Lookback", 300, 2000, 1000)
max_positions = st.sidebar.number_input("Max Open Positions", 1, 5, 2)
stop_loss_pct = st.sidebar.number_input("Stop-Loss %", 0.0, 10.0, DEFAULT_STOP_LOSS*100.0, step=0.1) / 100
take_profit_pct = st.sidebar.number_input("Take-Profit %", 0.0, 10.0, DEFAULT_TAKE_PROFIT*100.0, step=0.1) / 100
fee_pct = st.sidebar.number_input("Trade Fee %", 0.0, 1.0, DEFAULT_FEE*100.0, step=0.01) / 100

threshold_slider = st.sidebar.slider(
    "Entry Threshold", 
    min_value=0.0, max_value=1.0, 
    value=DEFAULT_THRESHOLD, step=0.01,
    help="How strong must the signal be before we BUY/SELL?"
)

start_btn = st.sidebar.button("🚀 Start Trading Bot (Spot Only)")
if start_btn:
    st.success("Bot started! (Spot market only)")
else:
    st.info("Ready. Configure & click Start.")

if mode == "Conservative":
    risk_pct, trailing_stop_pct, scale_in = RISK_PER_TRADE * 0.5, 0.01, False
elif mode == "Aggressive":
    risk_pct, trailing_stop_pct, scale_in = RISK_PER_TRADE * 1.5, 0.05, True
else:
    risk_pct, trailing_stop_pct, scale_in = RISK_PER_TRADE, 0.03, True

@st.cache_data(show_spinner=False)
def cached_fetch_klines(pair, interval, limit):
    return fetch_klines(pair=pair, interval=interval, limit=limit)

def trade_logic(pair: str, current_capital):
    try:
        base, quote = validate_pair(pair)
    except ValueError as ve:
        logger.error(f"❌ Invalid trading pair '{pair}': {ve}")
        return None, current_capital

    logger.info(f"🔍 Analyzing {pair}")
    df = cached_fetch_klines(pair, interval, lookback)
    if df.empty or not validate_df(df):
        logger.warning(f"⚠️ Invalid/empty data for {pair}")
        return None, current_capital

    # Volatility regime filter (optional, stub)
    # vol = df["volatility"].rolling(20).mean().iloc[-1]
    # if vol < df["volatility"].quantile(0.2):
    #     logger.info(f"Skipping {pair} due to low volatility regime")
    #     return None, current_capital

    # Multi-timeframe confirmation (optional, stub)
    # h1_df = cached_fetch_klines(pair, "1h", int(lookback / 4))
    # indicator_h1 = ... # compute on h1_df
    # if indicator_h1.iloc[-1] < 0:
    #     return None, current_capital

    df = add_indicators(df)
    raw_signal = generate_signal(df)
    smoothed = smooth_signal(raw_signal)

    if autotune:
        threshold = adaptive_threshold(df, target_profit=0.01)
    else:
        threshold = threshold_slider

    logger.debug(f"Threshold for {pair}: {threshold}")
    latest_signal = smoothed.iloc[-1]

    # --- Backtest mode: read-only, separate log ---
    if backtest_mode:
        combined_df = run_backtest(
            smoothed, df["Close"], threshold,
            initial_capital=DEFAULT_CAPITAL,
            risk_pct=risk_pct,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
            fee_pct=fee_pct,
        )
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
        return None, current_capital

    # --- Live trading logic ---
    action = None
    if latest_signal > threshold and pair not in open_positions:
        action = "buy"
    elif pair in open_positions:
        action = "sell"
    else:
        return None, current_capital

    if action == "buy":
        if len(open_positions) >= max_positions:
            logger.info("🚫 Max open positions reached → skipping BUY")
            return None, current_capital

    price = df["Close"].iloc[-1]
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    if action == "buy":
        amount = (current_capital * risk_pct) / price
        record = {
            "timestamp": now,
            "pair": pair,
            "action": "BUY",
            "amount": amount,
            "entry_price": price,
        }
        trade_log.append(record)
        open_positions[pair] = {"amount": amount, "entry_price": price}
        logger.info(f"📥 BUY {pair} at {price:.2f}")
        save_json(open_positions, POSITIONS_FILE, indent=2)
        save_json(trade_log, TRADE_LOG_FILE, indent=2)
        return None, current_capital

    if action == "sell":
        position = open_positions.pop(pair)
        exit_price = price
        return_pct = (exit_price - position["entry_price"]) / position["entry_price"]
        record = {
            "timestamp": now,
            "pair": pair,
            "action": "SELL",
            "amount": position["amount"],
            "entry_price": position["entry_price"],
            "exit_price": exit_price,
            "return_pct": return_pct,
        }
        trade_log.append(record)
        logger.info(f"📤 SELL {pair} at {exit_price:.2f} → Return: {return_pct:.2%}")
        # Compound capital
        current_capital *= (1 + return_pct)
        save_json(open_positions, POSITIONS_FILE, indent=2)
        save_json(trade_log, TRADE_LOG_FILE, indent=2)
        save_json(current_capital, CAPITAL_FILE, indent=2)

        if not backtest_mode:
            result = place_order(
                pair=pair,
                action=action,
                amount=position["amount"],
                price=exit_price,
                is_dry_run=dry_run,
            )
            track_trade_result(result, pair, action.upper())
        return None, current_capital

    return None, current_capital

def display_dashboard(current_capital):
    st.subheader("📈 Live Dashboard")
    perf = compute_trade_metrics(trade_log, current_capital)
    st.metric("Total Return", f"{perf['total_return']:.2%}")
    st.metric("Win Rate",     f"{perf['win_rate']:.2%}")

    if open_positions:
        st.write("🟢 Open Positions")
        df_open = pd.DataFrame(open_positions).T.reset_index(drop=True)
        desired_cols = ["amount", "entry_price"]
        cols = [c for c in desired_cols if c in df_open.columns]
        st.dataframe(df_open[cols])
    else:
        st.info("No active trades.")

    if trade_log:
        st.write("📘 Trade History")
        st.dataframe(pd.DataFrame(trade_log))
    else:
        st.info("No trade history yet.")

def main_loop():
    global current_capital
    if backtest_mode:
        with st.spinner("Running backtest…"):
            for pair in TRADING_PAIRS:
                trade_logic(pair, DEFAULT_CAPITAL)
        return

    last_timestamps = {pair: None for pair in TRADING_PAIRS}
    while True:
        for pair in TRADING_PAIRS:
            df = cached_fetch_klines(pair, interval, lookback)
            if df.empty or not validate_df(df):
                continue
            newest = df.index[-1]
            if newest != last_timestamps[pair]:
                _, updated_capital = trade_logic(pair, current_capital)
                current_capital = updated_capital
                last_timestamps[pair] = newest
        display_dashboard(current_capital)
        time.sleep(1)

if start_btn:
    main_loop()
