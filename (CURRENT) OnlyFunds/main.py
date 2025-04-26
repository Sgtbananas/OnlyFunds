import os
import logging
import time
from datetime import datetime

import streamlit as st
import pandas as pd
from dotenv import load_dotenv

from core.core_data import fetch_klines, validate_df, add_indicators, TRADING_PAIRS
from core.core_signals import (
    generate_signal, smooth_signal, adaptive_threshold, track_trade_result,
)
from core.trade_execution import place_order, place_limit_order, cancel_order
from core.backtester import run_backtest
from core.grid_trader import GridTrader
from utils.helpers import (
    compute_trade_metrics, compute_grid_metrics, suggest_tuning, save_json, load_json, validate_pair, log_grid_trade
)
from data.logs.logs import log_trade

st.set_page_config(page_title="CryptoTrader AI", layout="wide")

load_dotenv()
DEFAULT_DRY_RUN = os.getenv("USE_DRY_RUN", "True").lower() == "true"
DEFAULT_CAPITAL = float(os.getenv("DEFAULT_CAPITAL", 10))
RISK_PER_TRADE = float(os.getenv("RISK_PER_TRADE", 0.01))
DEFAULT_THRESHOLD = float(os.getenv("DEFAULT_THRESHOLD", 0.1))
DEFAULT_STOP_LOSS = 0.005
DEFAULT_TAKE_PROFIT = 0.01
DEFAULT_FEE = 0.0004
MIN_SIZE = 0.0001

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

POSITIONS_FILE = "state/open_positions.json"
TRADE_LOG_FILE = "state/trade_log.json"
CAPITAL_FILE = "state/current_capital.json"

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

# === STRATEGY SELECTION ===
st.title("🧠 CryptoTrader AI Bot (SPOT Market Only)")
st.sidebar.header("⚙️ Configuration")

strategy_mode = st.sidebar.selectbox(
    "Strategy Mode",
    ["Signal Trading", "Grid Trading", "Grid Backtest"],  # Added grid backtest!
    index=0
)

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
max_positions = st.sidebar.number_input("Max Open Positions", 1, 30, 2)
stop_loss_pct = st.sidebar.number_input("Stop-Loss %", 0.0, 10.0, DEFAULT_STOP_LOSS*100.0, step=0.1) / 100
take_profit_pct = st.sidebar.number_input("Take-Profit %", 0.0, 10.0, DEFAULT_TAKE_PROFIT*100.0, step=0.1) / 100
fee_pct = st.sidebar.number_input("Trade Fee %", 0.0, 1.0, DEFAULT_FEE*100.0, step=0.01) / 100

# --- Grid Controls ---
if strategy_mode in ["Grid Trading", "Grid Backtest"]:
    st.sidebar.markdown("#### Grid Trading Settings")
    grid_pair = st.sidebar.selectbox("Grid Pair", TRADING_PAIRS, index=0)
    grid_levels = st.sidebar.number_input("Grid Levels (per side)", min_value=1, max_value=20, value=6)
    grid_pct = st.sidebar.number_input("Grid Spacing (%)", min_value=0.1, max_value=5.0, value=0.4, step=0.1) / 100
    grid_size = st.sidebar.number_input("Grid Order Size", min_value=0.0001, value=0.001)
    grid_direction = st.sidebar.selectbox("Grid Direction", ["both", "buy", "sell"], index=0)
    grid_trailing = st.sidebar.checkbox("Enable Trailing Grid", value=False)
    grid_adaptive = st.sidebar.checkbox("Enable Adaptive Spacing", value=False)
    grid_start_btn = st.sidebar.button("Start Grid Trader" if strategy_mode == "Grid Trading" else "Run Grid Backtest")
else:
    threshold_slider = st.sidebar.slider(
        "Entry Threshold",
        min_value=0.0, max_value=1.0,
        value=DEFAULT_THRESHOLD, step=0.01,
        help="How strong must the signal be before we BUY/SELL?"
    )
    start_btn = st.sidebar.button("🚀 Start Trading Bot (Spot Only)")

if "grid_traders" not in st.session_state:
    st.session_state["grid_traders"] = {}

def grid_dashboard(grid: GridTrader):
    st.subheader(f"🤖 Grid Trader for {grid.pair}")
    st.write(f"Base price: {grid.base:.4f} | Levels: {grid.levels} | Spacing: {grid.grid_pct:.4%} | Size: {grid.size}")
    metrics = compute_grid_metrics([o.to_dict() for o in grid.orders])
    st.write(f"Total Fills: {metrics['fills']}, Buys: {metrics['buy_fills']}, Sells: {metrics['sell_fills']}, PnL: {metrics['total_pnl']:.6f}")
    df_orders = pd.DataFrame([o.to_dict() for o in grid.orders])
    if not df_orders.empty:
        st.dataframe(df_orders[["side", "price", "size", "active", "filled", "fill_price", "fill_time", "order_id"]])
        st.write(f"Active Orders: {df_orders['active'].sum()}")
    else:
        st.info("No grid orders yet.")

def run_grid_trading():
    key = f"{grid_pair}"
    df = fetch_klines(pair=grid_pair, interval=interval, limit=lookback)
    if df.empty:
        st.error(f"No data for {grid_pair}")
        return
    current_price = df["Close"].iloc[-1]
    volatility = df["Close"].pct_change().rolling(20).std().iloc[-1] if grid_adaptive else None
    spacing = grid_pct
    if grid_adaptive and volatility is not None:
        spacing = min(max(volatility, 0.001), 0.03)
    grid = GridTrader(
        pair=grid_pair,
        base_price=current_price,
        grid_levels=int(grid_levels),
        grid_spacing_pct=spacing,
        size=grid_size,
        direction=grid_direction,
        trailing=grid_trailing,
        adaptive=grid_adaptive,
    )
    grid.build_orders()
    grid.place_orders(lambda pair, side, size, price:
        place_limit_order(pair, side, size, price, is_dry_run=dry_run)
    )
    st.session_state.grid_traders[key] = grid
    grid_dashboard(grid)

def step_grid_simulation(grid: GridTrader, price, timestamp=None):
    # Simulate fills for all orders at this price
    grid.check_and_fill_orders(price, timestamp=timestamp, log_func=log_grid_trade)
    # Optionally trail grid
    if grid.trailing:
        grid.trail_grid(price)

def run_grid_backtest():
    df = fetch_klines(pair=grid_pair, interval=interval, limit=lookback)
    if df.empty:
        st.error(f"No data for {grid_pair}")
        return
    prices = df["Close"]
    idx = df.index
    grid_kwargs = dict(
        pair=grid_pair,
        base_price=prices.iloc[0],
        grid_levels=int(grid_levels),
        grid_spacing_pct=grid_pct,
        size=grid_size,
        direction=grid_direction,
        trailing=grid_trailing,
        adaptive=grid_adaptive,
    )
    result_df = run_backtest(
        signal=None,
        prices=prices,
        grid_mode=True,
        grid_kwargs=grid_kwargs,
        log_func=log_grid_trade
    )
    st.subheader(f"Grid Backtest Results for {grid_pair}")
    st.dataframe(result_df)
    summary = result_df.iloc[0].to_dict()
    st.write(f"Total Fills: {summary.get('trades', 0)}, Buys: {summary.get('buy_fills', 0)}, Sells: {summary.get('sell_fills', 0)}, PnL: {summary.get('total_pnl', 0):.6f}")

def main_loop_signal_trading():
    global current_capital
    last_timestamps = {pair: None for pair in TRADING_PAIRS}
    while True:
        for pair in TRADING_PAIRS:
            df = fetch_klines(pair=pair, interval=interval, limit=lookback)
            if df.empty or not validate_df(df):
                continue
            newest = df.index[-1]
            if newest != last_timestamps[pair]:
                _, updated_capital = trade_logic(pair, current_capital)
                current_capital = updated_capital
                last_timestamps[pair] = newest
        display_dashboard(current_capital)
        time.sleep(1)

def trade_logic(pair: str, current_capital):
    try:
        base, quote = validate_pair(pair)
    except ValueError as ve:
        logger.error(f"❌ Invalid trading pair '{pair}': {ve}")
        return None, current_capital

    logger.info(f"🔍 Analyzing {pair}")
    df = fetch_klines(pair=pair, interval=interval, limit=lookback)
    if df.empty or not validate_df(df):
        logger.warning(f"⚠️ Invalid/empty data for {pair}")
        return None, current_capital

    df = add_indicators(df)
    raw_signal = generate_signal(df)
    smoothed = smooth_signal(raw_signal)

    if autotune:
        threshold = adaptive_threshold(df, target_profit=0.01)
    else:
        threshold = threshold_slider

    logger.debug(f"Threshold for {pair}: {threshold}")
    latest_signal = smoothed.iloc[-1]

    # ML confidence filter stub (to be implemented)
    # if enable_ml:
    #     features = [df["rsi"].iloc[-1], df["macd"].iloc[-1], df["ema_diff"].iloc[-1], df["Close"].pct_change().rolling(20).std().iloc[-1]]
    #     prob = ml_confidence(model, features)
    #     if prob < min_signal_conf:
    #         return None, current_capital

    # Sentiment filter stub
    # sentiment = fetch_sentiment_score(pair)
    # if sentiment < -0.2:
    #     return None, current_capital

    # --- Backtest mode: read-only, separate log ---
    if backtest_mode:
        combined_df = run_backtest(
            smoothed, df["Close"], threshold,
            initial_capital=DEFAULT_CAPITAL,
            risk_pct=RISK_PER_TRADE,
            stop_loss_pct=DEFAULT_STOP_LOSS,
            take_profit_pct=DEFAULT_TAKE_PROFIT,
            fee_pct=DEFAULT_FEE,
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
    perf = compute_trade_metrics(trade_log, DEFAULT_CAPITAL)
    current_capital_live = DEFAULT_CAPITAL * (1 + perf["total_return"]/100)
    net_profit = current_capital_live - DEFAULT_CAPITAL
    risk_from_pct = DEFAULT_CAPITAL * RISK_PER_TRADE
    risk_from_pf = max(net_profit * 0.05, 0.0)
    usd_to_risk = max(1.0, risk_from_pct, risk_from_pf)
    amount = usd_to_risk / price

    if amount < MIN_SIZE:
        logger.warning(f"Calculated amount {amount:.6f} below min size {MIN_SIZE} → skipping BUY")
        return None, current_capital

    if action == "buy":
        record = {
            "timestamp": now,
            "pair": pair,
            "action": "BUY",
            "amount": amount,
            "entry_price": price,
        }
        trade_log.append(record)
        open_positions[pair] = {"amount": amount, "entry_price": price}
        logger.info(f"📥 BUY {pair} at {price:.2f} (amount={amount:.6f})")
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
        logger.info(f"📤 SELL {pair} at {exit_price:.2f} → Return: {return_pct:.4%}")
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
    perf = compute_trade_metrics(trade_log, DEFAULT_CAPITAL)
    current_capital_live = DEFAULT_CAPITAL * (1 + perf["total_return"]/100)
    st.subheader("📈 Live Dashboard")
    st.metric("Starting Capital", f"{DEFAULT_CAPITAL:.2f} USDT")
    st.metric("Current Capital",  f"{current_capital_live:.4f} USDT")
    st.metric("Total Return",     f"{perf['total_return']:.2%}")
    st.metric("Win Rate",         f"{perf['win_rate']:.2%}")

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

# Main UI logic
if strategy_mode == "Grid Trading":
    if grid_start_btn:
        run_grid_trading()
    # Always show grid dashboard for the selected grid in session
    key = f"{grid_pair}"
    if key in st.session_state.grid_traders:
        grid_dashboard(st.session_state.grid_traders[key])

elif strategy_mode == "Grid Backtest":
    if grid_start_btn:
        run_grid_backtest()
else:
    if start_btn:
        main_loop_signal_trading()