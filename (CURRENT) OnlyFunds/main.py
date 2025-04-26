# ... [existing imports] ...
from core.grid_trader import GridTrader
from core.trade_execution import place_limit_order
# ... [rest of imports unchanged] ...

st.set_page_config(page_title="CryptoTrader AI", layout="wide")

# ... [env, logging, file setup unchanged] ...

# === STRATEGY SELECTION ===
st.title("🧠 CryptoTrader AI Bot (SPOT Market Only)")
st.sidebar.header("⚙️ Configuration")

strategy_mode = st.sidebar.selectbox(
    "Strategy Mode",
    ["Signal Trading", "Grid Trading"],  # Expand as you add more!
    index=0
)

# --- Common Controls ---
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

# --- Grid Controls (show only if Grid Trading) ---
if strategy_mode == "Grid Trading":
    st.sidebar.markdown("#### Grid Trading Settings")
    grid_pair = st.sidebar.selectbox("Grid Pair", TRADING_PAIRS, index=0)
    grid_levels = st.sidebar.number_input("Grid Levels (per side)", min_value=1, max_value=20, value=6)
    grid_pct = st.sidebar.number_input("Grid Spacing (%)", min_value=0.1, max_value=5.0, value=0.4, step=0.1) / 100
    grid_size = st.sidebar.number_input("Grid Order Size", min_value=0.0001, value=0.001)
    grid_direction = st.sidebar.selectbox("Grid Direction", ["both", "buy", "sell"], index=0)
    grid_start_btn = st.sidebar.button("Start Grid Trader")
else:
    threshold_slider = st.sidebar.slider(
        "Entry Threshold",
        min_value=0.0, max_value=1.0,
        value=DEFAULT_THRESHOLD, step=0.01,
        help="How strong must the signal be before we BUY/SELL?"
    )
    start_btn = st.sidebar.button("🚀 Start Trading Bot (Spot Only)")

# --- State for GridTrader ---
if "grid_trader" not in st.session_state:
    st.session_state["grid_trader"] = None

def grid_dashboard(grid: GridTrader):
    st.subheader(f"🤖 Grid Trader for {grid.pair}")
    st.write(f"Base price: {grid.base:.4f} | Levels: {grid.levels} | Spacing: {grid.grid_pct:.4%} | Size: {grid.size}")
    df_orders = pd.DataFrame(grid.orders)
    if not df_orders.empty:
        st.dataframe(df_orders[["side", "price", "size", "active", "order_id"]])
        st.write(f"Active Orders: {df_orders['active'].sum()}")
    else:
        st.info("No grid orders placed yet.")

def run_grid_trading():
    # Only one grid trader at a time for now, could expand to multi-pair
    if st.session_state.grid_trader is None or st.session_state.grid_trader.pair != grid_pair:
        # Get current price for center
        df = fetch_klines(pair=grid_pair, interval=interval, limit=lookback)
        if df.empty:
            st.error(f"No data for {grid_pair}")
            return
        current_price = df["Close"].iloc[-1]
        grid = GridTrader(
            pair=grid_pair,
            base_price=current_price,
            grid_levels=grid_levels,
            grid_spacing_pct=grid_pct,
            size=grid_size,
            direction=grid_direction,
        )
        grid.build_orders()
        st.session_state.grid_trader = grid
    grid = st.session_state.grid_trader
    # Place grid orders (simulate)
    grid.place_orders(lambda pair, side, size, price:
        place_limit_order(pair, side, size, price, is_dry_run=dry_run)
    )
    grid_dashboard(grid)
    # Optionally, could refresh/cancel orders here too

def main_loop_signal_trading():
    global current_capital
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

# MAIN UI LOGIC
if strategy_mode == "Grid Trading":
    if grid_start_btn:
        run_grid_trading()
    # Show grid dashboard, even if grid already started
    if st.session_state.grid_trader is not None:
        grid_dashboard(st.session_state.grid_trader)
else:
    if start_btn:
        main_loop_signal_trading()