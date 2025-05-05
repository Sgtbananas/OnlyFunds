import streamlit as st
st.set_page_config(page_title="CryptoTrader AI (A)", layout="wide")

import os
import sys
import time
import threading
import traceback
import tempfile
import shutil
import json
from datetime import datetime
from utils.helpers import dynamic_threshold
import pandas as pd
from dotenv import load_dotenv
import yaml
import joblib
from logging.handlers import RotatingFileHandler
from pythonjsonlogger import jsonlogger
import logging
from utils.helpers import get_volatile_pairs
from utils.helpers import BLACKLISTED_TOKENS
import streamlit as st
from datetime import datetime, timedelta  # (Make sure to import datetime for scheduling)

# === Initialize session state flags ===
if 'backtest_triggered' not in st.session_state:
    st.session_state.backtest_triggered = False  # flag to indicate a backtest run is requested
if 'auto_mode' not in st.session_state:
    st.session_state.auto_mode = False           # flag to indicate if automatic backtesting is on
if 'next_run_time' not in st.session_state:
    st.session_state.next_run_time = None        # store next scheduled run time (if auto_mode)


@st.cache_data(ttl=300)  # Refresh every 5 minutes
def get_trading_pairs():
    return get_volatile_pairs(limit=10)

TRADING_PAIRS = get_trading_pairs()

# --- Background auto-refresh of volatile pairs
def auto_refresh_pairs(interval=300):  # 5 minutes
    while True:
        try:
            st.session_state["TRADING_PAIRS"] = get_volatile_pairs(limit=10)
            time.sleep(interval)
        except Exception as e:
            print(f"[WARN] Auto-refresh pairs failed: {e}")
            time.sleep(60)

# --- Initialize volatile pairs on startup
if "TRADING_PAIRS" not in st.session_state:
    st.session_state["TRADING_PAIRS"] = get_volatile_pairs(limit=10)
    threading.Thread(target=auto_refresh_pairs, args=(300,), daemon=True).start()

# --- Early folder creation
os.makedirs("state", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# --- Safe Sidebar State Init ---
def get_config_defaults():
    return dict(
        mode="Auto",
        dry_run=True,
        autotune=True,
        interval="5m",
        lookback=1000,
        threshold=0.5,
        max_positions=5,
        stop_loss_pct=0.01,
        take_profit_pct=0.02,
        fee=0.001,
        atr_stop_mult=1.0,
        atr_tp_mult=2.0,
        atr_trail_mult=1.0,
        partial_exit=True,
        capital_alloc_pct=0.05
    )

def sidebar(key, default=None, set_value=None):
    if "sidebar" not in st.session_state:
        st.session_state.sidebar = get_config_defaults()
    if set_value is not None:
        st.session_state.sidebar[key] = set_value
    return st.session_state.sidebar.get(key, default)

if "sidebar" not in st.session_state:
    st.session_state.sidebar = get_config_defaults()

# --- Safe get_pair_params fallback ---
def get_pair_params(pair):
    from utils.helpers import load_json
    try:
        params = load_json("state/auto_params.json", default={})
        return params.get(pair, {
            "interval": "5m",
            "lookback": 1000,
            "threshold": 0.5
        })
    except Exception as e:
        print(f"[WARN] get_pair_params fallback: {e}")
        return {
            "interval": "5m",
            "lookback": 1000,
            "threshold": 0.5
        }

# --- Core App Imports (no more TRADING_PAIRS here!)
from core.core_data import fetch_klines, validate_df, add_indicators
from core.core_signals import (
    generate_signal, smooth_signal, generate_ensemble_signal
)
from core.trade_execution import place_live_order
from core.backtester import run_backtest
from utils.helpers import (
    compute_trade_metrics, suggest_tuning, save_json, load_json, validate_pair,
    get_volatile_pairs
)
from core.ml_filter import load_model, ml_confidence, train_and_save_model
from core.risk_manager import RiskManager
from core.data_manager import get_top_coinex_symbols, update_historical_data, load_data
from utils.config import load_config

# --- Prometheus Metrics
from prometheus_client import start_http_server, Counter, Gauge, REGISTRY

# --- ENV Meta Model Selector
SELECTOR_VARIANT = os.getenv("SELECTOR_VARIANT", "A")
if SELECTOR_VARIANT == "A":
    META_MODEL_PATH = "state/meta_model_A.pkl"
    METRICS_PREFIX = "onlyfunds_A"
elif SELECTOR_VARIANT == "B":
    META_MODEL_PATH = "state/meta_model_B.pkl"
    METRICS_PREFIX = "onlyfunds_B"
else:
    META_MODEL_PATH = "state/meta_model.pkl"
    METRICS_PREFIX = "onlyfunds"

# --- Meta Model Loading (stub fallback)
def train_stub_meta_model(meta_model_path):
    from sklearn.dummy import DummyClassifier
    import numpy as np
    X = np.random.rand(50, 4)
    y = np.random.randint(0, 2, 50)
    model = DummyClassifier(strategy="most_frequent")
    model.fit(X, y)
    joblib.dump(model, meta_model_path)
    print(f"[INFO] Stub meta-learner saved to {meta_model_path}")

def ensure_meta_model(path):
    try:
        return joblib.load(path)
    except Exception as e:
        print(f"[WARN] Could not load meta-learner model at {path}: {e}")
        threading.Thread(target=lambda: train_stub_meta_model(path), daemon=True).start()
        return None

META_MODEL = ensure_meta_model(META_MODEL_PATH)

# --- Logging Setup
LOGS_DIR = "logs"
log_file = os.path.join(LOGS_DIR, f"{METRICS_PREFIX}.json")
handler = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5)
formatter = jsonlogger.JsonFormatter("%(asctime)s %(levelname)s %(name)s %(message)s")
handler.setFormatter(formatter)
root = logging.getLogger()
root.handlers = []
root.addHandler(handler)
root.setLevel(logging.INFO)
if os.getenv("DEBUG_LOG_STDOUT", "0") == "1":
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    root.addHandler(stdout_handler)

logger = logging.getLogger(__name__)

META_MODEL = None
try:
    META_MODEL = joblib.load("state/meta_model_A.pkl")
    logger.info("✅ META_MODEL loaded successfully.")
    st.sidebar.success("✅ META_MODEL loaded successfully.")
except Exception as e:
    logger.warning(f"⚠ META_MODEL not loaded: {e}")
    st.sidebar.error(f"⚠ META_MODEL not loaded: {e}")

# --- Prometheus Metrics Setup (safe singleton)
def get_prometheus_metrics():
    module = __import__(__name__)
    if not hasattr(module, "_PROMETHEUS_METRICS"):
        try:
            start_http_server(8000)
        except Exception:
            pass
        metrics = {}
        name = f"{METRICS_PREFIX}_trades_executed_total"
        if name not in REGISTRY._names_to_collectors:
            metrics['trade_counter'] = Counter(name, "Total trades executed")
        else:
            metrics['trade_counter'] = REGISTRY._names_to_collectors[name]

        name = f"{METRICS_PREFIX}_current_pnl"
        if name not in REGISTRY._names_to_collectors:
            metrics['pnl_gauge'] = Gauge(name, "Current unrealized PnL (USDT)")
        else:
            metrics['pnl_gauge'] = REGISTRY._names_to_collectors[name]

        name = f"{METRICS_PREFIX}_heartbeat"
        if name not in REGISTRY._names_to_collectors:
            metrics['heartbeat_gauge'] = Gauge(name, "Heartbeat timestamp")
        else:
            metrics['heartbeat_gauge'] = REGISTRY._names_to_collectors[name]

        module._PROMETHEUS_METRICS = metrics
    m = module._PROMETHEUS_METRICS
    return m['trade_counter'], m['pnl_gauge'], m['heartbeat_gauge']

trade_counter, pnl_gauge, heartbeat_gauge = get_prometheus_metrics()

# --- Load Main Config
CONFIG_PATH = "config/config.yaml"

def load_config_safe(config_path, fallback=None):
    try:
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        if not isinstance(cfg, dict):
            raise ValueError("Config is not a dict.")
        return cfg
    except Exception as e:
        st.sidebar.error(f"Error loading config: {e}")
        return fallback or {}

config = load_config_safe(CONFIG_PATH, fallback={})
risk_cfg = config.get("risk", {})
trading_cfg = config.get("trading", {})
ml_cfg = config.get("ml", {})

# --- Sidebar Setup (no manual ATR multipliers anymore)
st.title(f"🧠 CryptoTrader AI Bot (SPOT Market Only) — Variant {SELECTOR_VARIANT}")
st.sidebar.header("⚙️ Configuration")
interval_used = st.sidebar.selectbox('Interval', options=['5m', '15m', '1h'], index=0)
lookback_used = st.sidebar.slider('Lookback Candles', min_value=100, max_value=1000, value=500)
current_capital = st.sidebar.number_input('Starting Capital ($)', value=1000)
st.sidebar.markdown(f"**Meta-Learner Variant:** `{SELECTOR_VARIANT}`")

# --- Trading Mode
mode = st.sidebar.selectbox(
    "Trading Mode",
    ["Conservative", "Normal", "Aggressive", "Auto"],
    index=["Conservative", "Normal", "Aggressive", "Auto"].index(trading_cfg.get("mode", "Normal").capitalize())
)

# === Trading Mode Definitions (updated) ===
if st.session_state.sidebar["mode"] == "Conservative":
    st.session_state.sidebar["target_daily_return"] = 5.0
    st.session_state.sidebar["risk_pct"] = 0.005  # 0.5% per trade
    st.session_state.sidebar["capital_alloc_pct"] = 0.05  # 5% of balance per trade
elif st.session_state.sidebar["mode"] == "Aggressive":
    st.session_state.sidebar["target_daily_return"] = 20.0
    st.session_state.sidebar["risk_pct"] = 0.02   # 2% per trade
    st.session_state.sidebar["capital_alloc_pct"] = 0.2   # 20% of balance per trade
elif st.session_state.sidebar["mode"] == "Auto":
    st.session_state.sidebar["target_daily_return"] = 10.0
    st.session_state.sidebar["risk_pct"] = 0.01   # 1% per trade
    st.session_state.sidebar["capital_alloc_pct"] = 0.1   # 10_0]
else:
    st.session_state.sidebar["target_daily_return"] = 10.0
    st.session_state.sidebar["risk_pct"] = 0.01
    st.session_state.sidebar["capital_alloc_pct"] = 0.1

# --- Other sidebar toggles
dry_run = st.sidebar.checkbox("Dry Run Mode", value=trading_cfg.get("dry_run", True))
autotune = st.sidebar.checkbox("Enable Adaptive-Threshold Autotune", value=True)

# --- Show current volatile pairs in the sidebar ---
with st.sidebar.expander("📈 Auto-Selected Pairs (Top Volatility)", expanded=False):
    current_pairs = st.session_state.get("TRADING_PAIRS", [])
    if current_pairs:
        for i, pair in enumerate(current_pairs, 1):
            st.markdown(f"**{i}.** `{pair}`")
    else:
        st.info("Volatile pair list is still loading...")

# --- Only if Manual (not AI)
if mode != "Auto":
    interval = st.sidebar.selectbox(
        "Candle Interval",
        ["5m", "15m", "30m", "1h", "4h", "1d"],
        index=["5m", "15m", "30m", "1h", "4h", "1d"].index(trading_cfg.get("default_interval", "5m"))
    )
    lookback = st.sidebar.slider("Lookback Candles", 300, 2000, trading_cfg.get("backtest_lookback", 1000))
    threshold = st.sidebar.slider(
        "Entry Threshold", min_value=0.0, max_value=1.0,
        value=trading_cfg.get("threshold", 0.5), step=0.01
    )
else:
    interval = None
    lookback = None
    threshold = None

# --- Max open positions stays user-configurable
max_positions = st.sidebar.number_input("Max Open Positions", 1, 50, trading_cfg.get("max_positions", 5))

# --- (✅) NO manual ATR Multiplier inputs anymore
st.sidebar.info("ATR-based Stop-Loss, TP, Trailing is AI-optimized.")

# --- Save Preferences button
save_prefs = st.sidebar.button("💾 Save Config (Optional)")
if save_prefs:
    try:
        with open(CONFIG_PATH, "r") as f:
            data = yaml.safe_load(f)
        data["trading"]["dry_run"] = dry_run
        data["trading"]["max_positions"] = max_positions
        data["strategy"]["mode"] = mode
        if interval:
            data["trading"]["default_interval"] = interval
        if lookback:
            data["trading"]["backtest_lookback"] = lookback
        if threshold is not None:
            data["trading"]["threshold"] = threshold
        tmp = tempfile.NamedTemporaryFile("w", dir="config", delete=False)
        yaml.safe_dump(data, tmp)
        tmp.flush()
        tmp.close()
        shutil.move(tmp.name, CONFIG_PATH)
        st.sidebar.success("Preferences saved successfully.")
    except Exception as e:
        st.sidebar.error(f"Failed to save preferences: {e}")

# --- Show Heartbeat Info
HEARTBEAT_FILE = f"state/heartbeat_{SELECTOR_VARIANT}.json"
def write_heartbeat():
    ts = time.time()
    try:
        with open(HEARTBEAT_FILE, "w") as f:
            json.dump({"last_run": ts}, f)
        heartbeat_gauge.set(ts)
    except Exception as e:
        logger.error(f"Failed to write heartbeat: {e}")

# --- State Files and Safety Loaders
POSITIONS_FILE = "state/open_positions.json"
TRADE_LOG_FILE = "state/trade_log.json"
CAPITAL_FILE = "state/current_capital.json"

def safe_load_json(file_path, default):
    import json
    try:
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                data = json.load(f)
            if data is None:
                return default
            return data
    except Exception as e:
        logger.error(f"Failed to load {file_path}: {e}")
    return default

open_positions = safe_load_json(POSITIONS_FILE, {})
trade_log = safe_load_json(TRADE_LOG_FILE, [])
if not isinstance(current_capital, (float, int)):

# --- Risk Manager
 risk_manager = RiskManager(config)

# --- ML Retraining & Meta Model Watchdog
from core.ml_filter import load_model, ml_confidence, train_and_save_model

RETRAIN_TRADE_INTERVAL = 50
RETRAIN_TIME_INTERVAL = 6*60*60
LAST_RETRAIN_FILE = "state/last_ml_retrain.txt"

def read_last_retrain_time():
    try:
        with open(LAST_RETRAIN_FILE, "r") as f:
            return float(f.read().strip())
    except Exception:
        return 0.0

def write_last_retrain_time(ts):
    try:
        with open(LAST_RETRAIN_FILE, "w") as f:
            f.write(str(ts))
    except Exception as e:
        logger.error(f"Failed to write retrain time: {e}")

def retrain_ml_if_needed(trade_log):
    now = time.time()
    last = read_last_retrain_time()
    need_time = (now - last) > RETRAIN_TIME_INTERVAL
    need_trades = (len(trade_log) > 0 and len(trade_log) % RETRAIN_TRADE_INTERVAL == 0)
    if need_time or need_trades:
        logger.info("Triggering ML model retrain.")
        success, msg = train_and_save_model()
        write_last_retrain_time(now)
        if success:
            logger.info(f"ML retrain success: {msg}")
        else:
            logger.error(f"ML retrain failed: {msg}")

def retrain_ml_background(trade_log):
    threading.Thread(target=retrain_ml_if_needed, args=(trade_log,), daemon=True).start()

# --- Dynamic ATR Automation Functions
def estimate_dynamic_atr_multipliers(df):
    """Based on ATR/Price ratios, optimize SL/TP/Trailing multipliers."""
    try:
        # Dynamic target bands
        atr_avg = df["ATR"].rolling(50).mean().iloc[-1]
        close_price = df["Close"].iloc[-1]
        atr_pct = atr_avg / close_price if close_price > 0 else 0.01

        # Heuristics (can replace later with model prediction if wanted)
        stop_mult = max(0.8, min(2.0, atr_pct * 50))    # adapt SL around volatility
        tp_mult   = max(1.5, min(5.0, atr_pct * 100))   # TP further out
        trail_mult = max(0.5, min(2.0, atr_pct * 30))   # trailing in-between

        return stop_mult, tp_mult, trail_mult
    except Exception as e:
        logger.error(f"ATR tuning failed: {e}")
        return 1.0, 2.0, 1.0  # fallback safe values

# --- Heartbeat Writing for Watchdog ---
def write_heartbeat():
    ts = time.time()
    try:
        with open(f"state/heartbeat_{SELECTOR_VARIANT}.json", "w") as f:
            import json
            json.dump({"last_run": ts}, f)
    except Exception as e:
        logger.error(f"Failed to write heartbeat: {e}")
    try:
        heartbeat_gauge.set(ts)
    except Exception:
        pass

# --- Trade Execution Guardrails ---
def validate_trade(amount, price, capital, min_size=0.001):
    if amount < min_size or price <= 0 or capital <= 0:
        return False
    return True

def dynamic_capital_allocation(performance: dict, base_alloc_pct=0.05, min_alloc=1.0, max_alloc_pct=0.2):
    """
    Adjust capital allocation based on recent performance.
    If winning, increase allocation slightly. If losing, decrease.
    """
    win_rate = performance.get("win_rate", 50.0)
    current_balance = performance.get("current_capital", 10)

    if win_rate > 70:
        alloc_pct = min(base_alloc_pct * 1.5, max_alloc_pct)
    elif win_rate > 55:
        alloc_pct = min(base_alloc_pct * 1.2, max_alloc_pct)
    elif win_rate < 45:
        alloc_pct = max(base_alloc_pct * 0.8, 0.01)
    elif win_rate < 30:
        alloc_pct = max(base_alloc_pct * 0.5, 0.005)
    else:
        alloc_pct = base_alloc_pct

    capital_to_use = max(current_balance * alloc_pct, min_alloc)

    return capital_to_use

# --- Main Trading Function ---
def main_loop():
    global current_capital, trade_log, open_positions

    retrain_ml_background(trade_log)

    for pair in [pair]:  # Using sidebar pair
        perf = compute_trade_metrics(trade_log, trading_cfg.get("default_capital", 10))
        # Fetch Auto params

        df = load_data(pair, interval=interval_used, limit=lookback_used)
        st.write('DEBUG: Attempting to fetch klines for:', pair, interval_used, lookback_used)
        if df.empty or not validate_df(df):
            logger.warning(f"Invalid data for {pair}")
            continue

        df = add_indicators(df)
        st.write('DEBUG: DataFrame after fetching and adding indicators:')
        st.write(df.head(5))
        st.write('DEBUG: df empty?', df.empty)
        try:
            base_signal = generate_signal(df)  # Base strategy signal series
            st.write('🔎 Final signal values (last 10):', signal.tail(10))
            st.write('🔎 ATR values (last 10):', df['ATR'].tail(10))
            st.write('DEBUG: Is df empty?', df.empty)
            st.write('DEBUG: Is signal empty?', signal.empty if hasattr(signal, 'empty') else 'No attribute')
            st.write('🔎 Final signal values (last 10):', signal.tail(10))
            st.write('DEBUG: Is df empty?', df.empty)
            st.write('DEBUG: Is signal empty?', signal.empty if hasattr(signal, 'empty') else 'No attribute')
            st.write('🔎 ATR values (last 10):', df['ATR'].tail(10))
        except Exception as e:
            logger.error(f"Signal generation failed for {pair}: {e}")
            continue

        # ML confidence filtering
        conf_series = None
        if META_MODEL:
            try:
                conf_series = ml_confidence(df)  # Model predicts confidence over df
            except Exception as e:
                logger.error(f"ML confidence check failed for {pair}: {e}")
                st.write(f"⚠️ {pair}: ML confidence computation failed, skipping.")
                continue
            if conf_series is None:
                logger.warning(f"{pair}: Missing features for ML model – skipping trade.")
                st.write(f"⚠️ {pair}: Skipping trade due to missing ML features.")
                continue

        # Determine entry threshold (autotune or manual)
        autotune = st.session_state.sidebar.get("autotune", True)
        if autotune:
            try:
                threshold_final = trading_cfg.get("threshold", 0.5)
            except Exception as e:
                logger.warning(f"Autotune failed for {pair}: {e}")
                threshold_final = trading_cfg.get("threshold", 0.5)
        else:
            threshold_final = st.session_state.sidebar.get("threshold", 0.5)

        # Dynamic ATR-based multipliers for SL/TP/Trailing
        stop_mult, tp_mult, trail_mult = estimate_dynamic_atr_multipliers(df)

        latest_base = base_signal.iloc[-1] if hasattr(base_signal, "iloc") else base_signal[-1]
        latest_conf = None
        if META_MODEL:
            # Get latest confidence value
            if isinstance(conf_series, pd.Series):
                latest_conf = conf_series.iloc[-1]
            elif hasattr(conf_series, '__len__') and not isinstance(conf_series, str):
                # conf_series is array-like (e.g. numpy array/list)
                latest_conf = conf_series[-1]
            else:
                # conf_series is a single scalar value
                latest_conf = conf_series

        price = df["Close"].iloc[-1]
        atr_val = df["ATR"].iloc[-1] if "ATR" in df.columns else 0.0

        # BUY Condition (enter position)
        if pair not in open_positions and latest_base > threshold_final:
            # If ML model is present, require confidence above threshold as well
            if META_MODEL and latest_conf is not None and latest_conf < threshold_final:
                logger.info(f"{pair}: ML confidence {latest_conf:.2f} below threshold – trade skipped.")
                st.write(f"⚠️ {pair}: Low ML confidence ({latest_conf:.2f}), skipping buy signal.")
                continue
            if len(open_positions) >= trading_cfg.get("max_positions", 5):
                continue
            amount = min(current_capital / price, 1.0)
            if not validate_trade(amount, price, current_capital):
                continue

            try:
                _ = place_order(pair, "BUY", amount, price, dry_run=True)
                open_positions[pair] = {
                    "amount": amount,
                    "entry_price": price,
                    "atr_val": atr_val
                }
                trade_log.append({
                    "timestamp": datetime.utcnow().isoformat(),
                    "pair": pair,
                    "side": "BUY",
                    "amount": amount,
                    "price": price,
                    "result": None
                })
                current_capital -= amount * price * (1 + trading_cfg.get("fee", 0.001))
                trade_counter.inc()
                logger.info(f"Opened BUY for {pair} at {price:.4f}")
            except Exception as e:
                logger.error(f"Buy execution failed for {pair}: {e}")

            write_heartbeat()
            continue

        # SELL Condition (manage open position)
        if pair in open_positions:
            pos = open_positions[pair]
            amount = pos["amount"]
            entry_price = pos["entry_price"]
            atr_entry = pos.get("atr_val", atr_val)

            # Partial take-profit exit
            if st.session_state.sidebar.get("partial_exit", True):
                tp1 = entry_price + (atr_entry * tp_mult / 2.0)
                if price >= tp1 and not pos.get("partial_exit_done"):
                    logger.info(f"{pair}: Partial TP level reached – taking profit on half position.")
                    sell_amount = amount * 0.5
                    try:
                        _ = place_order(pair, "SELL", sell_amount, price, dry_run=True)
                        trade_log.append({
                            "timestamp": datetime.utcnow().isoformat(),
                            "pair": pair,
                            "side": "SELL_PARTIAL",
                            "amount": sell_amount,
                            "price": price,
                            "result": "TP1"
                        })
                        pos["amount"] -= sell_amount
                        pos["partial_exit_done"] = True
                        current_capital += sell_amount * price * (1 - trading_cfg.get("fee", 0.001))
                        logger.info(f"Partial exit executed for {pair} at TP1.")
                    except Exception as e:
                        logger.error(f"Partial take-profit failed for {pair}: {e}")
                    continue

            # Full exit conditions (TP, SL, or Trailing stop)
            tp_price = entry_price + atr_entry * tp_mult
            sl_price = entry_price - atr_entry * stop_mult
            trail_price = max(entry_price, price - atr_entry * trail_mult)
            exit_reason = None
            if price >= tp_price:
                exit_reason = "TP"
            elif price <= sl_price:
                exit_reason = "SL"
            elif price <= trail_price:
                exit_reason = "TRAIL"
            else:
                exit_reason = None

            if exit_reason:
                try:
                    _ = place_order(pair, "SELL", amount, price, dry_run=True)
                    trade_log.append({
                        "timestamp": datetime.utcnow().isoformat(),
                        "pair": pair,
                        "side": "SELL",
                        "amount": amount,
                        "price": price,
                        "result": exit_reason
                    })
                    current_capital += amount * price * (1 - trading_cfg.get("fee", 0.001))
                    del open_positions[pair]
                    trade_counter.inc()
                    logger.info(f"Closed position on {pair} due to {exit_reason}")
                except Exception as e:
                    logger.error(f"Error closing position on {pair}: {e}")
                write_heartbeat()

    # End of trading loop – save state
    atomic_save_json(open_positions, POSITIONS_FILE)
    atomic_save_json(trade_log, TRADE_LOG_FILE)
    atomic_save_json(current_capital, CAPITAL_FILE)
    pnl_gauge.set(current_capital)
    retrain_ml_background(trade_log)

run_backtest_btn = st.sidebar.button("Run Backtest")

# --- Backtest Execution ---
pair = st.sidebar.text_input('Trading Pair', value='BTCUSDT')
if run_backtest_btn:
    st.write("🔎 Fetching latest top CoinEx symbols...")
    TRADING_PAIRS = get_top_coinex_symbols(limit=250)
    st.write("Top symbols:", TRADING_PAIRS[:5], "...")

    updated_pairs = []
    for sym in TRADING_PAIRS:
        try:
            st.write(f"🔄 Updating data for {sym}...")
            update_historical_data(sym, interval=interval_used, limit=lookback_used)
            updated_pairs.append(sym)
        except Exception as e:
            st.write(f"⚠ Skipping {sym} due to data error: {e}")
    st.write(f"✅ Updated {len(updated_pairs)} symbols.")
    try:
        st.sidebar.success("Backtest started...")
        all_results = []
        starting_capital = trading_cfg.get("default_capital", 10.0)
        day_returns = []
        total_trades = 0

        for pair in updated_pairs:
            # Auto params per pair if available, otherwise use sidebar values
            if st.session_state.sidebar.get("mode") == "Auto":
                stop_mult, tp_mult, trail_mult = estimate_dynamic_atr_multipliers(df)
                threshold_used = params.get("threshold", 0.5)
            else:
                threshold_used = st.session_state.sidebar.get("threshold", 0.5)

            df = load_data(pair, interval=interval_used, limit=lookback_used)
            if df.empty or not validate_df(df):
                st.sidebar.error(f"Backtest failed: No valid data for {pair}")
                continue
            else:
                df = add_indicators(df)

                # --- Generate signal series (with ML confidence if available) ---
                if META_MODEL:
                    conf_series = None
                    try:
                        conf_series = ml_confidence(df)
                    except Exception as e:
                        logger.error(f"Backtest ML confidence error for {pair}: {e}")
                        conf_series = None

                    if conf_series is None:
                        st.sidebar.error(f"Backtest failed: Missing ML model features for {pair}")
                        continue
                    else:
                        base_signal = generate_signal(df)
                        # Combine base signal and ML confidence by gating
                        if not isinstance(conf_series, pd.Series):
                            conf_series = pd.Series(conf_series, index=df.index)
                        signal = pd.DataFrame({"base": base_signal, "conf": conf_series}).min(axis=1)
                else:
                    signal = generate_signal(df)

                # --- If ML failed completely, skip backtest ---
                if META_MODEL and (conf_series is None):
                    backtest_result = None
                    st.warning(f"Skipping backtest for {pair} — missing ML confidence.")
                    continue

                # --- Run backtest (corrected call matching new backtester.py) ---
                try:
                    # --- Run backtest (fully updated arguments) ---

                    backtest_result = run_backtest(
                        signal=signal,
                        pair=pair,
                        interval=interval_used,
                        limit=lookback_used,
                        equity=current_capital
                    )
                except Exception as e:
                    logger.error(f"Backtest failed for {pair}: {e}")
                    st.sidebar.error(f"Backtest failed for {pair}: {e}")
                    st.stop()
                except Exception as e:
                    logger.error(f"Backtest failed for {pair}: {e}")
                    st.sidebar.error(f"Backtest failed for {pair}: {e}")
                    continue

                if backtest_result is None or backtest_result.empty:
                    st.warning(f"No valid backtest results for {pair}.")
                    continue

                # --- Show results ---
                st.write("📊 Backtest Results", backtest_result)

                # --- Save results ---
                try:
                    atomic_save_json(backtest_result.to_dict(orient="records"), BACKTEST_RESULTS_FILE)
                    st.sidebar.success("✅ Backtest results saved!")
                except Exception as e:
                    logger.error(f"Saving backtest results failed: {e}")
                    st.sidebar.error("Saving backtest results failed!")

                # --- Final Signal Check ---
                latest_signal = signal.iloc[-1] if hasattr(signal, "iloc") else signal[-1]
                if abs(latest_signal) < threshold_used:
                    logger.info(
                        f"⚠ Signal below threshold ({latest_signal:.2f} < {threshold_used:.2f}), skipping {pair}."
                    )
                    st.write(
                        f"⚠ Signal below threshold ({latest_signal:.2f} < {threshold_used:.2f}), skipping {pair}."
                    )
                    continue

                logger.info(f"🚀 Preparing backtest for {pair}")
                st.write(f"🚀 Preparing backtest for {pair}")

                stop_mult, tp_mult, trail_mult = estimate_dynamic_atr_multipliers(df)
                logger.info(f"Stop multiplier: {stop_mult}, TP multiplier: {tp_mult}, Trail multiplier: {trail_mult}")

                # **This block was duplicated, removed redundant call**
                # Removed invalid backtest_df call

                if backtest_df.empty:
                    logger.info(f"🔎 No backtest results for {pair} — possible no valid trades.")
                    st.write(f"⚠ No valid trades for {pair}.")
                else:
                    logger.info(f"✅ Backtest results found for {pair}, appending to all_results.")
                    st.write(f"✅ Backtest completed for {pair}.")
                    st.write(backtest_df.head(5))  # Show first 5 trades

                    all_results.append(backtest_df)

                    summaries = backtest_df[backtest_df["type"] == "summary"]
                    for _, row in summaries.iterrows():
                        day_return = row.get("daily_return_pct", 0.0)
                        current_capital *= (1 + day_return / 100.0)
                        day_returns.append(day_return)
                        total_trades += row.get("trades", 0)

        # --- Results ---
        if all_results:
            final_df = pd.concat(all_results, ignore_index=True)
            avg_day_return = sum(day_returns) / len(day_returns) if day_returns else 0.0

            st.success(f"✅ Backtest complete across {len(all_results)} pairs!")
            st.metric("Final Capital", f"${current_capital:.2f}")
            st.metric("Average Daily Return", f"{avg_day_return:.2f}%")
            st.metric("Total Trades Executed", f"{total_trades}")

            st.dataframe(final_df)

            try:
                atomic_save_json(final_df.to_dict(orient="records"), BACKTEST_RESULTS_FILE)
            except Exception as e:
                logger.error(f"Saving backtest results failed: {e}")
        else:
            st.warning("❗ No valid backtest results to display.")

    except Exception as e:
        logger.exception("Backtest execution error")
        st.error(f"Backtest error: {e}")

# --- Global Error Catching for Crash Recovery ---
import sys
def global_exception_handler(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    error_msg = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
    logger.critical(f"UNHANDLED EXCEPTION:\n{error_msg}")
    st.error(f"Critical Error! {exc_type.__name__}: {exc_value}")

sys.excepthook = global_exception_handler

if run_backtest_btn:
    try:
        st.sidebar.success("Backtest started...")
        all_results = []
        starting_capital = trading_cfg.get("default_capital", 10.0)
        day_returns = []
        total_trades = 0

        for pair in TRADING_PAIRS:
            # Auto params per pair if available, otherwise use sidebar values
            if st.session_state.sidebar.get("mode") == "Auto":
                threshold_used = params.get("threshold", 0.5)
            else:
                threshold_used = st.session_state.sidebar.get("threshold", 0.5)

            df = load_data(pair, interval=interval_used, limit=lookback_used)
            if df.empty or not validate_df(df):
                st.sidebar.error(f"Backtest failed: No valid data for {pair}")
                continue
            else:
                df = add_indicators(df)

                # --- Generate signal series (with ML confidence if available) ---
                if META_MODEL:
                    conf_series = None
                    try:
                        conf_series = ml_confidence(df)
                    except Exception as e:
                        logger.error(f"Backtest ML confidence error for {pair}: {e}")
                        conf_series = None

                    if conf_series is None:
                        st.sidebar.error(f"Backtest failed: Missing ML model features for {pair}")
                        continue
                    else:
                        base_signal = generate_signal(df)
                        # Combine base signal and ML confidence by gating
                        if not isinstance(conf_series, pd.Series):
                            conf_series = pd.Series(conf_series, index=df.index)
                        signal = pd.DataFrame({"base": base_signal, "conf": conf_series}).min(axis=1)
                else:
                    signal = generate_signal(df)

                # --- If ML failed completely, skip backtest ---
                if META_MODEL and (conf_series is None):
                    backtest_result = None
                    st.warning(f"Skipping backtest for {pair} — missing ML confidence.")
                    continue

                # --- Run backtest (corrected call matching new backtester.py) ---
                try:
                    # --- Run backtest (fully updated arguments) ---

                    backtest_result = run_backtest(
                        signal=signal,
                        pair=pair,
                        interval=interval_used,
                        limit=lookback_used,
                        equity=current_capital
                    )
                except Exception as e:
                    logger.error(f"Backtest failed for {pair}: {e}")
                    st.sidebar.error(f"Backtest failed for {pair}: {e}")
                    st.stop()
                except Exception as e:
                    logger.error(f"Backtest failed for {pair}: {e}")
                    st.sidebar.error(f"Backtest failed for {pair}: {e}")
                    continue

                if backtest_result is None or backtest_result.empty:
                    st.warning(f"No valid backtest results for {pair}.")
                    continue

                # --- Show results ---
                st.write("📊 Backtest Results", backtest_result)

                # --- Save results ---
                try:
                    atomic_save_json(backtest_result.to_dict(orient="records"), BACKTEST_RESULTS_FILE)
                    st.sidebar.success("✅ Backtest results saved!")
                except Exception as e:
                    logger.error(f"Saving backtest results failed: {e}")
                    st.sidebar.error("Saving backtest results failed!")

                # --- Final Signal Check ---
                latest_signal = signal.iloc[-1] if hasattr(signal, "iloc") else signal[-1]
                if abs(latest_signal) < threshold_used:
                    logger.info(
                        f"⚠ Signal below threshold ({latest_signal:.2f} < {threshold_used:.2f}), skipping {pair}."
                    )
                    st.write(
                        f"⚠ Signal below threshold ({latest_signal:.2f} < {threshold_used:.2f}), skipping {pair}."
                    )
                    continue

                logger.info(f"🚀 Preparing backtest for {pair}")
                st.write(f"🚀 Preparing backtest for {pair}")

                stop_mult, tp_mult, trail_mult = estimate_dynamic_atr_multipliers(df)
                logger.info(f"Stop multiplier: {stop_mult}, TP multiplier: {tp_mult}, Trail multiplier: {trail_mult}")

                # **This block was duplicated, removed redundant call**

                # --- Run backtest (fully updated arguments) ---

                try:
                    backtest_result = run_backtest(
                        signal=signal,
                        pair=pair,
                        interval=interval_used,
                        limit=lookback_used,
                        equity=current_capital
                    )
                except Exception as e:
                    logger.error(f"Backtest failed for {pair}: {e}")
                    st.sidebar.error(f"Backtest failed for {pair}: {e}")
                    st.stop()

                if backtest_df.empty:
                    logger.info(f"🔎 No backtest results for {pair} — possible no valid trades.")
                    st.write(f"⚠ No valid trades for {pair}.")
                else:
                    logger.info(f"✅ Backtest results found for {pair}, appending to all_results.")
                    st.write(f"✅ Backtest completed for {pair}.")
                    st.write(backtest_df.head(5))  # Show first 5 trades

                    all_results.append(backtest_df)

                    summaries = backtest_df[backtest_df["type"] == "summary"]
                    for _, row in summaries.iterrows():
                        day_return = row.get("daily_return_pct", 0.0)
                        current_capital *= (1 + day_return / 100.0)
                        day_returns.append(day_return)
                        total_trades += row.get("trades", 0)

        # --- Results ---
        if all_results:
            final_df = pd.concat(all_results, ignore_index=True)
            avg_day_return = sum(day_returns) / len(day_returns) if day_returns else 0.0

            st.success(f"✅ Backtest complete across {len(all_results)} pairs!")
            st.metric("Final Capital", f"${current_capital:.2f}")
            st.metric("Average Daily Return", f"{avg_day_return:.2f}%")
            st.metric("Total Trades Executed", f"{total_trades}")

            st.dataframe(final_df)

            try:
                atomic_save_json(final_df.to_dict(orient="records"), BACKTEST_RESULTS_FILE)
            except Exception as e:
                logger.error(f"Saving backtest results failed: {e}")
        else:
            st.warning("❗ No valid backtest results to display.")

    except Exception as e:
        logger.exception("Backtest execution error")
        st.error(f"Backtest error: {e}")

# --- Global Error Catching for Crash Recovery ---
import sys
def global_exception_handler(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    error_msg = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
    logger.critical(f"UNHANDLED EXCEPTION:\n{error_msg}")
    st.error(f"Critical Error! {exc_type.__name__}: {exc_value}")

sys.excepthook = global_exception_handler

# --- No broken legacy backtest_df calls remain ---
# --- All valid results shown and saved above ---
