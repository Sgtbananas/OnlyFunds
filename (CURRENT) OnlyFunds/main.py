import os
import sys
import time
import threading
import traceback
import tempfile
import shutil
import json
from datetime import datetime

# --- Early folder creation
os.makedirs("state", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# --- Streamlit First Call (must be before any other Streamlit stuff)
import streamlit as st
st.set_page_config(page_title="CryptoTrader AI (A)", layout="wide")

# --- Safe Sidebar State Init (must come early) ---
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
        partial_exit=True
    )

if "sidebar" not in st.session_state or not isinstance(st.session_state.sidebar, dict):
    st.session_state.sidebar = get_config_defaults()

import pandas as pd
from dotenv import load_dotenv
import yaml
import joblib
from logging.handlers import RotatingFileHandler
from pythonjsonlogger import jsonlogger
import logging

# --- Core App Imports
from core.core_data import fetch_klines, validate_df, add_indicators, TRADING_PAIRS
from core.core_signals import (
    generate_signal, smooth_signal, adaptive_threshold, track_trade_result, generate_ensemble_signal
)
from core.trade_execution import place_order
from core.backtester import run_backtest
from utils.helpers import (
    compute_trade_metrics, suggest_tuning, save_json, load_json, validate_pair, get_auto_pair_params
)
from core.ml_filter import load_model, ml_confidence, train_and_save_model
from core.risk_manager import RiskManager
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
st.sidebar.markdown(f"**Meta-Learner Variant:** `{SELECTOR_VARIANT}`")

# --- Trading Mode
mode = st.sidebar.selectbox(
    "Trading Mode",
    ["Conservative", "Normal", "Aggressive", "Auto"],
    index=["Conservative", "Normal", "Aggressive", "Auto"].index(trading_cfg.get("mode", "Normal").capitalize())
)

# --- Other sidebar toggles
dry_run = st.sidebar.checkbox("Dry Run Mode", value=trading_cfg.get("dry_run", True))
autotune = st.sidebar.checkbox("Enable Adaptive-Threshold Autotune", value=True)

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
current_capital = safe_load_json(CAPITAL_FILE, trading_cfg.get("default_capital", 10.0))
if not isinstance(current_capital, (float, int)):
    current_capital = trading_cfg.get("default_capital", 10.0)

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

# --- Main Trading Function ---
def main_loop():
    global current_capital, trade_log, open_positions

    retrain_ml_background(trade_log)

    for pair in TRADING_PAIRS:
        perf = compute_trade_metrics(trade_log, trading_cfg.get("default_capital", 10))

        # Fetch Auto params
        interval_used = trading_cfg.get("default_interval", "5m")
        lookback_used = trading_cfg.get("backtest_lookback", 1000)

        df = fetch_klines(pair, interval=interval_used, limit=lookback_used)
        if df.empty or not validate_df(df):
            logger.warning(f"Invalid data for {pair}")
            continue

        df = add_indicators(df)

        try:
            if META_MODEL:
                signal = generate_ensemble_signal(df, META_MODEL)
            else:
                signal = generate_signal(df)
        except Exception as e:
            logger.error(f"Signal generation failed for {pair}: {e}")
            continue

        # Dynamic ATR Stop/TP/Trail tuning per pair
        stop_mult, tp_mult, trail_mult = estimate_dynamic_atr_multipliers(df)

        latest_signal = signal.iloc[-1] if hasattr(signal, "iloc") else signal[-1]
        price = df["Close"].iloc[-1]
        atr_val = df["ATR"].iloc[-1]

        # BUY Condition
        if pair not in open_positions and latest_signal > 0.5:
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

        # SELL Condition
        if pair in open_positions:
            pos = open_positions[pair]
            amount = pos["amount"]
            entry_price = pos["entry_price"]
            atr_entry = pos.get("atr_val", atr_val)

            # Partial Exit Handling
            if st.session_state.sidebar.get("partial_exit", True):
                tp1 = entry_price + (atr_entry * tp_mult / 2)
                if price >= tp1 and pos.get("partial_exit_done") != True:
                    logger.info(f"{pair}: Partial TP1 reached!")
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
                        logger.info(f"Partial exit at TP1 for {pair}")
                    except Exception as e:
                        logger.error(f"Partial TP1 exit failed: {e}")
                    continue

            # Full Exit Conditions (TP or SL)
            tp_price = entry_price + atr_entry * tp_mult
            sl_price = entry_price - atr_entry * stop_mult
            trail_price = max(entry_price, price - atr_entry * trail_mult)

            if price >= tp_price:
                reason = "TP"
            elif price <= sl_price:
                reason = "SL"
            elif price <= trail_price:
                reason = "TRAIL"
            else:
                continue

            try:
                _ = place_order(pair, "SELL", amount, price, dry_run=True)
                trade_log.append({
                    "timestamp": datetime.utcnow().isoformat(),
                    "pair": pair,
                    "side": "SELL",
                    "amount": amount,
                    "price": price,
                    "result": reason
                })
                current_capital += amount * price * (1 - trading_cfg.get("fee", 0.001))
                del open_positions[pair]
                trade_counter.inc()
                logger.info(f"Closed position {pair} due to {reason}")
            except Exception as e:
                logger.error(f"Closing failed for {pair}: {e}")

            write_heartbeat()

    # Save session
    atomic_save_json(open_positions, POSITIONS_FILE)
    atomic_save_json(trade_log, TRADE_LOG_FILE)
    atomic_save_json(current_capital, CAPITAL_FILE)

    pnl_gauge.set(current_capital)
    retrain_ml_background(trade_log)
# --- Sidebar Controls for Execution ---
run_trading_btn = st.sidebar.button("▶️ Run Trading Cycle")
run_backtest_btn = st.sidebar.button("🧪 Run Backtest")

# --- Main App Logic Entrypoint ---
if run_trading_btn:
    try:
        st.success("Trading cycle running...")
        main_loop()
    except Exception as e:
        st.error(f"Trading loop failed: {e}")
        logger.error(f"Trading loop error: {e}")

if run_backtest_btn:
    try:
        st.sidebar.success("Backtest started...")

        # Pick first trading pair (or extend later to multi-pair backtest)
        pair = TRADING_PAIRS[0]

        # Auto params per pair (interval, lookback, threshold) if available
        if st.session_state.sidebar["mode"] == "Auto":
            params = get_pair_params(pair)
            interval_used = params.get("interval", "5m")
            lookback_used = params.get("lookback", 1000)
            threshold_used = params.get("threshold", 0.5)
        else:
            interval_used = st.session_state.sidebar.get("interval", "5m")
            lookback_used = st.session_state.sidebar.get("lookback", 1000)
            threshold_used = st.session_state.sidebar.get("threshold", 0.5)

        df = fetch_klines(pair, interval=interval_used, limit=lookback_used)

        if df.empty or not validate_df(df):
            st.sidebar.error(f"Backtest failed: No valid data for {pair}")
        else:
            df = add_indicators(df)

            # Use Ensemble or base signal
            if META_MODEL:
                signal = generate_ensemble_signal(df, META_MODEL)
            else:
                signal = generate_signal(df)

            # === NEW: Estimate ATR multipliers dynamically
            stop_mult, tp_mult, trail_mult = estimate_dynamic_atr_multipliers(df)

            # Run backtest
            backtest_result = run_backtest(
                signal=signal,
                prices=df["Close"],
                threshold=threshold_used,
                initial_capital=trading_cfg.get("default_capital", 10),
                risk_pct=risk_cfg.get("risk_pct", 0.01),
                stop_loss_atr_mult=stop_mult,
                take_profit_atr_mult=tp_mult,
                trailing_atr_mult=trail_mult,
                fee_pct=trading_cfg.get("fee", 0.001),
                partial_exit=True,  # Always enable partial exits in AI backtest
                atr=df.get("ATR"),
            )

            # Show backtest results
            st.write("📊 Backtest Results", backtest_result)

            try:
                atomic_save_json(backtest_result.to_dict(orient="records"), BACKTEST_RESULTS_FILE)
                st.sidebar.success("✅ Backtest results saved!")
            except Exception as e:
                logger.error(f"Saving backtest results failed: {e}")
                st.sidebar.error("Saving backtest results failed!")

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
