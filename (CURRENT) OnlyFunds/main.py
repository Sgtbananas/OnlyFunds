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
    generate_signal, smooth_signal, adaptive_threshold, track_trade_result, generate_ensemble_signal
)
from core.trade_execution import place_order
from core.backtester import run_backtest
from utils.helpers import (
    compute_trade_metrics, suggest_tuning, save_json, load_json, validate_pair,
    get_volatile_pairs
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
    st.session_state.sidebar["capital_alloc_pct"] = 0.1   # 10% of balance per trade
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
# --- Main Trading Function (patched version) ---
def main_loop():
    global current_capital, trade_log, open_positions

    retrain_ml_background(trade_log)

    starting_capital = trading_cfg.get("default_capital", 10)
    capital = starting_capital

    pairs = st.session_state.get("TRADING_PAIRS", ["BTCUSDT", "ETHUSDT", "LTCUSDT"])
    for pair in pairs:
        performance = compute_trade_metrics(trade_log, starting_capital)

        # Auto-fetch interval/lookback if Auto mode
        if st.session_state.sidebar["mode"] == "Auto":
            params = get_pair_params(pair)
            interval_used = params.get("interval", "5m")
            lookback_used = params.get("lookback", 1000)

            if st.session_state.sidebar.get("autotune", True):
                threshold_used = dynamic_threshold(df)
            else:
                threshold_used = params.get("threshold", 0.5)
        else:
            interval_used = st.session_state.sidebar.get("interval", "5m")
            lookback_used = st.session_state.sidebar.get("lookback", "1000")

            if st.session_state.sidebar.get("autotune", True):
                threshold_used = dynamic_threshold(df)
            else:
                threshold_used = st.session_state.sidebar.get("threshold", 0.5)

        df = fetch_klines(pair, interval=interval_used, limit=lookback_used)

        if df.empty or not validate_df(df):
            logger.warning(f"Invalid or empty data for {pair}")
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

        latest_signal = signal.iloc[-1] if hasattr(signal, "iloc") else signal[-1]

        stop_mult, tp_mult, trail_mult = estimate_dynamic_atr_multipliers(df)

        # Bias tuning based on Trading Mode
        if st.session_state.sidebar.get("mode") == "Aggressive":
            stop_mult = max(0.7, stop_mult * 0.8)   # tighten stops
            tp_mult = tp_mult * 1.2                 # stretch take profits
            trail_mult = trail_mult * 1.1           # loosen trailing stop
        elif st.session_state.sidebar.get("mode") == "Conservative":
            stop_mult = stop_mult * 1.2             # looser stops
            tp_mult = tp_mult * 0.9                 # smaller take profits
            trail_mult = trail_mult * 1.0           # trailing unchanged

        # Get latest signal value
        latest_signal = signal.iloc[-1] if hasattr(signal, "iloc") else signal[-1]

confidence = ml_confidence(df, META_MODEL)

if confidence < 0.7:
    logger.info(f"❌ Skipping {pair} - confidence too low: {confidence:.2f}")
    continue


        ### BUY CONDITION
        if pair not in open_positions and latest_signal > threshold_used:
            if len(open_positions) >= st.session_state.sidebar["max_positions"]:
                continue

            capital_alloc = dynamic_capital_allocation(performance)
            amount = capital_alloc / price

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
                atomic_save_json(current_capital, CAPITAL_FILE)
                trade_counter.inc()
                logger.info(f"Opened BUY for {pair} at {price:.4f}")
            except Exception as e:
                logger.error(f"Buy execution failed for {pair}: {e}")

            write_heartbeat()
            continue

        ### SELL CONDITION
        if pair in open_positions:
            position = open_positions[pair]
            amount = position["amount"]
            entry_price = position["entry_price"]
            atr_val = position.get("atr_val", None)

            if atr_val is None:
                logger.warning(f"No ATR recorded for {pair}, skipping sell logic.")
                continue

            stop_loss_price = entry_price - (atr_val * stop_mult)
            take_profit_price = entry_price + (atr_val * tp_mult)

            sell_reason = None
            current_price = price

            if current_price <= stop_loss_price:
                sell_reason = "STOP-LOSS"
            elif current_price >= take_profit_price:
                sell_reason = "TAKE-PROFIT"
            elif latest_signal < -threshold_used:
                sell_reason = "SIGNAL-EXIT"

            if sell_reason:
                try:
                    _ = place_order(pair, "SELL", amount, current_price, dry_run=True)
                    trade_log.append({
                        "timestamp": datetime.utcnow().isoformat(),
                        "pair": pair,
                        "side": "SELL",
                        "amount": amount,
                        "price": current_price,
                        "result": sell_reason
                    })
                    current_capital += amount * current_price * (1 - trading_cfg.get("fee", 0.001))
                    del open_positions[pair]
                    trade_counter.inc()
                    logger.info(f"Closed SELL {pair} at {current_price:.4f} by {sell_reason}")
                except Exception as e:
                    logger.error(f"Sell execution failed for {pair}: {e}")

            write_heartbeat()
            continue

    # Save all
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
confidence = ml_confidence(df, META_MODEL)

if confidence < 0.7:
    logger.info(f"❌ Skipping {pair} - confidence too low: {confidence:.2f}")
    continue

if run_backtest_btn:
    try:
        st.sidebar.success("📊 Backtest running on ALL pairs...")

        all_results = []
        starting_capital = trading_cfg.get("default_capital", 10.0)
        current_capital = starting_capital
        day_returns = []
        total_trades = 0

        pairs = st.session_state.get("TRADING_PAIRS", ["BTCUSDT", "ETHUSDT", "LTCUSDT"])
        for pair in pairs:
            # Fetch parameters
            if st.session_state.sidebar["mode"] == "Auto":
                p = get_pair_params(pair)
                interval_used = p.get("interval", "5m")
                lookback_used = p.get("lookback", 1000)
                threshold_used = p.get("threshold", 0.5)
            else:
                interval_used = st.session_state.sidebar.get("interval", "5m")
                lookback_used = st.session_state.sidebar.get("lookback", 1000)

                if st.session_state.sidebar.get("autotune", True):
                    threshold_used = dynamic_threshold(df)
                else:
                    threshold_used = st.session_state.sidebar.get("threshold", 0.5)

            df = fetch_klines(pair, interval=interval_used, limit=lookback_used)

            if df.empty or not validate_df(df):
                logger.warning(f"⚠️ Skipping {pair}: no valid data")
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

            if st.session_state.sidebar.get("autotune", True):
                try:
                    threshold_used = estimate_optimal_threshold(
                        df=df,
                        signal=signal,
                        prices=df["Close"]
                    )
                except Exception as e:
                    logger.warning(f"Threshold AI optimization failed for {pair}: {e}")
                    threshold_used = 0.5
            else:
                threshold_used = st.session_state.sidebar.get("threshold", 0.5)

            latest_signal = signal.iloc[-1] if hasattr(signal, "iloc") else signal[-1]
            price = df["Close"].iloc[-1]

            stop_mult, tp_mult, trail_mult = estimate_dynamic_atr_multipliers(df)
            # Continue your backtesting logic here...

            # Bias tuning by mode
            mode = st.session_state.sidebar.get("mode")
            if mode == "Aggressive":
                stop_mult = max(0.7, stop_mult * 0.8)
                tp_mult *= 1.2
                trail_mult *= 1.1
            elif mode == "Conservative":
                stop_mult *= 1.2
                tp_mult *= 0.9
                trail_mult *= 1.0

            backtest_df = run_backtest(
                signal=signal,
                prices=df["Close"],
                threshold=threshold_used,
                initial_capital=current_capital,
                risk_pct=risk_cfg.get("risk_pct", 0.01),
                stop_loss_atr_mult=stop_mult,
                take_profit_atr_mult=tp_mult,
                trailing_atr_mult=trail_mult,
                fee_pct=trading_cfg.get("fee", 0.001),
                partial_exit=st.session_state.sidebar.get("partial_exit", True),
                atr=df.get("ATR")
            )

            if not backtest_df.empty:
                all_results.append(backtest_df)

                summaries = backtest_df[backtest_df["type"] == "summary"]
                for _, row in summaries.iterrows():
                    day_return = row.get("daily_return_pct", 0.0)
                    current_capital *= (1 + day_return / 100.0)
                    day_returns.append(day_return)
                    total_trades += row.get("trades", 0)

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
