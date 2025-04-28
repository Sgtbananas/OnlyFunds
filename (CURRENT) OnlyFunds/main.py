import streamlit as st
st.set_page_config(page_title="CryptoTrader AI (A)", layout="wide")  # FIRST STREAMLIT CALL

import os
import sys
import logging
import time
import threading
import traceback
from datetime import datetime, date
import pandas as pd
from dotenv import load_dotenv

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

import joblib

# --- Robust meta-learner fallback ---
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

def train_stub_meta_model(meta_model_path):
    import numpy as np
    from sklearn.dummy import DummyClassifier
    X = np.random.rand(20, 4)
    y = np.random.randint(0, 2, 20)
    model = DummyClassifier(strategy="most_frequent")
    model.fit(X, y)
    joblib.dump(model, meta_model_path)
    print(f"[INFO] Stub meta-learner saved to {meta_model_path}")

def robust_load_meta_model(path):
    try:
        return joblib.load(path)
    except Exception as e:
        print(f"[WARN] Could not load meta-learner model at {path}: {e}")
        train_stub_meta_model(path)
        return joblib.load(path)

META_MODEL = robust_load_meta_model(META_MODEL_PATH)

# --- Logging: both file and (optional) console, with JSON formatting, rotation, and error notification ---
from logging.handlers import RotatingFileHandler
from pythonjsonlogger import jsonlogger

LOGS_DIR = "logs"
os.makedirs(LOGS_DIR, exist_ok=True)
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

# --- Prometheus singleton, must come before any metric use ---
from prometheus_client import start_http_server, Counter, Gauge, REGISTRY

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

# --- Load config, with runtime override support ---
CONFIG_PATH = "config/config.yaml"
config = load_config(CONFIG_PATH)
risk_cfg = config["risk"]
trading_cfg = config["trading"]
ml_cfg = config.get("ml", {})

# --- State init (atomic, with data validation) ---
def safe_load_json(file, default):
    try:
        if os.path.exists(file):
            data = load_json(file)
            return data if data is not None else default
        return default
    except Exception as e:
        logger.error(f"Failed to load {file}: {e}")
        return default

POSITIONS_FILE = "state/open_positions.json"
TRADE_LOG_FILE = "state/trade_log.json"
CAPITAL_FILE = "state/current_capital.json"
BACKTEST_RESULTS_FILE = "state/backtest_results.json"
OPTUNA_BEST_FILE = "state/optuna_best.json"
AUTO_PARAMS_FILE = "state/auto_params.json"
HEARTBEAT_FILE = f"state/heartbeat_{SELECTOR_VARIANT}.json"

os.makedirs("state", exist_ok=True)
open_positions = safe_load_json(POSITIONS_FILE, {})
trade_log = safe_load_json(TRADE_LOG_FILE, [])
current_capital = safe_load_json(CAPITAL_FILE, trading_cfg["default_capital"])
if not isinstance(current_capital, (float, int)):
    current_capital = trading_cfg["default_capital"]

risk_manager = RiskManager(config)

st.title(f"🧠 CryptoTrader AI Bot (SPOT Market Only) — Variant {SELECTOR_VARIANT}")
st.sidebar.header("⚙️ Configuration")
st.sidebar.markdown(f"**Meta-Learner Variant:** `{SELECTOR_VARIANT}`")

def get_config_defaults():
    return dict(
        mode=config.get("strategy", {}).get("mode", "Normal").capitalize(),
        dry_run=trading_cfg["dry_run"],
        autotune=config.get("strategy", {}).get("autotune", True),
        backtest_mode=False,
        interval=trading_cfg.get("default_interval", "5m"),
        lookback=trading_cfg.get("backtest_lookback", 1000),
        threshold=trading_cfg["threshold"],
        max_positions=trading_cfg["max_positions"],
        stop_loss_pct=risk_cfg["stop_loss_pct"],
        take_profit_pct=risk_cfg["take_profit_pct"],
        fee=trading_cfg["fee"],
        atr_stop_mult=1.0,
        atr_tp_mult=2.0,
        atr_trail_mult=1.0,
        partial_exit=True
    )

if "sidebar" not in st.session_state:
    st.session_state.sidebar = get_config_defaults()

def reset_sidebar():
    st.session_state.sidebar = get_config_defaults()

modes = ["Conservative", "Normal", "Aggressive", "Auto"]
mode_idx = 3 if st.session_state.sidebar["mode"] == "Auto" else modes.index(st.session_state.sidebar["mode"])
st.session_state.sidebar["mode"] = st.sidebar.selectbox("Trading Mode", modes, index=mode_idx)
st.session_state.sidebar["dry_run"] = st.sidebar.checkbox("Dry Run Mode (Simulated)", value=st.session_state.sidebar["dry_run"])
st.session_state.sidebar["autotune"] = st.sidebar.checkbox("Enable Adaptive-Threshold Autotune", value=st.session_state.sidebar["autotune"])
st.session_state.sidebar["backtest_mode"] = st.sidebar.checkbox("Enable Backtesting", value=st.session_state.sidebar["backtest_mode"])

if st.session_state.sidebar["mode"] == "Auto":
    interval = None
    lookback = None
    threshold = None
    try:
        auto_params = safe_load_json(AUTO_PARAMS_FILE, {})
    except Exception:
        auto_params = {}
    def get_pair_params(pair):
        default = {
            "interval": "1h",
            "lookback": 1000,
            "threshold": trading_cfg["threshold"]
        }
        params = get_auto_pair_params(auto_params, pair, today=date.today(), fallback=default)
        if "params" in params:
            return params
        return {
            "interval": params.get("interval", "1h"),
            "lookback": params.get("lookback", 1000),
            "threshold": params.get("threshold", trading_cfg["threshold"]),
        }
else:
    st.session_state.sidebar["interval"] = st.sidebar.selectbox(
        "Candle Interval", ["5m", "15m", "30m", "1h", "4h", "1d"], 
        index=["5m", "15m", "30m", "1h", "4h", "1d"].index(st.session_state.sidebar["interval"])
    )
    st.session_state.sidebar["lookback"] = st.sidebar.slider(
        "Historical Lookback", 300, 2000, st.session_state.sidebar["lookback"]
    )
    st.session_state.sidebar["threshold"] = st.sidebar.slider(
        "Entry Threshold", min_value=0.0, max_value=1.0,
        value=st.session_state.sidebar["threshold"], step=0.01,
        help="How strong must the signal be before we BUY/SELL?"
    )

st.session_state.sidebar["max_positions"] = st.sidebar.number_input(
    "Max Open Positions", 1, 30, st.session_state.sidebar["max_positions"]
)
st.session_state.sidebar["stop_loss_pct"] = st.sidebar.number_input(
    "Stop-Loss %", 0.0, 10.0, st.session_state.sidebar["stop_loss_pct"]*100.0, step=0.1
) / 100
st.session_state.sidebar["take_profit_pct"] = st.sidebar.number_input(
    "Take-Profit %", 0.0, 10.0, st.session_state.sidebar["take_profit_pct"]*100.0, step=0.1
) / 100
st.session_state.sidebar["fee"] = st.sidebar.number_input(
    "Trade Fee %", 0.0, 1.0, st.session_state.sidebar["fee"]*100.0, step=0.01
) / 100
st.session_state.sidebar["atr_stop_mult"] = st.sidebar.number_input(
    "ATR Stop Multiplier", 0.1, 5.0, st.session_state.sidebar["atr_stop_mult"], step=0.1
)
st.session_state.sidebar["atr_tp_mult"] = st.sidebar.number_input(
    "ATR Take Profit Multiplier", 0.1, 10.0, st.session_state.sidebar["atr_tp_mult"], step=0.1
)
st.session_state.sidebar["atr_trail_mult"] = st.sidebar.number_input(
    "ATR Trailing Stop Multiplier", 0.1, 5.0, st.session_state.sidebar["atr_trail_mult"], step=0.1
)
st.session_state.sidebar["partial_exit"] = st.sidebar.checkbox(
    "Enable Partial Exit at TP1", value=st.session_state.sidebar["partial_exit"]
)

def persist_sidebar_overrides(sidebar: dict, config_path: str = CONFIG_PATH):
    import yaml
    with open(config_path, "r") as f:
        data = yaml.safe_load(f)
    if "trading" in data:
        data["trading"]["dry_run"] = sidebar["dry_run"]
        data["trading"]["max_positions"] = sidebar["max_positions"]
        data["trading"]["fee"] = sidebar["fee"]
        if "interval" in sidebar:
            data["trading"]["default_interval"] = sidebar["interval"]
        if "lookback" in sidebar:
            data["trading"]["backtest_lookback"] = sidebar["lookback"]
        if "threshold" in sidebar:
            data["trading"]["threshold"] = sidebar["threshold"]
    if "risk" in data:
        data["risk"]["stop_loss_pct"] = sidebar["stop_loss_pct"]
        data["risk"]["take_profit_pct"] = sidebar["take_profit_pct"]
    if "strategy" in data:
        data["strategy"]["mode"] = sidebar["mode"]
        data["strategy"]["autotune"] = sidebar["autotune"]
    with open(config_path, "w") as f:
        yaml.safe_dump(data, f)
    st.sidebar.success("Preferences saved to config/config.yaml!")

if st.sidebar.button("💾 Save Preferences"):
    persist_sidebar_overrides(st.session_state.sidebar, CONFIG_PATH)
    reset_sidebar()  # Reset to new config

if st.sidebar.button("🔄 Retrain ML Model from Trade Log"):
    with st.spinner("Retraining ML model..."):
        success, msg = train_and_save_model()
        if success:
            st.sidebar.success(msg)
        else:
            st.sidebar.error(msg)

st.sidebar.button("🔃 Reset Sidebar to Config Defaults", on_click=reset_sidebar)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "🟢 **Prometheus metrics** are available at [localhost:8000](http://localhost:8000/). "
    "To enable monitoring, use `config/prometheus.yml` for your Prometheus server."
)

def global_exception_handler(exc_type, exc_value, exc_traceback):
    tb = ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback))
    logger.critical(f"UNHANDLED EXCEPTION:\n{tb}")
    USER = os.getenv("ALERT_USER")
    PASS = os.getenv("ALERT_PASS")
    ALERT_EMAIL = os.getenv("ALERT_EMAIL")
    if USER and PASS and ALERT_EMAIL:
        try:
            import smtplib
            with smtplib.SMTP("smtp.gmail.com", 587) as s:
                s.starttls()
                s.login(USER, PASS)
                msg = f"Subject: OnlyFunds CRASHED!\n\n{tb}"
                s.sendmail(USER, ALERT_EMAIL, msg)
        except Exception as e:
            logger.error(f"Failed to send alert email: {e}")

sys.excepthook = global_exception_handler

def atomic_save_json(obj, file):
    import tempfile, shutil
    dir = os.path.dirname(file) or "."
    tmp = tempfile.NamedTemporaryFile("w", dir=dir, delete=False)
    try:
        import json
        json.dump(obj, tmp, indent=2)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp.close()
        shutil.move(tmp.name, file)
    except Exception as e:
        logger.error(f"Failed to atomically save {file}: {e}")
        try:
            os.remove(tmp.name)
        except Exception:
            pass

# --- The rest of your main logic goes here ---
# (trading loop, dashboard, etc., using the above robust utilities and patterns)