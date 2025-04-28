# ─── Imports and Initial Setup ─────────────────────────────────────────────────
import os
import sys
import time
import threading
import traceback
import logging
from datetime import datetime

import streamlit as st
st.set_page_config(page_title="CryptoTrader AI (A)", layout="wide")

import pandas as pd
import numpy as np
import yaml
import joblib
from dotenv import load_dotenv

from core.core_data import fetch_klines, validate_df, add_indicators, TRADING_PAIRS
from core.core_signals import (
    generate_signal, smooth_signal, adaptive_threshold, generate_ensemble_signal
)
from core.trade_execution import place_order
from core.backtester import run_backtest
from core.ml_filter import load_model, ml_confidence, train_and_save_model
from core.risk_manager import RiskManager
from utils.helpers import (
    compute_trade_metrics, suggest_tuning, save_json, load_json, validate_pair, get_auto_pair_params
)
from utils.config import load_config

# ─── Meta-learner (Auto Ensemble Model) ────────────────────────────────────────
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

def load_or_stub_meta_model(path):
    try:
        return joblib.load(path)
    except Exception:
        # Train dummy fallback
        from sklearn.dummy import DummyClassifier
        model = DummyClassifier(strategy="most_frequent")
        X = np.random.rand(10, 4)
        y = np.random.randint(0, 2, 10)
        model.fit(X, y)
        joblib.dump(model, path)
        return model

META_MODEL = load_or_stub_meta_model(META_MODEL_PATH)

# ─── Logging Setup ─────────────────────────────────────────────────────────────
from logging.handlers import RotatingFileHandler
from pythonjsonlogger import jsonlogger

LOGS_DIR = "logs"
os.makedirs(LOGS_DIR, exist_ok=True)
log_file = os.path.join(LOGS_DIR, f"{METRICS_PREFIX}.json")
handler = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5)
formatter = jsonlogger.JsonFormatter("%(asctime)s %(levelname)s %(name)s %(message)s")
handler.setFormatter(formatter)
root = logging.getLogger()
root.handlers.clear()
root.addHandler(handler)
root.setLevel(logging.INFO)

logger = logging.getLogger(__name__)
# ─── Prometheus Metrics Setup ───────────────────────────────────────────────────
from prometheus_client import start_http_server, Counter, Gauge, REGISTRY

def get_prometheus_metrics():
    module = __import__(__name__)
    if not hasattr(module, "_PROMETHEUS_METRICS"):
        try:
            start_http_server(8000)
        except Exception:
            pass
        metrics = {}
        def safe_counter(name, doc):
            if name not in REGISTRY._names_to_collectors:
                return Counter(name, doc)
            return REGISTRY._names_to_collectors[name]
        def safe_gauge(name, doc):
            if name not in REGISTRY._names_to_collectors:
                return Gauge(name, doc)
            return REGISTRY._names_to_collectors[name]

        metrics['trade_counter'] = safe_counter(f"{METRICS_PREFIX}_trades_executed_total", "Total trades executed")
        metrics['pnl_gauge'] = safe_gauge(f"{METRICS_PREFIX}_current_pnl", "Current unrealized PnL (USDT)")
        metrics['heartbeat_gauge'] = safe_gauge(f"{METRICS_PREFIX}_heartbeat", "Heartbeat timestamp")

        module._PROMETHEUS_METRICS = metrics
    m = module._PROMETHEUS_METRICS
    return m['trade_counter'], m['pnl_gauge'], m['heartbeat_gauge']

trade_counter, pnl_gauge, heartbeat_gauge = get_prometheus_metrics()

# ─── Config and Persistent State ───────────────────────────────────────────────
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

def safe_load_json(file, default):
    import json
    try:
        if os.path.exists(file):
            with open(file, "r") as f:
                data = json.load(f)
            return data if data is not None else default
        return default
    except Exception as e:
        st.sidebar.error(f"Failed to load {file}: {e}")
        return default

POSITIONS_FILE = "state/open_positions.json"
TRADE_LOG_FILE = "state/trade_log.json"
CAPITAL_FILE = "state/current_capital.json"
BACKTEST_RESULTS_FILE = "state/backtest_results.json"
AUTO_PARAMS_FILE = "state/auto_params.json"

os.makedirs("state", exist_ok=True)
open_positions = safe_load_json(POSITIONS_FILE, {})
trade_log = safe_load_json(TRADE_LOG_FILE, [])
current_capital = safe_load_json(CAPITAL_FILE, trading_cfg.get("default_capital", 10))
if not isinstance(current_capital, (float, int)):
    current_capital = trading_cfg.get("default_capital", 10)

risk_manager = RiskManager(config)

# ─── Sidebar (with Dynamic ATR Setting Placeholder) ─────────────────────────────
st.sidebar.header("⚙️ Configuration")
st.sidebar.markdown(f"**Meta-Learner Variant:** `{SELECTOR_VARIANT}`")

MODES = ["Conservative", "Normal", "Aggressive", "Auto"]
mode_idx = 3 if trading_cfg.get("mode", "Auto") == "Auto" else MODES.index(trading_cfg.get("mode", "Normal").capitalize())
mode = st.sidebar.selectbox("Trading Mode", MODES, index=mode_idx)

dry_run = st.sidebar.checkbox("Dry Run Mode (Simulated)", value=trading_cfg.get("dry_run", True))
autotune = st.sidebar.checkbox("Enable Adaptive-Threshold Autotune", value=True)

interval = st.sidebar.selectbox("Candle Interval", ["5m", "15m", "30m", "1h", "4h", "1d"], index=0)
lookback = st.sidebar.slider("Historical Lookback", 300, 2000, trading_cfg.get("backtest_lookback", 1000))

max_positions = st.sidebar.number_input("Max Open Positions", 1, 30, trading_cfg.get("max_positions", 5))
stop_loss_pct = st.sidebar.number_input("Static Stop Loss %", 0.0, 10.0, risk_cfg.get("stop_loss_pct", 0.01)*100.0, step=0.1) / 100
take_profit_pct = st.sidebar.number_input("Static Take Profit %", 0.0, 10.0, risk_cfg.get("take_profit_pct", 0.02)*100.0, step=0.1) / 100
fee_pct = st.sidebar.number_input("Trade Fee %", 0.0, 1.0, trading_cfg.get("fee", 0.001)*100.0, step=0.01) / 100

# New ATR MULTIPLIERS (automated later)
atr_stop_mult = st.sidebar.number_input("ATR Stop Multiplier", 0.1, 5.0, 1.0, step=0.1)
atr_tp_mult = st.sidebar.number_input("ATR Take Profit Multiplier", 0.1, 10.0, 2.0, step=0.1)
atr_trail_mult = st.sidebar.number_input("ATR Trailing Stop Multiplier", 0.1, 5.0, 1.0, step=0.1)

partial_exit = st.sidebar.checkbox("Enable Partial Exit at TP1", value=True)
# ─── Utility Functions ──────────────────────────────────────────────────────────
def atomic_save_json(obj, file):
    dir = os.path.dirname(file) or "."
    try:
        tmp = tempfile.NamedTemporaryFile("w", dir=dir, delete=False)
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

def validate_trade(amount, price, current_capital, min_size=0.001):
    if amount < min_size:
        logger.error(f"Trade validation failed: amount={amount:.6f} min_size={min_size}")
        return False
    if price <= 0:
        logger.error(f"Trade validation failed: price={price:.6f}")
        return False
    if current_capital <= 0:
        logger.error(f"Trade validation failed: current_capital={current_capital:.6f}")
        return False
    return True

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

# ─── Main Trading Logic ─────────────────────────────────────────────────────────
def main_loop():
    global current_capital, trade_log, open_positions
    retrain_ml_background(trade_log)

    for pair in TRADING_PAIRS:
        perf = compute_trade_metrics(trade_log, trading_cfg.get("default_capital", 100))

        df = fetch_klines(pair, interval, lookback)
        if df.empty or not validate_df(df):
            logger.warning(f"Data for {pair} invalid/empty.")
            continue
        df = add_indicators(df)

        try:
            if META_MODEL is not None:
                signal = generate_ensemble_signal(df, META_MODEL)
            else:
                signal = generate_signal(df)
        except Exception as e:
            logger.error(f"Signal generation failed for {pair}: {e}")
            continue

        # --- Autotune Threshold or Use Manual ---
        if autotune:
            try:
                threshold_final = adaptive_threshold(df, target_profit=0.01)
            except Exception as e:
                logger.warning(f"Autotune failed for {pair}: {e}")
                threshold_final = 0.5
        else:
            threshold_final = st.session_state.sidebar.get("threshold", 0.5)

        # --- ATR-Based Dynamic Settings (Overriding fixed stop_loss / tp) ---
        latest_atr = df["ATR"].iloc[-1] if "ATR" in df.columns else 0.001

        stop_loss_value = latest_atr * atr_stop_mult
        take_profit_value = latest_atr * atr_tp_mult
        trailing_stop_value = latest_atr * atr_trail_mult

        price = df["Close"].iloc[-1]
        min_size = 0.001

        # --- Managing Open Positions ---
        if pair in open_positions:
            pos = open_positions[pair]
            entry_price = pos["entry_price"]
            amount = pos["amount"]

            # Check if TP hit
            if price >= entry_price + take_profit_value:
                if partial_exit and "partial_exited" not in pos:
                    logger.info(f"{pair}: Partial Take Profit triggered.")
                    partial_amount = amount / 2
                    if validate_trade(partial_amount, price, current_capital, min_size):
                        result = place_order(pair, "SELL", partial_amount, price, dry_run)
                        logger.info(f"Partial exit: {result}")
                        pos["amount"] -= partial_amount
                        pos["partial_exited"] = True
                        current_capital += partial_amount * price * (1 - fee_pct)
                        atomic_save_json(open_positions, POSITIONS_FILE)
                        atomic_save_json(current_capital, CAPITAL_FILE)
                        trade_counter.inc()
                    continue
                else:
                    logger.info(f"{pair}: Full Take Profit triggered.")
                    if validate_trade(amount, price, current_capital, min_size):
                        result = place_order(pair, "SELL", amount, price, dry_run)
                        logger.info(f"Closed TP: {result}")
                        trade_log.append({
                            "pair": pair,
                            "side": "SELL",
                            "amount": amount,
                            "price": price,
                            "timestamp": datetime.utcnow().isoformat(),
                            "result": "TP"
                        })
                        current_capital += amount * price * (1 - fee_pct)
                        del open_positions[pair]
                        atomic_save_json(open_positions, POSITIONS_FILE)
                        atomic_save_json(trade_log, TRADE_LOG_FILE)
                        atomic_save_json(current_capital, CAPITAL_FILE)
                        trade_counter.inc()
                    continue

            # Check if SL hit
            if price <= entry_price - stop_loss_value:
                logger.info(f"{pair}: Stop Loss triggered.")
                if validate_trade(amount, price, current_capital, min_size):
                    result = place_order(pair, "SELL", amount, price, dry_run)
                    logger.info(f"Closed SL: {result}")
                    trade_log.append({
                        "pair": pair,
                        "side": "SELL",
                        "amount": amount,
                        "price": price,
                        "timestamp": datetime.utcnow().isoformat(),
                        "result": "SL"
                    })
                    current_capital += amount * price * (1 - fee_pct)
                    del open_positions[pair]
                    atomic_save_json(open_positions, POSITIONS_FILE)
                    atomic_save_json(trade_log, TRADE_LOG_FILE)
                    atomic_save_json(current_capital, CAPITAL_FILE)
                    trade_counter.inc()
                continue

        # --- Managing New Entries ---
        if pair not in open_positions:
            if len(open_positions) >= max_positions:
                continue
            latest_signal = signal.iloc[-1] if hasattr(signal, "iloc") else signal[-1]
            if latest_signal < threshold_final:
                continue

            amount = min(current_capital / price, 1.0)
            if validate_trade(amount, price, current_capital, min_size):
                result = place_order(pair, "BUY", amount, price, dry_run)
                logger.info(f"Placed BUY order: {result}")
                open_positions[pair] = {
                    "amount": amount,
                    "entry_price": price,
                    "timestamp": datetime.utcnow().isoformat()
                }
                trade_log.append({
                    "pair": pair,
                    "side": "BUY",
                    "amount": amount,
                    "price": price,
                    "timestamp": datetime.utcnow().isoformat(),
                    "result": None
                })
                current_capital -= amount * price * (1 + fee_pct)
                atomic_save_json(open_positions, POSITIONS_FILE)
                atomic_save_json(trade_log, TRADE_LOG_FILE)
                atomic_save_json(current_capital, CAPITAL_FILE)
                trade_counter.inc()

        pnl_gauge.set(current_capital)
        write_heartbeat()
        time.sleep(0.1)

    retrain_ml_background(trade_log)
    write_heartbeat()
# ─── Trading and Backtesting Buttons ───────────────────────────────────────────
run_trading_btn = st.sidebar.button("▶️ Run Trading Cycle")
run_backtest_btn = st.sidebar.button("🧪 Run Backtest")

# === Trading Mode ===
if run_trading_btn:
    try:
        main_loop()
    except Exception as e:
        tb = traceback.format_exc()
        logger.critical(f"Trading loop crashed:\n{tb}")
        st.error(f"Trading crashed: {e}")
        write_heartbeat()

# === Backtest Mode ===
if run_backtest_btn:
    try:
        st.sidebar.info("Backtest started...")

        pair = TRADING_PAIRS[0]  # You could loop or select specific pair if needed
        df = fetch_klines(pair, interval, lookback)
        if df.empty or not validate_df(df):
            st.sidebar.error(f"Backtest failed: Data for {pair} invalid or empty.")
        else:
            df = add_indicators(df)

            # Decide which signal generator to use
            if META_MODEL is not None:
                signal = generate_ensemble_signal(df, META_MODEL)
            else:
                signal = generate_signal(df)

            prices = df["Close"]

            # Fetch latest ATR
            latest_atr = df["ATR"].iloc[-1] if "ATR" in df.columns else 0.001

            stop_mult = st.session_state.sidebar.get("atr_stop_mult", 1.0)
            tp_mult = st.session_state.sidebar.get("atr_tp_mult", 2.0)
            trail_mult = st.session_state.sidebar.get("atr_trail_mult", 1.0)

            # Calculate realistic dynamic stops
            stop_loss_value = latest_atr * stop_mult
            take_profit_value = latest_atr * tp_mult
            trailing_stop_value = latest_atr * trail_mult

            bt_results = run_backtest(
                signal=signal,
                prices=prices,
                threshold=threshold_slider,
                initial_capital=trading_cfg.get("default_capital", 10.0),
                risk_pct=risk_cfg.get("risk_pct", 0.01),
                stop_loss_pct=stop_loss_value,
                take_profit_pct=take_profit_value,
                fee_pct=st.session_state.sidebar.get("fee", 0.001),
                verbose=True,
                partial_exit=st.session_state.sidebar.get("partial_exit", False),
                stop_loss_atr_mult=stop_mult,
                take_profit_atr_mult=tp_mult,
                atr=df["ATR"] if "ATR" in df.columns else None,
                trailing_atr_mult=trail_mult,
            )

            if bt_results is not None:
                st.success("Backtest Completed!")
                st.subheader(f"📈 Backtest Results for {pair}")
                st.dataframe(bt_results)
            else:
                st.error("Backtest failed: No valid results.")

    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"Backtest crashed:\n{tb}")
        st.sidebar.error(f"Backtest crash: {e}")
# ─── Global Exception Handler ──────────────────────────────────────────────────
def global_exception_handler(exc_type, exc_value, exc_traceback):
    tb = ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback))
    logger.critical(f"UNHANDLED EXCEPTION:\n{tb}")

    USER = os.getenv("ALERT_USER")
    PASS = os.getenv("ALERT_PASS")
    ALERT_EMAIL = os.getenv("ALERT_EMAIL")

    if USER and PASS and ALERT_EMAIL:
        try:
            import smtplib
            with smtplib.SMTP("smtp.gmail.com", 587) as server:
                server.starttls()
                server.login(USER, PASS)
                message = f"Subject: OnlyFunds CRASHED!\n\n{tb}"
                server.sendmail(USER, ALERT_EMAIL, message)
            logger.info("Crash alert email sent.")
        except Exception as email_error:
            logger.error(f"Failed to send alert email: {email_error}")

# Attach to sys
sys.excepthook = global_exception_handler
