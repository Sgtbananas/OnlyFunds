import os
import sys
import logging
import time
import threading
import traceback
from datetime import datetime, date, timedelta

import streamlit as st
st.set_page_config(page_title="CryptoTrader AI (A)", layout="wide")  # FIRST STREAMLIT CALL

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

# --- Meta-learner fallback (background thread stub training if missing) ---
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

def ensure_meta_model(path):
    try:
        return joblib.load(path)
    except Exception as e:
        print(f"[WARN] Could not load meta-learner model at {path}: {e}")
        def _train_and_reload():
            train_stub_meta_model(path)
        threading.Thread(target=_train_and_reload, daemon=True).start()
        return None  # will be available soon

META_MODEL = ensure_meta_model(META_MODEL_PATH)

# --- Logging: robust, rotating, JSON, optional console ---
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

# --- Prometheus singleton ---
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

# --- State + config ---
CONFIG_PATH = "config/config.yaml"
config = load_config(CONFIG_PATH)
risk_cfg = config["risk"]
trading_cfg = config["trading"]
ml_cfg = config.get("ml", {})

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

# --- Sidebar state: idiot-proof, autotune default, advanced threshold slider ---
def get_config_defaults():
    return dict(
        mode=config.get("strategy", {}).get("mode", "Normal").capitalize(),
        dry_run=trading_cfg["dry_run"],
        autotune=True,  # Idiot-proof default
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

with st.sidebar.expander("Advanced", expanded=False):
    # Entry Threshold only shown in advanced
    if st.session_state.sidebar["autotune"]:
        st.write("Entry Threshold is autotuned. Disable to override.")
    else:
        st.session_state.sidebar["threshold"] = st.slider(
            "Entry Threshold", min_value=0.0, max_value=1.0,
            value=st.session_state.sidebar["threshold"], step=0.01,
            help="How strong must the signal be before we BUY/SELL?"
        )

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

# --- Save preferences ---
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

st.sidebar.button("🔃 Reset Sidebar to Config Defaults", on_click=reset_sidebar)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "🟢 **Prometheus metrics** are available at [localhost:8000](http://localhost:8000/). "
    "To enable monitoring, use `config/prometheus.yml` for your Prometheus server."
)

# --- Hands-free ML Retraining ---
RETRAIN_TRADE_INTERVAL = 50   # retrain after every 50 trades
RETRAIN_TIME_INTERVAL = 6*60*60  # retrain after every 6 hours (in seconds)
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
        logger.error(f"Failed to write last retrain time: {e}")

def retrain_ml_if_needed(trade_log):
    now = time.time()
    last_ts = read_last_retrain_time()
    n_since = len(trade_log) % RETRAIN_TRADE_INTERVAL
    need_time = (now - last_ts) > RETRAIN_TIME_INTERVAL
    need_trades = len(trade_log) > 0 and n_since == 0
    if need_time or need_trades:
        logger.info("Auto ML retrain triggered.")
        success, msg = train_and_save_model()
        write_last_retrain_time(now)
        if success:
            logger.info(f"ML retrain: {msg}")
        else:
            logger.error(f"ML retrain error: {msg}")

def retrain_ml_background(trade_log):
    threading.Thread(target=retrain_ml_if_needed, args=(trade_log,), daemon=True).start()

def show_last_retrain_sidebar():
    last_ts = read_last_retrain_time()
    if last_ts > 0:
        st.sidebar.markdown(f"**Last ML retrain:** {datetime.fromtimestamp(last_ts).strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        st.sidebar.markdown("**Last ML retrain:** Never")

show_last_retrain_sidebar()

# --- Watchdog/heartbeat for auto-restart ---
def write_heartbeat():
    ts = time.time()
    try:
        with open(HEARTBEAT_FILE, "w") as f:
            import json
            json.dump({"last_run": ts}, f)
    except Exception as e:
        logger.error(f"Failed to write heartbeat: {e}")
    try:
        heartbeat_gauge.set(ts)
    except Exception:
        pass

def watchdog_loop(main_pid, heartbeat_file=HEARTBEAT_FILE, interval=10, max_stale=30):
    # This runs in the background and will forcibly restart the process if heartbeat is stale
    import psutil
    while True:
        try:
            time.sleep(interval)
            if not os.path.exists(heartbeat_file):
                continue
            with open(heartbeat_file, "r") as f:
                import json
                data = json.load(f)
                last = data.get("last_run", 0)
                age = time.time() - last
                if age > max_stale:
                    logger.critical("Watchdog: Heartbeat stale! Attempting self-restart.")
                    # In production, use supervisor/systemd to restart. Here, self-exec for dev.
                    psutil.Process(main_pid).terminate()
                    os.execv(sys.executable, [sys.executable] + sys.argv)
        except Exception as e:
            logger.error(f"Watchdog error: {e}")

def start_watchdog():
    main_pid = os.getpid()
    threading.Thread(target=watchdog_loop, args=(main_pid,), daemon=True).start()
    logger.info("Watchdog started.")

start_watchdog()

# --- Parameter validation before placing trade ---
def validate_trade(amount, price, current_capital, min_size=0.001):
    if amount < min_size:
        logger.error(f"Trade validation: amount {amount} < min_size {min_size}")
        return False
    if price <= 0:
        logger.error(f"Trade validation: price {price} <= 0")
        return False
    if current_capital <= 0:
        logger.error(f"Trade validation: current_capital {current_capital} <= 0")
        return False
    return True

# --- Self-tuning max positions ---
def self_tune_max_positions(win_rate, max_positions, trade_count, min_pos=1, max_pos=20, tune_every=10):
    # Increase if >60% win, decrease if <40% win, every tune_every trades
    if trade_count == 0 or trade_count % tune_every != 0:
        return max_positions
    if win_rate > 60.0 and max_positions < max_pos:
        logger.info(f"Self-tune: Increasing max_positions to {max_positions+1} (win_rate={win_rate:.1f}%)")
        return max_positions + 1
    if win_rate < 40.0 and max_positions > min_pos:
        logger.info(f"Self-tune: Decreasing max_positions to {max_positions-1} (win_rate={win_rate:.1f}%)")
        return max_positions - 1
    return max_positions

# --- Global error handler for email alerts ---
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

# --- Main trading loop with buy, sell/close, stop-loss, take-profit ---
def main_loop():
    global current_capital, trade_log, open_positions
    retrain_ml_background(trade_log)
    for pair in TRADING_PAIRS:
        perf = compute_trade_metrics(trade_log, trading_cfg["default_capital"])
        st.session_state.sidebar["max_positions"] = self_tune_max_positions(
            perf["win_rate"], st.session_state.sidebar["max_positions"], len(trade_log)
        )

        # Get trade params
        if st.session_state.sidebar["mode"] == "Auto":
            p = get_pair_params(pair)
            interval_used = p["interval"]
            lookback_used = p["lookback"]
            threshold_used = p["threshold"]
        else:
            interval_used = st.session_state.sidebar["interval"]
            lookback_used = st.session_state.sidebar["lookback"]
            threshold_used = st.session_state.sidebar["threshold"]

        # Fetch/validate data
        df = fetch_klines(pair, interval_used, lookback_used)
        if df.empty or not validate_df(df):
            logger.warning(f"Data for {pair} invalid/empty.")
            continue
        df = add_indicators(df)

        # Generate signal
        try:
            if META_MODEL is not None:
                signal = generate_ensemble_signal(df, META_MODEL)
            else:
                signal = generate_signal(df)
        except Exception as e:
            logger.error(f"Signal gen failed for {pair}: {e}")
            continue

        if st.session_state.sidebar["autotune"]:
            try:
                threshold_final = adaptive_threshold(df, target_profit=0.01)
            except Exception as e:
                logger.warning(f"Autotune failed for {pair}, fallback to default threshold: {e}")
                threshold_final = trading_cfg.get("threshold", 0.5)
        else:
            threshold_final = threshold_used

        price = df["Close"].iloc[-1]
        min_size = 0.001

        # --- SELL/CLOSE logic: If we have an open position, check for TP/SL or exit signal
        if pair in open_positions:
            pos = open_positions[pair]
            entry_price = pos["entry_price"]
            amount = pos["amount"]
            # Take-profit
            if price >= entry_price * (1 + st.session_state.sidebar["take_profit_pct"]):
                logger.info(f"{pair}: Take-profit hit. Closing position.")
                if validate_trade(amount, price, current_capital, min_size=min_size):
                    try:
                        result = place_order(pair, "SELL", amount, price, dry_run=st.session_state.sidebar["dry_run"])
                        logger.info(f"Closed (TP): {result}")
                        trade_log.append({
                            "pair": pair,
                            "side": "SELL",
                            "amount": amount,
                            "price": price,
                            "timestamp": datetime.utcnow().isoformat(),
                            "result": "TP"
                        })
                        current_capital += amount * price * (1 - st.session_state.sidebar["fee"])
                        del open_positions[pair]
                        atomic_save_json(open_positions, POSITIONS_FILE)
                        atomic_save_json(trade_log, TRADE_LOG_FILE)
                        atomic_save_json(current_capital, CAPITAL_FILE)
                        trade_counter.inc()
                    except Exception as e:
                        logger.error(f"TP close failed for {pair}: {e}")
                write_heartbeat()
                continue
            # Stop-loss
            if price <= entry_price * (1 - st.session_state.sidebar["stop_loss_pct"]):
                logger.info(f"{pair}: Stop-loss hit. Closing position.")
                if validate_trade(amount, price, current_capital, min_size=min_size):
                    try:
                        result = place_order(pair, "SELL", amount, price, dry_run=st.session_state.sidebar["dry_run"])
                        logger.info(f"Closed (SL): {result}")
                        trade_log.append({
                            "pair": pair,
                            "side": "SELL",
                            "amount": amount,
                            "price": price,
                            "timestamp": datetime.utcnow().isoformat(),
                            "result": "SL"
                        })
                        current_capital += amount * price * (1 - st.session_state.sidebar["fee"])
                        del open_positions[pair]
                        atomic_save_json(open_positions, POSITIONS_FILE)
                        atomic_save_json(trade_log, TRADE_LOG_FILE)
                        atomic_save_json(current_capital, CAPITAL_FILE)
                        trade_counter.inc()
                    except Exception as e:
                        logger.error(f"SL close failed for {pair}: {e}")
                write_heartbeat()
                continue
            # Exit on negative signal
            exit_signal = signal.iloc[-1] if hasattr(signal, 'iloc') else signal[-1]
            if exit_signal < 0:
                logger.info(f"{pair}: Negative signal, closing position.")
                if validate_trade(amount, price, current_capital, min_size=min_size):
                    try:
                        result = place_order(pair, "SELL", amount, price, dry_run=st.session_state.sidebar["dry_run"])
                        logger.info(f"Closed (SIG): {result}")
                        trade_log.append({
                            "pair": pair,
                            "side": "SELL",
                            "amount": amount,
                            "price": price,
                            "timestamp": datetime.utcnow().isoformat(),
                            "result": "SIG"
                        })
                        current_capital += amount * price * (1 - st.session_state.sidebar["fee"])
                        del open_positions[pair]
                        atomic_save_json(open_positions, POSITIONS_FILE)
                        atomic_save_json(trade_log, TRADE_LOG_FILE)
                        atomic_save_json(current_capital, CAPITAL_FILE)
                        trade_counter.inc()
                    except Exception as e:
                        logger.error(f"SIG close failed for {pair}: {e}")
                write_heartbeat()
                continue

        # --- BUY/OPEN logic: If not currently open and eligible
        if pair not in open_positions:
            # Only open if under max positions and positive signal
            if len(open_positions) >= st.session_state.sidebar["max_positions"]:
                logger.info(f"Max open positions reached ({len(open_positions)}). Skipping {pair}.")
                continue
            entry_signal = signal.iloc[-1] if hasattr(signal, 'iloc') else signal[-1]
            if entry_signal < threshold_final:
                logger.info(f"{pair}: Signal {entry_signal:.3f} below threshold {threshold_final:.3f}")
                continue
            # Position sizing (idiot-proof, never more than 1 unit for safety)
            amount = min(current_capital / price, 1.0)
            if not validate_trade(amount, price, current_capital, min_size=min_size):
                continue
            try:
                trade_result = place_order(pair, "BUY", amount, price, dry_run=st.session_state.sidebar["dry_run"])
                logger.info(f"Placed trade: {trade_result}")
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
                current_capital -= amount * price * (1 + st.session_state.sidebar["fee"])
                atomic_save_json(open_positions, POSITIONS_FILE)
                atomic_save_json(trade_log, TRADE_LOG_FILE)
                atomic_save_json(current_capital, CAPITAL_FILE)
                trade_counter.inc()
            except Exception as e:
                logger.error(f"Trade execution failed for {pair}: {e}")
        write_heartbeat()
        pnl_gauge.set(current_capital)
        time.sleep(0.1)

    retrain_ml_background(trade_log)
    write_heartbeat()

# --- Run/Backtest Buttons in Sidebar ---
run_trading_btn = st.sidebar.button("▶️ Run Trading Cycle")
run_backtest_btn = st.sidebar.button("🧪 Run Backtest")

if run_trading_btn:
    main_loop()

if run_backtest_btn:
    st.sidebar.info("Backtest started...")

    # Example: use the first pair and current sidebar params for backtest
    pair = TRADING_PAIRS[0]
    if st.session_state.sidebar["mode"] == "Auto":
        p = get_pair_params(pair)
        interval_used = p["interval"]
        lookback_used = p["lookback"]
        threshold_used = p["threshold"]
    else:
        interval_used = st.session_state.sidebar["interval"]
        lookback_used = st.session_state.sidebar["lookback"]
        threshold_used = st.session_state.sidebar["threshold"]

    df = fetch_klines(pair, interval_used, lookback_used)
    if df.empty or not validate_df(df):
        st.sidebar.error(f"Backtest failed: Data for {pair} invalid or empty.")
        bt_results = None
    else:
        df = add_indicators(df)

        # Generate signal for backtest (use meta-model if available)
        if META_MODEL is not None:
            signal = generate_ensemble_signal(df, META_MODEL)
        else:
            signal = generate_signal(df)
        prices = df["Close"]

        # Advanced risk/ATR/trailing/partial logic
        bt_results = run_backtest(
            signal=signal,
            prices=prices,
            threshold=threshold_used,
            initial_capital=trading_cfg.get("default_capital", 10.0),
            risk_pct=risk_cfg.get("risk_pct", 0.01),
            stop_loss_pct=st.session_state.sidebar["stop_loss_pct"],
            take_profit_pct=st.session_state.sidebar["take_profit_pct"],
            fee_pct=st.session_state.sidebar["fee"],
            verbose=False,
            partial_exit=st.session_state.sidebar.get("partial_exit", False),
            stop_loss_atr_mult=st.session_state.sidebar.get("atr_stop_mult", None),
            take_profit_atr_mult=st.session_state.sidebar.get("atr_tp_mult", None),
            atr=df["ATR"] if "ATR" in df.columns else None,
            trailing_atr_mult=st.session_state.sidebar.get("atr_trail_mult", None)
        )

        st.sidebar.success("Backtest complete!")

    if bt_results is not None:
        st.write("Backtest Results", bt_results)
    else:
        st.write("No results to show (backtest failed).")