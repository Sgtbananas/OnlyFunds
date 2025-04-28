import os
import logging
import time
from datetime import datetime, date

import streamlit as st
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

try:
    META_MODEL = joblib.load(META_MODEL_PATH)
except Exception as e:
    META_MODEL = None
    print(f"[WARN] Could not load meta-learner model at {META_MODEL_PATH}: {e}")

from pythonjsonlogger import jsonlogger
LOGS_DIR = "logs"
os.makedirs(LOGS_DIR, exist_ok=True)
handler = logging.FileHandler(os.path.join(LOGS_DIR, f"{METRICS_PREFIX}.json"))
formatter = jsonlogger.JsonFormatter("%(asctime)s %(levelname)s %(name)s %(message)s")
handler.setFormatter(formatter)
root = logging.getLogger()
root.addHandler(handler)
root.setLevel(logging.INFO)

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

config = load_config()
risk_cfg = config["risk"]
trading_cfg = config["trading"]
ml_cfg = config.get("ml", {})

st.set_page_config(page_title=f"CryptoTrader AI ({SELECTOR_VARIANT})", layout="wide")
logger = logging.getLogger(__name__)

POSITIONS_FILE = "state/open_positions.json"
TRADE_LOG_FILE = "state/trade_log.json"
CAPITAL_FILE = "state/current_capital.json"
BACKTEST_RESULTS_FILE = "state/backtest_results.json"
OPTUNA_BEST_FILE = "state/optuna_best.json"
AUTO_PARAMS_FILE = "state/auto_params.json"
HEARTBEAT_FILE = f"state/heartbeat_{SELECTOR_VARIANT}.json"

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
            current_capital = trading_cfg["default_capital"]
    else:
        current_capital = trading_cfg["default_capital"]
except Exception as e:
    logger.warning(f"Failed to load {CAPITAL_FILE}: {e}")
    current_capital = trading_cfg["default_capital"]

risk_manager = RiskManager(config)

st.title(f"🧠 CryptoTrader AI Bot (SPOT Market Only) — Variant {SELECTOR_VARIANT}")
st.sidebar.header("⚙️ Configuration")
st.sidebar.markdown(f"**Meta-Learner Variant:** `{SELECTOR_VARIANT}`")

modes = ["Conservative", "Normal", "Aggressive", "Auto"]
mode_idx = 3 if config.get("strategy", {}).get("mode", "Normal").capitalize() == "Auto" else modes.index(config.get("strategy", {}).get("mode", "Normal").capitalize())
mode = st.sidebar.selectbox("Trading Mode", modes, index=mode_idx)
dry_run = st.sidebar.checkbox("Dry Run Mode (Simulated)", value=trading_cfg["dry_run"])
autotune = st.sidebar.checkbox("Enable Adaptive-Threshold Autotune", value=False)
backtest_mode = st.sidebar.checkbox("Enable Backtesting", value=False)

if mode == "Auto":
    interval = None
    lookback = None
    threshold = None
    try:
        auto_params = load_json(AUTO_PARAMS_FILE)
    except Exception:
        auto_params = {}
    def get_pair_params(pair):
        default = {
            "interval": "1h",
            "lookback": 1000,
            "threshold": trading_cfg["threshold"]
        }
        params = get_auto_pair_params(auto_params, pair, today=date.today(), fallback=default)
        # If full param dict is present (from patched walkforward), unpack
        if "params" in params:
            return params
        # Legacy fallback
        return {
            "interval": params.get("interval", "1h"),
            "lookback": params.get("lookback", 1000),
            "threshold": params.get("threshold", trading_cfg["threshold"]),
        }
else:
    interval = st.sidebar.selectbox("Candle Interval",
                                    ["5m", "15m", "30m", "1h", "4h", "1d"],
                                    index=0)
    lookback = st.sidebar.slider("Historical Lookback", 300, 2000, 1000)
    threshold = st.sidebar.slider(
        "Entry Threshold",
        min_value=0.0, max_value=1.0,
        value=trading_cfg["threshold"], step=0.01,
        help="How strong must the signal be before we BUY/SELL?"
    )
max_positions = st.sidebar.number_input("Max Open Positions", 1, 30, trading_cfg["max_positions"])
stop_loss_pct = st.sidebar.number_input("Stop-Loss %", 0.0, 10.0, risk_cfg["stop_loss_pct"]*100.0, step=0.1) / 100
take_profit_pct = st.sidebar.number_input("Take-Profit %", 0.0, 10.0, risk_cfg["take_profit_pct"]*100.0, step=0.1) / 100
fee_pct = st.sidebar.number_input("Trade Fee %", 0.0, 1.0, trading_cfg["fee"]*100.0, step=0.01) / 100

# --- New risk management config for ATR/trailing ---
atr_stop_mult = st.sidebar.number_input("ATR Stop Multiplier", 0.1, 5.0, 1.0, step=0.1)
atr_tp_mult = st.sidebar.number_input("ATR Take Profit Multiplier", 0.1, 10.0, 2.0, step=0.1)
atr_trail_mult = st.sidebar.number_input("ATR Trailing Stop Multiplier", 0.1, 5.0, 1.0, step=0.1)
partial_exit = st.sidebar.checkbox("Enable Partial Exit at TP1", value=True)

if st.sidebar.button("🔄 Retrain ML Model from Trade Log"):
    with st.spinner("Retraining ML model..."):
        success, msg = train_and_save_model()
        if success:
            st.sidebar.success(msg)
        else:
            st.sidebar.error(msg)

def load_backtest_results():
    if os.path.exists(BACKTEST_RESULTS_FILE):
        try:
            results = load_json(BACKTEST_RESULTS_FILE)
            df = pd.DataFrame(results)
            if not df.empty and "pair" in df and "threshold" in df:
                if "timestamp" in df.columns:
                    df = df.sort_values("timestamp", ascending=False)
                else:
                    df = df.iloc[::-1]
            return df
        except Exception as e:
            st.warning(f"Failed to load backtest results: {e}")
            return pd.DataFrame()
    else:
        return pd.DataFrame()

def load_optuna_best():
    if os.path.exists(OPTUNA_BEST_FILE):
        try:
            return load_json(OPTUNA_BEST_FILE)
        except Exception as e:
            st.warning(f"Failed to load Optuna best results: {e}")
            return None
    return None

def load_auto_params():
    if os.path.exists(AUTO_PARAMS_FILE):
        try:
            return load_json(AUTO_PARAMS_FILE)
        except Exception as e:
            st.warning(f"Failed to load Auto Params: {e}")
            return None
    return None

st.sidebar.markdown("---")
st.sidebar.subheader("📊 Backtest & Optimization Results")

if st.sidebar.button("⏬ Show Backtest Results Table"):
    with st.spinner("Loading backtest results..."):
        df_results = load_backtest_results()
        if not df_results.empty:
            st.subheader("Recent Backtest Results (Most Recent on Top)")
            st.dataframe(df_results, use_container_width=True)
            st.download_button(
                "Download Backtest Results as CSV",
                df_results.to_csv(index=False),
                file_name="backtest_results.csv"
            )
        else:
            st.info("No backtest results found.")

if st.sidebar.button("🏆 Show Optuna Best Params"):
    best = load_optuna_best()
    if best:
        st.subheader("Optuna Best Hyperparameters")
        st.json(best)
    else:
        st.info("No Optuna best parameters found.")

if st.sidebar.button("🧠 Show Auto Params"):
    auto_params = load_auto_params()
    if auto_params:
        st.subheader("Auto (Per-Pair) Hyperparameters")
        st.json(auto_params)
    else:
        st.info("No Auto (per-pair) parameters found.")

start_btn = st.sidebar.button("🚀 Start Trading Bot (Spot Only)")
if start_btn:
    st.success(f"Bot started! (Spot market only, Variant {SELECTOR_VARIANT})")
else:
    st.info("Ready. Configure & click Start.")

if mode == "Conservative":
    risk_pct = 0.0025
    min_signal_conf = ml_cfg.get("min_signal_conf", 0.7)
    max_positions = min(max_positions, 3)
    enable_ml = ml_cfg.get("enabled", True)
elif mode == "Aggressive":
    risk_pct = 0.02
    min_signal_conf = ml_cfg.get("min_signal_conf", 0.4)
    max_positions = max(max_positions, 20)
    enable_ml = ml_cfg.get("enabled", True)
elif mode == "Auto":
    risk_pct = risk_cfg["per_trade"]
    min_signal_conf = ml_cfg.get("min_signal_conf", 0.5)
    enable_ml = ml_cfg.get("enabled", True)
else:
    risk_pct = risk_cfg["per_trade"]
    min_signal_conf = ml_cfg.get("min_signal_conf", 0.5)
    enable_ml = ml_cfg.get("enabled", True)

@st.cache_data(show_spinner=False)
def cached_fetch_klines(pair, interval, limit):
    return fetch_klines(pair=pair, interval=interval, limit=limit)

def write_heartbeat():
    ts = time.time()
    heartbeat_gauge.set(ts)
    with open(HEARTBEAT_FILE, "w") as f:
        import json
        json.dump({"last_run": ts}, f)

def trade_logic(pair: str, current_capital):
    # --- Get per-pair, per-day param dict ---
    if mode == "Auto":
        p = get_pair_params(pair)
        # Unpack all params from walkforward_optuna (or fallback)
        pair_interval = p.get("interval", "1h")
        pair_lookback = p.get("lookback", 1000)
        # Multi-param: threshold and indicator/signal settings (from walkforward param dict)
        best_params = p.get("params", {}) if "params" in p else {}
        pair_threshold = best_params.get("threshold", p.get("threshold", trading_cfg["threshold"]))
        indicator_params = {
            k: v for k, v in best_params.items() if k in [
                "rsi_window", "macd_fast", "macd_slow", "macd_signal", "boll_window",
                "boll_std", "ema_span", "volatility_window", "atr_window"
            ]
        }
        signal_params = {
            "regime_kwargs": {
                "vol_window": best_params.get("regime_vol_window"),
                "mom_window": best_params.get("regime_mom_window"),
                "vol_thresh": best_params.get("regime_vol_thresh"),
                "mom_thresh": best_params.get("regime_mom_thresh"),
            },
            "trend_kwargs": {
                "ema_fast_window": best_params.get("trend_ema_fast_window"),
                "ema_slow_window": best_params.get("trend_ema_slow_window"),
            },
            "reversion_kwargs": {
                "rsi_low": best_params.get("reversion_rsi_low"),
                "rsi_high": best_params.get("reversion_rsi_high"),
            }
        }
        smoothing_window = best_params.get("smoothing_window", 5)
        interval_used = pair_interval
        lookback_used = pair_lookback
        threshold_used = pair_threshold
    else:
        interval_used = interval
        lookback_used = lookback
        threshold_used = threshold
        indicator_params = {}
        signal_params = {}
        smoothing_window = 5

    try:
        base, quote = validate_pair(pair)
    except ValueError as ve:
        logger.error(f"❌ Invalid trading pair '{pair}': {ve}")
        return None, current_capital

    logger.info(f"🔍 Analyzing {pair}")
    df = cached_fetch_klines(pair, interval_used, lookback_used)
    if df.empty or not validate_df(df):
        logger.warning(f"⚠️ Invalid/empty data for {pair}")
        return None, current_capital

    df = add_indicators(df, indicator_params=indicator_params)
    if df.empty:
        logger.warning(f"⚠️ add_indicators produced empty df for {pair}")
        return None, current_capital

    # Multi-param signals
    if META_MODEL is not None:
        raw_signal = generate_ensemble_signal(df, meta_model=META_MODEL, **signal_params)
    else:
        raw_signal = generate_signal(df, indicator_params=signal_params)
    smoothed = smooth_signal(raw_signal, smoothing_window=smoothing_window)

    if autotune and mode != "Auto":
        threshold_final = adaptive_threshold(df, target_profit=0.01)
    else:
        threshold_final = threshold_used

    logger.debug(f"Threshold for {pair}: {threshold_final}")
    latest_signal = smoothed.iloc[-1]

    # ML confidence filter
    if enable_ml:
        try:
            features = [
                df["rsi"].iloc[-1] if "rsi" in df else 0,
                df["macd"].iloc[-1] if "macd" in df else 0,
                df["ema_diff"].iloc[-1] if "ema_diff" in df else 0,
                df["Close"].pct_change().rolling(20).std().iloc[-1] if "Close" in df else 0
            ]
            model = load_model()
            if model is not None:
                prob = ml_confidence(features, model=model)
                logger.info(f"ML confidence for {pair}: {prob:.2f} (min required: {min_signal_conf})")
                st.sidebar.write(f"ML prob for {pair}: {prob:.2f}")
                if prob < min_signal_conf:
                    logger.info(f"ML filter blocked trade on {pair} (prob={prob:.2f})")
                    return None, current_capital
            else:
                logger.info("ML model not found, skipping ML filter.")
        except Exception as e:
            logger.warning(f"ML filter error: {e}")

    # --- Backtest mode: read-only, separate log ---
    if backtest_mode:
        combined_df = run_backtest(
            smoothed, df["Close"], threshold_final,
            initial_capital=trading_cfg["default_capital"],
            risk_pct=risk_pct,
            stop_loss_atr_mult=atr_stop_mult,
            take_profit_atr_mult=atr_tp_mult,
            fee_pct=fee_pct,
            atr=df["atr"] if "atr" in df else None,
            partial_exit=partial_exit,
            trailing_atr_mult=atr_trail_mult
        )
        summary_df = (
            combined_df
            .loc[combined_df["type"] == "summary"]
            .drop(columns=["type"])
        )
        trades_df = (
            combined_df
            .loc[combined_df["type"] != "summary"]
            .drop(columns=["type"])
        )
        st.write(f"📊 Backtest Summary for {pair}:")
        st.dataframe(summary_df)
        st.write(f"📘 Trade Details for {pair}:")
        st.dataframe(trades_df)
        return None, current_capital

    # --- LIVE TRADING LOGIC with ATR/partial exit/trailing stop ---
    action = None
    price = df["Close"].iloc[-1]
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    atr_value = df["atr"].iloc[-1] if "atr" in df else None
    volatility = df["volatility"].iloc[-30:].values if "volatility" in df else None

    perf = compute_trade_metrics(trade_log, trading_cfg["default_capital"])
    equity_curve = [trading_cfg["default_capital"]]
    for trade in trade_log:
        if "return_pct" in trade:
            equity_curve.append(equity_curve[-1] * (1 + trade["return_pct"]))
    if risk_manager.check_max_drawdown(equity_curve, risk_cfg["max_drawdown_pct"]):
        logger.warning("⚠️ Max global drawdown exceeded! No more trades today.")
        st.error("Max global drawdown exceeded! Trading halted.")
        return None, current_capital
    if risk_manager.check_daily_loss(trade_log, perf["current_capital"], risk_cfg["max_daily_loss_pct"]):
        logger.warning("⚠️ Max daily loss exceeded! No more trades today.")
        st.error("Max daily loss exceeded! Trading halted.")
        return None, current_capital

    # --- Entry logic ---
    if latest_signal > threshold_final and pair not in open_positions:
        if len(open_positions) >= max_positions:
            logger.info("🚫 Max open positions reached → skipping BUY")
            return None, current_capital
        # Volatility-adjusted position sizing
        amount = risk_manager.position_size(
            perf["current_capital"], price, risk_pct, volatility=volatility, v_adj=True
        )
        min_size = risk_cfg.get("min_size", 0.001)
        if amount < min_size:
            logger.warning(f"Calculated amount {amount:.6f} below min size {min_size} → skipping BUY")
            return None, current_capital
        stop_loss = price - atr_stop_mult * atr_value if atr_value else price * (1 - stop_loss_pct)
        take_profit = price + atr_tp_mult * atr_value if atr_value else price * (1 + take_profit_pct)
        record = {
            "timestamp": now,
            "pair": pair,
            "action": "BUY",
            "amount": amount,
            "entry_price": price,
            "interval": interval_used,
            "lookback": lookback_used,
            "threshold": threshold_final,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "partial_exit": False,
            "trailing_stop": None
        }
        trade_log.append(record)
        open_positions[pair] = {
            "amount": amount,
            "entry_price": price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "partial_exit": False,
            "trailing_stop": None
        }
        logger.info(f"📥 BUY {pair} at {price:.2f} (amount={amount:.6f}) SL={stop_loss:.2f} TP={take_profit:.2f}")
        save_json(open_positions, POSITIONS_FILE, indent=2)
        save_json(trade_log, TRADE_LOG_FILE, indent=2)
        new_trades = 1
        trade_counter.inc(new_trades)
        return None, current_capital

    # --- Exit logic for open positions (partial exit & trailing stop) ---
    if pair in open_positions:
        position = open_positions[pair]
        entry_price = position["entry_price"]
        amount = position["amount"]
        stop_loss = position.get("stop_loss", entry_price * (1 - stop_loss_pct))
        take_profit = position.get("take_profit", entry_price * (1 + take_profit_pct))
        trailing_stop = position.get("trailing_stop")
        is_partial = position.get("partial_exit", False)
        exit_price = price
        exit_amount = 0
        # ATR for trailing logic
        atr_this = atr_value if atr_value else (entry_price * take_profit_pct)
        # Partial exit at TP1
        if partial_exit and not is_partial and price >= take_profit:
            exit_amount = amount / 2
            record = {
                "timestamp": now,
                "pair": pair,
                "action": "SELL_PARTIAL",
                "amount": exit_amount,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "return_pct": (exit_price - entry_price) / entry_price,
                "reason": "partial_take_profit"
            }
            trade_log.append(record)
            # Let remainder ride with trailing stop
            open_positions[pair]["amount"] = amount - exit_amount
            open_positions[pair]["partial_exit"] = True
            open_positions[pair]["trailing_stop"] = price - atr_trail_mult * atr_this
            logger.info(f"🟡 PARTIAL SELL {pair} at {exit_price:.2f} (amount={exit_amount:.6f}); trailing stop set at {open_positions[pair]['trailing_stop']:.2f}")
            save_json(open_positions, POSITIONS_FILE, indent=2)
            save_json(trade_log, TRADE_LOG_FILE, indent=2)
            trade_counter.inc(1)
            return None, current_capital
        # Trailing stop for remainder after partial
        if partial_exit and is_partial:
            # Dynamically update trailing stop
            new_trailing = price - atr_trail_mult * atr_this
            if trailing_stop is None or new_trailing > trailing_stop:
                open_positions[pair]["trailing_stop"] = new_trailing
                trailing_stop = new_trailing
            if price <= trailing_stop:
                exit_amount = amount
                record = {
                    "timestamp": now,
                    "pair": pair,
                    "action": "SELL_TRAILING",
                    "amount": exit_amount,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "return_pct": (exit_price - entry_price) / entry_price,
                    "reason": "trailing_stop"
                }
                trade_log.append(record)
                open_positions.pop(pair)
                logger.info(f"🔴 TRAILING SELL {pair} at {exit_price:.2f} (amount={exit_amount:.6f}); trailing stop hit")
                save_json(open_positions, POSITIONS_FILE, indent=2)
                save_json(trade_log, TRADE_LOG_FILE, indent=2)
                save_json(compute_trade_metrics(trade_log, trading_cfg["default_capital"])["current_capital"], CAPITAL_FILE, indent=2)
                trade_counter.inc(1)
                return None, compute_trade_metrics(trade_log, trading_cfg["default_capital"])["current_capital"]
        # Full exit: stop loss or signal reversal
        if (price <= stop_loss) or (not partial_exit and price >= take_profit) or (latest_signal < threshold_final):
            exit_amount = amount
            reason = "stop_loss" if price <= stop_loss else "take_profit" if price >= take_profit else "signal_flip"
            record = {
                "timestamp": now,
                "pair": pair,
                "action": "SELL",
                "amount": exit_amount,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "return_pct": (exit_price - entry_price) / entry_price,
                "reason": reason
            }
            trade_log.append(record)
            open_positions.pop(pair)
            logger.info(f"📤 SELL {pair} at {exit_price:.2f} (amount={exit_amount:.6f}) → Return: {(exit_price-entry_price)/entry_price:.4%} [{reason}]")
            save_json(open_positions, POSITIONS_FILE, indent=2)
            save_json(trade_log, TRADE_LOG_FILE, indent=2)
            save_json(compute_trade_metrics(trade_log, trading_cfg["default_capital"])["current_capital"], CAPITAL_FILE, indent=2)
            trade_counter.inc(1)
            return None, compute_trade_metrics(trade_log, trading_cfg["default_capital"])["current_capital"]

    return None, current_capital

def display_dashboard(current_capital):
    perf = compute_trade_metrics(trade_log, trading_cfg["default_capital"])
    pnl_gauge.set(perf["total_return"])
    st.subheader(f"📈 Live Dashboard — Variant {SELECTOR_VARIANT}")
    st.metric("Starting Capital", f"{trading_cfg['default_capital']:.2f} USDT")
    st.metric("Current Capital",  f"{perf['current_capital']:.4f} USDT")
    st.metric("Total Return",     f"{perf['total_return']:.2f}%")
    st.metric("Win Rate",         f"{perf['win_rate']:.2f}%")

    if open_positions:
        st.write("🟢 Open Positions")
        df_open = pd.DataFrame(open_positions).T.reset_index(drop=True)
        desired_cols = ["amount", "entry_price", "stop_loss", "take_profit", "trailing_stop", "partial_exit"]
        cols = [c for c in desired_cols if c in df_open.columns]
        st.dataframe(df_open[cols].iloc[::-1])
    else:
        st.info("No active trades.")

    if trade_log:
        st.write("📘 Trade History (Most Recent First)")
        df_trades = pd.DataFrame(trade_log)
        st.dataframe(df_trades.iloc[::-1])
    else:
        st.info("No trade history yet.")

def main_loop():
    global current_capital
    if backtest_mode:
        with st.spinner("Running backtest…"):
            for pair in TRADING_PAIRS:
                trade_logic(pair, trading_cfg["default_capital"])
        return

    last_timestamps = {pair: None for pair in TRADING_PAIRS}
    while True:
        for pair in TRADING_PAIRS:
            df = cached_fetch_klines(pair, interval if mode != "Auto" else get_pair_params(pair)["interval"],
                                     lookback if mode != "Auto" else get_pair_params(pair)["lookback"])
            if df.empty or not validate_df(df):
                continue
            newest = df.index[-1]
            if newest != last_timestamps[pair]:
                _, updated_capital = trade_logic(pair, current_capital)
                current_capital = updated_capital
                last_timestamps[pair] = newest
        display_dashboard(current_capital)
        write_heartbeat()
        time.sleep(1)

if start_btn:
    try:
        main_loop()
    except Exception:
        import traceback, smtplib
        tb = traceback.format_exc()
        logger.error(f"CRASHED: {tb}")
        USER = os.getenv("ALERT_USER")
        PASS = os.getenv("ALERT_PASS")
        ALERT_EMAIL = os.getenv("ALERT_EMAIL")
        if USER and PASS and ALERT_EMAIL:
            try:
                with smtplib.SMTP("smtp.gmail.com", 587) as s:
                    s.starttls()
                    s.login(USER, PASS)
                    msg = f"Subject: OnlyFunds CRASHED!\n\n{tb}"
                    s.sendmail(USER, ALERT_EMAIL, msg)
            except Exception as e:
                logger.error(f"Failed to send alert email: {e}")
        raise
