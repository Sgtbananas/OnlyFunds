import os
import logging
import time
from datetime import datetime

import streamlit as st
import pandas as pd
from dotenv import load_dotenv

from core.core_data       import fetch_klines, validate_df, add_indicators, TRADING_PAIRS
from core.core_signals    import (
    generate_signal,
    smooth_signal,
    adaptive_threshold,
    track_trade_result,
)
from core.trade_execution import place_order
from core.backtester      import run_backtest
from utils.helpers import compute_trade_metrics, suggest_tuning

# ─── Streamlit UI ────────────────────────────────────────────────────────────
st.set_page_config(page_title="CryptoTrader AI", layout="wide")
st.title("🧠 CryptoTrader AI Bot")
st.sidebar.header("⚙️ Configuration")

backtest_mode = st.sidebar.checkbox("Enable Backtesting", value=False)  # Default False for safety

# ─── Core Trade Logic ────────────────────────────────────────────────────────
def trade_logic(pair: str):
    logger.info(f"🔍 Analyzing {pair}")
    df = fetch_klines(pair=pair, interval=interval, limit=lookback)
    if df.empty or not validate_df(df):
        logger.warning(f"⚠️ Invalid/empty data for {pair}")
        return

    # add indicators
    df = add_indicators(df)

    # pick a threshold
    threshold = 0.5
    if autotune:
        threshold = adaptive_threshold(df)
        logger.debug(f"Adaptive threshold for {pair}: {threshold}")

    # compute + smooth
    raw     = generate_signal(df)
    smoothed = smooth_signal(raw)

    # Backtest Mode Execution
    if backtest_mode:
        summary, trade_log = run_backtest(smoothed, df["Close"], threshold)
        st.subheader(f"📊 Backtest: {pair}")
        
        # Display results
        if summary.empty:
            st.warning(f"No trades executed during backtesting for {pair}.")
        else:
            st.write("Summary:")
            st.dataframe(summary)
            st.write("Trade Log:")
            st.dataframe(trade_log)

        return