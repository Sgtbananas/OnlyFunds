# core/dashboard.py

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

def plot_price_and_signal(df: pd.DataFrame, pair: str):
    """
    Plots price and smoothed signal on a dual-axis chart using Plotly.
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df.index,
        y=df["Close"],
        name="Price",
        line=dict(color="deepskyblue"),
        yaxis="y1"
    ))

    if "signal" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df["signal"],
            name="Signal",
            line=dict(color="orange", dash="dot"),
            yaxis="y2"
        ))

    fig.update_layout(
        title=f"Price and Signal Chart: {pair}",
        xaxis=dict(title="Time"),
        yaxis=dict(title="Price", side="left"),
        yaxis2=dict(title="Signal", overlaying="y", side="right", range=[-1, 1]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=500,
    )

    st.plotly_chart(fig, use_container_width=True)

def display_position_summary(open_positions: dict, trade_log: list):
    """
    Displays current open positions and recent trade history.
    """
    st.subheader("🔓 Open Positions")
    if open_positions:
        st.dataframe(pd.DataFrame(open_positions).T)
    else:
        st.info("No active positions.")

    st.subheader("📘 Trade History")
    if trade_log:
        st.dataframe(pd.DataFrame(trade_log))
    else:
        st.info("No trades recorded yet.")

def show_dashboard_for_pair(df: pd.DataFrame, pair: str):
    """
    Wrapper to display both the plot and position info.
    """
    st.markdown(f"## 📊 Dashboard for {pair}")
    plot_price_and_signal(df, pair)
