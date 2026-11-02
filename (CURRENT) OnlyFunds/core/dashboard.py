"""Plotly chart builders for the Streamlit UI (no Streamlit calls here)."""
from __future__ import annotations

from typing import Optional

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

UP, DOWN = "#26a69a", "#ef5350"


def _x(df: pd.DataFrame):
    return df["timestamp"] if "timestamp" in df.columns else df.index


def price_chart(sig: pd.DataFrame, pair: str, trades: Optional[pd.DataFrame] = None, height: int = 720) -> go.Figure:
    """Candles + EMAs + Bollinger, with RSI and MACD panels. ``sig`` = generate_signals output."""
    x = _x(sig)
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.6, 0.2, 0.2], vertical_spacing=0.03,
                        subplot_titles=(pair, "RSI", "MACD"))
    fig.add_trace(go.Candlestick(x=x, open=sig["Open"], high=sig["High"], low=sig["Low"], close=sig["Close"],
                                 name="Price", increasing_line_color=UP, decreasing_line_color=DOWN), 1, 1)
    for col, color in (("ema_fast", "#f5a623"), ("ema_slow", "#4a90e2"), ("ema_trend", "#9b59b6")):
        if col in sig:
            fig.add_trace(go.Scatter(x=x, y=sig[col], name=col, line=dict(width=1, color=color)), 1, 1)
    if "bb_upper" in sig:
        fig.add_trace(go.Scatter(x=x, y=sig["bb_upper"], name="BB upper", line=dict(width=0.8, color="gray", dash="dot")), 1, 1)
        fig.add_trace(go.Scatter(x=x, y=sig["bb_lower"], name="BB lower", line=dict(width=0.8, color="gray", dash="dot"),
                                 fill="tonexty", fillcolor="rgba(128,128,128,0.08)"), 1, 1)
    if "entry" in sig:
        e = sig[sig["entry"]]
        fig.add_trace(go.Scatter(x=_x(e), y=e["Low"] * 0.998, mode="markers", name="entry signal",
                                 marker=dict(symbol="triangle-up", size=8, color=UP)), 1, 1)
    if trades is not None and len(trades):
        fig.add_trace(go.Scatter(x=pd.to_datetime(trades["entry_time"]), y=trades["entry_price"], mode="markers",
                                 name="buy", marker=dict(symbol="circle", size=9, color="#2ecc71",
                                                         line=dict(width=1, color="black"))), 1, 1)
        fig.add_trace(go.Scatter(x=pd.to_datetime(trades["exit_time"]), y=trades["exit_price"], mode="markers",
                                 name="sell", text=trades["exit_reason"],
                                 marker=dict(symbol="x", size=9, color="#e74c3c")), 1, 1)
    fig.add_trace(go.Scatter(x=x, y=sig["rsi"], name="RSI", line=dict(width=1, color="#f39c12")), 2, 1)
    for lvl in (30, 70):
        fig.add_hline(y=lvl, line=dict(width=0.6, dash="dash", color="gray"), row=2, col=1)
    fig.add_trace(go.Bar(x=x, y=sig["macd_hist"], name="MACD hist",
                         marker_color=[UP if v >= 0 else DOWN for v in sig["macd_hist"].fillna(0)]), 3, 1)
    fig.add_trace(go.Scatter(x=x, y=sig["macd"], name="MACD", line=dict(width=1, color="#3498db")), 3, 1)
    fig.add_trace(go.Scatter(x=x, y=sig["macd_signal"], name="Signal", line=dict(width=1, color="#e67e22")), 3, 1)
    fig.update_layout(height=height, xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=30, b=10),
                      legend=dict(orientation="h", y=1.04, x=0))
    return fig


def equity_chart(equity: pd.DataFrame, title: str = "Equity", x_col: str = "time") -> go.Figure:
    fig = go.Figure()
    if len(equity):
        x = pd.to_datetime(equity[x_col], errors="coerce") if x_col in equity else equity.index
        fig.add_trace(go.Scatter(x=x, y=equity["equity"], name="equity", line=dict(color="#2980b9")))
        peak = equity["equity"].cummax()
        fig.add_trace(go.Scatter(x=x, y=peak, name="peak", line=dict(color="gray", width=0.8, dash="dot")))
    fig.update_layout(title=title, height=320, margin=dict(l=10, r=10, t=40, b=10))
    return fig


def signal_chart(sig: pd.DataFrame, threshold: float) -> go.Figure:
    x = _x(sig)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=sig["signal"], name="signal", line=dict(color="#8e44ad")))
    fig.add_trace(go.Scatter(x=x, y=sig["confidence"], name="confidence", line=dict(color="#16a085", width=1)))
    fig.add_hline(y=threshold, line=dict(dash="dash", color=UP))
    fig.update_layout(height=250, margin=dict(l=10, r=10, t=20, b=10), yaxis=dict(range=[-1, 1]))
    return fig
