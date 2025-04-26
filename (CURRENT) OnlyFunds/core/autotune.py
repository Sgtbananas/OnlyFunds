import pandas as pd
import numpy as np
from typing import List, Dict, Any

def calculate_performance(
    trade_log: List[Dict[str, Any]],
    initial_capital: float
) -> Dict[str, float]:
    """
    Returns:
      - total_pnl      : final equity curve minus initial
      - win_rate       : fraction of trades with positive return
      - average_return : mean of individual returns
      - max_drawdown   : worst peak-to-trough drawdown
      - sharpe_ratio   : (mean/std)*sqrt(N)
    Expects trade_log items with 'entry_price','exit_price','amount','action','timestamp'.
    """
    if not trade_log:
        return dict.fromkeys(
            ["total_pnl", "win_rate", "average_return", "max_drawdown", "sharpe_ratio"],
            0.0
        )

    df = pd.DataFrame(trade_log)
    # compute per-trade return
    df["return_pct"] = np.where(
        df["action"].str.upper() == "BUY",
        (df["exit_price"] - df["entry_price"]) / df["entry_price"],
        (df["entry_price"] - df["exit_price"]) / df["entry_price"],
    )

    # equity curve
    equity = initial_capital * (1 + df["return_pct"].cumsum())
    peak = equity.cummax()
    drawdown = (equity - peak) / peak

    total_pnl      = equity.iloc[-1] - initial_capital
    win_rate       = (df["return_pct"] > 0).mean()
    average_return = df["return_pct"].mean()
    max_drawdown   = drawdown.min()

    std = df["return_pct"].std(ddof=0)
    sharpe = (average_return / std) * np.sqrt(len(df)) if std and len(df) > 1 else 0.0

    return {
        "total_pnl": total_pnl,
        "win_rate": win_rate,
        "average_return": average_return,
        "max_drawdown": max_drawdown,
        "sharpe_ratio": sharpe
    }

def suggest_tuning_parameters(metrics: Dict[str, float], thresholds: list = None) -> List[str]:
    """
    Based on metrics, suggests parameter tweaks:
     - win_rate < 0.5   -> raise threshold
     - max_drawdown < -0.05 -> tighten stops or lower size
     - sharpe_ratio >1.5 -> lower threshold for more trades
     - total_pnl < 0   -> switch to Conservative
     - thresholds volatile -> recommend smoothing/bounding
    """
    suggestions = []
    if metrics["win_rate"] < 0.5:
        suggestions.append("Win rate below 50% → consider raising entry threshold.")
    if metrics["max_drawdown"] < -0.05:
        suggestions.append("Drawdown >5% → tighten stop-loss or reduce size.")
    if metrics["sharpe_ratio"] > 1.5:
        suggestions.append("High Sharpe → consider lowering threshold to capture more trades.")
    if metrics["total_pnl"] < 0:
        suggestions.append("Overall loss → switch to Conservative mode or review signals.")

    # Threshold volatility check (for autotune)
    if thresholds is not None and len(thresholds) > 3:
        std = np.std(thresholds)
        if std > 0.1:
            suggestions.append("Autotune threshold is volatile. Consider smoothing or bounding the threshold.")
        elif std < 0.01:
            suggestions.append("Autotune threshold is very stable. Manual or hybrid may be sufficient.")

    if not suggestions:
        suggestions.append("Performance stable → maintain current settings.")
    return suggestions

def adaptive_threshold(signals: pd.Series, prices: pd.Series) -> float:
    """
    Dynamically determines the optimal threshold for trading signals.
    Uses backtesting results to iterate and find the threshold that maximizes performance.
    """
    from core.backtester import run_backtest

    best_threshold = 0.5
    best_sharpe = -float("inf")
    tested_thresholds = []
    sharpes = []

    # Test multiple thresholds
    for threshold in np.arange(0.1, 1.1, 0.1):
        summary, _ = run_backtest(signals, prices, threshold)  # Extract summary only
        sharpe = summary.iloc[0]["sharpe"]
        tested_thresholds.append(threshold)
        sharpes.append(sharpe)
        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_threshold = threshold

    print(f"Optimal threshold: {best_threshold} with Sharpe Ratio: {best_sharpe}")
    print(f"Thresholds tested: {tested_thresholds}")
    print(f"Sharpes: {sharpes}")
    return best_threshold