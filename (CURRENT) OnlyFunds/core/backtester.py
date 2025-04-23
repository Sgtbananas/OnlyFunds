import pandas as pd
import numpy as np

def run_backtest(signals: pd.Series, prices: pd.Series, threshold: float = 0.5) -> tuple:
    """
    Backtests trading strategy based on signal crossing threshold.
    Entry on signal > threshold (long), < -threshold (short).
    Exit when signal crosses back to neutral zone.
    """
    position = 0
    entry_price = 0
    returns = []
    trade_log = []

    for i in range(1, len(signals)):
        signal = signals.iloc[i]
        price = prices.iloc[i]

        print(f"Step {i}: Signal={signal}, Price={price}, Position={position}")

        if position == 0:
            if signal > threshold:
                position = 1
                entry_price = price
                print(f"Opened LONG at {entry_price}")
            elif signal < -threshold:
                position = -1
                entry_price = price
                print(f"Opened SHORT at {entry_price}")

        elif position == 1:
            if signal < 0:
                ret = (price - entry_price) / entry_price
                returns.append(ret)
                trade_log.append({
                    "entry": entry_price,
                    "exit": price,
                    "side": "LONG",
                    "return": round(ret, 5)
                })
                print(f"Closed LONG at {price}, Return: {ret}")
                position = 0

        elif position == -1:
            if signal > 0:
                ret = (entry_price - price) / entry_price
                returns.append(ret)
                trade_log.append({
                    "entry": entry_price,
                    "exit": price,
                    "side": "SHORT",
                    "return": round(ret, 5)
                })
                print(f"Closed SHORT at {price}, Return: {ret}")
                position = 0

    # Update warning to check both returns and trade_log
    if not returns and not trade_log:
        print("⚠️ No trades executed during backtesting. Check your signals or thresholds.")
        return pd.DataFrame([{
            "trades": 0,
            "win_rate": 0.0,
            "avg_return": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0
        }]), pd.DataFrame()

    returns = np.array(returns)
    cum_returns = np.cumsum(returns)
    rolling_max = np.maximum.accumulate(cum_returns)
    drawdown = rolling_max - cum_returns
    max_dd = drawdown.max()

    win_rate = (returns > 0).mean()
    avg_return = returns.mean()
    std_return = returns.std() if returns.std() > 0 else 1e-8
    sharpe_ratio = avg_return / std_return

    # Summary metrics
    summary = pd.DataFrame([{
        "trades": len(returns),
        "win_rate": round(win_rate * 100, 2),
        "avg_return": round(avg_return * 100, 2),
        "sharpe": round(sharpe_ratio, 2),
        "max_drawdown": round(max_dd * 100, 2)
    }])

    # Include the detailed trade log in the return value
    return summary, pd.DataFrame(trade_log)
