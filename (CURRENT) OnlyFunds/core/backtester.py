import pandas as pd
import numpy as np
import logging
from core.grid_trader import GridTrader
from core.meta_learner import select_strategy
from core import risk_manager
from core.core_data import add_indicators
import os

def load_backtest_data(pair, interval='5m', limit=1000):
    """
    Load backtest data from CSV or fetch it if not available.
    """
    from core.core_data import fetch_klines
    file_path = os.path.join('data/historical', f'{pair}_{interval}_{limit}.csv')

    if os.path.exists(file_path):
        logging.info(f"Loading data for {pair} from {file_path}")
        df = pd.read_csv(file_path)
    else:
        logging.warning(f"Data file not found for {pair} at {file_path}. Attempting to fetch live data.")
        df = fetch_klines(pair, interval, limit)
        if df.empty:
            logging.error(f"No data fetched for {pair} at {interval}")
            return pd.DataFrame()

    # Always ensure indicators + ATR applied
    df = add_indicators(df)

    # Patch: check if ATR column is missing or empty
    if "ATR" not in df.columns or df["ATR"].isnull().all():
        logging.warning("ATR missing or invalid after add_indicators. Forcing recalculation.")
        df = add_indicators(df)

    if "ATR" not in df.columns or df["ATR"].isnull().all():
        logging.error("ATR still missing after recalculation. Backtest will abort for this pair.")
        return pd.DataFrame()

    return df

def run_backtest(signal, pair="BTCUSDT", interval="5m", limit=1000, equity=1000,
                 performance_dict=None, meta_model=None,
                 risk_pct=0.01, atr_multiplier=2,
                 grid_mode=False, grid_kwargs=None,
                 fee_pct=0.002, verbose=False, log_every_n=100):
    """
    Run the backtest for the selected strategy.
    """
    data = load_backtest_data(pair, interval, limit)

    if data.empty or "ATR" not in data.columns:
        logging.error("No valid data available for backtesting or ATR missing.")
        return None

    prices = data[['Close', 'ATR']]
    signals = signal

    # --- STRATEGY SELECTION ---
    try:
        strategy, params = select_strategy(performance_dict, meta_model)
        if not params or not isinstance(params, dict):
            params = {}
    except Exception as e:
        logging.error(f"Strategy selection failed: {e}. Using fallback strategy.")
        strategy = "default"
        params = {}

    capital = equity
    position = None
    entry_price = None
    stop_loss = None
    take_profit = None
    trades = []
    equity_curve = [capital]

    for i in range(len(signals)):
        sig = signals.iloc[i]
        price = prices['Close'].iloc[i]
        atr = prices['ATR'].iloc[i] if not pd.isna(prices['ATR'].iloc[i]) else 0

        if verbose and i % log_every_n == 0:
            logging.debug(f"Step {i}: Signal={sig}, Price={price}, Capital={capital}")

        # --- Entry ---
        if position is None and sig > 0:
            position_size = risk_manager.position_size(
                capital, price,
                risk_pct=risk_pct,
                volatility=atr,
                v_adj=True
            )
            entry_price = price * (1 + fee_pct)
            stop_loss = entry_price - (atr_multiplier * atr)
            take_profit = entry_price + (atr_multiplier * atr)
            position = {
                "size": position_size,
                "entry_price": entry_price
            }
            capital -= position_size * entry_price
            trades.append({
                "type": "entry",
                "price": entry_price,
                "bar": i
            })

        # --- Exit conditions ---
        if position is not None:
            exit = False
            exit_reason = ""

            if price <= stop_loss:
                exit = True
                exit_reason = "stop_loss"
            elif price >= take_profit:
                exit = True
                exit_reason = "take_profit"
            elif sig < 0:
                exit = True
                exit_reason = "signal_flip"

            if exit:
                exit_price = price * (1 - fee_pct)
                pnl = position["size"] * (exit_price - entry_price)
                capital += position["size"] * exit_price
                trades.append({
                    "type": "exit",
                    "price": exit_price,
                    "reason": exit_reason,
                    "profit": pnl,
                    "bar": i
                })
                position = None
                entry_price = None
                stop_loss = None
                take_profit = None

        equity_curve.append(capital)

    # Close any open position at end
    if position is not None:
        exit_price = prices['Close'].iloc[-1] * (1 - fee_pct)
        pnl = position["size"] * (exit_price - entry_price)
        capital += position["size"] * exit_price
        trades.append({
            "type": "forced_exit",
            "price": exit_price,
            "profit": pnl,
            "bar": len(signals)-1
        })
        position = None

    # --- Metrics ---
    trades_df = pd.DataFrame(trades)

    if not trades_df.empty:
        trade_returns = trades_df[trades_df['type'] == 'exit']['profit'] / equity
        win_rate = (trade_returns > 0).mean() * 100
        avg_return = trade_returns.mean()
        total_pnl = trades_df[trades_df['type'].isin(['exit', 'forced_exit'])]['profit'].sum()
        sharpe = (trade_returns.mean() / trade_returns.std()) if trade_returns.std() != 0 else 0
    else:
        win_rate = 0
        avg_return = 0
        total_pnl = 0
        sharpe = 0

    curve = np.array(equity_curve)
    peak = np.maximum.accumulate(curve)
    drawdown = (peak - curve) / peak
    max_drawdown = np.max(drawdown) * 100 if len(drawdown) > 0 else 0

    summary = {
        "type": "summary",
        "final_capital": capital,
        "total_trades": len(trades_df),
        "win_rate": win_rate,
        "average_return": avg_return,
        "total_pnl": total_pnl,
        "max_drawdown": max_drawdown,
        "sharpe_ratio": sharpe
    }

    summary_df = pd.DataFrame([summary])
    combined_df = pd.concat([summary_df, trades_df], ignore_index=True)

    logging.info(f"Backtest complete. Trades: {len(trades_df)}, Win rate: {win_rate:.2f}%, Max DD: {max_drawdown:.2f}%")

    return combined_df
