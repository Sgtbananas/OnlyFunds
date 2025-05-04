import pandas as pd
import numpy as np
import logging
from core.grid_trader import GridTrader
from core.meta_learner import select_strategy
from core import risk_manager
import os
import yaml

# --- Load defaults from YAML automatically ---
with open('config/config.yaml', 'r') as f:
    CONFIG = yaml.safe_load(f)

DEFAULT_INTERVAL = CONFIG.get('DEFAULT_INTERVAL', '5m')
DEFAULT_LOOKBACK = CONFIG.get('DEFAULT_LOOKBACK', 1000)
DEFAULT_EQUITY = CONFIG.get('DEFAULT_EQUITY', 1000)
DEFAULT_ATR_MULTIPLIER = CONFIG.get('DEFAULT_ATR_MULTIPLIER', 2)
DEFAULT_RISK_PCT = CONFIG.get('DEFAULT_RISK_PCT', 0.01)
DEFAULT_FEE = CONFIG.get('DEFAULT_FEE', 0.002)

def load_backtest_data(pair, interval=DEFAULT_INTERVAL, limit=DEFAULT_LOOKBACK):
    """
    Load backtest data from CSV or fetch it if not available.
    """
    file_path = os.path.join('data/historical', f'{pair}_{interval}_{limit}.csv')

    if os.path.exists(file_path):
        logging.info(f"Loading data for {pair} from {file_path}")
        return pd.read_csv(file_path)
    else:
        logging.warning(f"Data file not found for {pair} at {file_path}. Attempting to fetch live data.")
        from core.core_data import fetch_klines, add_indicators
        df = fetch_klines(pair, interval, limit)
        if df.empty:
            logging.error(f"No data fetched for {pair} at {interval}")
            return pd.DataFrame()
        df = add_indicators(df)
        os.makedirs('data/historical', exist_ok=True)
        df.to_csv(file_path, index=False)
        logging.info(f"Fetched and saved new data for {pair} at {interval}")
        return df

def run_backtest(signal,
                 pair="BTCUSDT",
                 interval=DEFAULT_INTERVAL,
                 limit=DEFAULT_LOOKBACK,
                 equity=DEFAULT_EQUITY,
                 performance_dict=None,
                 meta_model=None,
                 risk_pct=DEFAULT_RISK_PCT,
                 atr_multiplier=DEFAULT_ATR_MULTIPLIER,
                 grid_mode=False,
                 grid_kwargs=None,
                 fee_pct=DEFAULT_FEE,
                 verbose=False,
                 log_every_n=100):
    """
    Run the backtest for the selected strategy.
    """
    data = load_backtest_data(pair, interval, limit)

    if data.empty:
        logging.error("No valid data available for backtesting.")
        return None

    prices = data[['Close', 'ATR']]
    signals = signal

    strategy, _ = select_strategy(performance_dict, meta_model)

    capital = equity
    position = None
    entry_price = None
    stop_loss = None
    take_profit = None
    trailing_stop = None
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
            trailing_stop = None
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

            # Stop Loss
            if price <= stop_loss:
                exit = True
                exit_reason = "stop_loss"

            # Take Profit
            elif price >= take_profit:
                exit = True
                exit_reason = "take_profit"

            # Signal flip
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
                trailing_stop = None

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
        trade_returns = trades_df[trades_df['type'].str.contains('exit')]['profit'] / equity
        win_rate = (trade_returns > 0).mean() * 100
        avg_return = trade_returns.mean()
        total_pnl = trades_df[trades_df['type'].str.contains('exit')]['profit'].sum()
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
