import pandas as pd
import numpy as np
import logging
from core.grid_trader import GridTrader
from core.meta_learner import select_strategy
import os

def load_backtest_data(pair, interval='5m', limit=1000):
    """
    Load backtest data from CSV or fetch it if not available.
    :param pair: Trading pair (e.g., 'BTCUSDT')
    :param interval: Time interval (e.g., '5m')
    :param limit: Number of data points to fetch
    :return: DataFrame containing historical market data
    """
    # Path to the historical data file
    file_path = os.path.join('data/historical', f'{pair}_{interval}_{limit}.csv')

    if os.path.exists(file_path):
        logging.info(f"Loading data for {pair} from {file_path}")
        return pd.read_csv(file_path)
    else:
        logging.warning(f"Data file not found for {pair} at {file_path}")
        return pd.DataFrame()  # Return empty dataframe if file is not found

def run_backtest(signal, pair="BTCUSDT", interval="5m", limit=1000, equity=1000, performance_dict=None, meta_model=None, 
                 risk_pct=0.01, atr_multiplier=2, grid_mode=False, grid_kwargs=None, fee_pct=0.002, verbose=False, log_every_n=100):
    """
    Run the backtest for the selected strategy.
    :param signal: The signal data for the backtest
    :param pair: Trading pair (e.g., 'BTCUSDT')
    :param interval: Time interval for the historical data (default is '5m')
    :param limit: Number of data points to fetch (default is 1000)
    :param equity: Initial equity for backtesting
    :param performance_dict: Dictionary for performance evaluation
    :param meta_model: Model for strategy selection
    :param risk_pct: Percentage of equity to risk per trade
    :param atr_multiplier: ATR multiplier for setting stop loss and take profit
    :param grid_mode: Whether to use grid trading mode
    :param grid_kwargs: Additional parameters for grid trading
    :param fee_pct: Trading fee percentage
    :param verbose: Verbosity flag for logging
    :param log_every_n: Frequency of logging
    :return: DataFrame containing the backtest results
    """
    # Fetch backtest data
    data = load_backtest_data(pair, interval, limit)

    if data.empty:
        logging.error("No valid data available for backtesting.")
        return None  # No valid data, cannot proceed with backtest

    # Extract the necessary price data for backtesting
    prices = data[['Close', 'ATR']]  # Assuming 'ATR' column exists
    signals = signal  # Assuming signal is already pre-processed

    # Select strategy using meta-learner or fallback to default strategy
    strategy, _ = select_strategy(performance_dict, meta_model)

    # Apply risk management for position sizing
    position_size = risk_manager.position_size(equity, prices['Close'].iloc[-1], risk_pct=risk_pct, volatility=prices['ATR'].iloc[-1], v_adj=True)

    # Apply ATR-based stop-loss and take-profit levels
    entry_price = prices['Close'].iloc[-1]
    stop_loss_price = entry_price - (atr_multiplier * prices['ATR'].iloc[-1])  # Example ATR-based stop loss
    take_profit_price = entry_price + (atr_multiplier * prices['ATR'].iloc[-1])  # Example ATR-based take profit

    # Simulate the trade based on the selected strategy
    if strategy == "trend_following":
        # Apply trend-following logic (e.g., EMA crossover)
        pass
    elif strategy == "mean_reversion":
        # Apply mean-reversion logic
        pass

    # Loop through each price and simulate the backtest
    trades = []
    position = None
    for i, price in enumerate(prices['Close']):
        if price <= stop_loss_price:
            log_trade(entry_price, 'stop_loss', price, profit_loss=price - entry_price)
            break
        elif price >= take_profit_price:
            log_trade(entry_price, 'take_profit', price, profit_loss=price - entry_price)
            break
    else:
        log_trade(entry_price, 'continue', prices['Close'].iloc[-1], profit_loss=prices['Close'].iloc[-1] - entry_price)
    
    # Handle grid mode (if grid_mode is True)
    if grid_mode:
        if grid_kwargs is None:
            raise ValueError("grid_kwargs must be provided for grid_mode backtest.")
        grid = GridTrader(**grid_kwargs)
        grid.build_orders()
        grid_orders = []
        price_series = prices['Close'].values
        idx_series = prices.index
        for i, price in enumerate(price_series):
            ts = idx_series[i] if hasattr(idx_series, "__getitem__") else None
            grid.check_and_fill_orders(price, timestamp=ts, log_func=log_func)
        grid_orders = [o.to_dict() for o in grid.orders]
        fills = [o for o in grid_orders if o["filled"]]
        pnl = 0
        buy_fills = [o for o in fills if o["side"] == "buy"]
        sell_fills = [o for o in fills if o["side"] == "sell"]
        min_fills = min(len(buy_fills), len(sell_fills))
        if min_fills > 0:
            pnl = sum(
                (sell_fills[i]["fill_price"] - buy_fills[i]["fill_price"]) * buy_fills[i]["size"]
                for i in range(min_fills)
            )
        summary = {
            "type": "summary",
            "trades": len(fills),
            "buy_fills": len(buy_fills),
            "sell_fills": len(sell_fills),
            "total_pnl": pnl,
        }
        summary_df = pd.DataFrame([summary])
        fills_df = pd.DataFrame(fills)
        combined_df = pd.concat([summary_df, fills_df], ignore_index=True)
        return combined_df

    # Standard backtest without grid mode
    else:
        trades = []
        position = None
        entry_price = None
        capital = equity
        equity_curve = [capital]
        partial_out = False
        trailing_stop = None

        for i in range(len(signals)):
            sig = signals.iloc[i]
            price = prices['Close'].iloc[i]
            this_atr = prices['ATR'].iloc[i] if prices['ATR'] is not None else None

            adjusted_entry_price = price * (1 + fee_pct) if position is None else entry_price
            adjusted_exit_price = price * (1 - fee_pct) if position is not None else price

            if verbose and (i % log_every_n == 0):
                logging.debug(f"Step {i}: Signal={sig:.4f}, Price={price:.2f}, Position={position}, Capital={capital:.2f}")

            # --- LONG entry ---
            if sig > 0 and position is None:
                position_size = (capital * risk_pct) / adjusted_entry_price
                position = {
                    "size": position_size,
                    "entry_price": adjusted_entry_price,
                    "bar": i,
                }
                entry_price = adjusted_entry_price
                position_cost = position_size * adjusted_entry_price
                capital -= position_cost
                partial_out = False
                trailing_stop = None
                # Dynamic/ATR stops
                stop_loss = None
                take_profit = None
                if stop_loss_atr_mult is not None and this_atr is not None:
                    stop_loss = adjusted_entry_price - stop_loss_atr_mult * this_atr
                else:
                    stop_loss = adjusted_entry_price * (1 - risk_pct)
                if take_profit_atr_mult is not None and this_atr is not None:
                    take_profit = adjusted_entry_price + take_profit_atr_mult * this_atr
                else:
                    take_profit = adjusted_entry_price * (1 + risk_pct)
                trades.append({
                    "type": "open",
                    "entry_bar": i,
                    "entry_price": adjusted_entry_price,
                    "size": position_size,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit
                })

            # --- LONG exit conditions ---
            elif position is not None:
                stop_loss = entry_price * (1 - risk_pct) if this_atr is None else entry_price - stop_loss_atr_mult * this_atr
                take_profit = entry_price * (1 + risk_pct) if this_atr is None else entry_price + take_profit_atr_mult * this_atr

                if price >= take_profit:
                    profit = position["size"] * (price - entry_price)
                    trades.append({
                        "type": "partial_exit",
                        "exit_bar": i,
                        "exit_price": price,
                        "profit": profit,
                        "return": (price - entry_price) / entry_price,
                        "amount": position["size"] / 2,
                        "capital": capital + (position["size"] / 2) * price
                    })
                    capital += (position["size"] / 2) * price
                    position["size"] /= 2  # Keep half of the position open

                if price <= stop_loss or price >= take_profit or sig < 0:
                    profit = position["size"] * (price - entry_price)
                    trades.append({
                        "type": "full_exit",
                        "exit_bar": i,
                        "exit_price": price,
                        "profit": profit,
                        "return": (price - entry_price) / entry_price,
                        "amount": position["size"],
                        "capital": capital + position["size"] * price,
                        "reason": "stop_loss" if price <= stop_loss else "take_profit" if price >= take_profit else "signal_flip"
                    })
                    capital += position["size"] * price
                    equity_curve.append(capital)
                    position = None
                    entry_price = None
                    trailing_stop = None

        trades_df = pd.DataFrame(trades)
        # Handle missing 'return' column robustly
        if not trades_df.empty and "return" in trades_df.columns:
            exit_mask = trades_df["type"].str.contains("exit")
            avg_return = trades_df.loc[exit_mask, "return"].mean()
            win_rate = (trades_df.loc[exit_mask, "return"] > 0).mean() * 100
            total_pnl = trades_df.loc[exit_mask, "profit"].sum()
            returns = trades_df.loc[exit_mask, "return"]
            sharpe_ratio = (returns.mean() / returns.std()) if returns.std() != 0 else 0
        else:
            avg_return = 0
            win_rate = 0
            total_pnl = 0
            sharpe_ratio = 0
        curve = np.array(equity_curve)
        peak = np.maximum.accumulate(curve)
        drawdown = (peak - curve) / peak
        max_drawdown = np.max(drawdown) * 100 if len(drawdown) > 0 else 0

        summary = {
            "type": "summary",
            "trades": len(trades_df),
            "avg_return": avg_return,
            "win_rate": win_rate,
            "capital": capital,
            "total_pnl": total_pnl,
            "max_drawdown": max_drawdown,
            "sharpe_ratio": sharpe_ratio
        }
        summary_df = pd.DataFrame([summary])
        combined_df = pd.concat([summary_df, trades_df], ignore_index=True)

        if trades_df.empty:
            logging.warning("No trades executed during backtest.")
        else:
            logging.info(f"Backtest complete: {summary['trades']} trades, Avg Return: {avg_return:.2%}")

        return combined_df
