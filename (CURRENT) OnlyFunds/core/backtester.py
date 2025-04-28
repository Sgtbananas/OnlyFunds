import pandas as pd
import numpy as np
import logging
from core.grid_trader import GridTrader

def run_backtest(
    signal: pd.Series,
    prices: pd.Series,
    threshold: float = 0.05,
    initial_capital: float = 10.0,
    risk_pct: float = 0.01,
    stop_loss_atr_mult: float = 1.0,
    take_profit_atr_mult: float = 2.0,
    fee_pct: float = 0.0004,
    verbose: bool = False,
    log_every_n: int = 50,
    grid_mode: bool = False,
    grid_kwargs: dict = None,
    log_func=None,
    indicator_params: dict = None,
    high: pd.Series = None,
    low: pd.Series = None,
    atr: pd.Series = None,
    partial_exit: bool = True,
    trailing_atr_mult: float = 1.0,
) -> pd.DataFrame:
    """
    Backtest with support for ATR-based stop-loss, take-profit, partial exits, trailing stop and fee modeling.
    If grid_mode, runs a grid strategy simulation instead.
    indicator_params: dict that can be passed through to indicator logic (for multi-param search).
    Returns DataFrame: [summary row, trade rows], summary always includes: 
        type, trades, avg_return, win_rate, capital, total_pnl, max_drawdown, sharpe_ratio
    """
    if not grid_mode:
        trades = []
        position = None
        entry_price = None
        capital = initial_capital
        equity_curve = [capital]
        partial_out = False
        trailing_stop = None

        for i in range(len(signal)):
            sig = signal.iloc[i]
            price = prices.iloc[i]
            this_atr = atr.iloc[i] if atr is not None else None

            # --- LONG entry ---
            if sig > threshold and position is None:
                position_size = (capital * risk_pct) / price
                position = {
                    "size": position_size,
                    "entry_price": price,
                    "bar": i,
                }
                entry_price = price
                position_cost = position_size * price
                capital -= position_cost
                partial_out = False
                trailing_stop = None
                stop_loss = price - stop_loss_atr_mult * this_atr if this_atr is not None else price * (1 - 0.01)
                take_profit = price + take_profit_atr_mult * this_atr if this_atr is not None else price * (1 + 0.02)
                trades.append({
                    "type": "open",
                    "entry_bar": i,
                    "entry_price": price,
                    "size": position_size,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit
                })
            # --- LONG exit conditions ---
            elif position is not None:
                stop_loss = entry_price - stop_loss_atr_mult * this_atr if this_atr is not None else entry_price * (1 - 0.01)
                take_profit = entry_price + take_profit_atr_mult * this_atr if this_atr is not None else entry_price * (1 + 0.02)
                # Partial exit at first TP
                if partial_exit and not partial_out and price >= take_profit:
                    # Sell half, keep half for trailing
                    half_size = position["size"] / 2
                    profit = half_size * (price - entry_price)
                    capital += half_size * price
                    trades.append({
                        "type": "partial_exit",
                        "exit_bar": i,
                        "exit_price": price,
                        "profit": profit,
                        "return": (price - entry_price) / entry_price,
                        "amount": half_size,
                        "capital": capital
                    })
                    position["size"] = half_size
                    partial_out = True
                    trailing_stop = price - trailing_atr_mult * this_atr if this_atr is not None else price * (1 - 0.01)
                # Trailing exit for remainder
                elif partial_exit and partial_out and price <= trailing_stop:
                    # Sell remainder at trailing stop
                    profit = position["size"] * (price - entry_price)
                    capital += position["size"] * price
                    trades.append({
                        "type": "trailing_exit",
                        "exit_bar": i,
                        "exit_price": price,
                        "profit": profit,
                        "return": (price - entry_price) / entry_price,
                        "amount": position["size"],
                        "capital": capital
                    })
                    equity_curve.append(capital)
                    position = None
                    entry_price = None
                    partial_out = False
                    trailing_stop = None
                # Full exit (SL or signal flip)
                elif ((not partial_exit) and (price <= stop_loss or sig < threshold)) or (partial_exit and not partial_out and price <= stop_loss):
                    profit = position["size"] * (price - entry_price)
                    capital += position["size"] * price
                    trades.append({
                        "type": "full_exit",
                        "exit_bar": i,
                        "exit_price": price,
                        "profit": profit,
                        "return": (price - entry_price) / entry_price,
                        "amount": position["size"],
                        "capital": capital,
                        "reason": "stop_loss" if price <= stop_loss else "signal_flip"
                    })
                    equity_curve.append(capital)
                    position = None
                    entry_price = None
                    partial_out = False
                    trailing_stop = None
                # Update trailing stop if in partial
                elif partial_exit and partial_out:
                    new_trailing = price - trailing_atr_mult * this_atr if this_atr is not None else price * (1 - 0.01)
                    if new_trailing > trailing_stop:
                        trailing_stop = new_trailing

        trades_df = pd.DataFrame(trades)
        if not trades_df.empty:
            avg_return = trades_df.loc[trades_df["type"].str.contains("exit"), "return"].mean()
            win_rate = (trades_df.loc[trades_df["type"].str.contains("exit"), "return"] > 0).mean() * 100
            total_pnl = trades_df.loc[trades_df["type"].str.contains("exit"), "profit"].sum()
            curve = np.array(equity_curve)
            peak = np.maximum.accumulate(curve)
            drawdown = (peak - curve) / peak
            max_drawdown = np.max(drawdown) * 100 if len(drawdown) > 0 else 0
            returns = trades_df.loc[trades_df["type"].str.contains("exit"), "return"]
            sharpe_ratio = (returns.mean() / returns.std()) if returns.std() != 0 else 0
        else:
            avg_return = 0
            win_rate = 0
            total_pnl = 0
            max_drawdown = 0
            sharpe_ratio = 0

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

    else:
        if grid_kwargs is None:
            raise ValueError("grid_kwargs must be provided for grid_mode backtest.")
        grid = GridTrader(**grid_kwargs)
        grid.build_orders()
        grid_orders = []
        price_series = prices.values
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
