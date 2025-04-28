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
    stop_loss_pct: float = 0.005,
    take_profit_pct: float = 0.01,
    fee_pct: float = 0.0004,
    verbose: bool = False,
    log_every_n: int = 50,
    grid_mode: bool = False,
    grid_kwargs: dict = None,
    log_func=None,
    indicator_params: dict = None,
    # The following args are for ATR/partial/trailing logic (added in advanced risk features)
    stop_loss_atr_mult: float = None,
    take_profit_atr_mult: float = None,
    atr: pd.Series = None,
    partial_exit: bool = False,
    trailing_atr_mult: float = None,
) -> pd.DataFrame:
    """
    Backtest with support for stop-loss, take-profit, ATR/trailing/partial-exit, and fee modeling.
    If grid_mode, runs a grid strategy simulation instead.
    indicator_params: dict that can be passed through to indicator logic (for multi-param search).
    Returns DataFrame: [summary row, trade rows], summary always includes: 
        type, trades, avg_return, win_rate, capital, total_pnl, max_drawdown, sharpe_ratio
    """
    # NOTE: indicator_params is for future use/compat with walk-forward hyperopt
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

            adjusted_entry_price = price * (1 + fee_pct) if position is None else entry_price
            adjusted_exit_price = price * (1 - fee_pct) if position is not None else price

            if verbose and (i % log_every_n == 0):
                logging.debug(f"Step {i}: Signal={sig:.4f}, Price={price:.2f}, Position={position}, Capital={capital:.2f}")

            # --- LONG entry ---
            if sig > threshold and position is None:
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
                    stop_loss = adjusted_entry_price * (1 - stop_loss_pct)
                if take_profit_atr_mult is not None and this_atr is not None:
                    take_profit = adjusted_entry_price + take_profit_atr_mult * this_atr
                else:
                    take_profit = adjusted_entry_price * (1 + take_profit_pct)
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
                # Recompute stops dynamically for each bar if ATR mode
                stop_loss = None
                take_profit = None
                if stop_loss_atr_mult is not None and this_atr is not None:
                    stop_loss = entry_price - stop_loss_atr_mult * this_atr
                else:
                    stop_loss = entry_price * (1 - stop_loss_pct)
                if take_profit_atr_mult is not None and this_atr is not None:
                    take_profit = entry_price + take_profit_atr_mult * this_atr
                else:
                    take_profit = entry_price * (1 + take_profit_pct)

                # Partial exit at first TP
                if partial_exit and not partial_out and price >= take_profit:
                    # Sell half, keep half for trailing
                    half_size = position["size"] / 2
                    profit = half_size * (price - entry_price)
                    trades.append({
                        "type": "partial_exit",
                        "exit_bar": i,
                        "exit_price": price,
                        "profit": profit,
                        "return": (price - entry_price) / entry_price,
                        "amount": half_size,
                        "capital": capital + half_size * price
                    })
                    capital += half_size * price
                    position["size"] = half_size
                    partial_out = True
                    # Set trailing stop for remainder
                    if trailing_atr_mult is not None and this_atr is not None:
                        trailing_stop = price - trailing_atr_mult * this_atr
                    else:
                        trailing_stop = price * (1 - stop_loss_pct)
                # Trailing exit for remainder
                elif partial_exit and partial_out and trailing_stop is not None:
                    # Dynamically update trailing stop if price moves up
                    if trailing_atr_mult is not None and this_atr is not None:
                        new_trailing = price - trailing_atr_mult * this_atr
                    else:
                        new_trailing = price * (1 - stop_loss_pct)
                    if new_trailing > trailing_stop:
                        trailing_stop = new_trailing
                    if price <= trailing_stop:
                        profit = position["size"] * (price - entry_price)
                        trades.append({
                            "type": "trailing_exit",
                            "exit_bar": i,
                            "exit_price": price,
                            "profit": profit,
                            "return": (price - entry_price) / entry_price,
                            "amount": position["size"],
                            "capital": capital + position["size"] * price
                        })
                        capital += position["size"] * price
                        equity_curve.append(capital)
                        position = None
                        entry_price = None
                        partial_out = False
                        trailing_stop = None
                # Full exit (SL or signal flip or TP if not using partial_exit)
                elif ((not partial_exit) and (price <= stop_loss or price >= take_profit or sig < threshold)) or (partial_exit and not partial_out and price <= stop_loss) or (partial_exit and not partial_out and sig < threshold):
                    # If partial_exit and not partial_out, SL or signal flip triggers full exit.
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
                    partial_out = False
                    trailing_stop = None
                # Update trailing stop if in partial
                elif partial_exit and partial_out:
                    if trailing_atr_mult is not None and this_atr is not None:
                        new_trailing = price - trailing_atr_mult * this_atr
                    else:
                        new_trailing = price * (1 - stop_loss_pct)
                    if new_trailing > trailing_stop:
                        trailing_stop = new_trailing

        trades_df = pd.DataFrame(trades)
        # --- PATCH: Robustly handle missing 'return' column ---
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