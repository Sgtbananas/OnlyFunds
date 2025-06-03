import pandas as pd
import numpy as np
import logging
from core.grid_trader import GridTrader
from core.meta_learner import select_strategy
from core import risk_manager
from core.core_data import add_indicators

logger = logging.getLogger(__name__)

def run_backtest(signal, pair="BTCUSDT", interval="5m", limit=1000, equity=1000,
                 performance_dict=None, meta_model=None,
                 risk_pct=0.01, atr_multiplier=2,
                 grid_mode=False, grid_kwargs=None,
                 fee_pct=0.002, verbose=False, log_every_n=100, data=None):
    import logging

    """
    Run the backtest for the selected strategy.
    """

    # --- Validate data ---
    if data is None:
        logger.error("Backtest called without data.")
        return None

    df = data.copy()

    # Ensure indicators are present
    required_cols = ["ATR", "Close"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logger.warning(f"Data missing columns {missing_cols} — recalculating indicators for {pair}.")
        df = add_indicators(df)

    if df.empty or "ATR" not in df.columns:
        logger.error(f"Backtest aborted: No ATR data available for {pair}.")
        return None

    prices = df["Close"]
    atr = df["ATR"]

    if isinstance(signal, pd.Series):
        signals = signal
        logging.info(f'SIGNAL DEBUG: mean={signals.mean():.4f}, >0 count={(signals > 0).sum()}, max={signals.max()}')
    else:
        signals = pd.Series(signal, index=df.index)

    strategy, _ = select_strategy(performance_dict, meta_model)

    capital = equity
    position = None
    trades = []
    equity_curve = [capital]

    for i in range(len(signals)):
        sig = signals.iloc[i]
        price = prices.iloc[i]
        this_atr = atr.iloc[i] if not pd.isna(atr.iloc[i]) else 0

        if verbose and i % log_every_n == 0:
            logger.debug(f"Step {i}: Signal={sig}, Price={price}, Capital={capital}")

        # --- Entry ---
        if position is None and sig > 0.05:
            position_size = risk_manager.position_size(
                capital, price,
                risk_pct=risk_pct,
                volatility=this_atr,
                v_adj=True
            )
            entry_price = price * (1 + fee_pct)
            stop_loss = entry_price - (atr_multiplier * this_atr)
            take_profit = entry_price + (atr_multiplier * this_atr)
            position = {
                "size": position_size,
                "entry_price": entry_price,
                "stop_loss": stop_loss,
                "take_profit": take_profit
            }
            capital -= position_size * entry_price
            trades.append({
                "type": "entry",
                "price": entry_price,
                "bar": i
            })

        # --- Exit ---
        if position:
            exit = False
            exit_reason = ""

            if price <= position["stop_loss"]:
                exit = True
                exit_reason = "stop_loss"
            elif price >= position["take_profit"]:
                exit = True
                exit_reason = "take_profit"
            elif sig < 0:
                exit = True
                exit_reason = "signal_flip"

            if exit:
                exit_price = price * (1 - fee_pct)
                pnl = position["size"] * (exit_price - position["entry_price"])
                capital += position["size"] * exit_price
                trades.append({
                    "type": "exit",
                    "price": exit_price,
                    "reason": exit_reason,
                    "profit": pnl,
                    "bar": i
                })
                position = None

        equity_curve.append(capital)

    # --- Forced exit if still open ---
    if position:
        exit_price = prices.iloc[-1] * (1 - fee_pct)
        pnl = position["size"] * (exit_price - position["entry_price"])
        capital += position["size"] * exit_price
        trades.append({
            "type": "forced_exit",
            "price": exit_price,
            "profit": pnl,
            "bar": len(signals)-1
        })

    # --- Metrics ---
    trades_df = pd.DataFrame(trades)

    if not trades_df.empty:
        profits = trades_df[trades_df["type"].isin(["exit", "forced_exit"])]["profit"]
        total_pnl = profits.sum()
        win_rate = (profits > 0).mean() * 100 if not profits.empty else 0
        avg_return = profits.mean() / equity if not profits.empty else 0
    else:
        total_pnl = 0
        win_rate = 0
        avg_return = 0

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
        "max_drawdown": max_drawdown
    }

    summary_df = pd.DataFrame([summary])
    combined_df = pd.concat([summary_df, trades_df], ignore_index=True)

    if trades_df.empty:
        logger.warning("No trades executed during backtest.")
    else:
        logger.info(f"Backtest complete: {len(trades_df)} trades, Win rate: {win_rate:.2f}%, Max DD: {max_drawdown:.2f}%")

    return combined_df
