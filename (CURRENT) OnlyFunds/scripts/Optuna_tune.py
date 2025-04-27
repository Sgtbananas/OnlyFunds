"""
Optuna hyperparameter search for OnlyFunds.
- Maximizes total return across all pairs.
- Saves best params and value to state/optuna_best.json.
"""
import os
import json
import optuna
from core.core_data import fetch_klines, add_indicators, TRADING_PAIRS
from core.core_signals import generate_signal
from core.backtester import run_backtest

def objective(trial):
    # Suggest hyperparameters
    threshold      = trial.suggest_float("threshold", 0.005, 0.2, log=True)
    risk_pct       = trial.suggest_float("risk_pct", 0.002, 0.03)
    stop_loss_pct  = trial.suggest_float("stop_loss_pct", 0.002, 0.02)
    take_profit_pct= trial.suggest_float("take_profit_pct", 0.002, 0.03)

    # aggregate metric across pairs
    total_return = 0
    for pair in TRADING_PAIRS:
        df = fetch_klines(pair=pair, interval="15m", limit=1000)
        if df.empty:
            continue
        df = add_indicators(df)
        sig = generate_signal(df)
        summary_df = run_backtest(
            signal=sig,
            prices=df["Close"],
            threshold=threshold,
            initial_capital=10.0,
            risk_pct=risk_pct,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
        )
        summary = summary_df.loc[summary_df["type"]=="summary"].iloc[0]
        # Prefer total_pnl if present, else fallback to capital/return
        total_return += summary.get("total_pnl", summary.get("capital", 0) - 10.0)
    # We want to maximize total_return
    return total_return

if __name__ == "__main__":
    os.makedirs("state", exist_ok=True)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)

    best = {
        "best_params": study.best_params,
        "best_value": study.best_value,
    }
    with open("state/optuna_best.json", "w") as f:
        json.dump(best, f, indent=2)
    print("✅ Best params saved to state/optuna_best.json")