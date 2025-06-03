import os
import json
from core.core_data import fetch_klines, add_indicators, TRADING_PAIRS

def ensure_state_dir():
    """
    Ensure that the state directory exists.
    """
    state_dir = os.path.join(os.path.dirname(__file__), "..", "state")
    os.makedirs(state_dir, exist_ok=True)
    return state_dir

def run_backtests():
    """
    Run a simple backtest for each trading pair and write results to state/backtest_results.json.
    If no results are found, write a dummy result so the UI always displays something.
    """
    results = []
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    print("DEBUG: Current working directory:", os.getcwd())
    print("DEBUG: Files in ./data/:", os.listdir(data_dir) if os.path.exists(data_dir) else "NO DATA DIR")

    for pair in TRADING_PAIRS:
        interval = "15m"
        limit = 1000
        print(f"DEBUG: Running backtest for {pair}")
        df = fetch_klines(pair, interval, limit)
        print(f"DEBUG: {pair} df.shape={df.shape}")
        if df.empty:
            print(f"WARNING: No data for {pair}")
            continue

        df = add_indicators(df)
        if "Close" in df.columns and "ATR" in df.columns:
            # Dummy strategy: enter at first bar, exit at last bar
            if len(df) >= 2:
                entry_price = df.iloc[0]["Close"]
                exit_price = df.iloc[-1]["Close"]
                profit = exit_price - entry_price
                results.append({
                    "pair": pair,
                    "entry": float(entry_price),
                    "exit": float(exit_price),
                    "profit": float(profit)
                })
            else:
                print(f"WARNING: Not enough data to perform backtest for {pair}")
        else:
            print(f"WARNING: Data missing required columns for {pair}")

    # If no results, insert a dummy result so UI always displays something
    if not results:
        results = [{
            "pair": "DUMMY",
            "entry": 100.0,
            "exit": 110.0,
            "profit": 10.0
        }]
        print("INFO: No real results, writing dummy result to ensure UI displays output.")

    state_dir = ensure_state_dir()
    out_path = os.path.join(state_dir, "backtest_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"DEBUG: Wrote backtest results to {out_path}")
    print(f"DEBUG: Results: {results}")

if __name__ == "__main__":
    run_backtests()