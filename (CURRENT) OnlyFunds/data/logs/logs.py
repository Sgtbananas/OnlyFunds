# logs.py

import os
import json
import csv
import logging
from datetime import datetime

LOGS_DIR = "data/logs"
os.makedirs(LOGS_DIR, exist_ok=True)

TRADE_JSON_FILE = os.path.join(LOGS_DIR, "trade_logs.json")
BACKTEST_JSON_FILE = os.path.join(LOGS_DIR, "backtest_logs.json")

# CSV files use timestamp in filename for overlays, and a "latest" pointer for analysis scripts
def get_csv_name(suffix="trades"):
    # Use UTC timestamp for unique file naming per run
    timestamp = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%S")
    return os.path.join(LOGS_DIR, f"{suffix}_{timestamp}.csv")

# Latest pointer for easy analysis script loading
LATEST_CSV_FILE = os.path.join(LOGS_DIR, "trades_latest.csv")

# Initialize logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

CSV_FIELDNAMES = [
    "timestamp", "pair", "action", "amount", "entry_price", "exit_price", "return_pct",
    "exit_reason", "capital", "threshold"
]

def log_trade(trade_data: dict, threshold_value=None, exit_reason=None, capital=None):
    """
    Log trade results to both a JSON and a CSV file.
    - threshold_value: the threshold used for this trade (manual or autotuned)
    - exit_reason: (optional) string describing why the trade was closed
    - capital: (optional) capital after trade
    """
    trade_data = trade_data.copy()
    trade_data["timestamp"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    if threshold_value is not None:
        trade_data["threshold"] = threshold_value
    if exit_reason is not None:
        trade_data["exit_reason"] = exit_reason
    if capital is not None:
        trade_data["capital"] = capital

    # JSON logging (append to list)
    try:
        if os.path.exists(TRADE_JSON_FILE):
            with open(TRADE_JSON_FILE, "r") as f:
                logs = json.load(f)
        else:
            logs = []
        logs.append(trade_data)
        with open(TRADE_JSON_FILE, "w") as f:
            json.dump(logs, f, indent=2)
    except Exception as e:
        logging.error(f"Error logging trade to JSON: {e}")

    # CSV logging (append, create new overlay file per run if needed)
    csv_file = LATEST_CSV_FILE
    is_new = not os.path.exists(csv_file)
    try:
        with open(csv_file, "a", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=CSV_FIELDNAMES)
            if is_new:
                writer.writeheader()
            # Ensure all keys needed (if missing, leave blank)
            row = {k: trade_data.get(k, "") for k in CSV_FIELDNAMES}
            writer.writerow(row)
    except Exception as e:
        logging.error(f"Error logging trade to CSV: {e}")

def start_new_csv_run():
    """
    Call this at the start of each bot run/session to rotate/overlay logs.
    """
    new_csv = get_csv_name("trades")
    # Point the "latest" symlink or copy to new csv for analysis scripts
    try:
        open(new_csv, "a").close()
        if os.path.exists(LATEST_CSV_FILE):
            os.remove(LATEST_CSV_FILE)
        # Symlink if possible, else copy
        try:
            os.symlink(new_csv, LATEST_CSV_FILE)
        except Exception:
            import shutil
            shutil.copy(new_csv, LATEST_CSV_FILE)
        logging.info(f"Started new CSV run: {new_csv}")
    except Exception as e:
        logging.error(f"Error rotating CSV log: {e}")

def log_backtest(results: dict):
    """Log backtest results to a JSON file."""
    results = results.copy()
    results["timestamp"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    try:
        if os.path.exists(BACKTEST_JSON_FILE):
            with open(BACKTEST_JSON_FILE, "r") as f:
                logs = json.load(f)
        else:
            logs = []
        logs.append(results)
        with open(BACKTEST_JSON_FILE, "w") as f:
            json.dump(logs, f, indent=2)
    except Exception as e:
        logging.error(f"Error logging backtest: {e}")