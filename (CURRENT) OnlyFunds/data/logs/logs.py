# logs.py

import os
import json
import logging
from datetime import datetime

LOGS_DIR = "data/logs"
os.makedirs(LOGS_DIR, exist_ok=True)

# Initialize logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def log_trade(trade_data: dict, log_file=None):
    """
    Log general trade results to a JSON file.
    If log_file is not specified, defaults to data/logs/trade_logs.json.
    """
    if log_file is None:
        log_file = os.path.join(LOGS_DIR, "trade_logs.json")
    trade_data = trade_data.copy()
    trade_data["timestamp"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    try:
        if os.path.exists(log_file):
            with open(log_file, "r") as f:
                logs = json.load(f)
        else:
            logs = []

        logs.append(trade_data)
        with open(log_file, "w") as f:
            json.dump(logs, f, indent=2)
    except Exception as e:
        logging.error(f"Error logging trade: {e}")

def log_grid_trade(trade_data, log_file=None):
    """
    Log grid trade results to a dedicated grid trade log JSON file.
    If log_file is not specified, defaults to data/logs/grid_trade_logs.json.
    """
    if log_file is None:
        log_file = os.path.join(LOGS_DIR, "grid_trade_logs.json")
    trade_data = trade_data.copy()
    trade_data["timestamp"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    try:
        if os.path.exists(log_file):
            with open(log_file, "r") as f:
                logs = json.load(f)
        else:
            logs = []
        logs.append(trade_data)
        with open(log_file, "w") as f:
            json.dump(logs, f, indent=2)
    except Exception as e:
        logging.error(f"Error logging grid trade: {e}")

def log_backtest(results: dict):
    """Log backtest results to a JSON file."""
    log_file = os.path.join(LOGS_DIR, "backtest_logs.json")
    results = results.copy()
    results["timestamp"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    try:
        if os.path.exists(log_file):
            with open(log_file, "r") as f:
                logs = json.load(f)
        else:
            logs = []

        logs.append(results)
        with open(log_file, "w") as f:
            json.dump(logs, f, indent=2)
    except Exception as e:
        logging.error(f"Error logging backtest: {e}")

# For backward compatibility: alias log_trade to log_grid_trade if needed
# log_trade = log_grid_trade  # Uncomment ONLY if you want both to be the same
