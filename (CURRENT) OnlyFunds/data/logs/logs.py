# logs.py

import os
import json
import csv
import logging
from datetime import datetime

LOGS_DIR = "data/logs"
os.makedirs(LOGS_DIR, exist_ok=True)

def log_trade(trade_data: dict, strategy: str = "signal"):
    """Log trade results to a JSON file, with optional strategy field."""
    log_file = os.path.join(LOGS_DIR, "trade_logs.json")
    trade_data = trade_data.copy()
    trade_data["timestamp"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    trade_data["strategy"] = strategy
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

def log_grid_trade(trade_data: dict):
    """Log grid trade results to a dedicated grid trade log JSON (and optionally CSV)."""
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