# logs.py

import os
import json
import logging
from datetime import datetime

LOGS_DIR = "data/logs"
os.makedirs(LOGS_DIR, exist_ok=True)

# Initialize logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def log_trade(trade_data: dict):
    """Log trade results to a JSON file."""
    log_file = os.path.join(LOGS_DIR, "trade_logs.json")
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

def log_backtest(results: dict):
    """Log backtest results to a JSON file."""
    log_file = os.path.join(LOGS_DIR, "backtest_logs.json")
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
