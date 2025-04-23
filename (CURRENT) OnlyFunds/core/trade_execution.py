# core/trade_execution.py

import os
import time
import hmac
import json
import hashlib
import logging
import requests
from dotenv import load_dotenv

load_dotenv()
API_KEY    = os.getenv("COINEX_API_KEY")
SECRET_KEY = os.getenv("COINEX_SECRET_KEY")
API_BASE   = os.getenv("API_BASE_URL", "https://api.coinex.com/v1")

HEADERS = {"Content-Type":"application/json"}

def _sign(params: dict) -> str:
    s = "&".join(f"{k}={v}" for k,v in sorted(params.items()))
    return hmac.new(SECRET_KEY.encode(), s.encode(), hashlib.sha256).hexdigest()

def _auth_post(endpoint: str, payload: dict) -> dict:
    payload["access_id"] = API_KEY
    payload["tonce"]     = int(time.time()*1000)
    sig = _sign(payload)
    h = HEADERS.copy(); h["Authorization"] = sig
    url = f"{API_BASE}/{endpoint}"
    r = requests.post(url, data=payload, headers=h, timeout=15)
    r.raise_for_status()
    return r.json()

def place_order(pair: str, action: str,
                amount: float, price: float = None,
                is_dry_run: bool = True) -> dict:
    """
    :action: 'buy' or 'sell'
    """
    if is_dry_run:
        logging.info(f"[DRY] {action.upper()} {amount} {pair} @ {'MKT' if price is None else price}")
        return {"dry_run":True, "pair":pair, "action":action, "amount":amount, "price":price}
    try:
        typ = "market" if price is None else "limit"
        res = _auth_post("order/limit", {
            "market": pair.lower(),
            "type":  typ,
            "amount": amount,
            "price":  price or 0,
            "side":   action.lower()
        })
        logging.info(f"Executed {action} on {pair}: {res}")
        return res
    except Exception as e:
        logging.error(f"place_order error: {e}")
        return {"error":str(e), "pair":pair, "action":action}
