"""CoinEx Spot API v2 client.

Implemented from the official docs (https://docs.coinex.com/api/v2):
- Base URL ``https://api.coinex.com/v2``; envelope ``{"code", "data", "message"}``,
  ``code == 0`` is success.
- Signing: ``HMAC-SHA256(secret, METHOD + request_path(with /v2 and query) + body + timestamp_ms)``,
  lowercase hex, sent as ``X-COINEX-KEY`` / ``X-COINEX-SIGN`` / ``X-COINEX-TIMESTAMP``.
- Public: GET /spot/market, /spot/ticker, /spot/kline, /spot/depth.
- Private: GET /assets/spot/balance, POST /spot/order, POST /spot/cancel-order,
  GET /spot/order-status.

Safety:
- Order placement and cancellation raise ``TradingLockedError`` unless the client
  was constructed with ``allow_trading=True``. The OF2 app never does this:
  live trading is locked (see ``core.trade_execution``).
- Order placement is never retried automatically. A network failure while
  placing an order raises ``OrderStateUnknown`` so the caller reconciles
  instead of risking a duplicate order.
- API secrets are never logged or included in ``repr``.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import math
import os
import re
import threading
import time
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode

import pandas as pd
import requests

logger = logging.getLogger(__name__)

BASE_URL = "https://api.coinex.com"
API_PREFIX = "/v2"

# Interval names used across OF2 -> CoinEx kline "period" values.
INTERVAL_MAP = {
    "1m": "1min", "3m": "3min", "5m": "5min", "15m": "15min", "30m": "30min",
    "1h": "1hour", "2h": "2hour", "4h": "4hour", "6h": "6hour", "12h": "12hour",
    "1d": "1day", "3d": "3day", "1w": "1week",
}
INTERVAL_SECONDS = {
    "1m": 60, "3m": 180, "5m": 300, "15m": 900, "30m": 1800, "1h": 3600, "2h": 7200,
    "4h": 14400, "6h": 21600, "12h": 43200, "1d": 86400, "3d": 259200, "1w": 604800,
}
MAX_KLINE_LIMIT = 1000
DEPTH_LIMITS = (5, 10, 20, 50)

# Codes that are safe to retry for idempotent (read) requests.
RETRYABLE_CODES = {3008, 4001, 4002, 4213}
RATE_LIMIT_CODE = 4213


class CoinExError(Exception):
    def __init__(self, code: Optional[int], message: str, http_status: Optional[int] = None):
        super().__init__(f"CoinEx error {code}: {message}")
        self.code = code
        self.message = message
        self.http_status = http_status


class CoinExRateLimitError(CoinExError):
    pass


class CoinExAuthError(CoinExError):
    pass


class CoinExNetworkError(CoinExError):
    pass


class OrderStateUnknown(CoinExError):
    """Order request was sent but no reliable response was received."""


class TradingLockedError(RuntimeError):
    """Raised whenever code tries to place or cancel a real order while locked."""


AUTH_ERROR_CODES = {4005, 4006, 4007, 4008, 4010, 4011, 4017}


@dataclass(frozen=True)
class MarketInfo:
    market: str
    base_ccy: str
    quote_ccy: str
    base_precision: int
    quote_precision: int
    min_amount: float
    maker_fee_rate: float
    taker_fee_rate: float
    status: str
    api_trading: bool

    @property
    def tradable(self) -> bool:
        return self.status == "online" and self.api_trading

    def floor_amount(self, amount: float) -> float:
        """Round an order amount *down* to the market's base precision."""
        q = 10 ** self.base_precision
        return math.floor(amount * q + 1e-9) / q

    def round_price(self, price: float) -> float:
        return round(price, self.quote_precision)

    @classmethod
    def from_api(cls, d: Dict[str, Any]) -> "MarketInfo":
        return cls(
            market=d["market"],
            base_ccy=d.get("base_ccy", ""),
            quote_ccy=d.get("quote_ccy", ""),
            base_precision=int(d.get("base_ccy_precision", 8)),
            quote_precision=int(d.get("quote_ccy_precision", 8)),
            min_amount=float(d.get("min_amount", 0) or 0),
            maker_fee_rate=float(d.get("maker_fee_rate", 0.002) or 0.002),
            taker_fee_rate=float(d.get("taker_fee_rate", 0.002) or 0.002),
            status=str(d.get("status", "")),
            api_trading=bool(d.get("is_api_trading_available", False)),
        )


def sign_request(secret: str, method: str, request_path: str, body: str, timestamp_ms: str) -> str:
    prepared = f"{method.upper()}{request_path}{body}{timestamp_ms}"
    return hmac.new(secret.encode("latin-1"), prepared.encode("latin-1"), hashlib.sha256).hexdigest().lower()


class CoinExClient:
    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None,
                 session: Optional[requests.Session] = None, timeout: float = 10.0,
                 max_retries: int = 3, backoff: float = 0.5, min_interval: float = 0.05,
                 allow_trading: bool = False, base_url: str = BASE_URL):
        self._key = api_key if api_key is not None else os.getenv("COINEX_API_KEY", "")
        self._secret = api_secret if api_secret is not None else (
            os.getenv("COINEX_API_SECRET", "") or os.getenv("COINEX_SECRET_KEY", ""))
        self.session = session or requests.Session()
        self.timeout = timeout
        self.max_retries = max(1, int(max_retries))
        self.backoff = backoff
        self.min_interval = min_interval
        self.allow_trading = bool(allow_trading)
        self.base_url = base_url.rstrip("/")
        self._lock = threading.Lock()
        self._last_call = 0.0
        self._markets: Dict[str, MarketInfo] = {}
        self._markets_ts = 0.0

    def __repr__(self) -> str:
        return f"CoinExClient(has_credentials={self.has_credentials}, allow_trading={self.allow_trading})"

    @property
    def has_credentials(self) -> bool:
        return bool(self._key and self._secret)

    # ------------------------------------------------------------------ core
    def _throttle(self) -> None:
        with self._lock:
            wait = self.min_interval - (time.monotonic() - self._last_call)
            if wait > 0:
                time.sleep(wait)
            self._last_call = time.monotonic()

    def _build(self, method: str, path: str, params: Optional[Dict], body: Optional[Dict], signed: bool):
        request_path = API_PREFIX + path
        if params:
            request_path += "?" + urlencode({k: v for k, v in params.items() if v is not None})
        body_str = json.dumps(body, separators=(",", ":")) if body is not None else ""
        headers = {"Content-Type": "application/json; charset=utf-8", "Accept": "application/json"}
        if signed:
            if not self.has_credentials:
                raise CoinExAuthError(None, "API credentials not configured (set COINEX_API_KEY / COINEX_API_SECRET)")
            ts = str(int(time.time() * 1000))
            headers["X-COINEX-KEY"] = self._key
            headers["X-COINEX-TIMESTAMP"] = ts
            headers["X-COINEX-SIGN"] = sign_request(self._secret, method, request_path, body_str, ts)
        return self.base_url + request_path, body_str, headers

    def _parse(self, resp) -> Any:
        if resp.status_code != 200:
            # Gateway errors come back as HTML; keep only the <title> text.
            m = re.search(r"<title>(.*?)</title>", resp.text or "", re.I | re.S)
            detail = m.group(1).strip() if m else (resp.text or "")[:200]
            raise CoinExError(None, f"HTTP {resp.status_code} from CoinEx gateway: {detail}", resp.status_code)
        try:
            payload = resp.json()
        except ValueError:
            raise CoinExError(None, f"Non-JSON response: {resp.text[:200]}", resp.status_code)
        code = payload.get("code")
        if code != 0:
            msg = str(payload.get("message", ""))
            if code == RATE_LIMIT_CODE:
                raise CoinExRateLimitError(code, msg, resp.status_code)
            if code in AUTH_ERROR_CODES:
                raise CoinExAuthError(code, msg, resp.status_code)
            raise CoinExError(code, msg, resp.status_code)
        return payload.get("data")

    def _request(self, method: str, path: str, params: Optional[Dict] = None, body: Optional[Dict] = None,
                 signed: bool = False, retry: bool = True) -> Any:
        attempts = self.max_retries if retry else 1
        last_exc: Optional[Exception] = None
        for attempt in range(attempts):
            url, body_str, headers = self._build(method, path, params, body, signed)
            self._throttle()
            try:
                resp = self.session.request(method, url, data=body_str or None, headers=headers,
                                            timeout=self.timeout)
            except requests.RequestException as exc:
                last_exc = CoinExNetworkError(None, f"{type(exc).__name__}: {exc}")
                logger.warning("CoinEx %s %s network error (attempt %d/%d): %s",
                               method, path, attempt + 1, attempts, type(exc).__name__)
            else:
                try:
                    return self._parse(resp)
                except CoinExError as exc:
                    retryable = (exc.code in RETRYABLE_CODES) or (exc.http_status in (429, 502, 503, 504))
                    if not retryable:
                        raise
                    last_exc = exc
                    logger.warning("CoinEx %s %s retryable error %s (attempt %d/%d)",
                                   method, path, exc.code, attempt + 1, attempts)
            if attempt + 1 < attempts:
                time.sleep(self.backoff * (2 ** attempt))
        assert last_exc is not None
        raise last_exc

    # ---------------------------------------------------------------- public
    def get_markets(self, refresh: bool = False, ttl: float = 600.0) -> Dict[str, MarketInfo]:
        if refresh or not self._markets or time.time() - self._markets_ts > ttl:
            data = self._request("GET", "/spot/market") or []
            self._markets = {d["market"]: MarketInfo.from_api(d) for d in data if "market" in d}
            self._markets_ts = time.time()
        return self._markets

    def get_market(self, market: str) -> MarketInfo:
        markets = self.get_markets()
        if market not in markets:
            raise CoinExError(None, f"Unknown market {market}")
        return markets[market]

    def get_tickers(self, markets: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        params = {"market": ",".join(markets)} if markets else None
        return self._request("GET", "/spot/ticker", params=params) or []

    def get_ticker(self, market: str) -> Dict[str, Any]:
        data = self.get_tickers([market])
        if not data:
            raise CoinExError(None, f"No ticker for {market}")
        return data[0]

    def get_klines(self, market: str, interval: str = "5m", limit: int = 500,
                   drop_unclosed: bool = True, now_ms: Optional[int] = None) -> pd.DataFrame:
        """OHLCV candles, oldest first, with a UTC ``timestamp`` column.

        ``drop_unclosed`` removes the still-forming last candle so decisions
        are made only on completed bars.
        """
        if interval not in INTERVAL_MAP:
            raise ValueError(f"Unsupported interval {interval}; use one of {list(INTERVAL_MAP)}")
        limit = int(max(1, min(MAX_KLINE_LIMIT, limit)))
        data = self._request("GET", "/spot/kline",
                             params={"market": market, "period": INTERVAL_MAP[interval], "limit": limit}) or []
        df = klines_to_frame(data)
        if drop_unclosed and not df.empty:
            now_ms = now_ms if now_ms is not None else int(time.time() * 1000)
            close_ms = df["created_at"] + INTERVAL_SECONDS[interval] * 1000
            df = df[close_ms <= now_ms].reset_index(drop=True)
        return df

    def get_depth(self, market: str, limit: int = 20, interval: str = "0") -> Dict[str, Any]:
        if limit not in DEPTH_LIMITS:
            raise ValueError(f"depth limit must be one of {DEPTH_LIMITS}")
        return self._request("GET", "/spot/depth", params={"market": market, "limit": limit, "interval": interval})

    # --------------------------------------------------------------- private
    def get_spot_balance(self) -> Dict[str, Dict[str, float]]:
        data = self._request("GET", "/assets/spot/balance", signed=True) or []
        return {d["ccy"]: {"available": float(d.get("available", 0)), "frozen": float(d.get("frozen", 0))}
                for d in data}

    def get_order_status(self, market: str, order_id: int) -> Dict[str, Any]:
        return self._request("GET", "/spot/order-status", params={"market": market, "order_id": int(order_id)},
                             signed=True)

    def _require_trading(self) -> None:
        if not self.allow_trading:
            raise TradingLockedError("Real order placement is locked in OnlyFunds 2.")

    def place_order(self, market: str, side: str, order_type: str, amount: float,
                    price: Optional[float] = None, client_id: Optional[str] = None,
                    ccy: Optional[str] = None) -> Dict[str, Any]:
        self._require_trading()
        if side not in ("buy", "sell"):
            raise ValueError("side must be 'buy' or 'sell'")
        if order_type not in ("limit", "market", "maker_only", "ioc", "fok"):
            raise ValueError(f"invalid order type {order_type}")
        if order_type != "market" and price is None:
            raise ValueError("price required for non-market orders")
        info = self.get_market(market)
        if not info.tradable:
            raise CoinExError(None, f"{market} is not available for API trading (status={info.status})")
        if not (ccy and ccy == info.quote_ccy):  # amount in base currency
            amount = info.floor_amount(amount)
            if amount <= 0 or amount < info.min_amount:
                raise CoinExError(3127, f"amount {amount} below market minimum {info.min_amount}")
        body: Dict[str, Any] = {"market": market, "market_type": "SPOT", "side": side, "type": order_type,
                                "amount": format_decimal(amount)}
        if price is not None:
            body["price"] = format_decimal(info.round_price(price))
        if client_id:
            body["client_id"] = client_id
        if ccy:
            body["ccy"] = ccy
        try:
            return self._request("POST", "/spot/order", body=body, signed=True, retry=False)
        except CoinExNetworkError as exc:
            raise OrderStateUnknown(None, f"order {client_id} state unknown: {exc.message}")

    def cancel_order(self, market: str, order_id: int) -> Dict[str, Any]:
        self._require_trading()
        return self._request("POST", "/spot/cancel-order",
                             body={"market": market, "market_type": "SPOT", "order_id": int(order_id)},
                             signed=True, retry=False)


def format_decimal(x: float) -> str:
    """Shortest exact decimal string for a float (no binary-float artefacts, no exponent)."""
    s = format(Decimal(repr(float(x))), "f")
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s or "0"


def klines_to_frame(data: List[Dict[str, Any]]) -> pd.DataFrame:
    cols = ["timestamp", "created_at", "Open", "High", "Low", "Close", "Volume", "Value"]
    if not data:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(data)
    df = df.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close",
                            "volume": "Volume", "value": "Value"})
    for c in ("Open", "High", "Low", "Close", "Volume", "Value"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["created_at"] = pd.to_numeric(df["created_at"]).astype("int64")
    df["timestamp"] = pd.to_datetime(df["created_at"], unit="ms", utc=True)
    df = df.drop_duplicates("created_at").sort_values("created_at").reset_index(drop=True)
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    return df[[c for c in cols if c in df.columns]]


def top_usdt_markets(client: CoinExClient, limit: int = 200, min_quote_volume: float = 0.0,
                     exclude: Optional[set] = None) -> List[str]:
    """Tradable USDT spot markets ranked by 24h quote volume (``value``)."""
    exclude = exclude or set()
    markets = client.get_markets()
    tickers = client.get_tickers()
    ranked = []
    leveraged = ("3L", "3S", "5L", "5S", "UP", "DOWN", "BULL", "BEAR")
    for t in tickers:
        m = t.get("market", "")
        info = markets.get(m)
        if not info or info.quote_ccy != "USDT" or not info.tradable or m in exclude:
            continue
        if any(info.base_ccy.endswith(x) for x in leveraged):
            continue
        try:
            value = float(t.get("value", 0) or 0)
        except (TypeError, ValueError):
            continue
        if value >= min_quote_volume:
            ranked.append((m, value))
    ranked.sort(key=lambda x: x[1], reverse=True)
    return [m for m, _ in ranked[:limit]]
