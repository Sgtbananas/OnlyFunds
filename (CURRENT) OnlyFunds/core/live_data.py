"""Live public market data (Binance spot): research/paper use only, never order placement.

Sources (official public endpoints, no API key):
  REST  https://api.binance.com/api/v3/{klines, ticker/bookTicker, depth, ticker/24hr, time}
  WS    wss://stream.binance.com:9443/stream?streams=<sym>@kline_<interval>/<sym>@bookTicker

Guarantees:
- Only COMPLETED candles are used for signals (REST drops the forming candle; WS uses kline.x == true).
- Quotes carry the exchange event time and the local receive time; anything older than
  ``max_quote_age`` is reported stale and must not be treated as current.
- Candle continuity is checked; gaps are filled only with real candles fetched from the exchange
  (REST backfill), never fabricated. Malformed / inconsistent messages are rejected and counted.
- Rate limits: the REST client tracks Binance's X-MBX-USED-WEIGHT-1M header, honours Retry-After on
  429/418, backs off before the 1-minute weight budget is exhausted, and retries transient errors.
- All times are UTC.
"""
from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
import requests

from core.coinex_client import INTERVAL_SECONDS
from core.history import COLUMNS, binance_rows_to_frame

logger = logging.getLogger(__name__)

REST_BASE = "https://api.binance.com"
WS_BASE = "wss://stream.binance.com:9443/stream?streams="
WEIGHT_LIMIT_PER_MIN = 6000  # Binance spot REQUEST_WEIGHT limit (exchangeInfo); we stay well below it


class StaleDataError(RuntimeError):
    pass


class FeedError(RuntimeError):
    pass


def utc_now_ms() -> int:
    return int(time.time() * 1000)


# --------------------------------------------------------------------------------- REST
class BinancePublicClient:
    def __init__(self, session: Optional[requests.Session] = None, timeout: float = 10.0, max_retries: int = 3,
                 min_interval: float = 0.05, weight_budget: float = 0.5, sleep: Callable[[float], None] = time.sleep):
        self.session = session or requests.Session()
        self.timeout, self.max_retries, self.min_interval = timeout, max_retries, min_interval
        self.weight_budget = weight_budget * WEIGHT_LIMIT_PER_MIN
        self.sleep = sleep
        self.used_weight = 0
        self._last = 0.0
        self._lock = threading.Lock()
        self.stats = {"requests": 0, "retries": 0, "rate_limited": 0, "errors": 0}

    def _get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries):
            with self._lock:
                wait = self.min_interval - (time.monotonic() - self._last)
                if wait > 0:
                    self.sleep(wait)
                if self.used_weight > self.weight_budget:  # slow down before Binance has to tell us
                    self.sleep(60 - datetime.now(timezone.utc).second + 1)
                    self.used_weight = 0
                self._last = time.monotonic()
            self.stats["requests"] += 1
            try:
                r = self.session.get(REST_BASE + path, params=params, timeout=self.timeout)
            except requests.RequestException as exc:
                last_exc = FeedError(f"network error on {path}: {type(exc).__name__}")
                self.stats["retries"] += 1
                self.sleep(min(2 ** attempt, 10))
                continue
            w = r.headers.get("X-MBX-USED-WEIGHT-1M") if hasattr(r, "headers") else None
            if w and str(w).isdigit():
                self.used_weight = int(w)
            if r.status_code in (418, 429):
                self.stats["rate_limited"] += 1
                retry_after = float(r.headers.get("Retry-After", 60)) if hasattr(r, "headers") else 60.0
                last_exc = FeedError(f"rate limited ({r.status_code}); retry after {retry_after}s")
                self.sleep(retry_after)
                continue
            if r.status_code >= 500:
                last_exc = FeedError(f"HTTP {r.status_code} on {path}")
                self.stats["retries"] += 1
                self.sleep(min(2 ** attempt, 10))
                continue
            if r.status_code != 200:
                self.stats["errors"] += 1
                raise FeedError(f"HTTP {r.status_code} on {path}: {r.text[:200]}")
            try:
                return r.json()
            except ValueError:
                self.stats["errors"] += 1
                raise FeedError(f"malformed JSON from {path}")
        self.stats["errors"] += 1
        raise last_exc or FeedError(f"request to {path} failed")

    def server_time_ms(self) -> int:
        return int(self._get("/api/v3/time")["serverTime"])

    def klines(self, symbol: str, interval: str, limit: int = 500, now_ms: Optional[int] = None) -> pd.DataFrame:
        if interval not in INTERVAL_SECONDS:
            raise ValueError(f"unsupported interval {interval}")
        rows = self._get("/api/v3/klines", {"symbol": symbol, "interval": interval, "limit": int(min(limit, 1000))})
        df = binance_rows_to_frame(rows, interval, now_ms=now_ms if now_ms is not None else utc_now_ms())
        df.attrs.update({"source": "binance-rest", "symbol": symbol, "interval": interval,
                         "collected_utc": datetime.now(timezone.utc).isoformat()})
        return df

    def klines_range(self, symbol: str, interval: str, start_ms: int, end_ms: int) -> pd.DataFrame:
        """Completed candles with open time in [start_ms, end_ms) (used for backfill after a gap)."""
        rows = self._get("/api/v3/klines", {"symbol": symbol, "interval": interval, "startTime": start_ms,
                                            "endTime": end_ms - 1, "limit": 1000})
        return binance_rows_to_frame(rows, interval, now_ms=utc_now_ms())

    def book_ticker(self, symbol: str) -> Dict[str, Any]:
        d = self._get("/api/v3/ticker/bookTicker", {"symbol": symbol})
        return {"bid": float(d["bidPrice"]), "ask": float(d["askPrice"]), "bid_qty": float(d["bidQty"]),
                "ask_qty": float(d["askQty"])}

    def depth(self, symbol: str, limit: int = 100) -> Dict[str, Any]:
        d = self._get("/api/v3/depth", {"symbol": symbol, "limit": limit})
        return {"bids": [(float(p), float(q)) for p, q in d["bids"]],
                "asks": [(float(p), float(q)) for p, q in d["asks"]], "last_update_id": d.get("lastUpdateId")}

    def ticker_24h(self, symbol: str) -> Dict[str, Any]:
        d = self._get("/api/v3/ticker/24hr", {"symbol": symbol})
        return {"last": float(d["lastPrice"]), "volume": float(d["volume"]), "quote_volume": float(d["quoteVolume"]),
                "price_change_pct": float(d["priceChangePercent"]), "trades": int(d["count"])}


# --------------------------------------------------------------------------------- snapshot
@dataclass
class MarketSnapshot:
    source: str
    symbol: str
    received_ms: int
    bid: float
    ask: float
    bid_qty: float
    ask_qty: float
    exchange_ms: Optional[int] = None
    depth_bid_usd: Dict[str, float] = field(default_factory=dict)   # within N bps of mid
    depth_ask_usd: Dict[str, float] = field(default_factory=dict)
    # True when the fetched order book did not reach the band edge, i.e. the figure is only a lower bound
    depth_lower_bound: Dict[str, bool] = field(default_factory=dict)
    quote_volume_24h: Optional[float] = None

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2

    @property
    def spread_bps(self) -> float:
        return (self.ask - self.bid) / self.mid * 1e4

    def age_s(self, now_ms: Optional[int] = None) -> float:
        return ((now_ms if now_ms is not None else utc_now_ms()) - self.received_ms) / 1000

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d.update({"mid": self.mid, "spread_bps": self.spread_bps,
                  "received_utc": datetime.fromtimestamp(self.received_ms / 1000, timezone.utc).isoformat()})
        return d


def validate_quote(bid: float, ask: float) -> None:
    if not (bid > 0 and ask > 0):
        raise FeedError(f"non-positive quote bid={bid} ask={ask}")
    if bid >= ask:
        raise FeedError(f"crossed/locked book bid={bid} ask={ask}")


def depth_within(levels: List, mid: float, bps: float, side: str) -> float:
    lim = mid * (1 - bps / 1e4) if side == "bid" else mid * (1 + bps / 1e4)
    ok = (lambda p: p >= lim) if side == "bid" else (lambda p: p <= lim)
    return float(sum(p * q for p, q in levels if ok(p)))


def take_snapshot(client: BinancePublicClient, symbol: str, with_depth: bool = True,
                  depth_levels: int = 1000) -> MarketSnapshot:
    bt = client.book_ticker(symbol)
    validate_quote(bt["bid"], bt["ask"])
    snap = MarketSnapshot("binance-rest", symbol, utc_now_ms(), bt["bid"], bt["ask"], bt["bid_qty"], bt["ask_qty"])
    if with_depth:
        d = client.depth(symbol, depth_levels)
        for bps in (10, 50):
            snap.depth_bid_usd[f"{bps}bps"] = depth_within(d["bids"], snap.mid, bps, "bid")
            snap.depth_ask_usd[f"{bps}bps"] = depth_within(d["asks"], snap.mid, bps, "ask")
            far_bid = d["bids"][-1][0] if d["bids"] else snap.mid
            far_ask = d["asks"][-1][0] if d["asks"] else snap.mid
            snap.depth_lower_bound[f"{bps}bps"] = bool(far_bid > snap.mid * (1 - bps / 1e4)
                                                       or far_ask < snap.mid * (1 + bps / 1e4))
    return snap


# --------------------------------------------------------------------------------- candles
class CandleBuffer:
    """Completed candles for one (symbol, interval). Rejects malformed/out-of-order data; detects gaps."""

    def __init__(self, symbol: str, interval: str, maxlen: int = 2000):
        self.symbol, self.interval, self.maxlen = symbol, interval, maxlen
        self.step_ms = INTERVAL_SECONDS[interval] * 1000
        self.rows: List[Dict[str, Any]] = []
        self.stats = {"accepted": 0, "duplicates": 0, "rejected": 0, "gaps": 0, "missing_candles": 0}
        self.pending_gaps: List[tuple] = []   # (start_ms, end_ms) to backfill from the exchange

    @staticmethod
    def valid(c: Dict[str, Any]) -> bool:
        try:
            o, h, l, cl, v = (float(c[k]) for k in ("Open", "High", "Low", "Close", "Volume"))
        except (KeyError, TypeError, ValueError):
            return False
        return min(o, h, l, cl) > 0 and v >= 0 and h >= max(o, cl, l) and l <= min(o, cl, h)

    def add(self, c: Dict[str, Any]) -> str:
        if not self.valid(c) or int(c["created_at"]) % self.step_ms != 0:
            self.stats["rejected"] += 1
            return "rejected"
        t = int(c["created_at"])
        if self.rows:
            last = self.rows[-1]["created_at"]
            if t == last:
                self.stats["duplicates"] += 1
                return "duplicate"
            if t < last:
                if any(r["created_at"] == t for r in self.rows):
                    self.stats["duplicates"] += 1
                    return "duplicate"
                self.rows.append(self._norm(c))   # a backfilled candle filling a known gap
                self.rows.sort(key=lambda r: r["created_at"])
                self.stats["accepted"] += 1
                return "backfilled"
            if t - last > self.step_ms:
                missing = (t - last) // self.step_ms - 1
                self.stats["gaps"] += 1
                self.stats["missing_candles"] += int(missing)
                self.pending_gaps.append((last + self.step_ms, t))
        self.rows.append(self._norm(c))
        self.stats["accepted"] += 1
        if len(self.rows) > self.maxlen:
            self.rows = self.rows[-self.maxlen:]
        return "accepted"

    @staticmethod
    def _norm(c: Dict[str, Any]) -> Dict[str, Any]:
        out = {k: float(c[k]) for k in ("Open", "High", "Low", "Close", "Volume")}
        out["created_at"] = int(c["created_at"])
        out["QuoteVolume"] = float(c.get("QuoteVolume", 0.0) or 0.0)
        out["Trades"] = int(c.get("Trades", 0) or 0)
        return out

    def extend_frame(self, df: pd.DataFrame) -> None:
        for rec in df.to_dict("records"):
            self.add(rec)

    def missing_in_last(self, n: int) -> int:
        t = [r["created_at"] for r in self.rows[-n:]]
        return int(sum(max(0, (b - a) // self.step_ms - 1) for a, b in zip(t, t[1:])))

    def to_frame(self) -> pd.DataFrame:
        if not self.rows:
            return pd.DataFrame(columns=COLUMNS)
        df = pd.DataFrame(self.rows)
        df["timestamp"] = pd.to_datetime(df["created_at"], unit="ms", utc=True)
        return df[COLUMNS].reset_index(drop=True)


def parse_ws_kline(msg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Completed candle from a Binance kline stream message, or None (still forming / not a kline)."""
    k = msg.get("k") if isinstance(msg, dict) else None
    if not k or not k.get("x"):
        return None
    return {"created_at": int(k["t"]), "Open": k["o"], "High": k["h"], "Low": k["l"], "Close": k["c"],
            "Volume": k["v"], "QuoteVolume": k.get("q", 0), "Trades": k.get("n", 0), "symbol": k.get("s"),
            "interval": k.get("i"), "event_ms": msg.get("E")}


def parse_ws_book_ticker(msg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(msg, dict) or "b" not in msg or "a" not in msg or "s" not in msg:
        return None
    return {"symbol": msg["s"], "bid": float(msg["b"]), "ask": float(msg["a"]), "bid_qty": float(msg["B"]),
            "ask_qty": float(msg["A"])}


# --------------------------------------------------------------------------------- live market
class LiveMarket:
    """REST-first live market data with optional WebSocket streaming and health tracking."""

    def __init__(self, symbols: List[str], interval: str, client: Optional[BinancePublicClient] = None,
                 history: int = 600, max_quote_age_s: float = 15.0, use_websocket: bool = False):
        self.symbols, self.interval = list(symbols), interval
        self.client = client or BinancePublicClient()
        self.history, self.max_quote_age_s = history, max_quote_age_s
        self.buffers = {s: CandleBuffer(s, interval, maxlen=max(history, 2000)) for s in symbols}
        self.quotes: Dict[str, MarketSnapshot] = {}
        self.health: Dict[str, Any] = {"ws_messages": 0, "ws_reconnects": 0, "ws_malformed": 0, "ws_last_ms": None,
                                       "rest_backfills": 0, "stale_quotes": 0}
        self._ws: Optional["BinanceStream"] = None
        if use_websocket:
            self._ws = BinanceStream(symbols, interval, self._on_ws)
            self._ws.start()

    # REST refresh of completed candles (also fills gaps from the exchange)
    def refresh_candles(self, symbol: str) -> pd.DataFrame:
        buf = self.buffers[symbol]
        df = self.client.klines(symbol, self.interval, limit=min(self.history, 1000))
        buf.extend_frame(df)
        while buf.pending_gaps:
            start, end = buf.pending_gaps.pop(0)
            buf.extend_frame(self.client.klines_range(symbol, self.interval, start, end))
            self.health["rest_backfills"] += 1
        return buf.to_frame()

    def completed_klines(self, symbol: str, interval: Optional[str] = None, limit: Optional[int] = None) -> pd.DataFrame:
        if interval and interval != self.interval:
            raise ValueError(f"LiveMarket configured for {self.interval}, asked for {interval}")
        df = self.refresh_candles(symbol)
        out = df.tail(limit or self.history).reset_index(drop=True)
        out.attrs.update({"source": "binance-live", "symbol": symbol, "interval": self.interval})
        return out

    def snapshot(self, symbol: str, with_depth: bool = True) -> MarketSnapshot:
        ws_q = self.quotes.get(symbol)
        if ws_q is not None and ws_q.age_s() <= self.max_quote_age_s and not with_depth:
            return ws_q
        snap = take_snapshot(self.client, symbol, with_depth)
        self.quotes[symbol] = snap
        return snap

    def quote(self, symbol: str) -> MarketSnapshot:
        """Current best bid/ask; raises StaleDataError rather than returning an old price."""
        snap = self.snapshot(symbol, with_depth=False)
        if snap.age_s() > self.max_quote_age_s:
            self.health["stale_quotes"] += 1
            raise StaleDataError(f"{symbol} quote is {snap.age_s():.1f}s old")
        return snap

    def _on_ws(self, kind: str, payload: Dict[str, Any]) -> None:
        self.health["ws_messages"] += 1
        self.health["ws_last_ms"] = utc_now_ms()
        if kind == "kline" and payload["symbol"] in self.buffers:
            self.buffers[payload["symbol"]].add(payload)
        elif kind == "quote":
            try:
                validate_quote(payload["bid"], payload["ask"])
            except FeedError:
                self.health["ws_malformed"] += 1
                return
            self.quotes[payload["symbol"]] = MarketSnapshot("binance-ws", payload["symbol"], utc_now_ms(),
                                                           payload["bid"], payload["ask"], payload["bid_qty"],
                                                           payload["ask_qty"])
        elif kind == "malformed":
            self.health["ws_malformed"] += 1
        elif kind == "reconnect":
            self.health["ws_reconnects"] += 1

    def feed_health(self) -> Dict[str, Any]:
        h = dict(self.health)
        h["buffers"] = {s: dict(b.stats, rows=len(b.rows)) for s, b in self.buffers.items()}
        h["rest"] = dict(self.client.stats)
        if self._ws:
            h["ws_alive"] = self._ws.alive()
        return h

    def close(self) -> None:
        if self._ws:
            self._ws.stop()


class BinanceStream:
    """Background WebSocket reader with stale detection and reconnect (exponential back-off)."""

    def __init__(self, symbols: List[str], interval: str, callback: Callable[[str, Dict[str, Any]], None],
                 stale_after_s: float = 30.0, max_backoff_s: float = 60.0):
        streams = "/".join(f"{s.lower()}@kline_{interval}/{s.lower()}@bookTicker" for s in symbols)
        self.url = WS_BASE + streams
        self.callback, self.stale_after_s, self.max_backoff_s = callback, stale_after_s, max_backoff_s
        self._running = False
        self._last_msg = 0.0
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=lambda: asyncio.run(self._run()), daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False

    def alive(self) -> bool:
        return self._running and (time.monotonic() - self._last_msg) < self.stale_after_s

    def handle_raw(self, raw: str) -> None:
        try:
            msg = json.loads(raw)
            data = msg.get("data", msg)
        except (ValueError, AttributeError):
            self.callback("malformed", {})
            return
        self._last_msg = time.monotonic()
        k = parse_ws_kline(data) if isinstance(data, dict) and data.get("e") == "kline" else None
        if k:
            self.callback("kline", k)
            return
        q = parse_ws_book_ticker(data)
        if q:
            self.callback("quote", q)
        elif not (isinstance(data, dict) and data.get("e") == "kline"):
            self.callback("malformed", {})

    async def _run(self) -> None:
        import websockets
        backoff = 1.0
        while self._running:
            try:
                async with websockets.connect(self.url, ping_interval=20, ping_timeout=20) as ws:
                    backoff = 1.0
                    while self._running:
                        raw = await asyncio.wait_for(ws.recv(), timeout=self.stale_after_s)
                        self.handle_raw(raw)
            except Exception as exc:  # stale (timeout), disconnect, network
                if not self._running:
                    break
                logger.warning("Binance stream reconnecting after %s", type(exc).__name__)
                self.callback("reconnect", {})
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, self.max_backoff_s)
