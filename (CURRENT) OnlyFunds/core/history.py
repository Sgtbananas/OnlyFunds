"""Historical OHLCV collection from exchange public APIs, with provenance and quality checks.

Primary source: Binance spot klines (GET https://api.binance.com/api/v3/klines; max 1000 candles
per request; columns: open time, open, high, low, close, volume, close time, quote volume, trades,
taker buy base, taker buy quote, ignore). Secondary source for cross-checks: OKX
(GET https://www.okx.com/api/v5/market/history-candles).

Rules:
- Only candles the exchange returned are stored. Missing candles are **reported, never filled**.
- Every dataset gets a manifest: source URL, exchange, symbol, interval, download time (UTC),
  coverage, row count, SHA-256 of the CSV, and the quality report.
- Data from Binance/OKX says nothing about CoinEx prices, liquidity or fees.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import requests

from core.coinex_client import INTERVAL_SECONDS

logger = logging.getLogger(__name__)

APP_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
HISTORY_DIR = os.path.join(APP_ROOT, "data", "historical")
BINANCE_URL = "https://api.binance.com/api/v3/klines"
OKX_URL = "https://www.okx.com/api/v5/market/history-candles"
BINANCE_LIMIT = 1000
OKX_BAR = {"1h": "1H", "4h": "4H", "1d": "1Dutc", "15m": "15m", "5m": "5m"}

# Large, liquid USDT pairs as of Jan 2019 (start of the data), not today's winners: includes
# coins that later faded or were delisted (EOS) to limit survivorship bias.
DEFAULT_UNIVERSE = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "LTCUSDT", "ADAUSDT", "EOSUSDT", "XLMUSDT",
                    "TRXUSDT"]
COLUMNS = ["timestamp", "created_at", "Open", "High", "Low", "Close", "Volume", "QuoteVolume", "Trades"]


def _ms(ts: str) -> int:
    return int(pd.Timestamp(ts, tz="UTC").value // 10**6)


def fetch_binance(symbol: str, interval: str, start: str, end: str,
                  get: Callable = requests.get, pause: float = 0.25, max_retries: int = 4) -> pd.DataFrame:
    """All completed candles with open time in [start, end), oldest first."""
    if interval not in INTERVAL_SECONDS:
        raise ValueError(f"unsupported interval {interval}")
    step = INTERVAL_SECONDS[interval] * 1000
    cursor, end_ms = _ms(start), _ms(end)
    rows: List[list] = []
    while cursor < end_ms:
        params = {"symbol": symbol, "interval": interval, "startTime": cursor, "endTime": end_ms - 1,
                  "limit": BINANCE_LIMIT}
        for attempt in range(max_retries):
            try:
                r = get(BINANCE_URL, params=params, timeout=20)
                if r.status_code in (418, 429) or r.status_code >= 500:
                    raise requests.HTTPError(f"HTTP {r.status_code}")
                if r.status_code != 200:
                    raise RuntimeError(f"Binance HTTP {r.status_code}: {r.text[:200]}")
                batch = r.json()
                break
            except (requests.RequestException, ValueError) as exc:
                if attempt + 1 == max_retries:
                    raise RuntimeError(f"Binance request failed for {symbol}: {exc}")
                time.sleep(2 ** attempt)
        if not batch:
            break
        rows.extend(batch)
        last_open = int(batch[-1][0])
        if last_open + step <= cursor:  # no progress: defensive stop
            break
        cursor = last_open + step
        if len(batch) < BINANCE_LIMIT:
            break
        time.sleep(pause)
    return binance_rows_to_frame(rows, interval, now_ms=int(time.time() * 1000))


def binance_rows_to_frame(rows: List[list], interval: str, now_ms: Optional[int] = None) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=COLUMNS)
    df = pd.DataFrame(rows).iloc[:, :9]
    df.columns = ["created_at", "Open", "High", "Low", "Close", "Volume", "close_time", "QuoteVolume", "Trades"]
    for c in ("Open", "High", "Low", "Close", "Volume", "QuoteVolume"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["created_at"] = df["created_at"].astype("int64")
    df["Trades"] = pd.to_numeric(df["Trades"], errors="coerce").astype("Int64")
    if now_ms is not None:  # drop a still-forming candle
        df = df[df["created_at"] + INTERVAL_SECONDS[interval] * 1000 <= now_ms]
    df["timestamp"] = pd.to_datetime(df["created_at"], unit="ms", utc=True)
    return df[COLUMNS].reset_index(drop=True)


def fetch_okx_closes(symbol: str, interval: str, start: str, end: str, get: Callable = requests.get,
                     pause: float = 0.12) -> pd.Series:
    """Close prices from OKX for a cross-exchange sanity check (pages backwards, 100 per call)."""
    inst = symbol.replace("USDT", "-USDT")
    after, start_ms = _ms(end), _ms(start)
    out: Dict[int, float] = {}
    while after > start_ms:
        r = get(OKX_URL, params={"instId": inst, "bar": OKX_BAR[interval], "after": after, "limit": 100},
                timeout=20)
        data = r.json().get("data", []) if r.status_code == 200 else []
        if not data:
            break
        for row in data:
            if int(row[8]) == 1:  # confirmed candle
                out[int(row[0])] = float(row[4])
        after = min(int(row[0]) for row in data)
        time.sleep(pause)
    s = pd.Series(out, dtype=float).sort_index()
    return s[s.index >= start_ms]


# ------------------------------------------------------------------ quality
def quality_report(df: pd.DataFrame, interval: str) -> Dict[str, Any]:
    """Validate a candle table. Nothing is modified; problems are counted and listed."""
    step = INTERVAL_SECONDS[interval] * 1000
    rep: Dict[str, Any] = {"rows": int(len(df))}
    if df.empty:
        rep["ok"] = False
        rep["problems"] = ["no data"]
        return rep
    t = df["created_at"].to_numpy("int64")
    rep["first_candle"] = str(pd.Timestamp(t.min(), unit="ms", tz="UTC"))
    rep["last_candle"] = str(pd.Timestamp(t.max(), unit="ms", tz="UTC"))
    expected = int((t.max() - t.min()) // step) + 1
    rep["expected_rows"] = expected
    rep["duplicates"] = int(len(t) - len(np.unique(t)))
    rep["unsorted"] = bool((np.diff(t) <= 0).any())
    rep["misaligned"] = int((t % step != 0).sum())
    d = np.diff(np.unique(t))
    gaps = d[d > step]
    rep["missing_candles"] = int(((gaps // step) - 1).sum())
    rep["gap_count"] = int(len(gaps))
    order = np.argsort(t)
    gap_idx = np.where(np.diff(t[order]) > step)[0]
    largest = sorted(((int((t[order][i + 1] - t[order][i]) // step) - 1, i) for i in gap_idx), reverse=True)[:10]
    rep["largest_gaps"] = [{"after": str(pd.Timestamp(t[order][i], unit="ms", tz="UTC")), "missing": n}
                           for n, i in largest]
    o, h, l, c, v = (df[k].to_numpy(float) for k in ("Open", "High", "Low", "Close", "Volume"))
    rep["nan_values"] = int(np.isnan(np.c_[o, h, l, c, v]).sum())
    rep["non_positive_prices"] = int(((np.c_[o, h, l, c]) <= 0).sum())
    rep["high_below_open_close_or_low"] = int((h < np.maximum(np.maximum(o, c), l) - 1e-12).sum())
    rep["low_above_open_close_or_high"] = int((l > np.minimum(np.minimum(o, c), h) + 1e-12).sum())
    rep["negative_volume"] = int((v < 0).sum())
    rep["zero_volume_candles"] = int((v == 0).sum())
    cs = c[order]
    ret = np.abs(np.diff(np.log(np.where(cs > 0, cs, np.nan))))  # invalid prices are counted above, not logged
    rep["max_abs_hourly_log_return"] = float(np.nanmax(ret)) if np.isfinite(ret).any() else 0.0
    rep["extreme_moves_over_25pct"] = int((np.nan_to_num(ret) > np.log(1.25)).sum())
    problems = [k for k in ("duplicates", "misaligned", "nan_values", "non_positive_prices",
                            "high_below_open_close_or_low", "low_above_open_close_or_high", "negative_volume")
                if rep[k]]
    if rep["unsorted"]:
        problems.append("unsorted")
    rep["problems"] = problems          # hard errors: dataset rejected
    rep["warnings"] = [k for k in ("missing_candles", "zero_volume_candles", "extreme_moves_over_25pct") if rep[k]]
    rep["ok"] = not problems
    return rep


def cross_check(primary: pd.DataFrame, secondary_close: pd.Series, tolerance: float = 0.01) -> Dict[str, Any]:
    """Compare closes of overlapping candles from two exchanges."""
    p = primary.set_index("created_at")["Close"]
    common = p.index.intersection(secondary_close.index)
    if not len(common):
        return {"overlap": 0}
    dev = (p[common] / secondary_close[common] - 1).abs()
    return {"overlap": int(len(common)), "median_abs_deviation": float(dev.median()),
            "p99_abs_deviation": float(dev.quantile(0.99)), "max_abs_deviation": float(dev.max()),
            "share_over_tolerance": float((dev > tolerance).mean()), "tolerance": tolerance}


# ------------------------------------------------------------------ storage
def dataset_paths(symbol: str, interval: str, exchange: str = "binance", root: Optional[str] = None):
    base = os.path.join(root or HISTORY_DIR, exchange)
    return os.path.join(base, f"{symbol}_{interval}.csv"), os.path.join(base, f"{symbol}_{interval}.manifest.json")


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# Cost assumptions recorded with every dataset: published exchange facts are labelled as such and
# separated from modelling assumptions. None of this is verified for a specific account.
COST_ASSUMPTIONS = {
    "binance": {
        "fee_rate_per_side": {"value": 0.001, "status": "published Binance base spot fee (no VIP/BNB discount); "
                                                       "not verified for any account"},
        "slippage_pct": {"value": 0.0005, "status": "assumption, not measured"},
        "stop_slippage_pct": {"value": 0.002, "status": "assumption, not measured"},
    },
    "okx": {
        "fee_rate_per_side": {"value": 0.001, "status": "published OKX base spot taker fee; not verified"},
        "slippage_pct": {"value": 0.0005, "status": "assumption, not measured"},
        "stop_slippage_pct": {"value": 0.002, "status": "assumption, not measured"},
    },
    "evaluation_default": {
        "fee_rate_per_side": {"value": 0.002, "status": "CoinEx default taker fee as shown in the CoinEx API v2 "
                                                       "documentation example; used for all OF2 evaluations; "
                                                       "not verified for any account"},
    },
}


class DatasetExists(RuntimeError):
    pass


def save_dataset(df: pd.DataFrame, symbol: str, interval: str, request: Dict[str, Any],
                 exchange: str = "binance", root: Optional[str] = None,
                 extra: Optional[Dict[str, Any]] = None, overwrite: bool = False) -> Dict[str, Any]:
    """Write a dataset and its manifest. Refuses to replace an existing dataset unless ``overwrite``:
    validation runs reference datasets by checksum, so silent replacement would break provenance."""
    csv_path, man_path = dataset_paths(symbol, interval, exchange, root)
    if not overwrite and (os.path.exists(man_path) or os.path.exists(csv_path)):
        raise DatasetExists(f"{csv_path} already exists; refusing to replace it (use an explicit refresh)")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    out = df.copy()
    out["timestamp"] = out["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    out.to_csv(csv_path, index=False, lineterminator="\n")
    manifest = {
        "exchange": exchange, "symbol": symbol, "interval": interval,
        "source_url": BINANCE_URL if exchange == "binance" else OKX_URL,
        "request": request, "downloaded_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rows": int(len(df)), "sha256": sha256_file(csv_path), "quality": quality_report(df, interval),
        "note": "Exchange-sourced candles; missing candles are not filled. Not CoinEx data.",
        "cost_assumptions": {"exchange": COST_ASSUMPTIONS.get(exchange),
                             "evaluation_default": COST_ASSUMPTIONS["evaluation_default"]},
    }
    if extra:
        manifest.update(extra)
    with open(man_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)
    return manifest


def load_dataset(symbol: str, interval: str = "1h", exchange: str = "binance", root: Optional[str] = None,
                 verify: bool = True) -> pd.DataFrame:
    """Load a stored dataset; with ``verify`` the CSV must match its manifest checksum."""
    csv_path, man_path = dataset_paths(symbol, interval, exchange, root)
    if verify:
        with open(man_path, encoding="utf-8") as f:
            man = json.load(f)
        if sha256_file(csv_path) != man["sha256"]:
            raise ValueError(f"{csv_path} does not match its manifest checksum (modified or corrupted)")
    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["created_at"] = df["created_at"].astype("int64")
    return df
