"""Live data layer tests (no network: fake HTTP session, Binance-format messages)."""
import json

import numpy as np
import pandas as pd
import pytest

from core import live_data as L
from core.core_signals import generate_signals
from core.history import binance_rows_to_frame
from tests.synthetic import synthetic_regime

H = 3_600_000


class Resp:
    def __init__(self, payload, status=200, headers=None):
        self._p, self.status_code, self.headers = payload, status, headers or {}
        self.text = json.dumps(payload) if payload is not None else "x"

    def json(self):
        if self._p is None:
            raise ValueError("bad json")
        return self._p


class Session:
    def __init__(self, responses):
        self.responses, self.calls = list(responses), []

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, params))
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def client(responses):
    return L.BinancePublicClient(session=Session(responses), sleep=lambda s: None, min_interval=0)


def kline_row(t, o=100.0, h=101.0, l=99.0, c=100.5, v=10.0):
    return [t, str(o), str(h), str(l), str(c), str(v), t + H - 1, str(v * c), 5, "0", "0", "0"]


# ------------------------------------------------------------------ REST
def test_klines_drop_forming_candle_and_use_history_schema():
    now = 10 * H + 5
    rows = [kline_row(k * H) for k in range(11)]          # last one (10H) still forming at now
    c = client([Resp(rows)])
    df = c.klines("BTCUSDT", "1h", 11, now_ms=now)
    assert len(df) == 10 and df["created_at"].iloc[-1] == 9 * H
    assert list(df.columns) == list(binance_rows_to_frame(rows, "1h").columns)
    assert c.session.calls[0][0].endswith("/api/v3/klines")


def test_rate_limit_retry_after_and_weight_tracking():
    slept = []
    c = L.BinancePublicClient(session=Session([Resp({}, 429, {"Retry-After": "7"}),
                                               Resp({"serverTime": 123}, 200, {"X-MBX-USED-WEIGHT-1M": "42"})]),
                              sleep=slept.append, min_interval=0)
    assert c.server_time_ms() == 123
    assert 7.0 in slept and c.used_weight == 42 and c.stats["rate_limited"] == 1


def test_transient_errors_retried_then_raised():
    import requests
    c = client([requests.ConnectionError("x"), Resp({}, 503), Resp({}, 502)])
    with pytest.raises(L.FeedError):
        c.server_time_ms()
    assert c.stats["retries"] == 3


def test_malformed_and_client_errors_raise():
    with pytest.raises(L.FeedError):
        client([Resp(None)]).server_time_ms()
    with pytest.raises(L.FeedError):
        client([Resp({"code": -1121, "msg": "Invalid symbol."}, 400)]).book_ticker("NOPE")


def test_snapshot_spread_and_depth():
    c = client([Resp({"bidPrice": "99.9", "askPrice": "100.1", "bidQty": "2", "askQty": "3"}),
                Resp({"lastUpdateId": 1, "bids": [["99.9", "1"], ["99.0", "5"]], "asks": [["100.1", "1"], ["101", "5"]]})])
    s = L.take_snapshot(c, "BTCUSDT")
    assert s.mid == pytest.approx(100.0) and s.spread_bps == pytest.approx(20.0)
    assert s.depth_bid_usd["10bps"] == pytest.approx(99.9) and s.depth_bid_usd["50bps"] == pytest.approx(99.9)
    assert s.depth_ask_usd["50bps"] == pytest.approx(100.1)
    assert s.depth_lower_bound == {"10bps": False, "50bps": False}  # fetched book reaches 1% on both sides


def test_depth_flagged_as_lower_bound_when_book_too_short():
    c = client([Resp({"bidPrice": "99.99", "askPrice": "100.01", "bidQty": "2", "askQty": "3"}),
                Resp({"lastUpdateId": 1, "bids": [["99.99", "1"], ["99.95", "5"]], "asks": [["100.01", "1"], ["100.05", "5"]]})])
    s = L.take_snapshot(c, "BTCUSDT")
    assert s.depth_lower_bound == {"10bps": True, "50bps": True}


@pytest.mark.parametrize("bid,ask", [(0, 1), (101, 100), (100, 100)])
def test_invalid_quotes_rejected(bid, ask):
    with pytest.raises(L.FeedError):
        L.validate_quote(bid, ask)


def test_stale_quote_is_never_returned_as_current():
    lm = L.LiveMarket(["BTCUSDT"], "1h", client=client([]), max_quote_age_s=5)
    lm.quotes["BTCUSDT"] = L.MarketSnapshot("binance-ws", "BTCUSDT", L.utc_now_ms() - 60_000, 99, 101, 1, 1)
    lm.client = client([Resp({"bidPrice": "99", "askPrice": "101", "bidQty": "1", "askQty": "1"})])
    assert lm.quote("BTCUSDT").age_s() < 5          # refreshed from REST, not the 60s-old WS quote
    lm.snapshot = lambda s, with_depth=True: L.MarketSnapshot("x", s, L.utc_now_ms() - 60_000, 99, 101, 1, 1)
    with pytest.raises(L.StaleDataError):
        lm.quote("BTCUSDT")


# ------------------------------------------------------------------ candle buffer
def cand(t, **kw):
    d = {"created_at": t, "Open": 100, "High": 101, "Low": 99, "Close": 100.5, "Volume": 1}
    d.update(kw)
    return d


def test_buffer_rejects_malformed_duplicate_and_detects_gaps():
    b = L.CandleBuffer("X", "1h")
    assert b.add(cand(0)) == "accepted"
    assert b.add(cand(0)) == "duplicate"
    assert b.add(cand(H, High=98)) == "rejected"          # high below close
    assert b.add(cand(H + 5)) == "rejected"               # misaligned timestamp
    assert b.add(cand(H, Close=-1)) == "rejected"
    assert b.add(cand(4 * H)) == "accepted"               # 3 candles missing
    assert b.stats["missing_candles"] == 3 and b.stats["gaps"] == 1
    assert b.pending_gaps == [(H, 4 * H)] and b.missing_in_last(10) == 3
    for t in (H, 2 * H, 3 * H):
        assert b.add(cand(t)) == "backfilled"
    assert b.missing_in_last(10) == 0 and list(b.to_frame()["created_at"]) == [0, H, 2 * H, 3 * H, 4 * H]


def test_live_market_backfills_gaps_from_exchange_only():
    rows = [kline_row(k * H) for k in range(20)]
    without = [r for r in rows if r[0] not in (5 * H, 6 * H)]
    c = client([Resp(without), Resp([kline_row(5 * H), kline_row(6 * H)])])
    lm = L.LiveMarket(["BTCUSDT"], "1h", client=c, history=50)
    df = lm.refresh_candles("BTCUSDT")
    assert list(df["created_at"]) == [k * H for k in range(20)]
    assert lm.health["rest_backfills"] == 1
    assert c.session.calls[1][1]["startTime"] == 5 * H and c.session.calls[1][1]["endTime"] == 7 * H - 1


# ------------------------------------------------------------------ websocket messages
def test_ws_messages_parsed_completed_only():
    got = []
    s = L.BinanceStream(["BTCUSDT"], "1m", lambda kind, p: got.append((kind, p)))
    forming = {"stream": "btcusdt@kline_1m", "data": {"e": "kline", "E": 1, "s": "BTCUSDT", "k": {
        "t": 60_000, "T": 119_999, "s": "BTCUSDT", "i": "1m", "o": "1", "c": "2", "h": "2", "l": "1", "v": "3",
        "n": 4, "x": False, "q": "5"}}}
    closed = json.loads(json.dumps(forming))
    closed["data"]["k"]["x"] = True
    quote = {"stream": "btcusdt@bookTicker", "data": {"u": 1, "s": "BTCUSDT", "b": "99", "B": "1", "a": "100", "A": "2"}}
    for m in (forming, closed, quote):
        s.handle_raw(json.dumps(m))
    s.handle_raw("{not json")
    kinds = [k for k, _ in got]
    assert kinds == ["kline", "quote", "malformed"]
    assert got[0][1]["created_at"] == 60_000 and got[1][1]["ask"] == 100.0
    assert "btcusdt@kline_1m" in s.url and "btcusdt@bookTicker" in s.url


def test_ws_callbacks_update_market_and_health():
    lm = L.LiveMarket(["BTCUSDT"], "1h", client=client([]))
    lm._on_ws("kline", {**cand(0), "symbol": "BTCUSDT"})
    lm._on_ws("quote", {"symbol": "BTCUSDT", "bid": 99.0, "ask": 100.0, "bid_qty": 1.0, "ask_qty": 1.0})
    lm._on_ws("quote", {"symbol": "BTCUSDT", "bid": 101.0, "ask": 100.0, "bid_qty": 1.0, "ask_qty": 1.0})  # crossed
    lm._on_ws("reconnect", {})
    h = lm.feed_health()
    assert h["buffers"]["BTCUSDT"]["rows"] == 1 and h["ws_malformed"] == 1 and h["ws_reconnects"] == 1
    assert lm.quotes["BTCUSDT"].bid == 99.0


# ------------------------------------------------------------------ historical vs live consistency
def test_same_completed_candles_give_same_signals_in_both_pipelines():
    hist = synthetic_regime("switching", 900, seed=3, interval="1h")
    live = L.CandleBuffer("SYN", "1h", maxlen=5000)
    for rec in hist.to_dict("records"):                  # candles arriving one by one
        live.add(rec)
    lf = live.to_frame()
    pd.testing.assert_frame_equal(lf[["created_at", "Open", "High", "Low", "Close", "Volume"]],
                                  hist[["created_at", "Open", "High", "Low", "Close", "Volume"]].astype(
                                      {"Volume": float}), check_dtype=False)
    a, b = generate_signals(hist), generate_signals(lf)
    for col in ("signal", "confidence", "entry", "exit", "regime", "strategy", "ATR", "rsi"):
        assert (a[col].astype(str) == b[col].astype(str)).all() if a[col].dtype == object else \
            np.allclose(a[col].astype(float), b[col].astype(float), equal_nan=True), col
