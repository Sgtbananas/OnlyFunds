"""CoinEx client tests. All HTTP is mocked: nothing reaches the network."""
import hashlib
import hmac
import json

import pytest
import requests

from core import coinex_client as cx


class FakeResp:
    def __init__(self, payload=None, status=200, text=None):
        self.status_code = status
        self._payload = payload
        self.text = text if text is not None else json.dumps(payload)

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def request(self, method, url, data=None, headers=None, timeout=None):
        self.calls.append({"method": method, "url": url, "data": data, "headers": headers})
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def ok(data):
    return FakeResp({"code": 0, "data": data, "message": "OK"})


MARKET = {"market": "BTCUSDT", "base_ccy": "BTC", "quote_ccy": "USDT", "base_ccy_precision": 8,
          "quote_ccy_precision": 2, "min_amount": "0.0005", "maker_fee_rate": "0.002", "taker_fee_rate": "0.002",
          "status": "online", "is_api_trading_available": True}


def client(responses, **kw):
    kw.setdefault("api_key", "KEY")
    kw.setdefault("api_secret", "SECRET")
    return cx.CoinExClient(session=FakeSession(responses), min_interval=0, backoff=0, **kw)


def test_signature_matches_documented_construction(monkeypatch):
    monkeypatch.setattr(cx.time, "time", lambda: 1700490703.564)
    c = client([ok([{"ccy": "USDT", "available": "10.5", "frozen": "1"}])])
    bal = c.get_spot_balance()
    assert bal == {"USDT": {"available": 10.5, "frozen": 1.0}}
    call = c.session.calls[0]
    h = call["headers"]
    assert call["url"] == "https://api.coinex.com/v2/assets/spot/balance"
    assert h["X-COINEX-KEY"] == "KEY" and h["X-COINEX-TIMESTAMP"] == "1700490703564"
    expected = hmac.new(b"SECRET", b"GET/v2/assets/spot/balance1700490703564", hashlib.sha256).hexdigest()
    assert h["X-COINEX-SIGN"] == expected and len(expected) == 64


def test_signature_includes_query_and_body():
    s = cx.sign_request("s", "post", "/v2/spot/order?x=1", '{"a":1}', "123")
    assert s == hmac.new(b"s", b'POST/v2/spot/order?x=1{"a":1}123', hashlib.sha256).hexdigest()


def test_signed_request_without_credentials_fails():
    c = cx.CoinExClient(api_key="", api_secret="", session=FakeSession([]), min_interval=0)
    with pytest.raises(cx.CoinExAuthError):
        c.get_spot_balance()
    assert c.session.calls == []


def test_secrets_never_in_repr():
    c = client([])
    assert "SECRET" not in repr(c) and "KEY" not in repr(c)


def test_klines_parsed_sorted_and_unclosed_dropped():
    rows = [{"market": "BTCUSDT", "created_at": t, "open": "1", "high": "2", "low": "0.5", "close": "1.5",
             "volume": "3", "value": "4"} for t in (600_000, 0, 300_000)]
    c = client([ok(rows)])
    df = c.get_klines("BTCUSDT", "5m", limit=3, now_ms=700_000)  # the 600000 candle closes at 900000
    assert list(df["created_at"]) == [0, 300_000]
    assert df["Close"].dtype == float
    url = c.session.calls[0]["url"]
    assert "period=5min" in url and "limit=3" in url and "market=BTCUSDT" in url


def test_kline_limit_clamped_and_interval_validated():
    c = client([ok([])])
    c.get_klines("BTCUSDT", "1h", limit=5000)
    assert "limit=1000" in c.session.calls[0]["url"]
    with pytest.raises(ValueError):
        c.get_klines("BTCUSDT", "7m")


def test_error_codes_mapped():
    with pytest.raises(cx.CoinExAuthError):
        client([FakeResp({"code": 4006, "message": "sig"})]).get_spot_balance()
    with pytest.raises(cx.CoinExError) as e:
        client([FakeResp({"code": 3639, "message": "bad"})]).get_tickers()
    assert e.value.code == 3639
    with pytest.raises(cx.CoinExError):
        client([FakeResp(None, status=502, text="gateway")] * 3).get_tickers()


def test_rate_limit_and_busy_are_retried():
    c = client([FakeResp({"code": 4213, "message": "too frequent"}), FakeResp({"code": 3008, "message": "busy"}),
                ok([{"market": "BTCUSDT", "last": "1"}])])
    assert c.get_tickers() == [{"market": "BTCUSDT", "last": "1"}]
    assert len(c.session.calls) == 3


def test_network_errors_retried_then_raised():
    c = client([requests.ConnectionError("down")] * 3)
    with pytest.raises(cx.CoinExNetworkError):
        c.get_markets()
    assert len(c.session.calls) == 3


def test_market_info_precision():
    c = client([ok([MARKET])])
    m = c.get_market("BTCUSDT")
    assert m.tradable and m.min_amount == 0.0005
    assert m.floor_amount(0.123456789) == 0.12345678
    assert m.round_price(101.239) == 101.24
    with pytest.raises(cx.CoinExError):
        c.get_market("NOPEUSDT")


def test_trading_locked_by_default():
    c = client([])
    with pytest.raises(cx.TradingLockedError):
        c.place_order("BTCUSDT", "buy", "market", 10)
    with pytest.raises(cx.TradingLockedError):
        c.cancel_order("BTCUSDT", 1)
    assert c.session.calls == []


def test_order_request_shape_when_unlocked_in_test_only():
    """Mocked: verifies the documented body; no request leaves the process."""
    c = client([ok([MARKET]), ok({"order_id": 13400, "market": "BTCUSDT"})], allow_trading=True)
    res = c.place_order("BTCUSDT", "buy", "limit", 0.001, price=10000.123, client_id="of2-abc")
    assert res["order_id"] == 13400
    call = c.session.calls[1]
    assert call["method"] == "POST" and call["url"].endswith("/v2/spot/order")
    body = json.loads(call["data"])
    assert body == {"market": "BTCUSDT", "market_type": "SPOT", "side": "buy", "type": "limit",
                    "amount": "0.001", "price": "10000.12", "client_id": "of2-abc"}


def test_order_network_failure_is_not_retried():
    c = client([ok([MARKET]), requests.Timeout("t"), ok({})], allow_trading=True)
    with pytest.raises(cx.OrderStateUnknown):
        c.place_order("BTCUSDT", "sell", "market", 0.001, client_id="of2-x")
    assert len(c.session.calls) == 2  # markets + exactly one order attempt


def test_order_validation():
    c = client([ok([MARKET])], allow_trading=True)
    with pytest.raises(ValueError):
        c.place_order("BTCUSDT", "hold", "market", 1)
    with pytest.raises(ValueError):
        c.place_order("BTCUSDT", "buy", "limit", 1)


def test_top_usdt_markets_ranks_by_quote_volume():
    markets = [MARKET, {**MARKET, "market": "ETHUSDT", "base_ccy": "ETH"},
               {**MARKET, "market": "ETHBTC", "base_ccy": "ETH", "quote_ccy": "BTC"},
               {**MARKET, "market": "BTC3LUSDT", "base_ccy": "BTC3L"},
               {**MARKET, "market": "OFFUSDT", "base_ccy": "OFF", "status": "counting_down"}]
    tickers = [{"market": "BTCUSDT", "value": "100"}, {"market": "ETHUSDT", "value": "500"},
               {"market": "ETHBTC", "value": "9999"}, {"market": "BTC3LUSDT", "value": "9999"},
               {"market": "OFFUSDT", "value": "9999"}]
    c = client([ok(markets), ok(tickers)])
    assert cx.top_usdt_markets(c, limit=5) == ["ETHUSDT", "BTCUSDT"]


def test_depth_limit_validated():
    with pytest.raises(ValueError):
        client([]).get_depth("BTCUSDT", limit=7)


def test_format_decimal():
    assert cx.format_decimal(0.00010000) == "0.0001"
    assert cx.format_decimal(5.0) == "5"
    assert cx.format_decimal(10000.12) == "10000.12"
    assert cx.format_decimal(1e-05) == "0.00001"
    assert cx.format_decimal(0.1 + 0.2) == "0.30000000000000004"  # exact float value, never rounded up


def test_order_amount_floored_and_minimum_enforced():
    c = client([ok([MARKET]), ok({"order_id": 1})], allow_trading=True)
    c.place_order("BTCUSDT", "sell", "market", 0.0012345678912)
    assert json.loads(c.session.calls[1]["data"])["amount"] == "0.00123456"
    c2 = client([ok([MARKET])], allow_trading=True)
    with pytest.raises(cx.CoinExError):
        c2.place_order("BTCUSDT", "sell", "market", 0.0001)
    assert len(c2.session.calls) == 1


def test_gateway_html_error_is_readable():
    html = "<html><head><title>503 Service Temporarily Unavailable</title></head><body>x</body></html>"
    with pytest.raises(cx.CoinExError) as e:
        client([FakeResp(None, status=503, text=html)] * 3).get_markets()
    assert "503 Service Temporarily Unavailable" in str(e.value) and "<html>" not in str(e.value)
