"""Second-pass audit: risk limits verified independently, across several simultaneous positions."""
from datetime import timedelta

import pytest

from core.coinex_client import MarketInfo
from core.paper_trader import PaperSettings, PaperTrader
from core.risk_manager import RiskLimits, RiskManager
from core.trade_execution import PaperExecutor
from tests.synthetic import synthetic_ohlcv
from tests.test_paper_trader import no_network  # noqa: F401
from utils.config import Config, build_settings


class MultiMarket:
    def __init__(self, pairs=("AAA", "BBB", "CCC", "DDD"), n=2500, start_bar=400):
        self.dfs = {p: synthetic_ohlcv(n, seed=40 + k, start=10.0 * (k + 1)) for k, p in enumerate(pairs)}
        self.pairs = list(pairs)
        self.i = start_bar

    def now(self):
        df = next(iter(self.dfs.values()))
        return (df["timestamp"].iloc[self.i] + timedelta(minutes=5, seconds=1)).to_pydatetime()

    def klines(self, pair, interval, limit):
        return self.dfs[pair].iloc[max(0, self.i + 1 - limit): self.i + 1].reset_index(drop=True)

    def price(self, pair):
        return float(self.dfs[pair]["Open"].iloc[self.i + 1])


def trader(tmp_path, m, limits, executor=None, threshold=0.2):
    s = PaperSettings(interval="5m", lookback=400, risk=limits, threshold=threshold)
    return PaperTrader(s, str(tmp_path), data_fn=m.klines, price_fn=m.price, executor=executor, now_fn=m.now)


def limits(tmp_path, **kw):
    base = dict(kill_switch_file=str(tmp_path / "KS"), max_daily_loss_pct=0.5, max_drawdown_pct=0.5,
                max_cost_to_risk=10.0)  # mechanics test on 5m data: cost filter off
    base.update(kw)
    return RiskLimits(**base)


def test_limits_hold_at_every_step_with_simultaneous_positions(tmp_path):
    m = MultiMarket()
    L = limits(tmp_path, risk_per_trade=0.01, max_position_pct=0.15, max_exposure_pct=0.35, max_open_positions=3)
    t = trader(tmp_path, m, L)
    max_open = 0
    for _ in range(1200):
        out = t.step(m.pairs)
        eq = t.equity()
        prices = t.ledger["last_prices"]
        open_ = t.positions
        max_open = max(max_open, len(open_))
        assert len(open_) <= 3
        for d in out:
            if d["action"] == "BUY":
                notional = d["qty"] * d["price"]
                assert notional <= 0.15 * eq * 1.01 + 1e-9           # entry-time cap (eq after the buy ~ same)
        exposure_at_entry_prices = sum(p.qty * p.entry_price for p in open_.values())
        assert exposure_at_entry_prices <= 0.35 * t.ledger["peak_equity"] * 1.01
        assert t.ledger["cash"] >= -1e-9
        assert t.reconcile() == []
        m.i += 1
    assert max_open >= 2, "test never exercised simultaneous positions"


def test_every_stop_out_respects_risk_budget(tmp_path):
    m = MultiMarket()
    L = limits(tmp_path, risk_per_trade=0.005, max_position_pct=1.0, max_exposure_pct=1.0, max_open_positions=4)
    t = trader(tmp_path, m, L)
    equity_at_entry = {}
    checked = 0
    for _ in range(1500):
        for d in t.step(m.pairs):
            if d["action"] == "BUY":
                equity_at_entry[d["pair"]] = t.equity()
            if d["action"] == "SELL" and d["reason"] == "stop_loss" and d["pair"] in equity_at_entry:
                rec = t.ledger["trades"][-1] if t.ledger["trades"][-1]["pair"] == d["pair"] else None
                gap = rec and rec["exit_price"] < rec["stop_loss"] * (1 - 0.002) - 1e-9
                if rec and not gap:  # gaps through the stop are outside the risk budget by definition
                    assert -rec["pnl"] <= 0.005 * equity_at_entry[d["pair"]] * 1.02
                    checked += 1
        m.i += 1
    assert checked >= 3


def test_daily_loss_limit_blocks_entries_until_next_day(tmp_path):
    m = MultiMarket(pairs=("AAA",))
    t = trader(tmp_path, m, limits(tmp_path, max_daily_loss_pct=0.03))
    t.step(["AAA"])
    t.ledger["day_start_equity"] = t.equity() / 0.95  # pretend we are already down 5% today
    t.save()
    out = []
    for _ in range(200):
        out += t.step(["AAA"])
        m.i += 1
        if t.ledger["day"] != t.now_fn().date().isoformat():
            break
    blocked = [d for d in out if d["action"] == "BLOCKED"]
    assert blocked and all("daily loss" in d["reason"] for d in blocked)


def test_insufficient_cash_limits_size(tmp_path):
    r = RiskManager(RiskLimits(kill_switch_file="", risk_per_trade=0.05, max_position_pct=1, max_exposure_pct=1))
    d = r.position_size(equity=1000, cash=30, price=100, stop_price=99, fee_rate=0.002)
    assert d.ok and d.qty * 100 * 1.002 <= 30 + 1e-9 and "cash" in d.reason
    assert not r.position_size(equity=1000, cash=0.5, price=100, stop_price=99, fee_rate=0.002).ok


def test_exchange_minimum_and_precision_enforced_in_paper(tmp_path):
    m = MultiMarket(pairs=("AAA",))
    big_min = {"AAA": MarketInfo("AAA", "A", "USDT", 2, 4, 1_000.0, 0.002, 0.002, "online", True)}
    ex = PaperExecutor(markets=big_min)
    t = trader(tmp_path, m, limits(tmp_path), executor=ex)
    out = []
    for _ in range(1500):
        out += t.step(["AAA"])
        m.i += 1
    assert not any(d["action"] == "BUY" for d in out)
    assert any(d["action"] == "SKIP" and "min_amount" in d["reason"] for d in out)


def test_kill_switch_file_blocks_all_pairs(tmp_path):
    m = MultiMarket()
    L = limits(tmp_path)
    RiskManager(L).engage_kill_switch("test")
    t = trader(tmp_path, m, L)
    out = []
    for _ in range(300):
        out += t.step(m.pairs)
        m.i += 1
    assert not any(d["action"] == "BUY" for d in out)


def test_shipped_defaults_are_conservative():
    cfg = Config()
    assert cfg.profile == "Conservative"
    _, paper = build_settings(cfg)
    r = paper.risk
    assert r.risk_per_trade <= 0.005 and r.max_position_pct <= 0.10 and r.max_exposure_pct <= 0.30
    assert r.max_daily_loss_pct <= 0.03 and r.max_drawdown_pct <= 0.10
    for prof in ("Conservative", "Normal", "Aggressive"):
        _, p = build_settings(cfg, prof)
        assert p.risk.risk_per_trade <= 0.02 and p.risk.max_exposure_pct <= 0.6
    assert RiskLimits().risk_per_trade == 0.005
