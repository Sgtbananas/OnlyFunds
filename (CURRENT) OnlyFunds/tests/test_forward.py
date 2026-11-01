"""Forward paper engine, evidence-gated selection, shadow tracking and monitoring (offline, simulated clock)."""
import json
import os
from datetime import timedelta
from types import SimpleNamespace

import pandas as pd
import pytest
import requests

from core import monitoring
from core.forward import ForwardPaperEngine, ShadowTracker
from core.live_data import MarketSnapshot, StaleDataError
from core.paper_trader import PaperSettings
from core.positions import ExitRules
from core.risk_manager import RiskLimits
from core.selection import Approval, StrategySelector, load_approvals
from tests.synthetic import synthetic_ohlcv


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    monkeypatch.setattr(requests.Session, "request", lambda *a, **k: (_ for _ in ()).throw(AssertionError("network")))


class FakeLive:
    """Replays candles; current quote = next bar open +/- half spread."""

    def __init__(self, n=2200, i=600, spread_bps=4.0, seed=8):
        self.df = synthetic_ohlcv(n, seed=seed)
        self.i, self.spread, self.quotes, self.stale = i, spread_bps / 1e4, {}, False

    def now(self):
        return (self.df["timestamp"].iloc[self.i] + timedelta(minutes=5, seconds=1)).to_pydatetime()

    def completed_klines(self, pair, interval=None, limit=500):
        out = self.df.iloc[max(0, self.i + 1 - (limit or 500)): self.i + 1].reset_index(drop=True)
        out.attrs["source"] = "fake-live"
        return out

    def quote(self, pair):
        if self.stale:
            raise StaleDataError("quote too old")
        mid = float(self.df["Open"].iloc[self.i + 1])
        q = MarketSnapshot("fake", pair, 0, mid * (1 - self.spread / 2), mid * (1 + self.spread / 2), 1, 1)
        self.quotes[pair] = q
        return q

    def snapshot(self, pair, with_depth=True):
        return self.quote(pair)

    def feed_health(self):
        return {"ws_reconnects": 0, "stale_quotes": int(self.stale), "buffers": {}, "rest": {}}


def settings(strategy="auto", **risk):
    r = dict(kill_switch_file="", max_drawdown_pct=0.9, max_daily_loss_pct=0.9, max_cost_to_risk=10.0)
    r.update(risk)
    return PaperSettings(interval="5m", lookback=400, strategy=strategy, threshold=0.25, risk=RiskLimits(**r))


def engine(tmp_path, live, selector=None, strategy="auto"):
    return ForwardPaperEngine(["SYN"], settings(strategy), str(tmp_path), live,
                              selector=selector or StrategySelector([]), now_fn=live.now)


def run(eng, live, steps):
    outs = []
    for _ in range(steps):
        outs.append(eng.step())
        live.i += 1
    return outs


# ---------------------------------------------------------------- inactive by default
def test_no_approved_strategy_means_no_trades_but_full_records(tmp_path):
    live = FakeLive()
    eng = engine(tmp_path, live)
    outs = run(eng, live, 500)
    actions = [d["action"] for o in outs for d in o["trader"]]
    assert "BUY" not in actions and actions.count("REJECTED") > 0
    rej = [d for o in outs for d in o["trader"] if d["action"] == "REJECTED"]
    assert all("no strategy has passed validation" in d["reason"] for d in rej)
    sh = json.load(open(tmp_path / "shadow.json"))
    assert len(sh["open"]) + len(sh["resolved"]) >= len(rej)
    assert len(sh["resolved"]) > 0 and all("r_multiple" in x for x in sh["resolved"])
    obs = [json.loads(x) for f in os.listdir(tmp_path / "observations")
           for x in open(tmp_path / "observations" / f) if x.strip()]
    assert len(obs) == 500 and len({o["candle"]["created_at"] for o in obs}) == 500
    o = obs[-1]
    assert set(o["signals"]) == {"auto", "trend_following", "mean_reversion", "stacked"}
    assert o["selection"]["strategy"] is None and o["snapshot"]["bid"] < o["snapshot"]["ask"]


def test_no_duplicate_observations_or_shadow_entries_without_new_candle(tmp_path):
    live = FakeLive()
    eng = engine(tmp_path, live)
    run(eng, live, 50)
    sh_before = json.load(open(tmp_path / "shadow.json"))
    eng.step()
    eng.step()                      # same candle twice more
    sh_after = json.load(open(tmp_path / "shadow.json"))
    assert len(sh_after["open"]) + len(sh_after["resolved"]) == len(sh_before["open"]) + len(sh_before["resolved"])
    obs = [x for f in os.listdir(tmp_path / "observations") for x in open(tmp_path / "observations" / f) if x.strip()]
    assert len(obs) == 51


def test_restart_continues_without_duplicates(tmp_path):
    live = FakeLive()
    run(engine(tmp_path, live), live, 100)
    eng2 = engine(tmp_path, live)
    eng2.step()
    obs = [x for f in os.listdir(tmp_path / "observations") for x in open(tmp_path / "observations" / f) if x.strip()]
    assert len(obs) == 100 + 1
    assert eng2.state["steps"] == 101


# ---------------------------------------------------------------- with an approval (test-only)
def approve_all(strategy):
    return StrategySelector([Approval(strategy, "fixed", None, "test", "run", 0.2, 500,
                                      regimes=["strong_up", "weak_up", "range", "weak_down", "strong_down"],
                                      min_regime_confidence=0.0)])


def test_approved_strategy_trades_with_bid_ask_fills(tmp_path):
    live = FakeLive()
    eng = engine(tmp_path, live, selector=approve_all("trend_following"), strategy="trend_following")
    for _ in range(1200):
        out = eng.step()
        buys = [d for d in out["trader"] if d["action"] == "BUY"]
        if buys:
            b = buys[0]
            assert b["price"] == pytest.approx(b["ask"] * (1 + eng.settings.slippage_pct))
            assert eng.trader.reconcile() == []
            return
        live.i += 1
    pytest.fail("no entry with an approved strategy")


def test_stale_quote_blocks_entries_and_is_recorded(tmp_path):
    live = FakeLive()
    eng = engine(tmp_path, live, selector=approve_all("trend_following"), strategy="trend_following")
    live.stale = True
    outs = run(eng, live, 400)
    actions = [d for o in outs for d in o["trader"]]
    assert not any(d["action"] == "BUY" for d in actions)
    assert any(d["action"] == "DATA_STALE" and "quote" in d["reason"] for d in actions)


# ---------------------------------------------------------------- selector
def test_selector_rules():
    assert StrategySelector([]).select({"label": "x", "trend": "weak_up"}).strategy is None
    a = Approval("trend_following", "fixed", None, "p", "r", 0.1, 300, regimes=["weak_up"], min_regime_confidence=0.5)
    b = Approval("mean_reversion", "fixed", None, "p", "r2", 0.3, 300, regimes=["weak_up", "range"], min_regime_confidence=0.5)
    sel = StrategySelector([a, b])
    assert sel.select({"label": "uncertain", "trend": "weak_up", "confidence": 0.9}).strategy is None
    assert "no approved strategy" in sel.select({"label": "x", "trend": "strong_down", "confidence": 0.9}).reason
    assert sel.select({"label": "x", "trend": "weak_up", "confidence": 0.2}).strategy is None
    assert sel.select({"label": "x", "trend": "weak_up", "confidence": 0.9}).strategy == "mean_reversion"  # higher CI
    weak = Approval("stacked", "fixed", None, "p", "r3", -0.01, 300, regimes=["range"])
    small = Approval("stacked", "fixed", None, "p", "r4", 0.5, 20, regimes=["range"])
    assert StrategySelector([weak, small]).approvals == []          # no evidence -> filtered out


def test_load_approvals_requires_eligibility_and_credible_test(tmp_path):
    def write(run, credible, eligible):
        d = tmp_path / "plan" / run
        (d / "stage1").mkdir(parents=True)
        (d / "stage2").mkdir()
        (d / "stage1" / "eligibility.json").write_text(json.dumps(
            {"trend_following/fixed": {"strategy": "trend_following", "variant": "fixed", "params": None,
                                       "eligible": eligible, "approved_regimes": ["weak_up"]}}))
        (d / "stage2" / "result.json").write_text(json.dumps({"evaluated": [
            {"candidate": "trend_following/fixed", "credible_edge": credible, "ci95_low_r": 0.05, "trades": 300}]}))
    write("r1", True, True)
    write("r2", True, False)
    write("r3", False, True)
    got = load_approvals(str(tmp_path))
    assert [(a.run_id, a.regimes) for a in got] == [("r1", ["weak_up"])]
    assert load_approvals(str(tmp_path / "missing")) == []


def test_real_repository_has_no_approved_strategy():
    assert StrategySelector().approvals == []


# ---------------------------------------------------------------- shadow tracker
def test_shadow_outcome_exact(tmp_path):
    rules = ExitRules(1.0, 2.0, 1.0, use_trailing=False)
    st = ShadowTracker(str(tmp_path / "s.json"), rules, fee_rate=0.001, slippage_pct=0.0, stop_slippage_pct=0.0)
    H = 300_000
    base = {"pair": "X", "strategy": "s", "regime": "r", "action": "REJECTED", "reason": "test", "interval": "5m",
            "candle_ms": 0, "atr": 1.0, "ref_buy": 100.0}
    assert st.record(base) and st.record(base) is None              # duplicate ignored
    candles = pd.DataFrame({"created_at": [0, H, 2 * H], "Open": [100, 100, 100.5], "High": [100.5, 101, 102.5],
                            "Low": [99.5, 99.6, 100.2], "Close": [100, 100.5, 102]})
    done = st.update("X", candles)
    assert len(done) == 1 and done[0]["exit_reason"] == "take_profit" and done[0]["bars"] == 2
    risk = 100 * 1.001 - 99 * 0.999
    assert done[0]["r_multiple"] == pytest.approx((102 * 0.999 - 100 * 1.001) / risk)
    st.save()
    assert len(ShadowTracker(str(tmp_path / "s.json"), rules, 0.001, 0, 0).state["resolved"]) == 1


# ---------------------------------------------------------------- monitoring
def test_monitor_keeps_categories_separate(tmp_path):
    live = FakeLive()
    eng = engine(tmp_path, live)
    run(eng, live, 300)
    text = monitoring.report(str(tmp_path), str(tmp_path / "runs"))
    for section in ("Validation runs (v2)", "Forward paper trading", "Shadow: outcomes of rejected opportunities",
                    "Data feed reliability"):
        assert section in text
    sh = monitoring.shadow(str(tmp_path))
    assert sh["resolved"] > 0 and "by_rejection" in sh and "by_strategy" in sh
    fp = monitoring.forward_paper(str(tmp_path))
    assert fp["overall"]["trades"] == 0 and "REJECTED" in fp["decisions"]
