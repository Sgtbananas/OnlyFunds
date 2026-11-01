"""Regression tests for audit fixes P1-P4 (validation framework v2). Synthetic data, temp folders only."""
import json
import os
import stat

import numpy as np
import pandas as pd
import pytest

from core import eligibility as E
from core import provenance as P
from core.history import save_dataset
from core.run_store import ImmutableError, RunStore, write_once
from scripts import validation_run as VR
from tests.synthetic import synthetic_regime

SYMS = ["AAAUSDT", "BBBUSDT"]


@pytest.fixture
def env(tmp_path, monkeypatch):
    data_root = tmp_path / "data"
    for k, s in enumerate(SYMS):
        df = synthetic_regime("switching" if k else "range", 4000, seed=k, interval="1h")
        save_dataset(df, s, "1h", {"synthetic": True}, root=str(data_root))
    proto = tmp_path / "protocol.md"
    proto.write_text("protocol v2 test")
    plan = {"name": "testplan", "protocol": str(proto), "exchange": "binance", "interval": "1h", "symbols": SYMS,
            "data_root": str(data_root), "runs_root": str(tmp_path / "runs"),
            "periods": {"DEV": ["2025-01-01", "2025-03-15"], "VAL": ["2025-03-15", "2025-04-30"],
                        "TEST": ["2025-05-01", "2025-06-15"]},
            "walk_forward": [["2025-01-01", "2025-02-15", "2025-02-15", "2025-03-15"],
                             ["2025-01-15", "2025-03-15", "2025-03-15", "2025-04-15"]],
            "strategies": ["trend_following"], "grid": {"threshold": [0.3, 0.45]}, "ml": False}
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))
    monkeypatch.setattr(VR, "git_state", lambda: {"commit": "test", "branch": "t", "dirty": False, "dirty_files": []})
    monkeypatch.setattr(P, "git_state", lambda: {"commit": "test", "branch": "t", "dirty": False, "dirty_files": []})
    return {"plan": plan, "plan_path": str(plan_path), "tmp": tmp_path}


def eligible_everything(monkeypatch):
    monkeypatch.setattr(VR, "candidate_eligibility",
                        lambda val, wf, rules=None: {"eligible": True, "reasons": ["forced for test"], "evidence": {}})


# ------------------------------------------------------------------------- P1
def test_p1_manifest_records_everything(env):
    run = VR.stage1(env["plan_path"], argv=["test"])
    store = RunStore(env["plan"]["runs_root"])
    m = store.read_json("testplan", run, "stage1", "provenance.json")
    for key in ("run_id", "stage", "command", "git", "environment", "protocol", "plan", "effective_config", "data",
                "strategies", "fingerprint_components", "validation_fingerprint"):
        assert key in m
    assert m["git"]["commit"] == "test" and m["git"]["dirty"] is False
    assert set(m["fingerprint_components"]["implementation"]) == set(P.IMPLEMENTATION_FILES)
    assert all(v for v in m["fingerprint_components"]["implementation"].values())
    assert all(d["matches_manifest"] for d in m["data"].values())
    assert m["environment"]["packages"]["pandas"] == pd.__version__


@pytest.mark.parametrize("change", ["protocol", "plan", "data", "code"])
def test_p1_stage2_refuses_when_anything_changed(env, monkeypatch, change):
    eligible_everything(monkeypatch)
    extra_code = env["tmp"] / "impl_extra.py"
    extra_code.write_text("x = 1\n")
    monkeypatch.setattr(P, "IMPLEMENTATION_FILES", P.IMPLEMENTATION_FILES + [str(extra_code)])
    run = VR.stage1(env["plan_path"])
    if change == "protocol":
        open(env["plan"]["protocol"], "a").write("\nedited")
    elif change == "plan":
        plan = dict(env["plan"], grid={"threshold": [0.3]})
        open(env["plan_path"], "w").write(json.dumps(plan))
    elif change == "data":
        csv = os.path.join(env["plan"]["data_root"], "binance", "AAAUSDT_1h.csv")
        open(csv, "a").write("\n")
    else:
        extra_code.write_text("x = 2\n")
    with pytest.raises((VR.RefusedError,)):
        VR.stage2(env["plan_path"], run)
    assert not os.path.isdir(os.path.join(env["plan"]["runs_root"], "testplan", run, "stage2"))


def test_p1_dirty_tree_refused_unless_explicit(env, monkeypatch):
    monkeypatch.setattr(VR, "git_state", lambda: {"commit": "x", "branch": "t", "dirty": True, "dirty_files": ["a"]})
    with pytest.raises(VR.RefusedError):
        VR.stage1(env["plan_path"])
    run = VR.stage1(env["plan_path"], allow_dirty=True)
    m = RunStore(env["plan"]["runs_root"]).read_json("testplan", run, "stage1", "provenance.json")
    assert m["allow_dirty"] is True


# ------------------------------------------------------------------------- P2
def test_p2_write_once_refuses_overwrite_and_is_read_only(tmp_path):
    p = str(tmp_path / "a" / "x.json")
    write_once(p, {"a": 1})
    with pytest.raises(ImmutableError):
        write_once(p, {"a": 2})
    assert json.load(open(p)) == {"a": 1}
    assert not os.stat(p).st_mode & stat.S_IWRITE
    with pytest.raises(PermissionError):
        open(p, "w")


def test_p2_stage1_rerun_creates_new_run_and_keeps_old(env):
    r1 = VR.stage1(env["plan_path"])
    before = open(os.path.join(env["plan"]["runs_root"], "testplan", r1, "stage1", "eligibility.json")).read()
    r2 = VR.stage1(env["plan_path"])
    assert r1 != r2
    after = open(os.path.join(env["plan"]["runs_root"], "testplan", r1, "stage1", "eligibility.json")).read()
    assert before == after
    assert set(RunStore(env["plan"]["runs_root"]).list_runs("testplan")) == {r1, r2}


def test_p2_stage_dirs_immutable_and_supersede_is_explicit(env):
    store = RunStore(env["plan"]["runs_root"])
    r1 = VR.stage1(env["plan_path"])
    with pytest.raises(ImmutableError):
        store.stage_dir("testplan", r1, "stage1", create=True)
    with pytest.raises(ImmutableError):
        VR.stage1(env["plan_path"], supersedes=r1, reason="")
    r2 = VR.stage1(env["plan_path"], supersedes=r1, reason="re-run after data refresh")
    meta = store.read_json("testplan", r2, "run.json")
    assert meta["supersedes"] == r1 and meta["reason"]


def test_p2_stage2_cannot_run_twice(env, monkeypatch):
    eligible_everything(monkeypatch)
    run = VR.stage1(env["plan_path"])
    VR.stage2(env["plan_path"], run)
    with pytest.raises(ImmutableError):
        VR.stage2(env["plan_path"], run)


def test_p2_consumed_test_window_cannot_be_reused(env, monkeypatch):
    eligible_everything(monkeypatch)
    run = VR.stage1(env["plan_path"])
    VR.stage2(env["plan_path"], run)
    with pytest.raises(ImmutableError):
        VR.stage1(env["plan_path"])           # same TEST window -> refused before any work


def test_p2_v1_stage1_refuses_to_overwrite(tmp_path, monkeypatch):
    from scripts import real_market_validation as R
    (tmp_path / "frozen.json").write_text("{}")
    monkeypatch.setattr(R, "FROZEN", str(tmp_path / "frozen.json"))
    assert R.stage_devval(type("A", (), {"symbols": []})()) == 2
    assert (tmp_path / "frozen.json").read_text() == "{}"


def test_p2_v1_window_registered_as_consumed():
    store = RunStore()
    with pytest.raises(ImmutableError):
        store.check_test_window_unused("2025-06-01", "2025-07-01", ["BTCUSDT"], "1h")
    store.check_test_window_unused("2026-11-01", "2027-06-01", ["BTCUSDT"], "1h")   # fresh window is fine


# ------------------------------------------------------------------------- P3
def val_pool(**kw):
    base = {"trades": 300, "expectancy_r": 0.2, "ci_low": 0.05, "ci_high": 0.3, "profit_factor": 1.3, "breadth": 0.7,
            "eligible_coins": 8, "exp_r_without_best_coin": 0.15, "worst_max_dd_pct": 5.0, "mean_net_return_pct": 4.0}
    base.update(kw)
    return base


WF_OK = pd.DataFrame({"expectancy_r": [0.1, 0.2, -0.05, 0.1]})


def test_p3_eligibility_rules():
    assert E.candidate_eligibility(val_pool(), WF_OK)["eligible"]
    cases = {"trades": 99, "expectancy_r": -0.01, "ci_low": -0.01, "profit_factor": 1.05, "breadth": 0.5,
             "exp_r_without_best_coin": -0.01}
    for k, v in cases.items():
        d = E.candidate_eligibility(val_pool(**{k: v}), WF_OK)
        assert not d["eligible"] and len(d["reasons"]) == 1, (k, d)
    d = E.candidate_eligibility(val_pool(), pd.DataFrame({"expectancy_r": [0.1, -0.2, -0.05, 0.1]}))
    assert not d["eligible"] and "walk-forward" in d["reasons"][0]
    assert not E.candidate_eligibility(val_pool(), None)["eligible"]


def test_p3_v1_candidates_would_all_be_ineligible():
    """The real v1 VAL numbers (docs/validation/stage1_dev_val.md) fail P3 for every candidate."""
    v1 = [("auto/fixed", 255, -0.065, -0.190, 0.911, 0.40, -0.120), ("trend_following/fixed", 380, -0.082, -0.184,
          0.867, 0.286, -0.124), ("mean_reversion/fixed", 283, -0.084, -0.197, 0.858, 0.25, -0.104),
          ("stacked/fixed", 58, -0.194, -0.415, 0.653, 0.0, -0.283)]
    for name, n, e, lo, pf, br, wob in v1:
        d = E.candidate_eligibility(val_pool(trades=n, expectancy_r=e, ci_low=lo, profit_factor=pf, breadth=br,
                                             exp_r_without_best_coin=wob), pd.DataFrame({"expectancy_r": [-0.1] * 4}))
        assert not d["eligible"], name


def test_p3_stage2_skips_test_when_nothing_eligible(env):
    run = VR.stage1(env["plan_path"])
    elig = RunStore(env["plan"]["runs_root"]).read_json("testplan", run, "stage1", "eligibility.json")
    assert all(not d["eligible"] and d["reasons"] for d in elig.values())
    res = VR.stage2(env["plan_path"], run)
    assert res["evaluated"] == [] and "NOT consumed" in res["note"]
    VR.stage1(env["plan_path"])            # window still unused -> a later run is allowed


def test_p3_stage2_evaluates_only_eligible(env, monkeypatch):
    calls = []
    real = E.candidate_eligibility

    def only_fixed(val, wf, rules=None):
        calls.append(1)
        d = real(val, wf, rules)
        return {**d, "eligible": len(calls) == 1}            # first decision (fixed) eligible, tuned not
    monkeypatch.setattr(VR, "candidate_eligibility", only_fixed)
    run = VR.stage1(env["plan_path"])
    res = VR.stage2(env["plan_path"], run)
    assert [r["candidate"] for r in res["evaluated"]] == ["trend_following/fixed"]


# ------------------------------------------------------------------------- P4
def pooled_like(trades, exp, dd=5.0, by_year=None):
    per_coin = pd.DataFrame({"net_return_pct": [1.0, 2.0], "max_dd_pct": [dd, dd]})
    return {**val_pool(trades=trades, expectancy_r=exp, worst_max_dd_pct=dd), "per_coin": per_coin,
            "by_year": by_year if by_year is not None else pd.DataFrame(
                {"trades": [trades // 2, trades - trades // 2], "expectancy_r": [exp, exp]}, index=[2023, 2024])}


def test_p4_v1_single_trade_carry_is_rejected():
    """Exact v1 case: mean reversion + ML made 1 VAL trade (+1.232R) vs 283 without ML (-0.084R)."""
    base = pooled_like(283, -0.084)
    ml = pooled_like(1, 1.232, by_year=pd.DataFrame({"trades": [1], "expectancy_r": [1.232]}, index=[2023]))
    rng = np.random.default_rng(0)
    d = E.ml_promotion(base, ml, rng.normal(-0.084, 1, 283), np.array([1.232]),
                       {"useful": False, "validation_auc": 0.525},
                       E.candidate_eligibility(ml, WF_OK))
    assert not d["promoted"]
    text = " ".join(d["reasons"])
    assert "sample too small" in text and "not useful" in text and "fails the P3" in text
    assert d["evidence"]["ml_trades"] == 1


def test_p4_promotes_only_with_robust_improvement():
    rng = np.random.default_rng(1)
    base_r, ml_r = rng.normal(0.05, 1, 800), rng.normal(0.35, 1, 600)
    base = pooled_like(800, 0.05, dd=6.0)
    ml = pooled_like(600, 0.35, dd=5.0)
    elig = {"eligible": True, "reasons": ["ok"], "evidence": {}}
    good = E.ml_promotion(base, ml, base_r, ml_r, {"useful": True, "validation_auc": 0.6}, elig)
    assert good["promoted"], good["reasons"]
    worse_dd = E.ml_promotion(base, pooled_like(600, 0.35, dd=9.0), base_r, ml_r, {"useful": True}, elig)
    assert not worse_dd["promoted"] and any("drawdown" in r for r in worse_dd["reasons"])
    inconsistent = pooled_like(600, 0.35, dd=5.0, by_year=pd.DataFrame({"trades": [300, 300],
                                                                         "expectancy_r": [0.7, 0.0]}, index=[2023, 2024]))
    bad = E.ml_promotion(base, inconsistent, base_r, ml_r, {"useful": True}, elig)
    assert not bad["promoted"] and any("2024" in r for r in bad["reasons"])
    tiny_gain = E.ml_promotion(base, pooled_like(800, 0.055, dd=5.0), base_r, base_r + 0.005,
                               {"useful": True}, elig)   # same trades, +0.005R: not a measurable improvement
    assert not tiny_gain["promoted"] and any("improvement" in r for r in tiny_gain["reasons"])


def test_p4_ml_path_runs_and_records_evidence(env):
    plan = dict(env["plan"], ml=True)
    open(env["plan_path"], "w").write(json.dumps(plan))
    try:
        run = VR.stage1(env["plan_path"])
    except ValueError:
        pytest.skip("synthetic data too small for ML")
    d = RunStore(plan["runs_root"]).read_json("testplan", run, "stage1", "eligibility.json")["trend_following/fixed+ML"]
    assert d["eligible"] is False and d["reasons"]


def test_p1_fingerprint_ignores_line_endings(tmp_path):
    a, b = tmp_path / "a.py", tmp_path / "b.py"
    a.write_bytes(b"x = 1\ny = 2\n")
    b.write_bytes(b"x = 1\r\ny = 2\r\n")
    assert P.sha256_text(str(a)) == P.sha256_text(str(b))
    b.write_bytes(b"x = 1\r\ny = 3\r\n")
    assert P.sha256_text(str(a)) != P.sha256_text(str(b))
