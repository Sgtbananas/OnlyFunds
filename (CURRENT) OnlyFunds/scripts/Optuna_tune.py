"""Optuna search over strategy parameters, selected on the training segment only.

The best trial is then evaluated once on the unseen 30% test segment. If
optuna is not installed, use scripts/auto_tuner.py (grid search) instead.

    python scripts/Optuna_tune.py --pair BTCUSDT [--trials 50] [--interval 15m]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.autotune import _apply, _confirm  # noqa: E402
from core.backtester import run_backtest  # noqa: E402
from core.core_data import fetch_klines  # noqa: E402
from core.indicators import warmup_bars  # noqa: E402
from utils.config import APP_ROOT, build_settings, load_config  # noqa: E402


def main(argv=None):
    import optuna

    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", default="BTCUSDT")
    ap.add_argument("--interval")
    ap.add_argument("--trials", type=int, default=50)
    a = ap.parse_args(argv)
    cfg = load_config()
    base, _ = build_settings(cfg, None, {"interval": a.interval} if a.interval else None)
    df = fetch_klines(a.pair, base.interval, 1000)
    cut = int(len(df) * 0.7)
    train = df.iloc[:cut].reset_index(drop=True)
    test = df.iloc[max(0, cut - warmup_bars()):].reset_index(drop=True)

    def params_of(trial):
        return {"threshold": trial.suggest_float("threshold", 0.15, 0.6),
                "strategy": trial.suggest_categorical("strategy", ["auto", "trend_following", "mean_reversion",
                                                                   "stacked"]),
                "exit_rules.stop_atr_mult": trial.suggest_float("stop_atr_mult", 0.8, 3.0),
                "exit_rules.tp_atr_mult": trial.suggest_float("tp_atr_mult", 1.0, 5.0),
                "exit_rules.trail_atr_mult": trial.suggest_float("trail_atr_mult", 0.8, 4.0)}

    def objective(trial):
        m = run_backtest(train, _apply(base, params_of(trial)), a.pair).metrics
        if m["trades"] < 8:
            return -1e9
        return m["total_return_pct"] / max(m["max_drawdown_pct"], 1.0)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=a.trials)
    best = params_of(optuna.trial.FixedTrial(study.best_params))
    oos = run_backtest(test, _apply(base, best), a.pair)
    confirmed, reason = _confirm(oos, 3)
    out = {"pair": a.pair, "best_params": best, "train_score": study.best_value,
           "out_of_sample": oos.metrics, "confirmed": confirmed, "reason": reason}
    print(json.dumps(out, indent=2, default=str))
    path = os.path.join(APP_ROOT, "state", "optuna_best.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    return 0


if __name__ == "__main__":
    sys.exit(main())
