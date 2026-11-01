# Validation protocol v2 (2026-11-01)

v2 keeps the v1 method (`VALIDATION_PROTOCOL.md`) and fixes audit issues P1–P4
(`AUDIT_REPORT_3_VALIDATION.md`). It applies to **all future validations**. v1's results are frozen and are
not re-judged under v2.

## What a validation run is
- A **plan** file (`docs/validation_plans/*.json`) fixes, in advance:
  - the universe, exchange and interval
  - the DEV / VAL / TEST periods, which must be chronological and non-overlapping
  - the walk-forward folds, which must end before TEST
  - the strategies, tuning grid, promotion rules and cost overrides
- **Run with:** `scripts/validation_run.py stage1 --plan …`, then `stage2 --plan … --run-id …`.

## P1: provenance
Every stage writes `provenance.json` with:
- the git commit and whether the tree was clean (Stage 1 refuses a dirty tree unless `--allow-dirty`, which is recorded)
- the exact command
- the Python version and package versions
- SHA-256 of the protocol, the plan, every implementing file (`core/provenance.py: IMPLEMENTATION_FILES`), the
  config file, the effective backtest config, every dataset (checked against its manifest), and every promoted model
- strategy definitions

The **validation fingerprint** hashes protocol + plan + implementation + config + data. Stage 2 recomputes it.
If it differs, Stage 2 refuses and lists what changed: that's a **new validation version**, and it needs a new Stage 1.

## P2: immutable runs
- **Where:** each Stage 1 creates `docs/validation/runs/<plan>/<run_id>/`.
- **Write-once:** files are written once, then made read-only. Stage directories can't be recreated.
- **Replacing a run:** a new run with `--supersedes <run_id> --reason "…"`. The old run is untouched.
- **One use per final-test window:** a window is registered when consumed (`runs/_consumed_test_windows/`). Any later run whose TEST
  window overlaps it, for the same interval and any shared symbol, is refused. The v1 window
  (2025-01-01 → 2026-11-01, 1h, 9 coins) is registered.

## P3: eligibility before TEST
A candidate is **eligible** only if, on VAL, all of these hold:
- ≥ 100 pooled trades
- expectancy > 0R with the 95% bootstrap lower bound > 0
- profit factor ≥ 1.10
- positive on ≥ 60% of coins with ≥ 10 trades
- still positive without the best coin
- walk-forward positive in ≥ 3 of 4 folds

Each candidate's decision is recorded with its reasons and evidence (`eligibility.json`). Stage 2 evaluates **only** eligible
candidates. If none are eligible, TEST isn't evaluated and the window isn't consumed. A TEST verdict of
"credible edge" therefore always implies a VAL pass too.

## P4: ML promotion
An ML-gated variant is promoted only if **all** of these hold:
1. It is itself P3-eligible, which includes ≥ 100 VAL trades.
2. The model's held-out check said `useful`.
3. Net expectancy after fees and slippage beats the same strategy without ML, with the 95% bootstrap interval of
   the difference above zero.
4. Risk-adjusted return (mean net return ÷ max drawdown) is at least the non-ML value.
5. Worst drawdown is no higher than the non-ML value.
6. It beats the non-ML strategy in **every** VAL sub-period (calendar year), with ≥ 20 ML trades in each.

Otherwise ML stays disabled, and the evidence is recorded. ML is never assumed to help.

## Unchanged from v1
The criteria thresholds, costs, cost-to-risk filter, Conservative risk settings and the bootstrap method are
unchanged. Win rate is never a criterion.
