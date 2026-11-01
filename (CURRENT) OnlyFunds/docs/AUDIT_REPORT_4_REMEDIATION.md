# Audit remediation: P1–P4 (2026-11-01)

Fixes for the four protocol weaknesses in `AUDIT_REPORT_3_VALIDATION.md`. The v1 results were not altered,
re-run or re-judged. New rules apply to future validations through `VALIDATION_PROTOCOL_V2.md` and
`scripts/validation_run.py`.

Checkpoint before any change: tag `checkpoint/pre-remediation-56897c8` (pushed).

| Issue | Fix | Where | Regression tests (`tests/test_remediation.py`) |
|---|---|---|---|
| **P1** Fingerprint covered the protocol text only | Provenance manifest for every stage; fingerprint over protocol + plan + every implementing file + config + effective config + data checksums; Stage 2 refuses any change and lists it; dirty tree refused unless `--allow-dirty` (recorded) | `core/provenance.py`; `scripts/validation_run.py: stage1/stage2` | `test_p1_manifest_records_everything`, `test_p1_stage2_refuses_when_anything_changed[protocol, plan, data, code]`, `test_p1_dirty_tree_refused_unless_explicit`, `test_p1_fingerprint_ignores_line_endings` |
| **P2** Stage 1 could overwrite frozen results | Unique run ids and one directory per run; write-once, read-only files; stage directories can't be recreated; `--supersedes` + reason creates a new run; consumed TEST windows registered (v1 included); v1 Stage 1 refuses to overwrite | `core/run_store.py`; `scripts/real_market_validation.py: stage_devval`; `docs/validation/runs/_consumed_test_windows/` | `test_p2_write_once_refuses_overwrite_and_is_read_only`, `test_p2_stage1_rerun_creates_new_run_and_keeps_old`, `test_p2_stage_dirs_immutable_and_supersede_is_explicit`, `test_p2_stage2_cannot_run_twice`, `test_p2_consumed_test_window_cannot_be_reused`, `test_p2_v1_stage1_refuses_to_overwrite`, `test_p2_v1_window_registered_as_consumed`, plus `tests/test_frozen_v1.py` (6 hashes) |
| **P3** TEST verdict ignored the VAL result | Eligibility gate on VAL (sample size, expectancy and CI, PF, breadth, best-coin dependence, walk-forward) with reasons and evidence per candidate; Stage 2 evaluates only eligible candidates; with none eligible, TEST isn't evaluated and not consumed | `core/eligibility.py: candidate_eligibility`; `scripts/validation_run.py: stage2` | `test_p3_eligibility_rules`, `test_p3_v1_candidates_would_all_be_ineligible`, `test_p3_stage2_skips_test_when_nothing_eligible`, `test_p3_stage2_evaluates_only_eligible` |
| **P4** ML carried forward on 1 trade | ML promoted only if P3-eligible itself, the model is `useful`, the expectancy gain over non-ML has a bootstrap lower bound > 0, risk-adjusted return and drawdown are no worse, and it wins in every VAL sub-period with ≥ 20 trades; evidence recorded | `core/eligibility.py: ml_promotion`; `scripts/validation_run.py: stage1` | `test_p4_v1_single_trade_carry_is_rejected` (exact v1 numbers), `test_p4_promotes_only_with_robust_improvement`, `test_p4_ml_path_runs_and_records_evidence` |

**Applied retroactively?** No.
- The v1 TEST results stay exactly as published.
- `test_p3_v1_candidates_would_all_be_ineligible` shows that under v2 rules none of the v1 candidates would have
  reached TEST at all. The v1 TEST evaluation would simply not have happened, which matches its "no edge" result.

**P1 limitation for v1, stated honestly:** v1's implementing code was committed only after its final test, so no
fingerprint can prove which code produced it (`docs/validation/README.md`). What can be shown:
- the protocol predates the run
- the only edit between stages was a console-encoding fix
- re-running Stage 1 reproduces the frozen results exactly
