# OnlyFunds 2 (OF2) — handover report

Written 2026-11-01 (UTC) at the end of a long Claude Code conversation, and updated the same day once the deployment
code was tested and committed. Everything below was checked against the repository and machine at writing time.

## 0. Most important facts
1. **No forward paper session is running. The 24-hour supervised trial has NOT been started.** It waits for explicit
   approval from the user. `state/forward_deployments/` does not exist yet.
2. **Live trading is locked and must stay locked.** No strategy has a demonstrated edge.
3. **The deployment code is committed and pushed** (§1). The full suite passed (261 tests), and the launch, stop and crash
   recovery were verified on a disposable deployment in a scratch folder (§6).

## 1. Git state
- Repo: `Sgtbananas/OnlyFunds`. App folder: `(CURRENT) OnlyFunds/`. Branch: `of2-completion`. `main` is untouched at `5a19aaa`.
- Commits after the first handover (`e1aa54e`):
  - `4523055`: forward paper hardening (crash safety, endpoint allowlist, exchange clock)
  - `08e27c5`: long-running deployment (session manager, supervisor, reports, trial report, bat files)
  - the commit containing this update to the handover document
- Working tree clean after these commits. All of them pushed to `origin/of2-completion`.
- Recovery tag before the audit-remediation work: `checkpoint/pre-remediation-56897c8` (pushed).

## 2. PR status
- **PR #1** (`of2-completion` → `main`) is open, and its head follows `of2-completion`.
- **Do not merge without explicit approval from the user.**
- The PR description was last updated during the real-market validation phase; it doesn't describe the
  remediation, live-data or deployment work.

## 3. Running processes and sessions
| PID | What | Notes |
|---|---|---|
| 5732, 6400 | `python.exe`, started 31/10/2026, command line not readable | **not OF2 sessions** (they predate the session code); leave them alone |

**OF2 sessions:** none running. All test sessions ran with `OF2_STATE_DIR` pointing at temporary scratch folders, and
none is part of a real deployment:
- `20261101T075056Z-e470ff70`: live session runner, 3/3 steps OK, 8 coins observed
- an earlier 13-step forward smoke run
- disposable deployment `disposable-test` (§6): sessions `20261101T080929Z-f7ff3e8b` (killed on purpose) and
  `20261101T081032Z-b636be0f` (stopped with the STOP script)

## 4. Forward paper trading: start time, runtime, health
Not started, so there's no start time, runtime or health yet. The startup health report is created automatically when
the first session starts: `state/forward_deployments/binance-1h/reports/startup/<session_id>.md`.

## 5. Where data and logs will live (`(CURRENT) OnlyFunds/state/`, git-ignored)
For deployment `binance-1h` (the default in the `.bat` files), under `state/forward_deployments/binance-1h/`:

| Path | Content |
|---|---|
| `paper/paper_ledger.json` (+ `.bak`), `paper/archive/*.jsonl` | simulated balance, positions, trades, decisions (overflow archived, never deleted) |
| `shadow.json` | rejected opportunities and their hypothetical outcomes (separate from paper) |
| `observations/YYYY-MM-DD.jsonl` | one line per symbol per completed candle (market snapshot, regime, signals, selection) |
| `feed_health/`, `steps/` (daily jsonl) | feed counters; every step's success and errors |
| `engine_state.json` | last observed candle per symbol (restart without duplicates) |
| `sessions/<session_id>/` | write-once `session_start.json` / `session_end.json` (or `session_end_recovered.json` after a crash), `heartbeat.json`, `session.log` (UTC) |
| `reports/{startup,daily,weekly,trial}/*.md` | write-once reports |
| `incidents/*.json` | unexpected_termination, data_outage, outage_recovered, ledger_corrupt, session_crash |
| `supervisor.log` | supervisor start/restart events (UTC) |
| `STOP` | create this file to stop gracefully |

**Other locations:**
- **Historical candles:** `data/historical/binance/*.csv` are git-ignored; their manifests are committed. Re-create them with
  `python scripts/fetch_history.py` (1h) and `--intervals 4h 1d`.
- **Validation:** v1 results are frozen in `docs/validation/`; v2 runs go in `docs/validation/runs/`; the consumed-window registry is in
  `docs/validation/runs/_consumed_test_windows/`.

## 6. Work completed (chronological; details in the docs named)
1. **Rebuild** of the trading engine, CoinEx v2 client (orders locked), backtester, risk manager, paper trader, ML filter and Streamlit UI (`docs/AUDIT_REPORT.md`).
2. **Second-pass audit:** 24 issues fixed (`docs/AUDIT_REPORT_2.md`).
3. **Real-market validation** under a pre-registered protocol (Binance 1h, 9 coins, 2019–2026): **no strategy has a credible edge** (`docs/REAL_MARKET_EVALUATION.md`, `docs/VALIDATION_PROTOCOL.md`, `docs/validation/`).
4. **Independent audit** of that validation: conclusion supported (`docs/AUDIT_REPORT_3_VALIDATION.md`, `docs/audit/`).
5. **P1–P4 remediation:**
   - provenance fingerprint
   - immutable runs
   - the eligibility gate before the final test
   - the ML promotion rules
   - `docs/AUDIT_REPORT_4_REMEDIATION.md`, `docs/VALIDATION_PROTOCOL_V2.md`, `scripts/validation_run.py`
6. **Data pipeline:** 1h / 4h / 1d data and cost provenance (`docs/data_quality/`).
7. **Live Binance public data layer** (`core/live_data.py`).
8. **Regime detector study:** it describes the past and **does not predict** the future (`docs/REGIME_STUDY.md`).
9. **Forward paper engine:** evidence-gated selection, shadow tracking, separated monitoring (`docs/LIVE_DATA_AND_FORWARD_PAPER.md`).
10. **Deployment readiness:** 10 defects found by tracing the production path and fixed, plus the session manager, supervisor, reports and trial report (`docs/FORWARD_DEPLOYMENT.md`).

**Test status:**
- **Full suite:** 261 passed, pytest exit 0, in 18 min 11 s, on the code committed in `4523055` + `08e27c5`. The only
  warnings are 2 pandas FutureWarnings from `core/forward_reports.py:90`.
- **Paper regression run:** 29 passed. The two lock-concurrency tests passed 3 of 3 repeats after the Windows
  lock-release retry fix.
- **Disposable deployment check** (state in a scratch folder, real Binance public data):
  - `START_FORWARD_PAPER.bat disposable-test` started the supervisor and a session.
  - Force-killing the session made the supervisor restart it 10 s later. The new session wrote `session_end_recovered.json`
    for the old one and an `unexpected_termination` incident.
  - `STOP_FORWARD_PAPER.bat disposable-test` stopped the session gracefully (`session_end.json`, reason "STOP file present"), and the
    supervisor exited.
  - Trial report checks: records_persisted, no_duplicates, recovery_tested, no_live_orders_possible and
    no_positions_without_approval all true. Uptime was false only because the sessions were killed and stopped on purpose.
  - 8 observations across two sessions and 5 steps: one per coin for the single completed candle, all unique. 0 positions.

## 7. Outstanding tasks, errors, warnings, blockers
**Tasks, in order:**
1. **Wait for the user's explicit approval**, then start the 24h trial from the clean, committed tree with `START_FORWARD_PAPER.bat`.
2. Optional: kill the session process once during the trial, to repeat the crash-recovery proof on the real deployment.
3. After 24 hours, run `python run_forward_session.py --deployment binance-1h --trial-report --since <start UTC>`
   and review the verdict and checks.

**Known warnings and limitations:**
- **Test hygiene only:** `tests/test_forward.py` still opens some files without closing them (ResourceWarnings), and
  `core/forward_reports.py:90` raises a pandas FutureWarning (fillna downcasting). Neither affects results.
- **Sleep pauses the session:** the PC must stay awake. Sleep or hibernation pauses it; on wake, gaps are backfilled with real candles
  and entries are blocked until the data is clean.
- **Daily/weekly reports need a running session:** they're only written while a session is running across a UTC day or week boundary. Missed ones can
  be produced on demand: `--report daily --date YYYY-MM-DD`.
- **Feed counters reset per session:** reports state that market-data deltas are within-session.
- **CoinEx is unreachable from this machine** (HTTP 503). Exchange integration is deferred to the final stage and is unverified.
- **Strategies:** no strategy has passed validation, so the forward system will open **no** positions (by design).

**Blockers to the trial:** none technical. It waits only for the user's approval.

## 8. Safety protections currently active
- **Live trading locked:** `core/trade_execution.py: LIVE_TRADING_LOCKED = True`. No live executor exists, and
  `CoinExClient.place_order/cancel_order` raise unless `allow_trading=True`, which no app code sets (static test).
- **Forward market-data client is allowlisted:** read-only `GET`s to 5 Binance public endpoints only
  (`core/live_data.py: ALLOWED_PATHS`). Anything else raises before any request is sent.
- **Forward path is clean:** no credentials are read, and a static test forbids order, signing or credential code there.
- **Evidence-gated entries:** entries need a v2-validated approval (`core/selection.py`). None exists, so the system is inactive. Forward sessions never write
  validation runs, so there's no automatic promotion.
- **Shadow is hypothetical only:** it never touches the ledger or executor.
- **Risk controls:**
  - Conservative profile: 0.5% risk per trade, ≤ 10% per position, ≤ 30% exposure, 3% daily loss limit, 10% drawdown halt and kill switch
  - cost-to-risk filter
  - stale-quote, stale-candle and missing-candle entry guards
  - duplicate-order and duplicate-position protection
  - ledger reconciliation gate
  - corrupt ledger never silently reset
- **Frozen v1 results are hash-checked** by `tests/test_frozen_v1.py`, and the v1 test window is registered as consumed.

## 9. Commands (run from `(CURRENT) OnlyFunds/`)
| Purpose | Command |
|---|---|
| Start (supervised, auto-restart) | `START_FORWARD_PAPER.bat` (= `python scripts\forward_supervisor.py --deployment binance-1h --clear-stop -- --interval 1h --every 60 --websocket`, in a minimised window) |
| Start in the foreground (one session) | `python run_forward_session.py --deployment binance-1h --interval 1h --every 60 --websocket` |
| Stop gracefully | `STOP_FORWARD_PAPER.bat`, or create `state\forward_deployments\binance-1h\STOP` |
| Resume after a stop | delete the `STOP` file (`START_FORWARD_PAPER.bat` does this), then start again; evidence continues in the same deployment folder |
| Health now | open the newest `state\forward_deployments\binance-1h\sessions\*\heartbeat.json`; `supervisor.log`; `incidents\` |
| Is it running? | PowerShell: `Get-CimInstance Win32_Process -Filter "Name='python.exe'" \| Select ProcessId,CommandLine` (look for `run_forward_session.py` / `forward_supervisor.py`) |
| Daily / weekly report | `python run_forward_session.py --deployment binance-1h --report daily --date 2026-11-02` (`--report weekly --date <Monday>`) |
| Trial report | `python run_forward_session.py --deployment binance-1h --trial-report --since 2026-11-01T00:00:00Z` |
| Separated monitor | `python -c "from core.monitoring import report; print(report('state/forward_deployments/binance-1h', 'docs/validation/runs'))"` |
| Tests | `python -m pytest` |

**Don't** run two sessions on the same deployment. A second session refuses to start while the first one's PID is alive.

## 10. Recommended next steps
1. Finish §7 steps 1–3. Then keep the session running for weeks without changing strategy parameters.
2. Read the weekly reports: shadow outcomes by rejection reason, and feed reliability. Treat fewer than 100 outcomes as
   INSUFFICIENT evidence, not as negative evidence.
3. **New strategy research** (only if wanted):
   - run it under `docs/VALIDATION_PROTOCOL_V2.md` with `scripts/validation_run.py`
   - use a **fresh final-test window after 2026-10**: the 2025-01 → 2026-11 1h window is consumed
   - with CoinEx-level fees, focus on ideas whose gross edge comfortably exceeds about 0.16–0.6R of costs per trade
4. CoinEx integration is the **last** stage, and needs separate explicit approval before any live capability.

## 11. Decisions and constraints a new conversation must preserve
- Never enable live trading, place real orders, request or store API secrets, or merge to `main` without explicit approval.
- Never modify the frozen v1 artefacts (`docs/validation/*`, `docs/VALIDATION_PROTOCOL.md`, `docs/REAL_MARKET_EVALUATION.md`)
  or reuse the consumed TEST window. New validations go through the v2 runner (immutable runs, provenance, P3/P4 gates).
- Don't tune strategies on forward or shadow evidence and then present the same evidence as validation. Keep historical,
  validation, forward-paper and shadow results separate, and never combine them into one figure.
- Selection is evidence-gated: with no approval, OF2 stays inactive. That's the intended behaviour.
- Binance data never establishes CoinEx profitability, liquidity or execution.
- Datasets are never silently replaced (`save_dataset` refuses); reports and run records are write-once.
- Work on `of2-completion`; commit each coherent, tested change separately and push; keep PR #1 open.
- Environment the results were verified on: Windows 11, Python 3.9.13, pandas 2.2.3, numpy 2.0.2, scikit-learn 1.6.1.
