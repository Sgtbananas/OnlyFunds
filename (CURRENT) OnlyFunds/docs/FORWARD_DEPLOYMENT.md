# Long-running forward paper deployment

Paper trading only. **No exchange order endpoint is reachable, no credentials are used, live trading stays locked.**
The objective is reliable evidence collection, not trades. With no validated strategy, OF2 observes, logs and
shadow-tracks every opportunity, and opens no positions.

## Start / stop (Windows)
| Action | How |
|---|---|
| Start (with automatic crash restart) | double-click `START_FORWARD_PAPER.bat` → `scripts/forward_task.ps1 -Action start` → Task Scheduler task `\OnlyFunds\OF2 Forward Paper binance-1h` → `pythonw scripts/forward_supervisor.py` → `run_forward_session.py --interval 1h --every 60 --websocket` (8 coins) |
| Stop gracefully | `STOP_FORWARD_PAPER.bat` (creates `STOP`; the session writes its end record and exits; the supervisor exits too). **Do not** use "End task" in Task Scheduler: it kills without an end record. |
| Status | `STATUS_FORWARD_PAPER.bat` (task state, supervisor/session PIDs, heartbeat age, last log lines); details in `sessions/<id>/heartbeat.json`, `supervisor.log`, `session_console.log`, `incidents/`, `reports/` |
| Reports on demand | `python run_forward_session.py --report daily --date YYYY-MM-DD`; `--trial-report --since <UTC>` |
| Remove the task | `powershell -NoProfile -ExecutionPolicy Bypass -File scripts\forward_task.ps1 -Action unregister` (refused while running; records untouched) |
| Start at logon (optional, done by you) | Task Scheduler → the task above → Triggers → New → "At log on". The task has no triggers by default. |

### Why the Task Scheduler (process lifecycle)
The first 24h trial (session `20261101T091214Z-7a6be833`) was launched with `start` from a Claude Code tool shell and both
processes disappeared about 3 minutes later with no exit logged. They were part of the process tree and job objects of
Claude's tool processes. The exact trigger could not be reproduced, but the fix removes that dependency entirely. The
supervisor is now started by the Task Scheduler service: its parent is `svchost.exe` (Schedule), it has no console window to
close, and terminals, editors and Claude Code are not its ancestors.

**Task settings:**
- current user, runs only while logged on (no stored password)
- no execution time limit
- one instance only (plus `supervisor.pid` guard)
- runs on battery
- the Task Scheduler re-launches the supervisor (3×, 1 min apart) only if the supervisor itself fails

**Verified 2026-11-01 on a disposable deployment:**
- the supervisor's parent was `svchost.exe`
- a force-killed session was restarted after 10 s, with an `unexpected_termination` incident
- supervisor and session survived a Claude Code turn ending
- a second start was refused

Still to verify: that the supervisor and session survive the Claude Code app being closed entirely.

Logging off, restarting Windows, sleep and hibernation still stop or pause the deployment. After a reboot, run
`START_FORWARD_PAPER.bat` again, or add the "At log on" trigger.

The PC must stay on and awake. Sleep or hibernation pauses the process. On wake, the session detects the gap, backfills
real candles, and the missing-candle guard blocks entries until the data is clean again.

## Deployment folder `state/forward_deployments/<name>/` (git-ignored)
| Path | Written how | Content |
|---|---|---|
| `paper/paper_ledger.json` (+ `.bak`) | atomic + fsync | simulated balances, positions (including trailing-stop state), trades, decisions |
| `paper/archive/*.jsonl` | append-only | decisions / equity points moved out of the ledger: **never deleted** |
| `shadow.json` | atomic + fsync | rejected opportunities and their hypothetical outcomes (never mixed with paper) |
| `observations/YYYY-MM-DD.jsonl` | append | one line per symbol per completed candle |
| `feed_health/`, `steps/` (daily jsonl) | append | feed counters and every step's success/errors |
| `sessions/<session_id>/` | write-once start/end records, heartbeat, UTC log | one folder per process run; previous sessions are never modified |
| `reports/{startup,daily,weekly,trial}/` | write-once | reports are never overwritten |
| `incidents/*.json` | write-once | unexpected termination, data outage, outage recovered, ledger corruption, session crash |

## Deployment-readiness audit: production path traced, defects fixed
| # | Defect found | Fix |
|---|---|---|
| D1 | Ledger trimmed decisions to the latest 3,000 and equity points to 5,000: **evidence deleted** | overflow moved to append-only `paper/archive/` before trimming (`PaperTrader._archive_overflow`) |
| D2 | Candle completeness judged by the local clock | Binance server-time offset, re-synced hourly (`BinancePublicClient.now_ms`) |
| D3 | Recovery after a long outage fetched one 1,000-candle page and left the rest | backfill pages through the whole gap; what the exchange doesn't have stays missing (`LiveMarket.refresh_candles`) |
| D4 | Lock file of a crashed process blocked every step for 5 minutes | lock holder PID checked; a dead holder is released immediately (`pid_alive`, `ledger_lock`) |
| D5 | Safety relied on the absence of order code, with no enforcement | market-data client refuses any path outside a 5-endpoint read-only allowlist (`ALLOWED_PATHS`) |
| D6 | Snapshot errors swallowed into the observation | counted in engine state and logged |
| D7 | Local-time log timestamps; one ever-growing feed-health file | UTC log formatter; daily feed-health files |
| D8 | No session identity, end record, crash detection, back-off or reports | `core/session.py`, `core/forward_reports.py`, `core/trial.py`, supervisor |
| D9 | Long-running code leaked file handles (`json.load(open(...))`) | `read_json` helper with explicit close everywhere |
| D10 | A same-PID session would have been declared crashed instead of refused | any live PID (including our own) blocks a second session |

**Already correct (verified by tracing, not just tests):**
- **No duplicates:** the per-pair last-processed-candle (ledger) and per-pair last-observed-candle (engine state) stop duplicate trades and observations.
- **Positions survive restarts:** open positions persist with their trailing state.
- **Shadow stays separate:** the shadow tracker has no access to the executor.
- **UTC throughout:** all timestamps are UTC.
- **Guards hold:** the stale-quote, stale-candle and gap guards block entries.

## Paper-only safety (enforced and tested)
- **No order endpoint:** `core/live_data.py: ALLOWED_PATHS` holds market-data GETs only. Any other path raises before a request is made.
- **No order capability in the forward path:** a static test asserts no forward file contains order or signing code, CoinEx client use, credentials or POST calls.
- **No live executor:** `LIVE_TRADING_LOCKED = True`; there is no live executor and no credentials are read.
- **No bypassing validation:** entries pass `StrategySelector`, which accepts only v2-validated approvals. There are none, so it stays inactive (tested).
- **No automatic promotion:** forward sessions never write validation runs (tested); approvals come only from immutable v2 run folders.
- **Shadow can't become paper:** shadow outcomes are computed from candles only and never touch the ledger or executor.

## Session lifecycle
- **Start:**
  1. Detect and record any previous session that ended without an end record, and file an incident.
  2. Refuse to start if a previous session's process is still alive.
  3. Create a new session folder.
  4. Write the start record: git commit and dirty flag, config fingerprint, data source, universe, environment, PID.
  5. Write the startup health report.
- **Loop:**
  1. Run a step and log its result.
  2. Write the heartbeat.
  3. On consecutive failed steps, back off exponentially (bounded at 15 min); after 3 failures, file a data-outage incident, and an outage-recovered incident when it clears.
  4. Write daily and weekly reports at UTC rollovers.
- **Stop:** `STOP` file, Ctrl+C or SIGTERM/SIGBREAK cause a graceful end record. A crash writes an incident and is restarted by the supervisor (bounded back-off of 10 s–15 min).

## Evidence wording in reports
- **Fewer than 100 outcomes:** **INSUFFICIENT**, which is not a negative result.
- **100 or more:** a bootstrap 95% interval decides:
  - **NEGATIVE** if the whole interval is below 0
  - **POSITIVE (forward sample, not a validation)** only if it's wholly above 0
  - otherwise **INCONCLUSIVE**

Strategy parameters are not changed during evidence collection. Forward evidence is never used to tune and then re-presented
as independent validation.
