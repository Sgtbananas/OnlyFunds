# OF2 safety, stop-risk cap and shadow-monitoring review (2026-11-02)

**Scope.** This is an engineering review of `of2-completion` at `d303f11`. No strategy was developed or changed. T50, RS3, X1, X2 and
their pass criteria are unchanged, and their fingerprints still verify. Live trading stays locked and no credentials were used.
No research result changed. Updated after the corrections review (notifications, integrity guarantees, stale prices). **The fixes listed below are uncommitted, pending your approval.**

## 1. Live-trading lock

| Check | Result | Evidence |
|---|---|---|
| Can live trading be enabled by accident? | **No.** The lock is the code constant `LIVE_TRADING_LOCKED = True` (`core/trade_execution.py:26`). No config key, environment variable or UI control reads or sets it. `config.yaml` `trading.mode` only accepts `paper`, and `Config.save` refuses any other value. | `tests/test_safety.py` asserts the constant |
| Can any exchange order be submitted? | **No call site exists.** The only order code is `CoinExClient.place_order` / `cancel_order`, and nothing in the app, the scripts or the shadow tracker calls them. `place_live_order` and `get_executor("live")` raise. `run_live.py` exits with code 2. The legacy grid trader takes an injected function and is not wired to anything. | grep of every call site |
| **Weakness found and fixed** | `place_order` / `cancel_order` were guarded only by the client's own `allow_trading` flag, which ignored the global lock. One line, `CoinExClient(allow_trading=True)`, with credentials in the environment, would have sent a real order. | fix W1 |
| Does shadow tracking need credentials? | **No.** It reads only public Binance daily candles. The wrapper strips `COINEX_*`, `BINANCE_API*`, `API_KEY*` and `API_SECRET*` from the environment before importing the tracker. | `test_shadow_task.py` |
| Does the shadow process have an execution path? | **No.** The tracking files contain no order, signing or credential code. The health check scans for these patterns on every run, and a test blocks every exchange-client constructor during a full update. | `shadow_health.py` check 8 |
| Can a setting silently bypass the protections? | **No.** After fix W1, unlocking needs two deliberate code changes: the constant, and `allow_trading=True`. The lock fails closed: if it cannot be read, it counts as locked. | `test_coinex_client.py` |

**Minor note, not a defect.** The `assert LIVE_TRADING_LOCKED` lines in the paper scripts are removed under `python -O`. That
does not matter, because those paths only ever construct `PaperExecutor`.

## 2. The 10% / 8% stop-risk cap (`open_risk_cap=(0.10, 0.08)`; research only, not frozen, not active anywhere)

**How it works.** At each daily close, stop risk = Σ max(qty × (close − stop), 0) over open positions: the loss if every stop
filled at its stop price at once. If that is above 10% of equity, positions are sold at the next open, largest contributor
first, until stop risk is back to 8%. The cap only ever sells.

| Question | Finding |
|---|---|
| Are all open positions included? | All positions staying overnight are included. Positions already exiting at the next open are excluded; they leave at that open. **Defect fixed (W2):** a held coin with no price bar that day was left out of the total. It is now counted at its last close, and other positions are cut, because it cannot be sold. This changed no published result: no held coin lacked a bar in 2019–24, and all 16 published capped results reproduce to 0.00 points. |
| Pending signals and orders? | **Not counted when the cap decides.** Entries decided at the same close fill at the next open, after the trims. Each new entry adds about 0.5% of equity in initial risk (`risk_per_trade`), and at most 3 positions can be open, so the overshoot is bounded at about 1.5%. |
| Is sizing based on current equity? | **Yes.** Entries are sized on equity marked at the fill open, after that day's exits and trims. Sizing includes both fees and stop slippage, and rounds quantity down. The cap uses equity at the close. |
| Correlated positions? | The sum of stop risks already assumes every stop is hit together, so correlation cannot push a stop-out above the cap. It makes that outcome likely, though: the coins held correlate at 0.52–0.70. Correlation matters for losses *beyond* the stops (see gaps). Correlation-weighted caps were tested in Rounds 3–4 and rejected. |
| Can rounding cause a breach? | In simulation, no: quantities are continuous. Live sizing rounds down, which lowers risk. A rounded-down trim could leave stop risk above target by less than one lot. |
| Missing data or stale prices? | **Fail-safe since correction 3 (W5).** A held coin with no bar that day is still counted in every aggregate figure: stop risk, gross exposure and correlation exposure, all at its last close, which is never treated as current. Each day's risk record says how many positions lack a current price, which ones, the age of the oldest price, and whether the risk figure is reliable. While any held position lacks a current price, no new entry is approved at the close and none is filled at the open; each blocked entry is counted. Its own risk cannot be cut until it trades again, so other positions are cut instead. A delisted coin cannot be sold at all, so its real loss can still exceed its stop. |
| Is the cap exceeded in practice? | It is an end-of-day cap with a one-day lag. Days above 10%: DEV 8 of 1,461 (max 11.2%), VAL 3 of 731 (max **17.0%**, TRX's +75% day on 2024-12-03), HOLDOUT 5 of 2,192 (max 11.7%). Every breach was trimmed at the next open, and none lasted more than two consecutive closes. |
| Costs in the stop-risk figure? | The figure excludes the exit fee and stop slippage. Adding them raises the maximum by 0.02–0.06 points. |
| Gaps and sudden jumps | **This limitation must be stated explicitly.** Crypto trades around the clock, so a daily open almost equals the previous close. In 2019–24 no T50 stop was ever gapped through: 0 of 49 stop exits in the capped books and 0 of 49 in the uncapped shared-account books, and no trade did worse than −1.00R in any of 446 per-coin trades. The simulation assumes stops are watched continuously and fill at the stop minus 0.2% (0.5% under stress costs). It does not model an intraday crash or thin liquidity that fills far below the stop. The only proxy is the 30%-gap figure: 0.3 × gross exposure, at most about 9% of equity. On the upside, a one-day jump (TRX) can push stop risk well past the cap before any end-of-day rule can act. |
| Kill switch | It never trips at base costs. **Correction to the Round 4 summary:** under stress costs, the capped HOLDOUT book's drawdown is 10.26%, above the 10% kill threshold. "Never trips the kill switch" holds at base costs only. |

## 3. Shadow monitoring

**Verified:**
- **Schedule.** One task, `\OnlyFunds\OF2 Shadow Daily Update`, runs at 00:30 UTC. Its UTC-pinned start boundary shows as 11:30 local and does not drift with daylight saving.
- **Overlap and catch-up.** A second instance is never started (`IgnoreNew`), and a missed start runs as soon as possible (`StartWhenAvailable`).
- **Run state.** Last run 2026-11-02 00:30 UTC, result 0, no missed runs. Each run also takes an exclusive OS lock (exit 9).
- **Crashes.** A crash is journaled with its traceback (exit 10).
- **Fingerprints.** All 120 snapshots verify against their recorded SHA-256. The T50, RS3, X1 and X2 fingerprints match.
- **Runtime.** The 1,000-run random benchmark takes about 6 minutes on today's data and about 11 minutes after 24 months of forward data, far inside the 2-hour time limit.

**Weaknesses found and fixed (W3, W4):**
- **Nobody would have seen a failure.** The health check had to be run by hand.
- **Problems the health check ignored:**
  - the frozen tracker's own anomaly flags (`DATA_REVISION`, `COMPARATOR_MISMATCH`);
  - abnormal run statuses;
  - experimental steps that exited non-zero;
  - missed days;
  - evaluation lag;
  - corrupt or gappy snapshot data;
  - a coin that silently drops out (the frozen tracker treats an empty Binance reply as a delisting and reuses the old snapshot);
  - package-version changes.
- **No tamper evidence.** The forward record had no check against deletion or editing.
- **Silent wrapper failure.** An exception in the wrapper after the tracker had run would end a `pythonw` process silently.

**Is exit code 11 a notification? No.** Task Scheduler shows "Last Run Result" only to someone who opens the task.
The task has no e-mail or display action, and `alerts.jsonl` is just a file. Before correction 1, a failure produced
no visible signal at all.

**What now happens on every scheduled run** (correction 1). All of this is in the wrapper, the health check and the
new notifier; the frozen tracker is not touched.
1. **Ledger.** New runs are added to the hash-chained ledger (section 3b).
2. **Health check.** Every finding is an issue with a stable key and a level:
   - **critical:** the record may be wrong, incomplete or altered, or tracking did not run. Examples: a fingerprint mismatch, a snapshot hash mismatch, a ledger or checkpoint problem, a crash or non-zero exit, a `DATA_REVISION` / `COMPARATOR_MISMATCH` anomaly, an abnormal run status, corrupt or gappy snapshot data, a missed day, order code found.
   - **warning:** worth a look, results not known to be wrong. Examples: evaluation lag, a coin dropping out between runs, a Python package change.
3. **Visible notification** (`research/shadow_notify.py`).
   - **Which issues.** One native Windows toast per run in which an issue is new, or has escalated from warning to critical.
   - **Format.** The title gives the level, e.g. "OF2 shadow CRITICAL: 2 new issues". The next line gives the date and time in UTC and local time, plus the open critical and warning counts. Then up to three failed checks, each tagged [CRITICAL] or [WARNING], and the full path to the report.
   - **Critical toasts** stay on screen until dismissed. Every toast also stays in the Windows notification centre.
   - **Delivery check.** Delivery is confirmed by reading the toast history back. If the toast fails, `msg.exe` shows a message box instead.
   - **Dependencies.** Built-in Windows PowerShell 5.1 and `msg.exe` only. No packages, no network, no service.
4. **No repeats.** `state/shadow/alert_state.json` holds the open issues with their first-seen and notified times. An issue that stays open is not notified again. When it clears, it is logged as resolved, and if it later reappears it is notified once more.
5. **Records.** `state/shadow/alerts.jsonl` is append-only. It records every raised, escalated and resolved issue, with the result of the notification. The record is written whether or not the notification succeeds. `HEALTH_LATEST.md` starts with CRITICAL, WARNING or OK and lists the open issues.
6. **Exit code.** The task exits with code 11 while any critical issue is open, unless the tracker itself failed, in which case its own code wins. Warnings alone exit 0.
7. **Checkpoint.** It is refreshed after the journal is written (section 3b).
8. **Provenance and wrapper errors.** As before: the wrapper's SHA-256 and the package versions are recorded, and any wrapper error is journaled as CRASHED.

**Tested with a real notification.** A simulated critical failure was run through the real wrapper on a temporary
state. The tracker exited 5, and its run record was marked "SIMULATED FAILURE (alert test only; nothing is wrong)".
It produced a critical toast, confirmed in the Windows toast history at 2026-11-02 02:35 UTC. The toast read
"OF2 shadow CRITICAL: 2 new issues", gave the time, and listed the abnormal run status and the exit-5 journal entry.
The test is opt-in (`OF2_TEST_REAL_NOTIFY=1`), so the normal suite never pops notifications. To check notifications
yourself at any time: `python research/shadow_notify.py --test critical`.

**Limits of the notification.** It appears only while you are logged in to this PC. Focus assist hides the banner
but keeps the toast in the notification centre. Turning off notifications for Windows PowerShell falls back to the
message box. Nothing can notify while the PC is off.

A dry run on today's real state, checkpoint included, found nothing to report. The five delisted coins and ICX,
stale since 3 September, appear as info.

### 3b. What the shadow-data integrity checks can and cannot detect (correction 2)

The ledger is a SHA-256 chain: each entry holds the hash of every file of one stored run and the hash of the
previous entry. No secret key is involved, so **nothing here is tamper-proof**. Anyone who can write to this PC
can rebuild a consistent chain. Before correction 2, deleting the whole ledger, or removing its last entries, went
undetected: the next run silently rebuilt it as if it were new.

**Correction:** a checkpoint now lives *outside* the project, in `%LOCALAPPDATA%\OnlyFunds\shadow_checkpoint\ledger_checkpoint.json`.
- **Contents.** The ledger's entry count and head hash, and the byte length and hash of `task_journal.jsonl` and `alerts.jsonl`. It holds only hashes, sizes and times: no market data, no credentials, nothing uploaded.
- **When it updates.** After each run.
- **When the state no longer matches:**
  - it refuses to update, so an alteration is never absorbed;
  - the ledger is not rebuilt;
  - the issue is raised as critical, and stays open, until a person looks.

| Change | Without the checkpoint | With the checkpoint |
|---|---|---|
| A stored run file edited | detected (file hash) | detected |
| A file or a whole run deleted | detected (listed in the ledger) | detected |
| A ledger entry edited | detected (chain) | detected |
| A ledger entry removed from the middle, or entries reordered | detected (chain and sequence numbers) | detected |
| The last ledger entries removed | **not detected** | detected ("truncated") |
| The entire ledger deleted | **not detected** (looks never created) | detected ("deleted"); not rebuilt |
| The journal or alert log edited, or cut short | not detected | detected (length and prefix hash) |
| Runs and ledger rewritten consistently, checkpoint left alone | not detected | detected ("rewritten") for anything recorded before the last checkpoint |
| Runs, ledger **and** checkpoint all rewritten consistently | not detected | **not detected**: no local mechanism can catch this |
| The checkpoint itself deleted | n/a | not detected, but harmless on its own: the next run writes a fresh one |

**Does the separate checkpoint add meaningful protection? Yes, but limited.**
- **What it catches:** the realistic accidents and casual edits. Examples: a clean-up deleting the git-ignored `state/` folder (for example `git clean -X`), a log trimmed by hand, a restored backup missing recent runs.
- **What it cannot catch:** a deliberate local rewrite of both locations.

The tests demonstrate both sides, including the undetectable case. The only further protection would be to keep
the ledger head off this machine, for example by writing it down. It is a short hash and reveals no data. That
needs no upload of shadow data, and is left to you.

## 4. Reproducibility (`python research/review_reproduce.py`; committed code and data, by an independent route where possible; about 25 s)

| Figure | Published | Recomputed | Route |
|---|---|---|---|
| T50 every-signal total, 25 coins, 2021–24 | 289 trades, +251.205R | 289, +251.205R | research engine **and** the frozen forward tracker's own engine (`forward_shadow.signal_book`) |
| GM2-gated total | 223 trades, +188.947R (−62.26R) | 223, +188.947R; 1,515 blocked days | the saved Round 1 M2 models, no retraining |
| Shared account (all 25 coins, 2021–24, no halt) | +41.20R, +20.99%, DD 19.23%, 78 positions | identical; kill switch (as configured) trips 2021-05-19 | frozen single-account simulator |
| Simultaneous stop-loss exposure (DEV) | 24.196% | 24.196% (24.21% with exit costs) | rebuilt from the trade log and closing prices only |
| 10% / 8% cap (DEV / VAL / HOLDOUT) | return / DD 4.89 / 4.23 / 3.37 | identical | trade-log stop risk matches the simulator's column to 4e-15 points |
| X1 vs X2, HOLDOUT (VAL) | 3.60 vs 2.32 (3.13 vs 3.28) | identical | frozen experimental simulator `portfolio_variants` |

### 4b. All published results re-run after the corrections

**All published results re-run** (`python research/review_published_unchanged.py`, about 7 minutes). Every published
single-account cell was recomputed with the corrected simulators and compared with the committed results files:
- Round 3: 96 cells, every candidate, split, halt mode and cost setting;
- Round 4: 88 cells, every book, including the regime-driven ones;
- reconciliation: 28 risk profiles and cells.

**Result: 3,632 numeric fields compared, largest absolute difference exactly 0, no mismatches.** No Round 4 book had a single day with an unpriced held position.

## 5. Forward-testing integrity
- **Frozen definitions.** No frozen file changed. `load_frozen()` reports no problems for T50/RS3 or X1/X2, and the fingerprints are unchanged: T50 `29139c64…`, RS3 `1ffc1d23…`, X1 `3de65f01…`, X2 `c2edc9f7…`.
- **Isolation of the edited files.** The wrapper, health check, notifier, CoinEx client and the two research simulators are outside every fingerprint list. They cannot alter what the tracker computes, and the tracker's own code is byte-identical.
- **Rules.** No filter was added and no parameter was touched. There are no shadow results yet (first evaluated day: 2 November), so nothing could have been tuned on them.
- **Separation.** The forward record lives only in `state/`, from the 2 November start. Pre-start candles are used only to warm up indicators. Reports never combine forward and historical results.
- **Preservation.** Every daily snapshot is kept. The tracker never deletes or overwrites a run. Deviations and failures are journaled, alerted and recorded in `alerts.jsonl`.

## 6. Remaining limitations
1. **Single copy.** `state/` is git-ignored and exists only on this PC. The checkpoint detects its loss but cannot restore it. Recommended: copy `state/shadow` off the machine regularly, and keep the ledger head somewhere else.
2. **No tamper-proofing.** A deliberate, consistent rewrite of the runs, the ledger and the checkpoint is undetectable locally (section 3b).
3. **The PC must be on and logged in.**
   - The task uses an interactive token. While the PC is off, nothing runs and nothing can notify; the missed day is reported, and notified, on the next run.
   - The next run recomputes every past day, so no evaluation is lost, but that day's snapshot is.
   - There is no watchdog off the machine.
4. **What the notification needs.** It works only while you are logged in, with notifications allowed for Windows PowerShell. If those are off, it falls back to a message box. Focus assist hides only the banner.
5. **Frozen-tracker behaviours, now flagged but not changeable without a re-freeze:**
   - it still evaluates a snapshot with missing days;
   - it treats an empty Binance reply as a delisting.
6. **Package versions are not pinned.** A numpy or pandas upgrade raises a warning, but is not prevented.
7. **What the stop-risk cap cannot cover:**
   - an intraday crash below the stops;
   - a one-day jump;
   - a coin that cannot be sold;
   - entries pending at the decision (about 1.5% at most).

   It is research-only, post hoc, not frozen, and not active anywhere.

## Fixes (uncommitted)

| ID | Files | Change |
|---|---|---|
| W1 | `core/coinex_client.py`, `tests/test_coinex_client.py` | The global lock overrides `allow_trading`. Non-GET requests are refused while locked. Fails closed. The mocked order-format tests lift the lock explicitly, in tests only. |
| W2, W5 | `research/portfolio_pm.py`, `research/portfolio_engine.py`, `tests/test_portfolio_pm.py`, `tests/test_portfolio_engine.py` | Every held position counts in the open-risk, gross and correlation caps, including one with no current price. Each day's risk record flags unpriced positions, price age and reliability. No entry is approved or filled while any held price is missing. |
| W3 | `research/shadow_health.py`, `tests/test_shadow_health.py` (fixture now a well-formed snapshot), `tests/test_shadow_monitoring.py` (new) | Extended checks; issues with keys and critical / warning levels; ledger; checkpoint outside the state folder. |
| W4, W6 | `research/shadow_task.py`, `research/shadow_notify.py` (new), `tests/test_shadow_alerts.py` (new) | Open-issue state, one notification per new or escalated issue, a resolution log, exit code 11 while a critical issue is open, checkpoint refresh, provenance, wrapper errors journaled. |
| — | `research/review_reproduce.py`, `research/review_published_unchanged.py` (both new), this document (new) | Rerunnable verification of sections 4 and 4b. |

**Tests.**
- Full suite: **373 passed, 1 skipped**, in 18 min 48 s (335 at `d303f11`; 39 new tests).
- The skipped test is the opt-in real-notification test. Run on its own with `OF2_TEST_REAL_NOTIFY=1`, it passed, and the toast was confirmed in the Windows history.
- The new simulator tests fail on the committed simulators and pass with the fixes.
- `research/review_published_unchanged.py`: 0 mismatches over 3,632 fields. `research/review_reproduce.py`: all six figures identical.
- Fingerprints verify, and `LIVE_TRADING_LOCKED` is True.

**The scheduled task runs the working tree.** The next run, 2026-11-03 00:30 UTC, will use the new wrapper, health
check and notifier even though they are uncommitted. The tracker and its fingerprints are unaffected, and the
journal records the wrapper's hash. On its first run it will:
- create the ledger, marking the six existing runs as backfilled;
- create the checkpoint folder under `%LOCALAPPDATA%\OnlyFunds`.
