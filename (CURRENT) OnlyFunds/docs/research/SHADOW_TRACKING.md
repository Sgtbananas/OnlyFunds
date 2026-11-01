# Forward shadow tracking of T50 and RS3: prospective validation only

**Neither strategy is approved for trading.**
- No exchange orders, no keys, no capital.
- `LIVE_TRADING_LOCKED` and the deployment are untouched.
- Prospective results are reported **separately** and are never combined with historical results.

## What is tracked
| | T50 | RS3 |
|---|---|---|
| Rules | daily close > previous 50-day high → buy next open; exit next open after close < previous 25-day low; 3-ATR(14) fixed stop; no TP/trailing | Monday close: top 3 by 28-day return, return > 0, BTC > 100-day SMA → buy next open; leave at the next Monday close if any condition fails; 3-ATR fixed stop |
| Universe (fixed before any forward data) | 25 coins: the 9 OF2 coins + the rule-selected 16-coin holdout. Coins without live Binance data (e.g. EOS, delisted 2025) are recorded as unavailable. | the 9 OF2 coins (as pre-registered) |
| Status before tracking | Failed Round 1 on trade count only (62 < 100 VAL trades); see `T50_SCRUTINY_SUMMARY.md` | Failed Round 1 (82 VAL trades; positive on 50% of coins < 60%) |

**Frozen fingerprints and full spec:** `research/shadow/FROZEN.json`.
- **Current fingerprints** (frozen 2026-11-01T12:21:54Z at commit `0acc195`):
  - T50 `29139c64b6c0ad9d7c0c9085b56ecba95199de0fa0d5d9b8cb49753ed04910c7`
  - RS3 `1ffc1d23ab8d26e22851733fe8ea731d8f854bf0549f1f1e1d5187a989ab0fec`
- **First freeze superseded:** the first freeze (12:20:09Z) was replaced before any forward evaluation. The reason
  is in `research/shadow/superseded/README.md`.

**Universe availability at the start (snapshot 2026-11-01):**
- **T50:** 20 of 25 coins have data. 19 are current. ICXUSDT stopped at 2026-09-03 and is kept as stale. EOS, NULS,
  OMG, WAVES and XMR return no data.
- **RS3:** 8 of 9 coins (EOS is delisted). Each fingerprint is the SHA-256 of the
strategy's rules, parameters, universe, costs, risk limits and benchmark definition, together with the SHA-256 of
every implementing file. `update` refuses to run if any of them changed: the strategy is stopped, and earlier
results are preserved.

## Why RS3 is tracked, and what it has NOT shown
RS3 has **not** demonstrated an edge. Its evidence (DEV/VAL 2019–2024 and holdout; data ≥ 2025 never loaded):

| Check | Result |
|---|---|
| Pooled expectancy | DEV +0.53R (102 trades, 95% lower bound +0.16R); VAL +0.34R (84, lower bound **+0.04R**); holdout +0.31R (210, lower bound +0.09R) |
| Random baskets (Round 1, 200 runs, DEV+VAL) | +0.45R vs random 95th percentile +0.23R. Not yet re-run with the 1,000-run matched method. |
| Without the largest winner | VAL +0.23R, lower bound **−0.02R** (no longer significant) |
| Without the top 10% of trades | **Negative everywhere** (DEV −0.04R, VAL −0.06R, holdout −0.14R) |
| By BTC market state (2019–24, main) | bull +0.57R (152 trades); sideways −0.10R (23); bear −0.12R (11) |
| Years | Positive in 2019, 2020, 2021, 2023, 2024 (each individually not significant except 2020); 2022 −0.74R (6 trades) |
| Pre-registered criteria | **Failed** (trade count, breadth) |

**Why track it anyway:**
- It is a different mechanism (cross-sectional ranking) with a weak but consistent positive result.
- Tracking costs nothing.
- It gives T50 a comparison under identical forward conditions.
- **Prior: lower than T50.**

**Update (2026-11-02, `05_rs3_scrutiny.md`, rule H2-RS3 registered before its results):** RS3 is **not** explained by one
or two trades.
- **Main 9, 2019–24:** 186 trades. Without the top 2 trades the mean is +0.35R, with a lower bound of +0.11R.
- **Against random 3-coin baskets on the same Mondays:** 100th percentile on both the main 9 and the holdout 16
  (p = 0.002 each).
- **Holdout without its top 2 trades:** +0.21R.
- **Weaknesses that remain:**
  - Negative without the top 10% of trades, as with T50.
  - The holdout lost money in 2023 and 2024.
  - VAL without its single largest trade is not significant.
- **Decision:** RS3 research continues. It is **not** promoted, and its frozen tracking is unchanged.

## Experimental variants tracked alongside (separate fingerprints; `ROUND2_PROGRAMME.md` section A)
- **The variants:** X1 `T50-PM` (frozen T50 + R3 position management) and X2 `T50-PM-B` (X1 + BTC > 100-day SMA entry
  gate). Both were frozen on 2026-11-01 at 14:23 UTC, before the first forward decision, in
  `research/shadow/EXPERIMENTAL_FROZEN.json`.
  - X1 fingerprint: `3de65f01…`
  - X2 fingerprint: `c2edc9f7…`
- **How they run:** `research/forward_experimental.py` runs after each successful frozen run, inside the same
  scheduled job and lock. It reads that run's data snapshot read-only and re-verifies its SHA-256 hashes. It writes
  only to `state/shadow_experimental/`.
- **Isolation:** a crash or fingerprint failure there is journaled and cannot fail the frozen run.
- **Comparator:** the frozen T50 portfolio book, recomputed and cross-checked against the frozen run's own equity.
  Any difference is logged as `COMPARATOR_MISMATCH`.
- **Decision:** only after ≥ 24 months and ≥ 40 closed positions per book, under the criteria pre-registered in
  `ROUND2_PROGRAMME.md`. These are experiments, not approved strategies.

## How it works
- **Daily:** `python research/forward_shadow.py update`.
  - **Schedule:** registered 2026-11-02 with the user's approval.
    - Task: `\OnlyFunds\OF2 Shadow Daily Update`, daily at **00:30 UTC**. The trigger is stored with an explicit
      UTC offset ("synchronize across time zones"), so it does not move when daylight saving changes.
    - It runs `pythonw research\shadow_task.py`, a wrapper around `forward_shadow.py update`. The wrapper is not
      part of the frozen fingerprint, and it changes nothing that is evaluated.
    - Register again: `powershell -NoProfile -ExecutionPolicy Bypass -File research
egister_shadow_task.ps1`.
      Add `-Remove` to delete the task; the records are untouched.
  - **Verification (2026-11-02), each check tested:**
    | Requirement | How it is met | Checked by |
    |---|---|---|
    | Daily at 00:30 UTC | Calendar trigger, `StartBoundary 2026-11-01T11:30:00+11:00` = 00:30 UTC, synchronised across time zones | task XML |
    | Independent of Claude Code | Launched by Task Scheduler: the process's parent is `svchost.exe`, not Claude or a terminal | live run, `Win32_Process` parent |
    | No overlapping runs | Scheduler `MultipleInstances IgnoreNew` (a second start while running was ignored: one journal line). The wrapper also takes an exclusive OS lock (`state/shadow/tracker.lock`), so a manual run during a scheduled one exits with code 9. | live run; `test_second_instance_is_refused` |
    | Missed runs and data gaps | `StartWhenAvailable` runs a missed day once the PC is back. Every run recomputes from the start, so nothing is lost. Missing days, stale and unavailable coins are reported in `run.json` and the journal; an unexpected download failure computes nothing (exit 5). | `test_network_failure_computes_nothing`, `test_stale_coin_stays_in_evaluation`, `test_delisted_mid_tracking_keeps_its_record` |
    | Snapshots preserved, revisions flagged | Each run keeps its own `runs/<id>/data/` with SHA-256 hashes; a changed earlier trade is logged as `DATA_REVISION` | `test_rerun_is_deterministic_and_revisions_are_flagged` |
    | Fingerprints verified every run | `update` re-hashes the spec and all 13 implementing files first; any change stops tracking (exit 3) and keeps earlier results | `test_changed_implementation_stops_tracking`; live run `problems: []` |
    | Failures are not silent | stdout and stderr go to `state/shadow/logs/task_<date>.log`. Every run appends to `state/shadow/task_journal.jsonl`: exit code, status, data warnings, and the traceback on a crash (exit 10). | `test_crash_is_journaled_with_traceback`, `test_full_update_through_wrapper_never_touches_the_exchange` |
    | No orders, keys or live execution | The tracker and wrapper contain no order or credential code. The wrapper deletes `COINEX_*` and other exchange-key variables before importing anything. A full offline run passes with the exchange client made to fail if created. Only public Binance GET requests for daily candles are made. | `test_shadow_code_has_no_order_or_credential_paths`, `test_wrapper_has_no_order_paths`, `test_credentials_are_removed_before_the_tracker_runs` |
  - **Missing a day loses nothing.** Every run recomputes from Binance's history, so a missed day only means a missing
    point-in-time snapshot in the audit trail.
  - Each run does the following.
  1. Verifies the frozen fingerprints.
  2. Downloads completed daily candles (Binance public REST) for every coin from 2026-05-01. Candles before the start
     date are used **only** for indicator warm-up and are never evaluated. They lie inside the consumed 2025–26
     window, and nothing is tuned on them.
  3. Saves the data snapshot with SHA-256 hashes under `state/shadow/runs/<run>/data/`.
  4. Recomputes both books for each strategy, deterministically, from **2026-11-02**. That is the first possible
     fill: the decision at the close of the 2026-11-01 candle, which closed after freezing.
     - **Signal book:** the unchanged per-coin engine. It records every entry decision (filled, blocked by the cost
       filter, skipped by sizing), every trade with prices, stop, size, initial risk, fees, exit reason and time,
       net P&L and R, and positions still open. This is the evidence sample.
     - **Portfolio book:** one 10,000 USDT account with the Conservative limits across coins. It logs every signal:
       accepted, filled with fill and open price, slippage cost, size, stop, initial risk, fee and equity; or rejected
       with the reason (no free slot, cost filter, sizing, daily loss, drawdown halt).
  5. Checks that closed trades before the previous run's last day are identical to that run's. Any difference is
     logged as `DATA_REVISION`. Nothing is overwritten.
  6. Writes `state/shadow/runs/<run>/run.json` and `state/shadow/reports/<run>.md`, plus `state/shadow/SHADOW_REPORT_LATEST.md`.
- **Random benchmark (frozen):**
  - T50: per coin, a daily entry probability equal to the share of 2019–2024 days on which T50's entry condition was
    true (values in `FROZEN.json`).
  - RS3: 3 random coins on each BTC-filter-on Monday.
  - Both use the same exits, stops, sizing and costs: 1,000 runs, seed 2611, recomputed over the same forward period.
- **Uncertainty:** 95% intervals come from a cluster bootstrap that resamples **calendar weeks of entry**, so trades
  opened in the same market move are not treated as independent. Results are also shown by BTC market state.

## Pass criteria (prospective sample only, unchanged)
1. ≥ 100 closed signal-book trades. **No historical trade is counted.**
2. Expectancy > 0R, with the week-clustered 95% lower bound > 0.
3. Profit factor ≥ 1.10.
4. Positive on ≥ 60% of coins with ≥ 10 trades, and positive without the best coin.
5. Above the 95th percentile of the frozen random benchmark over the same period.

**Expected time to 100 trades, from the 2019–24 frequency:**
- T50 on 25 coins: about 75 trades a year, so **~16 months**.
- RS3 on 9 coins: about 30 a year, so **~3 years**.

Interim reports describe progress and never change the rules.

## Known limitations
- **Data:** Binance prices and liquidity only. They say nothing definitive about CoinEx fills.
- **Daily close:** daily candles close at 00:00 UTC; a run before a candle completes simply waits.
- **Delisting:** a coin delisted during tracking is recorded as unavailable from then on. Its open signal-book position
  remains open in the record, marked at its last price.
- **Kill switch:** the portfolio book's 10% drawdown halt is permanent, as configured. If it trips, the book stops
  taking trades (each later signal is logged as rejected: "drawdown halt"). The signal book continues.
