# OF2 second-pass audit (2026-11-01, branch `of2-completion`)

The brief was to challenge the first rebuild instead of trusting its 109 passing tests. Method: write
adversarial tests that pin exact execution assumptions, confirm each suspected defect with a failing
test *before* fixing it, then fix and re-run. Software correctness and trading profitability are
reported separately: §7 answers the profitability question.

## 1. Deliverable verification (Part 1)
| Check | Result |
|---|---|
| Repository / PR | `Sgtbananas/OnlyFunds` PR #1, open, not draft |
| Source → target | `of2-completion` → `main` |
| Commits | `38672bb`, `3b9e661` (plus this audit's commit) |
| `main` untouched | yes, still `5a19aaa` |
| Credentials | none in the diff; the old config only had `your_api_key_here` placeholders, now removed |
| Tracked junk | no `.env`, `.pkl`, `state/`, caches or `__pycache__` |
| Deletions | all explained: `core__init__.py` (became `__init__.py`), a fake 15-row CSV, a runtime heartbeat file, 11 moves into `legacy/` |

**Discrepancy found:** the PR description said `python -m pytest` gave "109 passed". In fact that was two
runs (106, plus 3 safety tests run separately). This audit ends with one complete run (§9).

## 2. Backtesting (Part 2)
Hand-built price paths (flat 100, ATR exactly 1.0, a forced entry on a known bar) in
`tests/test_audit_backtest.py`.

**Passed verification unchanged:**
- entry at the next bar's open, never the signal bar's close
- signal exit at the next open
- gap through the stop or target fills at the open
- stop and target in the same candle → stop
- the entry candle itself can stop out
- trailing-stop sequence with exact levels
- fees on both legs and slippage on both fills
- reproducibility
- no look-ahead (future-tampering tests)

**Failed and fixed (each confirmed by a failing test first):**
| Defect | Effect | Fix |
|---|---|---|
| Sizing ignored fees and stop slippage | a stop-out lost **more** than `risk_per_trade` | per-unit loss now includes both fees and stop slippage; test asserts the realised loss ≤ budget |
| Stops filled with the same small slippage as calm exits | optimistic | separate `stop_slippage_pct` (default 0.2%) |
| No liquidity limit | any size assumed to fill (unrealistic partial-fill assumption) | size capped at 10% of the signal candle's volume (known at decision time) |
| Drawdown used closes only | intrabar dips invisible; drawdown understated | drawdown measured against each bar's low-marked equity |
| Buy-and-hold started before trading could begin and paid no costs | benchmark misaligned, especially out-of-sample | starts at the first tradable open; pays the same fees and slippage |
| Missing metrics | — | expectancy, Sortino, return/max-DD, trades/day, per-regime breakdown, missing-candle count |
| **Uneconomic trades** (found in §7) | at 5m, costs ≈ 0.7R per trade, so the strategy logic was swamped by fees | `max_cost_to_risk` = 0.25: skip trades whose round-trip costs exceed 25% of the stop distance; default interval → 1h |

**Remaining assumptions (documented, not hidden):**
- fills against order-book depth aren't modelled (only the volume cap)
- a gap through the stop can still lose more than the risk budget (bounded by `max_position_pct`)

## 3. CoinEx API (Part 3)
Re-verified against the official v2 docs.

**Matches the docs:**
- base URL, response envelope, and the three auth headers
- lowercase hex HMAC-SHA256 over method + path (with query) + body + timestamp
- kline periods and limit ≤ 1000; depth limits
- market fields; ticker for all markets when `market` is omitted
- order and cancel body fields; order-status values
- error codes 4213, 3008, 3127, 4005–4017; rate-limit groups

**Changed:**
- **Server-time sync** via `GET /time`. Signed requests are valid for only 5 s by default (`X-COINEX-WINDOWTIME`), so a drifting PC clock would have caused signature-expiry errors.
- **Read-only reconciliation queries** `GET /spot/pending-order` (the only documented lookup by `client_id`) and `GET /spot/finished-order`.
- **`GET /maintain/info`** added.

**Unresolved by the docs** (only live data can settle these):
- is kline `created_at` the open or the close time, and in what sort order? The "drop unfinished candle" logic is safe either way: it only delays by one bar.
- is `min_amount` in base or quote currency?
- which currency is ticker `value` in?
- is `/v2` part of the signed path? (The docs contradict themselves.)
- what does a market-buy `amount` mean without `ccy`?
- CoinEx doesn't store orders cancelled with no fills, which limits reconciliation.

**Connectivity:** one final plain check at 2026-11-01 03:12 UTC. `GET /v2/time` and `GET /v2/spot/market` both returned
**HTTP 503 from the CoinEx gateway**. No workaround was attempted. **Live integration is not verified; all
client tests use mocked HTTP.**

## 4. Paper trading (Part 4)
Attack tests are in `tests/test_audit_paper.py`.

**Failed and fixed:**
| Defect | Fix |
|---|---|
| A corrupt ledger was silently replaced with a fresh account at starting capital | last-good `.bak` written on every save, plus fsync; load falls back to the backup; if both are unreadable it **refuses to start** (`LedgerCorrupt`); the UI offers an explicit "move aside" |
| Structurally invalid ledgers (wrong types) were accepted | schema check on load |
| Stops found by replaying candles filled at the stop price even when the market was already far below | fill at `min(stop, current price)`: never better than the market at execution |
| The candle the entry happened in was never checked for stops | included, with its pre-entry open replaced by the entry price |
| **Candle replay never ran in any earlier test** (executor used the wall clock, trader used the simulated clock) | the trader stamps entries and exits with its own clock; the replay is now exercised |
| An inconsistent ledger still allowed new entries | reconciliation at the start of every step; any issue blocks entries (exits still run) |
| Missing candles not detected | any gap in the last 100 bars blocks entries (`DATA_GAP`) |
| A step longer than 5 min let another process steal the lock | lock heartbeat per pair; a lost race reports "busy" instead of crashing |
| Stops in paper used normal slippage, unlike the backtest | stop slippage applied; sizing uses the same cost and liquidity budget as the backtest |
| Ledger writes were slow (indented JSON) | compact C-encoder write: about 10× faster |
| **Lock race (found by the final full test run):** if the lock was released between two checks, `FileNotFoundError` escaped instead of "busy", and `run_paper.py` would have crashed whenever the app and runner collided | treated as busy (also `PermissionError` on Windows); the runner logs and continues on unexpected step errors, but stops deliberately on a corrupt ledger; deterministic regression test plus a 2-thread × 3,000-acquisition stress test |

**Passed:**
- restart recovery
- app and runner hammering one ledger from two threads (no duplicate client IDs or positions; reconciliation clean)
- network outage then recovery
- duplicate order IDs rejected after restart
- emergency stop persisting across restart
- stale data

## 5. Risk management (Part 5)
`tests/test_audit_risk.py` runs four pairs simultaneously and checks the limits at **every step**.

**Verified:**
- every stop-out stayed within its risk budget, apart from gaps
- position and exposure caps held with up to 3 open positions
- the daily loss limit blocks entries until the next UTC day
- insufficient cash limits size
- the exchange minimum and precision are enforced
- the kill switch blocks all pairs

**Defaults made conservative:** Conservative profile, 0.5% risk per trade, at most 10% per position and 30% exposure,
3 open positions, 3% daily loss, 10% drawdown halt. The Aggressive profile was reduced to 1.5% risk and 60% exposure.
No risk setting was raised.

## 6. ML (Part 6)
- **Defect:** usefulness was judged on precision (≈ win rate). A filter can raise win rate while cutting
  profit. **Fixed:** outcomes are now measured in R-multiples **after fees and stop slippage**. "Useful" requires AUC ≥ 0.55,
  at least 10 accepted trades, and accepted-trade expectancy > 0 and > the take-everything expectancy.
- Label intrabar ordering aligned exactly with the live exit rules.
- Leakage checks pass: causal features (future-tampering test), purge gap, and the model can't claim an edge on random data.

### UI
- **Defect:** an unreadable paper ledger called `st.stop()`, which also hid the ML and Risk & status tabs. **Fixed:** the
  error is shown in the Paper tab only, with an explicit "move corrupt ledger aside" action (`test_app.py`).
- The Backtest tab now shows expectancy, trades/day, Sharpe/Sortino, return/max-DD, exposure, per-regime results
  and a missing-candle warning. A "Compare strategies" panel runs the evaluation from the UI.

## 7. Profitability: is there a credible edge? (Part 6)
`scripts/evaluate_strategies.py` runs every strategy at fixed defaults and with walk-forward tuning. It
reports net return vs buy-and-hold, max drawdown, profit factor, expectancy, Sharpe/Sortino and trades/day,
plus ML on/off and in-sample vs out-of-sample degradation.

**No real market data was available** (CoinEx unreachable, and nothing was downloaded from elsewhere).
The evaluation ran on **controlled synthetic regimes at 1h** (full table: `docs/STRATEGY_EVALUATION_SYNTHETIC.md`). These test
whether the logic behaves as designed; they are **not** evidence about real markets.

**Findings:**
- **No strategy shows a credible edge in any regime** (0/24 pass the verdict). The ML filter is never useful.
  The optimiser never confirms a parameter set out-of-sample.
- **Trend-up regime:** every strategy badly underperforms buy-and-hold (best: stacked +2.6% vs +138%).
- **Down and high-volatility regimes:** strategies lose 1–10% while buy-and-hold loses 90%. They act as capital
  preservation, not as profit.
- **Mean reversion** is mildly positive in ranging and switching markets (profit factor 1.34–1.46, ~36 trades) but
  fails walk-forward. Not credible.
- **Logic sanity:** with stops wide relative to costs and a strong drift, trend-following is profitable on
  4/4 seeds and loses on a random walk on 4/4 (`tests/test_audit_costs.py`). So the engine *can* capture a real
  edge where one exists. The strategies' current edge is the open question.

## 8. Answers
1. **Passed independent verification:**
   - the core execution timing and exit mechanics
   - fee arithmetic
   - look-ahead protection
   - the risk limits under simultaneous positions
   - paper-trading restart and concurrency behaviour
   - CoinEx endpoint definitions (on paper)
2. **Failed:** 24 issues: 7 backtest, 11 paper trading, 2 ML, 2 CoinEx client, 1 risk defaults, 1 UI (§2–§6). Most made results look
   *better* than reality: risk budget overshoot, optimistic stop fills, missed stops, understated drawdown,
   misaligned benchmark, win-rate-based ML judgement, uneconomic trades.
3. **Fixed:** all 24, each with a regression test. (The risk-default change is covered by `test_shipped_defaults_are_conservative`.)
4. **Still unverified:**
   - live CoinEx anything
   - real-market strategy performance
   - order-book depth effects
   - the doc ambiguities in §3
5. **Is paper trading safe to begin?** **Yes, as a software test.** It cannot place real orders, its ledger is
   durable, and it refuses to trade on bad data. But it needs CoinEx data. While CoinEx returns 503 it will log
   `ERROR` / `DATA_STALE` and make no entries. Paper results will **not** show profitability (see 7).
6. **Is CoinEx integration genuinely operational?** **No.** Built and doc-verified, but never reached live.
7. **Does any strategy show a credible positive edge?** **No.** Not on any data tested. The honest default
   expectation is that these indicator strategies lose about their fees on real data too, until a real-data
   walk-forward says otherwise.
8. **Ready for the next phase?** For **real-data evaluation and exchange verification**, yes. The tooling is in
   place: CSV import, the evaluation script, walk-forward, the paper trader. For live trading, no.

## 9. Verification run
One complete run: `python -m pytest`. Result recorded in the PR description.
