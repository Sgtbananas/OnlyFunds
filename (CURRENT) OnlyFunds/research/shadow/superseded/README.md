# Superseded freezes

## FROZEN_20261101T122009Z.json (commit db82762), superseded 2026-11-01 before any forward day was evaluated
- **Fingerprints:** T50 `a4c799ac…`, RS3 `59ab2bac…`.
- **What happened:** its only run, at 12:20 UTC, ended with status "waiting". Under this freeze no forward
  trade, signal or statistic was ever computed.
- **Bug found by that first live data snapshot:**
  1. A coin that stopped updating (ICXUSDT, last candle 2026-09-03) was dropped from the evaluation. So was a coin
     that Binance stops returning mid-tracking. Its earlier trades would have vanished from later records,
     violating "never silently discard signals".
  2. A transient download failure would have produced an evaluation on a partial universe.
- **Fix (tracking code only):** stale coins stay in the evaluation up to their last candle. A coin that disappears
  reuses its last snapshot. Any other download failure aborts the run, and nothing is computed.
- **Unchanged:** no strategy rule, parameter, universe, cost, risk limit, benchmark or pass criterion. The
  new freeze differs only in the implementation hash of `research/forward_shadow.py` and the freeze metadata.
