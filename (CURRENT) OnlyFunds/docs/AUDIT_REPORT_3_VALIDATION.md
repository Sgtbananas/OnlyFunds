# Independent audit of the real-market validation (2026-11-01)

**Question:** is the conclusion "none of the four OF2 strategies has demonstrated a genuine trading edge"
supported by technically sound research?

**Answer: yes.** The data, engine and calculations verify independently. The conclusion is reproducible and
robust to every realistic change in assumptions tested. The protocol has **process** weaknesses (§4), all
in record-keeping and guard-rails. None of them could turn the negative result into a positive one.

## Scope and constraints honoured
- **Left unmodified:** the frozen protocol, the frozen decisions, the final TEST period, all strategy, risk and engine
  parameters, and all existing reports. The final TEST period was **not** re-run.
- **Evidence:** every check below uses DEV (2019–22) and VAL (2023–24) data, the committed TEST report (read only),
  or the data files.
- **New files only:**
  - `scripts/audit/okx_crosscheck.py`, `engine_audit.py`, `sensitivity.py`, `reproduce_stage1.py`
  - results in `docs/audit/*.json`
- **No engine, strategy, risk or protocol code changed** during this audit.

## 1. Results by area
| Area | Verdict | Basis |
|---|---|---|
| Data | **PASS** (coverage caveats) | checksums, independent recount, re-download byte-identical, 42,113-candle OKX cross-check |
| Backtest engine | **PASS** | 144 real-data equivalence cases, 10,940 trades recomputed independently, look-ahead and history checks |
| Validation protocol | **PARTIAL** | correct separation and ordering; weak guard-rails and audit trail (P1–P4) |
| Financial assumptions | **PASS** (unverified for CoinEx) | conservative where it matters; the conclusion survives every optimistic variant |
| Reproducibility | **PASS** (environment not pinned) | data re-downloaded byte-identical; Stage 1 rerun identical; ML model retrains identically |

## 2. Data audit
**Coins and rationale:**
- BTC, ETH, BNB, XRP, LTC, ADA, EOS, XLM, TRX vs USDT on Binance spot.
- The rationale is "large, liquid USDT pairs as of Jan 2019, the start of the data", including later laggards and delisted EOS.
- BCH was excluded because its Binance series breaks at the 2019 BCHABC→BCH change. That is a defensible but discretionary choice (see D3).

**Coverage** (independently recounted from the CSVs, not taken from the manifests):

| Coin | First candle (UTC) | Last candle (UTC) | Rows | Expected | Missing | Duplicates | OHLCV violations | Zero-volume |
|---|---|---|---|---|---|---|---|---|
| BTC, ETH, BNB, XRP, LTC, ADA, XLM, TRX | 2019-01-01 00:00 | 2026-10-31 23:00 | 68,604 each | 68,664 | 60 | 0 | 0 | 3 |
| EOS | 2019-01-01 00:00 | 2025-05-26 02:00 | 56,031 | 56,091 | 60 | 0 | 0 | 3 |

**Integrity checks:**
- All 9 SHA-256 checksums match their manifests, and all timestamps are hour-aligned and strictly increasing.
- A fresh download of BTC and EOS produced **byte-identical** files.

**Missing candles:** 20 exchange-wide outages, identical on every coin, totalling 60 candles.
- **When they fall:** 19 in DEV, 1 in VAL (2023-03-24 12:00, 1 candle), **none in TEST**.
- **Full list:** in this report's source data (`engine_audit.json` and the manifests). The largest is 2019-05-15 (+10 h, Binance maintenance after its hack).
- **Not filled:** nothing was invented or interpolated.

**Can missing-candle handling mislead?**
- Indicators treat the rows on either side of a gap as adjacent, which distorts them around a gap.
- This is mitigated: entries are blocked for 100 bars after any gap (12,991 bar-decisions blocked).
- 45 of 10,940 audited trades (0.4%) were held across a gap. Their exits were priced at the post-gap open with
  stop slippage, which is correct and pessimistic (0 fill errors).
- **Not material.**

**Delisted asset (EOS):**
- Handled correctly: it trades until its last candle (2025-05-26) and is then skipped (`core/validation.py: run_period`).
- It contributed 0–66 trades per TEST candidate.
- Its buy-and-hold uses the last traded close. Any delisting conversion (EOS→A) isn't modelled; that is minor.

**Cross-exchange check** (`docs/audit/okx_crosscheck.json`):
- **Sample:** all coins × 8 windows of 30 days, including COVID (2020-03), May 2021, LUNA (2022-05) and FTX (2022-11).
- **Size:** 42,113 candles.

| Statistic | Absolute close deviation Binance vs OKX |
|---|---|
| Median | 0.019% |
| 95th percentile | 0.106% |
| 99th percentile | 0.216% |
| Max | **4.98%** (ADA, 2020-03-12 10:00, COVID crash hour) |
| Share > 1% | 0.059% |

- **Where the large deviations are:** all at crash hours (2020-03-12, 2021-05-19, 2022-05-11), where real cross-exchange dislocations occur.
- **Per-coin maxima:** BTC 1.18%, ETH 4.21%, XRP 3.83%, TRX 3.86%, LTC 2.15%, XLM 2.16%, ADA 4.98%, BNB 0.16%.
- **Coverage gap:** **EOS couldn't be cross-checked at all** (no OKX history under that symbol), and **BNB only from 2023** (D2).

## 3. Backtest engine audit (`docs/audit/engine_audit.json`, DEV and VAL only)
**Scope:**
- The implementation (`core/backtester.py: run_backtest`, `core/positions.py: check_exit / close_trade_record / planned_risk`, `core/risk_manager.py: position_size`) was re-read line by line.
- It was then checked numerically on real data: 9 coins × 4 strategies × {fixed, tuned} × {DEV, VAL} = 144 runs, 10,940 trades.

| Check | Result |
|---|---|
| Array engine vs pre-optimisation engine (taken from git commit `bc0033a`) | **144/144 identical** (trades, metrics, decision logs), on top of the earlier 20 cases |
| Entry timing: fill at the open after the signal bar, plus slippage | 0 errors |
| PnL, fees, R-multiple recomputed from prices | max error 0.0 |
| Position ≤ 10% of equity; planned risk ≤ 0.5% of equity | 0 violations |
| Stop fills (level or gap open, minus stop slippage); TP never filled without being reached | 0 errors |
| Planned stop-outs below −1R | 0 |
| Max drawdown recomputed from the equity curve | exact |
| Buy-and-hold recomputed | exact (see note) |
| Indicator history: 500 bars vs full history | max signal difference 1.6e-7; 0 entry and 0 regime disagreements |
| Look-ahead: truncating future data on every coin | earlier trades identical (101 trades compared) |

**Note:** the first audit run reported buy-and-hold differences of up to 121 pp in DEV. That was a **bug in the audit
script** (it ignored the 51-bar indicator warm-up at the very start of the data). The engine was right; the script was
fixed and re-run.

**Issues found:**
- **E1, medium: the drawdown halt truncates runs.**
  - *Where:* `core/backtester.py: run_backtest`, `halted_reason`. A run that hits the 10% drawdown limit stops trading permanently.
  - *Scale:* 35 of 72 DEV runs, 3 VAL runs, and 3 TEST runs (stacked/tuned on XRP, ADA, XLM, per the committed TEST report).
  - *Effect:* the evaluation measures "strategy + risk system", not the raw strategy.
  - *Material?* **No.** With the halt disabled on DEV/VAL, every strategy stays negative (§5).
- **E2, low: take-profit fills on an exact touch.**
  - *Where:* `core/positions.py: check_exit`. A TP fills when the high merely reaches the level.
  - *Scale:* 75 of 2,268 TP exits (3.3%) had a high within 0.05% of the level, where a real limit fill is uncertain.
  - *Material?* No. This is optimistic, so correcting it can only make results worse. The worst case is under 0.02R per trade.
- **E3, low: open positions at a period end are left out of the R statistics.**
  - *Where:* `close_at_end=False`.
  - *Scale:* only 1 of 144 runs.
- **E4, low: coins are evaluated independently.**
  - *How:* 1,000 USDT each, with no cross-coin exposure cap and per-coin halts. Paper trading shares one equity, caps total exposure at 30%, and halts all pairs together.
  - *Effect:* per-trade R statistics are unaffected, but the set of trades in a portfolio would differ.
- **E5, info: no exchange precision rounding in backtests.** Negligible at these sizes.

**Paper vs backtest differences** (code-level):

| Aspect | Backtest | Paper |
|---|---|---|
| Entry price | next bar's open | live ticker at decision time (≈ same instant) |
| Exits found by candle replay | level or gap open | `min(level, current price)` (more pessimistic) |
| Equity, exposure cap, halt | per coin | shared across pairs; global kill switch |
| Cost filter price | decision-bar close | live ticker |
| Stale-data guard | not applicable | yes |

Fees, slippage, stop slippage, sizing formula, gap guard, volume cap and ML gate are the **same** in both.

## 4. Validation protocol audit
**Verified correct:**
- **Period separation:** `core/validation.py: segment` cuts at `timestamp < end`, and `trade_from` blocks trading and
  metrics before the start. Earlier candles serve only as indicator history, and their sufficiency was verified above. Trades can't cross into the next period.
- **Selection data:** DEV-only tuning for VAL; DEV+VAL tuning for TEST; ML trained on DEV with history=0 (`train_ml`).
- **Walk-forward:** chronological, tuned on the 2 prior years and scored on the year, with TEST years excluded (`walk_forward`).
- **Minimum trades:** 100 pooled (criteria and tuning) and 10 per coin (breadth), enforced in `criteria` and `tune`.
  NaN results fail every comparison.
- **Ordering, from git and file timestamps (UTC):**
  1. protocol committed 03:55:15 (`bc0033a`)
  2. Stage 1 ran 03:58:31 → ~04:17:50
  3. TEST ran 04:18:46
  4. the protocol is unchanged since (`git diff bc0033a a94145b` is empty)
- **No TEST information influenced selection:** the design choices (1h interval, cost filter, risk defaults, strategy
  definitions) were all made on synthetic data before any real data was downloaded.

**Issues:**
- **P1, medium: the fingerprint protects the protocol text only, not the code that implements it.**
  - *Where:* `core/validation.py: protocol_fingerprint`. It hashes `VALIDATION_PROTOCOL.md` only. `PERIODS`, `GRID`, `MIN_TRADES` and `criteria()` in `core/validation.py`, and the logic in `scripts/real_market_validation.py`, aren't covered.
  - *Audit trail:* that code was first committed **after** the TEST run (`a94145b`), so git can't prove it was unchanged between Stage 1 and TEST.
  - *Known change in between:* one — the console-encoding fix in `main()`.
  - *Supporting evidence:* the Stage 1 rerun with the committed code reproduces the frozen decisions exactly.
- **P2, medium: rerunning Stage 1 silently overwrites the frozen state.**
  - *Where:* `scripts/real_market_validation.py: stage_devval` overwrites `frozen_decisions.json`, `stage1_dev_val.md` and `ml_*.pkl` without refusing.
  - *Risk:* a rerun after TEST would rewrite the frozen state that TEST refers to. Git history preserves the originals.
- **P3, medium: the TEST "criteria" column ignores the VAL result.**
  - *Where:* `stage_test` evaluates criteria on TEST only.
  - *Risk:* the protocol requires a pass on VAL **and** TEST, but a candidate that failed VAL could still show "PASS" in the TEST table.
  - *Effect now:* no visible effect, because everything failed.
- **P4, medium: the ML carry-forward has no minimum trade count.**
  - *Where:* `stage_devval`, `ml_carry = p_ml["trades"] > 0 and ...`.
  - *Exact effect:*
    - On VAL, mean reversion + ML made **1 trade (+1.232R)** vs 283 trades (−0.084R) without ML, so it was carried.
    - Its internal DEV validation said `useful=False` (AUC 0.525), and nothing checked that.
    - On TEST it made **0 trades**, so its row fails "0 trades < 100".
    - ML variants are also exempt from the walk-forward criterion (`stage_test`: `wf_ok = ... or vname == "fixed+ML"`), as the protocol text allows.
    - Combined with P3, a thinly evidenced ML candidate *could* display PASS on TEST without valid VAL evidence.
  - *Effect now:* none. One extra failing row; no other candidate or result affected.
- **P5, low:** `frozen_at_utc` records when Stage 1 *started*, not when it froze.
- **P6, low:** walk-forward evidence for "fixed" candidates comes from the tuning procedure, not from the fixed parameters.
- **P7, low:** the one-shot TEST guard is the existence of `stage2_test.md`. `--rerun-test` is documented and labelled in the output, but deleting the file would bypass it.

## 5. Financial assumptions
| Assumption | Value | Status |
|---|---|---|
| Fee | 0.2% per side | **Documented example, not verified for an account**: CoinEx v2 API docs show `taker_fee_rate "0.002"` and `maker_fee_rate "0.002"` in the market example. Actual fees vary by VIP tier and CET discount. **Not verified** for any CoinEx account. |
| Binance fee | 0.1% (information-only scenario) | Common published spot rate; **not verified** in this work |
| Entry and normal-exit slippage | 0.05% | **Assumption**, not measured. Plausible for ~100 USDT orders on Binance majors; unknown for CoinEx. |
| Stop slippage | 0.2% | **Assumption** |
| Gap through stop | fill at the gap open minus stop slippage | modelled; realistic for stop-market orders |
| Take-profit | fill at the level (minus slippage) on touch | slightly optimistic (E2) |
| Liquidity | ≤ 10% of the signal candle's Binance volume | never binding at these sizes; **not** CoinEx liquidity |
| Risk | 0.5% per trade, ≤ 10% per position, 10% drawdown halt | shipped Conservative defaults |

**Robustness** (`docs/audit/sensitivity.json`; DEV and VAL only, fixed strategies, one assumption changed at a time).

VAL pooled expectancy with 95% bootstrap interval:

| Scenario | Trend following | Auto | Mean reversion | Stacked |
|---|---|---|---|---|
| As evaluated | −0.082R [−0.184, +0.025] n=380 | −0.065R [−0.190, +0.063] n=255 | −0.084R [−0.197, +0.031] n=283 | −0.194R [−0.415, +0.051] n=58 |
| No drawdown halt | identical | identical | identical | identical |
| No cost filter | −0.335R [−0.374, −0.295] | −0.358R | −0.326R | −0.343R |
| 0.1% fee, no slippage | −0.102R [−0.137, −0.066] n=2,930 | −0.125R | −0.130R | −0.120R |
| All optimistic combined | −0.101R [−0.132, −0.070] | −0.121R | −0.170R | −0.137R |
| Zero fees, zero slippage | **+0.061R [+0.024, +0.098]** | +0.046R | −0.042R | +0.025R |

On DEV, the same pattern holds: every realistic scenario is negative, with all 95% intervals below zero.

**What this shows:**
- **No realistic cost assumption produces a positive result.** Lower costs let more trades past the cost filter, which
  *tightens* the evidence. At 0.1% fees every 95% interval is entirely below zero on 520–3,900 VAL trades per strategy.
- **Trend following has a small, statistically detectable *gross* edge** of about +0.04–0.06R per trade before costs, on DEV and VAL alike.
  Realistic fees (≥ 0.1% per side ≈ 0.16R per trade) more than consume it. The strategies are uneconomic, not random.
- **The cost filter protects capital.** Removing it makes results much worse.

## 6. Reproducibility
| Item | Result |
|---|---|
| Historical data | CSVs are git-ignored but reproduced **byte-identically** by `scripts/fetch_history.py` (verified for BTC and EOS against the committed SHA-256). Depends on Binance continuing to serve the same history. |
| Manifests | committed; `core/history.py: load_dataset` refuses mismatching CSVs |
| Stage 1 | rerun with the committed code into a temp folder (`scripts/audit/reproduce_stage1.py`): **report identical apart from its run-time header**; frozen decisions identical; walk-forward identical (`docs/audit/stage1_reproduction.json`) |
| Model file `docs/validation/ml_mean_reversion.pkl` | **Not strictly required.** Retraining on DEV gives identical predictions on all VAL data (max difference 0.0). It exists so the TEST ML row can be reproduced without retraining. **Provenance** (only inside the pickle until now): trained 2026-11-01 04:13:05 UTC by Stage 1, strategy mean_reversion, 9 coins DEV 2019–2022, 8,762 train / 5,280 internal-validation samples, AUC 0.525, `useful=False`, SHA-256 prefix `6d4b704441b08c75`. Loading a pickle executes code, so only load this file from this repository. |
| Environment | **Not pinned** (`requirements.txt` uses `>=`). Verified with Python 3.9.13, pandas 2.2.3, numpy 2.0.2, scikit-learn 1.6.1 (R1). Other versions may change CSV bytes or model output. |
| TEST stage | Deliberately not re-run. `--stage test --rerun-test` reproduces it and labels the output as a reproduction. |

## 7. All issues, ranked
| # | Severity | Area | Issue | Where | Could it flip the negative result? |
|---|---|---|---|---|---|
| P1 | Medium | Protocol | Fingerprint covers protocol text, not code; validation code committed after TEST | `core/validation.py: protocol_fingerprint`; git `a94145b` | No: Stage 1 reproduces exactly; only a print fix changed in between |
| P3 | Medium | Protocol | TEST "PASS" ignores VAL failure | `scripts/real_market_validation.py: stage_test` | No: all candidates fail on TEST itself |
| P4 | Medium | Protocol | ML carry-forward without minimum trades (1-trade carry) | `stage_devval` (`ml_carry`) | No: the carried variant made 0 TEST trades |
| P2 | Medium | Protocol | Stage 1 rerun overwrites frozen state | `stage_devval` | No: originals in git |
| E1 | Medium | Engine | Permanent drawdown halt truncates long runs | `core/backtester.py: run_backtest` | No: still negative without the halt |
| R1 | Medium | Reproducibility | Environment not pinned | `requirements.txt` | No (affects re-verification, not the result) |
| D2 | Low–medium | Data | EOS never cross-checked; BNB only from 2023 | `scripts/audit/okx_crosscheck.py` | No: same exchange-wide gap pattern, 0 integrity errors |
| E2 | Low | Engine | TP fills on exact touch (3.3% of TP exits) | `core/positions.py: check_exit` | No: optimistic; correcting it worsens results |
| E4 | Low | Engine | Coins evaluated independently (no portfolio cap) | `core/validation.py: run_period` | No: per-trade R unaffected |
| D3 | Low | Data | Universe discretionary (BCH excluded; BSV, delisted Apr 2019, absent); no market-cap snapshot recorded | `core/history.py: DEFAULT_UNIVERSE` | Unlikely: affects early DEV only |
| D1 | Low | Data | 3 exchange zero-volume placeholder candles per coin | data | No |
| E3 | Low | Engine | Open position at period end excluded (1/144 runs) | `run_backtest` (`close_at_end=False`) | No |
| P5–P7 | Low | Protocol | Freeze-time label; walk-forward of fixed candidates; deletable one-shot guard | `stage_devval`, `stage_test` | No |
| E5 | Info | Engine | No precision rounding in backtests | `run_backtest` | No |

No issue found could plausibly turn the result positive. Several of them (E2, E4, the optimistic TP fills, the CoinEx-vs-Binance gap)
bias results **upward**, which makes the negative conclusion more conservative, not less.

## 8. Is the evaluation trustworthy?
**Yes, for the stated conclusion.** The data is genuine, verified and reproducible. The engine is verified on 10,940
real trades. Stage 1 reproduces exactly. The negative finding holds under every realistic cost and risk assumption, and
with much larger samples (520–3,900 VAL trades per strategy) than the original run.

**Remaining uncertainty:**
1. **CoinEx.** The exchange is still unverified (HTTP 503). CoinEx fees, spreads and liquidity could differ from Binance,
   most likely for the worse for an intraday strategy.
2. **Scope.** The result covers these four strategies at 1h on these 9 coins, 2019–2026. It doesn't show that no edge exists at other
   intervals, assets or fee levels. With near-zero costs, trend following shows a small gross signal.
3. **The final TEST window has been used once.** Any new strategy needs fresh, untouched data (after 2026-10).

## 9. Recommended fixes (not applied: this was an audit)
1. **P1:** fingerprint `core/validation.py` and `scripts/real_market_validation.py` together with the protocol, and commit the code before Stage 1.
2. **P3:** make the TEST criteria column require the VAL pass.
3. **P4:** require ≥ 100 VAL trades and `useful=True` before an ML variant is carried.
4. **P2:** make Stage 1 refuse to overwrite an existing frozen file.
5. **R1:** pin exact versions (`requirements.lock`).
6. **E1:** report evaluation metrics both with and without the drawdown halt.
