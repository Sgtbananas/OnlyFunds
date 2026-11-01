# Round 2 pre-registration (written after ML Round 1, before any Round 2 result)

**Status:** research only. Nothing is approved for trading. Live trading stays locked. The frozen T50 and RS3 shadow
tracking is unchanged; its fingerprints are verified by a test.

## What Round 1 established (`04_ml_round1.md`)
1. **Market-state and trade-scoring ML has no out-of-sample skill.**
   - All four classifiers lost to climatology and to the rule baseline out of sample. They are badly miscalibrated:
     the predicted-lowest decile was up 66% of the time.
   - Training AUC was 0.69–0.98 against 0.25–0.59 out of sample.
   - Every ML gate removed T50's large winners: removed trades averaged **+0.65R to +1.63R**. Every gate cut total R
     (−62R to −146R against +251R).
2. **T50's edge is "never miss the big trend".** A gate that skips many signals destroys it.
   - The only gates that helped skipped few signals in clearly bearish BTC conditions: G1 (+10.6R, random-skip
     p 0.003) and G3 (+2.9R, p 0.007).
   - Both were **inconclusive**: G1 improved only 2 of 4 years, and G3 had the wrong sign on the holdout.
3. **Position management is the most promising lever.** R1 and R3 improved return, drawdown, Sharpe and return/DD in
   both universes and both modes, and the top five winners kept more than 100% of their profit. They were
   **rejected**, though, because the holdout drawdown fell by less than the pre-registered 2 points
   (10.0 → 9.7–9.8%).

**Data-snooping budget.** Round 2 reuses 2019–2024, which Round 1 already examined: 12 hypotheses, plus Round 1 and
the T50 scrutiny before that. So **no historical Round 2 result can by itself establish an edge.** Historical tests
here can only reject ideas or show fragility. Only prospective (forward) data is uncontaminated, which is why
Section A exists.

## A. Experimental forward candidates (prospective; frozen before the 2026-11-01 daily close)
Two **separate experimental strategies**, each with its own fingerprint (`research/shadow/EXPERIMENTAL_FROZEN.json`).
They use the frozen T50 entry and exit signals unchanged. They are portfolio books only: 10,000 USDT, Conservative
limits, the same costs and the same Binance daily data as the frozen run.

| Id | Rule |
|---|---|
| X1 `T50-PM` | frozen T50 + R3 position management: concentration band (value > 15% of equity at a close → sell to 10% at the next open) + profit lock (once the close is ≥ 2R above the fill, stop = max(stop, highest close − 6 × ATR14)) |
| X2 `T50-PM-B` | X1 + skip new entries when BTC close ≤ its 100-day SMA (G1) |

- **Comparator:** the frozen T50 portfolio book of the same run (same data, same limits, same start).
- **First possible fill:** 2026-11-02, the same as the frozen books.
- **Infrastructure:**
  - `research/forward_experimental.py` runs after the frozen update in the same scheduled job.
  - It reads that run's data snapshot read-only, writes only to `state/shadow_experimental/`, and has its own
    fingerprint check.
  - A failure in it cannot affect the frozen run: it runs afterwards, in its own exception boundary, and is journaled
    separately.
- **Prospective decision**, after both ≥ 24 months **and** ≥ 40 closed portfolio positions per book. Criteria for
  each X, against the frozen T50 portfolio book:
  1. Max drawdown at least 2 points lower.
  2. Return / max DD not lower.
  3. Total return at least 80% of T50's.
  4. The five most profitable frozen-T50 positions keep at least 70% of their profit.
- **Interim reports are descriptive and change nothing.**

## B. Historical checks (2019–2024 only; cannot establish an edge, can only reject or show fragility)
| Id | Question | Pre-declared rule |
|---|---|---|
| H2-RS3 | Is RS3's record more than one or two trades? (registered before its results existed) | Keep researching RS3 only if: (a) main 9 mean R without its top 2 trades has a CI lower bound > 0; (b) RS3 beats the random-basket 95th percentile on main 9 **and** holdout 16; (c) holdout mean without top 2 > 0 |
| H2-PM-ROBUST | Is R3 fragile? | 12 pre-declared neighbours: band (hi, lo) ∈ {(0.12, 0.08), (0.15, 0.10), (0.20, 0.15)} × lock (kR, nATR) ∈ {(1.5, 5), (2, 6), (3, 8), off}. **Robust** only if ≥ 75% of neighbours raise return/DD above R0 in all four cells (main/holdout × as configured/no halt). This **cannot change X1's frozen parameters.** |
| H2-PM-COST | Does R3 survive higher costs? | Under 0.4% fee, 0.2% slippage and 0.5% stop slippage, R3's return/DD ≥ R0's in all four cells |
| H2-PM-GENERIC | Is R3 generic risk reduction? (descriptive) | Apply R0 and R3 to 100 matched random-entry books (main 9, as configured and no halt). Report the share of draws where R3 raises return/DD. |
| H2-ML7 | Does a sparse, heavily regularised market-state model help where the Round 1 models overfit? | Logistic, C = 0.1, 4 features (`btc_ma200_dist`, `btc_ret_60`, `breadth_ma50`, `btc_vol_20`), target `y_up`, the same purged and embargoed folds. Gate: skip a T50 entry only when P(up) < the training base rate − 0.10. **The Round 1 gate rules apply unchanged** (6 checks). Holm with H2-ML8. |
| H2-ML8 | Can expected adverse excursion be predicted? Noise stop-outs are the largest loss source. | Target: a T50 trade that is stopped out within 5 bars. Logistic, C = 0.1, on `atr_pct`, `vol_expansion`, `brk_atr`, `exit_dist_atr`, `btc_vol_20`. Gate: skip an entry whose predicted risk is above the fold's training 80th percentile. The same 6 gate checks. Holm with H2-ML7. |

**Disclosures:**
- H2-ML7 and H2-ML8 were informed by Round 1 on the same data (sparser models; gates that block few signals). Their
  evidence is weaker than Round 1's. Even if they pass, they become prospective candidates, not improvements.
- No neural networks or paid services are used.

## Not done
- No change to the frozen T50 or RS3 rules or fingerprints.
- No relabelling of a Round 1 rejection as a pass.
- No 2025–26 data.
- No tuning against forward results.
