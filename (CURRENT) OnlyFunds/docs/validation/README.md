# Validation results

## v1 (frozen, 2026-11-01): files in this folder
| File | SHA-256 of committed content (LF line endings) |
|---|---|
| `frozen_decisions.json` | `22d40434cdab831991af551c305569c27667961742b5590420f2f12d8513ca6f` |
| `stage1_dev_val.md` | `eec8664431815de3d1adff4f1d468a40004efa3eb0900df3ea1b647765d7267b` |
| `stage2_test.md` | `a8e4d216748d66949ae117b64dd7992c148a5bf208f2feaa43d3c4daf733a9d3` |
| `ml_mean_reversion.pkl` | `6d4b704441b08c7588f8bf12af75a9e004717863d619ea7408b89e7c2429be56` |
| `../VALIDATION_PROTOCOL.md` | `726b2c63bc27fcf13f3c9068d07a97771ebe95ab335d41f29723b54cdd5c0812` |

These files must never change. The checkpoint tag `checkpoint/pre-remediation-56897c8` marks them.
`tests/test_frozen_v1.py` fails if any hash changes.

**Provenance limitation (honest statement):**
- v1 ran before the P1 provenance system existed. Its fingerprint covered the protocol text only.
- The code that implemented v1 (`core/validation.py`, `scripts/real_market_validation.py`) was first committed in
  `a94145b`, **after** the final test ran.
- So **no fingerprint or commit can prove which exact code produced v1**.

What *is* established:
- The protocol was committed (`bc0033a`) before any real-data run.
- The only edit made between the two v1 stages was a console-encoding fix.
- Re-running Stage 1 with the committed code reproduces the frozen results exactly (`docs/audit/stage1_reproduction.json`).

The v1 final-test window is registered as consumed in `runs/_consumed_test_windows/v1-2026-11-01.json`.

## v2 and later: `runs/<plan>/<run_id>/`
Produced by `scripts/validation_run.py` under `../VALIDATION_PROTOCOL_V2.md`. Each run is immutable and carries its own
`provenance.json`.
