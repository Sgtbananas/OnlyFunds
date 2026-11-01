"""How well does the regime detector work on real data?  2019-2024 only (the used v1 TEST window
2025-2026 is excluded so it cannot influence anything).

    python scripts/regime_study.py --out docs/REGIME_STUDY.md [--intervals 1h 4h]

Thresholds are fixed a priori; this measures, it does not tune.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd  # noqa: E402

from core import regime as R  # noqa: E402
from core.history import DEFAULT_UNIVERSE, load_dataset  # noqa: E402

END = "2025-01-01"
HORIZON = {"1h": 48, "4h": 12, "1d": 5}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--intervals", nargs="+", default=["1h", "4h"])
    a = ap.parse_args(argv)
    if os.path.exists(a.out):
        print(f"{a.out} exists; not overwriting")
        return 2
    rows = []
    for iv in a.intervals:
        for sym in DEFAULT_UNIVERSE:
            df = load_dataset(sym, iv)
            df = df[pd.to_datetime(df["timestamp"], utc=True) < pd.Timestamp(END, tz="UTC")].reset_index(drop=True)
            for pers in (1, 4):
                det = R.detect(df, R.RegimeParams(persistence=pers))
                ev = R.evaluate(det, df["Close"], horizon=HORIZON[iv])
                dist = det["trend"].value_counts(normalize=True)
                rows.append({"interval": iv, "symbol": sym, "persistence": pers, **ev,
                             **{f"share_{k}": float(dist.get(k, 0)) for k in R.TREND_CLASSES}})
                print(rows[-1], flush=True)
    t = pd.DataFrame(rows)
    agg = t.groupby(["interval", "persistence"])[[
        "agreement_with_trailing_direction", "agreement_with_expost_direction", "chance_agreement",
        "transitions_per_1000_bars", "median_run_bars",
        "false_transition_share", "median_detection_delay_bars", "uncertain_share"]].mean().reset_index()
    fmt = lambda d: d.to_markdown(index=False, floatfmt=".3f") if hasattr(d, "to_markdown") else d.to_string()
    try:
        agg_md, all_md = fmt(agg), fmt(t[["interval", "symbol", "persistence", "agreement_with_trailing_direction",
                                           "agreement_with_expost_direction",
                                           "chance_agreement", "transitions_per_1000_bars", "false_transition_share",
                                           "median_detection_delay_bars", "uncertain_share"]])
    except ImportError:
        agg_md, all_md = agg.to_string(index=False), t.to_string(index=False)
    text = f"""# Regime detector study (Binance, 2019-01-01 -> 2024-12-31; TEST window excluded)

Detector: `core/regime.py` (causal, fixed thresholds, may answer "uncertain").
Reference: ex-post forward trend over {HORIZON} bars, in volatility units (**non-causal, evaluation only**).
Agreement is measured on direction (up / range / down):
- `agreement_with_trailing_direction`: vs the PAST {list(HORIZON.values())} bars (causal), i.e. does it describe the present?
- `agreement_with_expost_direction`: vs the NEXT bars (non-causal), i.e. does the current regime persist / predict?
- `chance_agreement`: what independent labels with the same mix would score against the ex-post label.

## Averages over 9 coins
```
{agg_md}
```

## Per coin
```
{all_md}
```
"""
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(text)
    print(agg.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
