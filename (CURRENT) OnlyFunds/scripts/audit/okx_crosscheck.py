"""Audit: compare Binance closes with OKX closes for every coin over many 30-day windows.

    python scripts/audit/okx_crosscheck.py --out docs/audit/okx_crosscheck.json

Read-only public market data. Windows include stress periods (COVID crash, May-2021 crash,
LUNA, FTX). Reports median / p95 / p99 / max absolute close deviation per coin and window.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from core.history import DEFAULT_UNIVERSE, fetch_okx_closes, load_dataset  # noqa: E402

WINDOWS = [("2019-07-01", "2019-07-31"), ("2020-03-01", "2020-03-31"), ("2021-05-01", "2021-05-31"),
           ("2022-05-01", "2022-05-31"), ("2022-11-01", "2022-11-30"), ("2023-08-01", "2023-08-31"),
           ("2024-08-01", "2024-08-31"), ("2025-04-01", "2025-04-30")]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    rows, pooled = [], []
    for sym in DEFAULT_UNIVERSE:
        b = load_dataset(sym).set_index("created_at")["Close"]
        for start, end in WINDOWS:
            try:
                okx = fetch_okx_closes(sym, "1h", start, end)
            except Exception as exc:  # network / not listed
                rows.append({"symbol": sym, "window": f"{start}..{end}", "overlap": 0, "note": str(exc)[:80]})
                continue
            common = b.index.intersection(okx.index)
            if not len(common):
                rows.append({"symbol": sym, "window": f"{start}..{end}", "overlap": 0,
                             "note": "no OKX data (not listed / outside OKX history / delisted)"})
                continue
            dev = (b[common] / okx[common] - 1).abs()
            pooled.append(dev.to_numpy())
            worst = dev.idxmax()
            rows.append({"symbol": sym, "window": f"{start}..{end}", "overlap": int(len(common)),
                         "median": float(dev.median()), "p95": float(dev.quantile(0.95)),
                         "p99": float(dev.quantile(0.99)), "max": float(dev.max()),
                         "max_at": str(pd.Timestamp(worst, unit="ms", tz="UTC")),
                         "share_over_1pct": float((dev > 0.01).mean())})
            print(rows[-1], flush=True)
    allv = np.concatenate(pooled) if pooled else np.array([])
    summary = {"candles_compared": int(len(allv)), "median": float(np.median(allv)) if len(allv) else None,
               "p95": float(np.quantile(allv, 0.95)) if len(allv) else None,
               "p99": float(np.quantile(allv, 0.99)) if len(allv) else None,
               "max": float(allv.max()) if len(allv) else None,
               "share_over_1pct": float((allv > 0.01).mean()) if len(allv) else None}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump({"windows": WINDOWS, "summary": summary, "rows": rows}, f, indent=1)
    print("SUMMARY", summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
