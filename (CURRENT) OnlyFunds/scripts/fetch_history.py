"""Download exchange-sourced historical candles with provenance and a quality report.

    python scripts/fetch_history.py                              # default universe, 1h, 2019-01-01 .. 2026-11-01
    python scripts/fetch_history.py --symbols BTCUSDT ETHUSDT --start 2019-01-01 --end 2026-11-01
    python scripts/fetch_history.py --cross-check 2023-01-01 2024-01-01   # compare closes with OKX

Writes data/historical/binance/<SYMBOL>_<interval>.csv (+ .manifest.json with the SHA-256 and
quality report) and docs/DATA_QUALITY_REPORT.md. Missing candles are reported, never filled.
"""
import argparse
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.history import (APP_ROOT, BINANCE_URL, DEFAULT_UNIVERSE, OKX_URL, cross_check,  # noqa: E402
                          fetch_binance, fetch_okx_closes, save_dataset)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="+", default=DEFAULT_UNIVERSE)
    ap.add_argument("--interval", default="1h")
    ap.add_argument("--start", default="2019-01-01")
    ap.add_argument("--end", default="2026-11-01", help="exclusive")
    ap.add_argument("--cross-check", nargs=2, metavar=("START", "END"))
    ap.add_argument("--cross-symbols", nargs="+", default=["BTCUSDT", "ETHUSDT"])
    ap.add_argument("--report", default=os.path.join(APP_ROOT, "docs", "DATA_QUALITY_REPORT.md"))
    a = ap.parse_args(argv)
    manifests = {}
    for sym in a.symbols:
        print(f"Downloading {sym} {a.interval} {a.start} .. {a.end} from Binance ...", flush=True)
        df = fetch_binance(sym, a.interval, a.start, a.end)
        extra = {}
        if a.cross_check and sym in a.cross_symbols:
            okx = fetch_okx_closes(sym, a.interval, *a.cross_check)
            extra["cross_check_okx"] = {"period": a.cross_check, "source_url": OKX_URL, **cross_check(df, okx)}
        m = save_dataset(df, sym, a.interval, {"url": BINANCE_URL, "symbol": sym, "interval": a.interval,
                                               "start": a.start, "end_exclusive": a.end}, extra=extra)
        q = m["quality"]
        print(f"  {m['rows']} rows, {q.get('first_candle')} .. {q.get('last_candle')}, "
              f"missing {q.get('missing_candles')}, problems {q.get('problems')}", flush=True)
        manifests[sym] = m
    write_report(manifests, a)
    return 0 if all(m["quality"]["ok"] for m in manifests.values()) else 1


def write_report(manifests, a):
    lines = [
        "# Historical data quality report", "",
        f"Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')} by `scripts/fetch_history.py`.", "",
        f"- Source: **Binance spot** public klines API (`{BINANCE_URL}`), interval `{a.interval}`, "
        f"requested {a.start} to {a.end} (exclusive).",
        "- Candles are stored exactly as returned. **Missing candles are reported, never filled.**",
        "- Each CSV has a manifest with its SHA-256; `core.history.load_dataset` refuses a CSV that doesn't match.",
        "- Universe = large USDT pairs as of Jan 2019 (start of data), including later laggards and one "
        "delisted coin (EOS), to limit survivorship bias. BCH was excluded: its Binance history breaks at the "
        "2019 BCHABC→BCH symbol change.",
        "- **This is Binance data. It does not establish CoinEx prices, liquidity, fees or execution.**", "",
        "| symbol | rows | first candle | last candle | expected | missing | gaps | largest gap (h) | zero-vol | "
        "moves >25%/h | max |log ret| | hard problems | sha256 (12) |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for sym, m in manifests.items():
        q = m["quality"]
        largest = q["largest_gaps"][0]["missing"] if q.get("largest_gaps") else 0
        lines.append(f"| {sym} | {m['rows']} | {q.get('first_candle')} | {q.get('last_candle')} | "
                     f"{q.get('expected_rows')} | {q.get('missing_candles')} | {q.get('gap_count')} | {largest} | "
                     f"{q.get('zero_volume_candles')} | {q.get('extreme_moves_over_25pct')} | "
                     f"{q.get('max_abs_hourly_log_return', 0):.3f} | {', '.join(q.get('problems', [])) or 'none'} | "
                     f"{m['sha256'][:12]} |")
    lines += ["", "## Largest gaps per symbol", ""]
    for sym, m in manifests.items():
        g = ", ".join(f"{x['after']} (+{x['missing']})" for x in m["quality"].get("largest_gaps", [])[:5])
        lines.append(f"- **{sym}**: {g or 'none'}")
    xc = {s: m["cross_check_okx"] for s, m in manifests.items() if "cross_check_okx" in m}
    if xc:
        lines += ["", "## Cross-exchange check (Binance vs OKX closes)", "",
                  "| symbol | period | overlapping candles | median abs dev | p99 abs dev | max abs dev | share > 1% |",
                  "|---|---|---|---|---|---|---|"]
        for s, c in xc.items():
            if c.get("overlap"):
                lines.append(f"| {s} | {c['period'][0]}..{c['period'][1]} | {c['overlap']} | "
                             f"{c['median_abs_deviation']:.5%} | {c['p99_abs_deviation']:.4%} | "
                             f"{c['max_abs_deviation']:.4%} | {c['share_over_tolerance']:.4%} |")
            else:
                lines.append(f"| {s} | {c.get('period')} | 0 | - | - | - | - |")
    os.makedirs(os.path.dirname(a.report), exist_ok=True)
    with open(a.report, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {a.report}")


if __name__ == "__main__":
    sys.exit(main())
