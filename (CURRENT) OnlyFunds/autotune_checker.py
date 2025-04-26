import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import Counter, defaultdict

def parse_log(log_path):
    trade_pattern = re.compile(
        r"(?P<datetime>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) - INFO - Exited LONG at (?P<exit_price>[\d.]+) → Return: (?P<return>[-\d.]+)%, Reason: (?P<reason>[^,]+), Capital: (?P<capital>[\d.]+)"
    )
    entry_pattern = re.compile(
        r"(?P<datetime>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) - INFO - Entered LONG at (?P<entry_price>[\d.]+), Size: (?P<size>[\d.]+), Capital: (?P<capital>[\d.]+)"
    )
    summary_pattern = re.compile(
        r"Backtest complete: (?P<trades>\d+) trades, Avg Return: (?P<avg_return>[-\d.]+)%"
    )
    threshold_pattern = re.compile(
        r"Threshold for .*?: (?P<threshold>[\d.]+)"
    )
    autotune_pattern = re.compile(
        r"Adaptive threshold for .*?: (?P<threshold>[\d.]+)"
    )

    trades = []
    entries = []
    trade_pairs = []
    summaries = []
    thresholds = []
    autotunes = []

    with open(log_path, "r") as f:
        for line in f:
            trade_match = trade_pattern.search(line)
            if trade_match:
                trades.append({
                    "datetime": trade_match.group("datetime"),
                    "exit_price": float(trade_match.group("exit_price")),
                    "return": float(trade_match.group("return")),
                    "reason": trade_match.group("reason"),
                    "capital": float(trade_match.group("capital")),
                })
            entry_match = entry_pattern.search(line)
            if entry_match:
                entries.append({
                    "datetime": entry_match.group("datetime"),
                    "entry_price": float(entry_match.group("entry_price")),
                    "size": float(entry_match.group("size")),
                    "capital": float(entry_match.group("capital")),
                })
            summary_match = summary_pattern.search(line)
            if summary_match:
                summaries.append({
                    "trades": int(summary_match.group("trades")),
                    "avg_return": float(summary_match.group("avg_return")),
                })
            threshold_match = threshold_pattern.search(line)
            if threshold_match:
                thresholds.append(float(threshold_match.group("threshold")))
            autotune_match = autotune_pattern.search(line)
            if autotune_match:
                autotunes.append(float(autotune_match.group("threshold")))

    # Pair up entries and exits for trade duration/hold calculations
    entry_idx = 0
    for trade in trades:
        if entry_idx < len(entries):
            entry = entries[entry_idx]
            trade_pairs.append({
                "entry_time": entry["datetime"],
                "entry_price": entry["entry_price"],
                "exit_time": trade["datetime"],
                "exit_price": trade["exit_price"],
                "return": trade["return"],
                "reason": trade["reason"],
                "size": entry["size"],
                "capital": trade["capital"]
            })
            entry_idx += 1

    return trades, summaries, thresholds, autotunes, trade_pairs

def analyze_trades(trades, trade_pairs, label=""):
    returns = [t["return"] for t in trades if t["return"] is not None]
    reasons = Counter([t["reason"] for t in trades if t["reason"] is not None])
    print(f"\n{label} # Trades: {len(returns)}")
    print(f"  Win rate: {sum(1 for r in returns if r > 0)/len(returns)*100:.2f}%")
    avg_win = np.mean([r for r in returns if r > 0]) if any(r > 0 for r in returns) else 0.0
    avg_loss = np.mean([r for r in returns if r < 0]) if any(r < 0 for r in returns) else 0.0
    print(f"  Avg win: {avg_win:.2f}%")
    print(f"  Avg loss: {avg_loss:.2f}%")
    print(f"  Win/Loss ratio: {abs(avg_win/avg_loss) if avg_loss else np.nan:.2f}")
    print(f"  Max win: {max(returns):.2f}%  Max loss: {min(returns):.2f}%")
    print(f"  Most common exit reasons: {reasons.most_common(5)}")
    print(f"  Average trade size: {np.mean([t['size'] for t in trade_pairs]):.5f}")
    if len(trade_pairs) > 1:
        # Calculate holding periods if possible
        times = [
            (
                pd.to_datetime(t["entry_time"]),
                pd.to_datetime(t["exit_time"])
            ) for t in trade_pairs
        ]
        hold_durations = [(exit_t - entry_t).total_seconds()/60 for entry_t, exit_t in times]
        print(f"  Avg Hold Time: {np.mean(hold_durations):.2f} min, Median: {np.median(hold_durations):.2f} min")
        plt.hist(hold_durations, bins=20, alpha=0.5)
        plt.title("Trade Holding Duration (minutes)")
        plt.xlabel("Minutes")
        plt.ylabel("Count")
        plt.show()

    plt.figure(figsize=(10,4))
    plt.hist(returns, bins=30, alpha=0.5)
    plt.title(f"Trade Return Distribution {label}")
    plt.xlabel("Return (%)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

    # Cumulative equity curve
    equity = np.cumprod([1 + r/100 for r in returns]) * 100
    plt.plot(equity)
    plt.title("Cumulative Equity Curve (%)")
    plt.xlabel("Trade #")
    plt.ylabel("Equity")
    plt.tight_layout()
    plt.show()

    # Reason breakdown pie
    plt.figure(figsize=(6,6))
    plt.pie([v for _,v in reasons.most_common()], labels=[k for k,_ in reasons.most_common()], autopct="%1.1f%%")
    plt.title("Exit Reason Breakdown")
    plt.show()

def analyze_thresholds(thresholds, autotunes):
    if thresholds:
        plt.plot(thresholds, label="Manual Thresholds")
    if autotunes:
        plt.plot(autotunes, label="Autotuned Thresholds")
    plt.legend()
    plt.title("Thresholds Over Time")
    plt.xlabel("Trade #")
    plt.ylabel("Threshold Value")
    plt.show()
    print(f"Manual thresholds range: {min(thresholds or [0]):.2f}–{max(thresholds or [0]):.2f}")
    print(f"Autotuned thresholds range: {min(autotunes or [0]):.2f}–{max(autotunes or [0]):.2f}")
    if autotunes:
        print("Autotune volatility (std dev):", np.std(autotunes))

def analyze_outliers(trades):
    returns = [t["return"] for t in trades if t["return"] is not None]
    if not returns:
        return
    q1, q3 = np.percentile(returns, [25, 75])
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outliers = [r for r in returns if r < lower or r > upper]
    print(f"Outlier trades (beyond 1.5 IQR): {outliers}")

def analyze_performance_by_reason(trades):
    grouped = defaultdict(list)
    for t in trades:
        grouped[t["reason"]].append(t["return"])
    print("\nPerformance by Exit Reason:")
    for reason, rets in grouped.items():
        print(f"  {reason:15} count={len(rets):3d} avg={np.mean(rets):.2f}% winrate={100*sum(r>0 for r in rets)/len(rets):.1f}%")

if __name__ == "__main__":
    log_path = "CMD log.txt"
    trades, summaries, thresholds, autotunes, trade_pairs = parse_log(log_path)

    print("--- TRADE & SUMMARY STATISTICS ---")
    analyze_trades(trades, trade_pairs, label="(Autotune/Manual Mixed)")

    if summaries:
        print("\nBacktest Summaries:")
        for s in summaries:
            print(f"  {s['trades']} trades, Avg Return: {s['avg_return']:.2f}%")

    print("\n--- THRESHOLD ANALYSIS ---")
    analyze_thresholds(thresholds, autotunes)

    print("\n--- OUTLIER ANALYSIS ---")
    analyze_outliers(trades)

    print("\n--- EXIT REASON PERFORMANCE ---")
    analyze_performance_by_reason(trades)