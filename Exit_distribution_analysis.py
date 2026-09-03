"""
exit_distribution_analysis.py
Answers: what fraction of closed trades exit at TP1 (partial) vs TP2 (runner)
vs SL vs time-exit, and what's the average hold time for each? Run against
your real trade_history.json rather than guessing.

    python exit_distribution_analysis.py trade_history.json
"""

import json
import sys
from pathlib import Path
from collections import defaultdict


def classify_exit(reason: str) -> str:
    r = reason.lower()
    if "tp2" in r: return "TP2 (Runner)"
    if "tp1" in r: return "TP1 (Partial)"
    if "sl hit" in r or "stopped out" in r: return "Stop Loss"
    if "break-even" in r or "breakeven" in r: return "Breakeven"
    if "time exit" in r and "pre-tp1" in r: return "Time Exit (Pre-TP1 Stall)"
    if "time exit" in r: return "Time Exit (Post-TP1)"
    if "kill switch" in r: return "Kill Switch"
    if "unverified" in r or "unrecorded" in r: return "Unverified (excluded)"
    if "manual close" in r: return "Manual Close"
    if "ghost" in r: return "Ghost/Unrecoverable"
    return "Other"


def analyze(history: list):
    buckets = defaultdict(list)
    for rec in history:
        if rec.get("pnl_unverified", False):
            continue
        if rec.get("signal") == "RECOVERED":
            continue
        reason = rec.get("close_reason", "")
        category = classify_exit(reason)
        duration_min = rec.get("duration")
        if duration_min is None:
            try:
                from datetime import datetime
                o = datetime.fromisoformat(str(rec.get("opened_at", "")).replace("Z", "+00:00"))
                c = datetime.fromisoformat(str(rec.get("closed_at", "")).replace("Z", "+00:00"))
                duration_min = (c - o).total_seconds() / 60.0
            except Exception:
                duration_min = None
        buckets[category].append({
            "pnl": float(rec.get("pnl", 0) or 0),
            "duration_min": duration_min,
            "symbol": rec.get("symbol"),
        })

    total = sum(len(v) for v in buckets.values())
    print("=" * 70)
    print(f"EXIT DISTRIBUTION  (n={total} verified closed trades)")
    print("=" * 70)
    print(f"{'Category':<28}{'Count':>7}{'% of Total':>12}{'Avg Hold (min)':>18}{'Avg PnL':>12}")
    for cat, recs in sorted(buckets.items(), key=lambda x: -len(x[1])):
        n = len(recs)
        pct = n / total * 100 if total else 0
        durations = [r["duration_min"] for r in recs if r["duration_min"] is not None]
        avg_dur = sum(durations) / len(durations) if durations else float("nan")
        avg_pnl = sum(r["pnl"] for r in recs) / n if n else 0
        print(f"{cat:<28}{n:>7}{pct:>11.1f}%{avg_dur:>17.0f}{avg_pnl:>12.4f}")

    tp1_n = len(buckets.get("TP1 (Partial)", []))
    tp2_n = len(buckets.get("TP2 (Runner)", []))
    sl_n = len(buckets.get("Stop Loss", []))
    print("\n" + "-" * 70)
    if tp1_n + tp2_n > 0:
        print(f"TP1-only vs TP2 runner ratio: {tp1_n} : {tp2_n}  "
              f"({tp2_n/(tp1_n+tp2_n)*100:.1f}% of TP-hitting trades ran to TP2)")
    if total > 0:
        print(f"Overall win-path rate (TP1+TP2+Breakeven) vs loss-path (SL): "
              f"{(tp1_n+tp2_n)/total*100:.1f}% vs {sl_n/total*100:.1f}%")
    print("=" * 70)


if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("trade_history.json")
    with open(path) as f:
        history = json.load(f)
    analyze(history)
