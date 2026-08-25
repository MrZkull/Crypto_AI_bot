"""
fg_override_audit.py
Ad-hoc audit of F&G-override trades vs. normal trades. Run directly:

    python fg_override_audit.py [path/to/trade_history.json]

Or imported by dashboard.py for the live /api/override_audit endpoint.
"""

import json
import sys
from pathlib import Path

def _is_override(rec: dict) -> bool:
    return bool(rec.get("fg_override", False))

def _closed(rec: dict) -> bool:
    reason = rec.get("close_reason", "")
    return "Ghost" not in reason  # exclude unrecoverable ghosts from PnL stats

def _stats(records: list) -> dict:
    closed = [r for r in records if _closed(r)]
    n = len(closed)
    if n == 0:
        return {"n": 0, "win_rate": None, "profit_factor": None, "net_pnl": 0.0}

    wins = [r for r in closed if float(r.get("pnl", 0) or 0) > 0]
    losses = [r for r in closed if float(r.get("pnl", 0) or 0) < 0]

    gross_win = sum(float(r.get("pnl", 0) or 0) for r in wins)
    gross_loss = abs(sum(float(r.get("pnl", 0) or 0) for r in losses))
    net_pnl = gross_win - gross_loss

    return {
        "n": n,
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(len(wins) / n * 100, 1),
        "profit_factor": round(gross_win / gross_loss, 2) if gross_loss > 0 else (float("inf") if gross_win > 0 else 0.0),
        "gross_win": round(gross_win, 4),
        "gross_loss": round(gross_loss, 4),
        "net_pnl": round(net_pnl, 4),
        "avg_pnl_per_trade": round(net_pnl / n, 4),
    }

def audit_fg_overrides(history: list) -> dict:
    override_recs = [r for r in history if _is_override(r)]
    normal_recs = [r for r in history if not _is_override(r)]

    override_stats = _stats(override_recs)
    normal_stats = _stats(normal_recs)

    verdict = "insufficient_data"
    if override_stats["n"] >= 15:
        if override_stats["win_rate"] is not None and normal_stats["win_rate"] is not None:
            gap = override_stats["win_rate"] - normal_stats["win_rate"]
            if gap >= -5:
                verdict = "performing_in_line_or_better"
            else:
                verdict = f"underperforming_baseline_by_{abs(gap):.1f}pts"

    return {
        "override": override_stats,
        "normal_baseline": normal_stats,
        "verdict": verdict,
        "note": "Verdict requires n>=15 override trades to be meaningful; treat anything below that as directional only.",
    }

def _print_report(result: dict):
    o, n = result["override"], result["normal_baseline"]
    print("=" * 60)
    print("F&G OVERRIDE AUDIT")
    print("=" * 60)
    print(f"\nOverride trades (n={o['n']}):")
    if o["n"] > 0:
        print(f"  Win rate:       {o['win_rate']}%  ({o['wins']}W / {o['losses']}L)")
        print(f"  Profit factor:  {o['profit_factor']}")
        print(f"  Net PnL:        {o['net_pnl']:+.4f}")
        print(f"  Avg PnL/trade:  {o['avg_pnl_per_trade']:+.4f}")
    else:
        print("  No override trades recorded yet.")

    print(f"\nNormal-signal baseline (n={n['n']}):")
    if n["n"] > 0:
        print(f"  Win rate:       {n['win_rate']}%  ({n['wins']}W / {n['losses']}L)")
        print(f"  Profit factor:  {n['profit_factor']}")
        print(f"  Net PnL:        {n['net_pnl']:+.4f}")
        print(f"  Avg PnL/trade:  {n['avg_pnl_per_trade']:+.4f}")

    print(f"\nVerdict: {result['verdict']}")
    print(f"({result['note']})")
    print("=" * 60)

if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("trade_history.json")
    if not path.exists():
        alt = Path("data") / path.name
        path = alt if alt.exists() else path

    with open(path) as f:
        history = json.load(f)

    result = audit_fg_overrides(history)
    _print_report(result)
