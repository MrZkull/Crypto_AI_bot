# inspect_calibration.py — Extracts per-symbol performance across confidence tiers
import json
from pathlib import Path

DOSSIER_FILE = "coin_dossier.json"

def audit_calibration():
    p = Path(DOSSIER_FILE)
    if not p.exists():
        print(f"File {DOSSIER_FILE} not found.")
        return

    dossier = json.loads(p.read_text())
    print(f"\n{'SYMBOL':<10} | {'AUDITED':<8} | {'OVERALL':<8} | {'45-55%':<12} | {'55-65%':<12} | {'65-75%':<12} | {'75%+':<12}")
    print("-" * 88)

    for sym, d in sorted(dossier.items()):
        total = d.get("total_audited", 0)
        correct = d.get("total_correct", 0)
        overall = f"{(correct/total*100):.1f}%" if total > 0 else "N/A"

        buckets = d.get("by_confidence_bucket", {})
        
        def fmt_b(key):
            b = buckets.get(key, {})
            n = b.get("n", 0)
            c = b.get("correct", 0)
            return f"{(c/n*100):.0f}% ({n})" if n > 0 else "—"

        print(f"{sym:<10} | {total:<8} | {overall:<8} | {fmt_b('45-55'):<12} | {fmt_b('55-65'):<12} | {fmt_b('65-75'):<12} | {fmt_b('75+'):<12}")

if __name__ == "__main__":
    audit_calibration()

