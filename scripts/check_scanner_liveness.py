#!/usr/bin/env python3
"""Validate the trade-scanner per-run liveness contract.

The validator is intentionally strict when scan_ran=true. A deliberate
should_scan=false path is reported as an intentional skip and passes. A stale
or stuck workflow with phase=started, failed, or missing counters fails.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Any


def validate(status: dict[str, Any], expected_symbols: int | None = None) -> tuple[bool, dict[str, Any]]:
    required = ("phase", "scan_ran", "symbols_attempted", "symbols_scored", "predictions_saved")
    missing = [k for k in required if k not in status]
    if missing:
        return False, {"reason": "MISSING_LIVENESS_FIELDS", "missing": missing}

    if status.get("phase") != "completed":
        return False, {"reason": "SCAN_NOT_COMPLETED", "phase": status.get("phase")}

    try:
        attempted = int(status["symbols_attempted"])
        scored = int(status["symbols_scored"])
        saved = int(status["predictions_saved"])
    except (TypeError, ValueError):
        return False, {"reason": "NON_INTEGER_LIVENESS_FIELDS", "status": status}

    if min(attempted, scored, saved) < 0:
        return False, {"reason": "NEGATIVE_LIVENESS_FIELD", "status": status}
    if scored > attempted:
        return False, {"reason": "SCORED_EXCEEDS_ATTEMPTED", "status": status}

    if not bool(status["scan_ran"]):
        if status.get("skip_reason") in (None, ""):
            return False, {"reason": "SKIP_REASON_MISSING", "status": status}
        return True, {"reason": "INTENTIONAL_SCAN_SKIP", "status": status}

    if expected_symbols is not None and attempted != int(expected_symbols):
        return False, {"reason": "INCOMPLETE_SYMBOL_ATTEMPTS", "status": status, "expected_symbols": int(expected_symbols)}
    if scored <= 0:
        return False, {"reason": "ZERO_SYMBOLS_SCORED", "status": status}
    if saved <= 0:
        return False, {"reason": "ZERO_PREDICTIONS_SAVED", "status": status}
    if saved < scored:
        return False, {"reason": "PERSISTENCE_BEHIND_ML", "status": status}

    return True, {"reason": "OK", "status": status}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", default="scan_status.json")
    ap.add_argument("--expected-symbols", type=int)
    args = ap.parse_args()
    status = json.loads(Path(args.status).read_text(encoding="utf-8"))
    ok, detail = validate(status, args.expected_symbols)
    print(json.dumps(detail, indent=2, sort_keys=True))
    return 0 if ok else 1

if __name__ == "__main__":
    raise SystemExit(main())
