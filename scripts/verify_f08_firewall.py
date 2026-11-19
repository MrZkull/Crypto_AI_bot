#!/usr/bin/env python3
"""Check F08 protected paths without scanning ignored historical evidence."""

from __future__ import annotations

import argparse
import subprocess
import sys


LOCKED_BASENAMES = {
    "feature_engineering.py",
    "market_data_integrity.py",
}


def is_f08_locked_path(path: str) -> bool:
    normalized = path.replace("\\", "/").strip("/").lower()
    parts = normalized.split("/")
    name = parts[-1]

    if name in LOCKED_BASENAMES:
        return True

    if name.startswith("phase2d_") or name.startswith("test_phase2d_"):
        return True

    if normalized == ".github/workflows/run_phase2d.yml":
        return True

    if "research_outputs" in parts:
        return True

    if len(parts) >= 2 and parts[0:2] == ["evidence", "phase2d"]:
        return True

    return False


def git(*args: str) -> list[str]:
    result = subprocess.run(
        ["git", *args],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "Git command failed")
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base",
        default="a30c8084445a41e9dd220513bfb089d2cbe47587",
    )
    args = parser.parse_args()

    try:
        git("rev-parse", "--verify", f"{args.base}^{{commit}}")
        tracked = git("diff", "--name-only", "--no-renames", args.base, "--")
        untracked = git("ls-files", "--others", "--exclude-standard")
    except Exception as exc:
        print(f"F08 FIREWALL ERROR: {exc}", file=sys.stderr)
        return 2

    paths = sorted(set(tracked).union(untracked))
    blocked = [path for path in paths if is_f08_locked_path(path)]

    if blocked:
        print("F08 FIREWALL BLOCKED:", file=sys.stderr)
        for path in blocked:
            print(f"  {path}", file=sys.stderr)
        return 1

    print(
        f"F08 FIREWALL PASS: checked {len(paths)} tracked/non-ignored "
        "untracked paths. Ignored archives and historical evidence were not enumerated."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())