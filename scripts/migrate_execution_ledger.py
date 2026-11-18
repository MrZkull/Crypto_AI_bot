"""Apply execution-ledger PostgreSQL schema migrations (privileged operation)."""
from __future__ import annotations

import os
import sys

from postgres_execution_ledger import apply_schema_migrations


def main() -> int:
    dsn = os.getenv("EXECUTION_LEDGER_MIGRATION_DSN", "").strip()
    if not dsn:
        print("EXECUTION_LEDGER_MIGRATION_DSN is required; no migration attempted.", file=sys.stderr)
        return 2
    sslmode = os.getenv("EXECUTION_LEDGER_SSLMODE", "require").strip().lower()
    try:
        version = apply_schema_migrations(dsn, sslmode=sslmode)
    except Exception as exc:
        print(f"Execution-ledger migration failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"Execution-ledger schema migration verified: version={version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
