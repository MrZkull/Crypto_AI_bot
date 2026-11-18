"""Fail-closed execution-ledger backend selection.

SQLite is suitable for local/single-host development only. Deployments that
require coordination across independent runners or hosts must explicitly select
PostgreSQL and set EXECUTION_LEDGER_SHARED_REQUIRED=true.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from exchange_reliability import ExecutionLedger
from postgres_execution_ledger import PostgresExecutionLedger


class ExecutionStorageConfigurationError(RuntimeError):
    pass


def _as_bool(value: object) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def create_execution_ledger(
    *,
    backend: Optional[str] = None,
    sqlite_path: Optional[str | Path] = None,
    dsn: Optional[str] = None,
    shared_required: Optional[bool] = None,
    sslmode: Optional[str] = None,
    require_fenced_writes: Optional[bool] = None,
):
    """Create the configured ledger without silently downgrading durability.

    Environment variables:
      EXECUTION_LEDGER_BACKEND=sqlite|postgresql
      EXECUTION_LEDGER_SHARED_REQUIRED=true|false
      EXECUTION_LEDGER_SQLITE_PATH=/persistent/path/ledger.sqlite3
      EXECUTION_LEDGER_DSN=<secret PostgreSQL DSN>
      EXECUTION_LEDGER_SSLMODE=require
      EXECUTION_LEDGER_REQUIRE_FENCED_WRITES=true|false

    If shared storage is required, SQLite is rejected. PostgreSQL connection or
    driver failures are raised to the caller; there is no fallback to a local
    file and no exchange action is performed by this factory.
    """
    selected = str(backend or os.getenv("EXECUTION_LEDGER_BACKEND", "sqlite")).strip().lower()
    require_shared = (
        _as_bool(os.getenv("EXECUTION_LEDGER_SHARED_REQUIRED", "false"))
        if shared_required is None
        else bool(shared_required)
    )
    if selected not in {"sqlite", "postgres", "postgresql"}:
        raise ExecutionStorageConfigurationError(
            f"Unsupported EXECUTION_LEDGER_BACKEND={selected!r}"
        )
    if require_shared and selected == "sqlite":
        raise ExecutionStorageConfigurationError(
            "Shared execution storage is required; SQLite cannot coordinate independent hosts"
        )
    if selected in {"postgres", "postgresql"}:
        database_dsn = dsn or os.getenv("EXECUTION_LEDGER_DSN", "")
        if not database_dsn.strip():
            raise ExecutionStorageConfigurationError(
                "PostgreSQL backend selected but EXECUTION_LEDGER_DSN is missing"
            )
        configured_fencing = (
            _as_bool(os.getenv("EXECUTION_LEDGER_REQUIRE_FENCED_WRITES", "false"))
            if require_fenced_writes is None
            else bool(require_fenced_writes)
        )
        # Shared-required mode may not silently opt out of token validation.
        fenced_writes = bool(require_shared or configured_fencing)
        return PostgresExecutionLedger(
            database_dsn,
            sslmode=sslmode or os.getenv("EXECUTION_LEDGER_SSLMODE", "require"),
            require_fenced_writes=fenced_writes,
        )
    path = sqlite_path or os.getenv(
        "EXECUTION_LEDGER_SQLITE_PATH", str(Path("data") / "execution_ledger.sqlite3")
    )
    if str(path).strip() in {"", ":memory:"} and require_shared:
        raise ExecutionStorageConfigurationError(
            "In-memory/local SQLite is forbidden when shared execution storage is required"
        )
    return ExecutionLedger(path)
