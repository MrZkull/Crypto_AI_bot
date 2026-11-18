from __future__ import annotations

import pytest

from execution_storage import (
    ExecutionStorageConfigurationError,
    create_execution_ledger,
)
from exchange_reliability import ExecutionLedger


def test_factory_creates_local_sqlite_when_shared_storage_is_not_required(tmp_path):
    path = tmp_path / "local" / "ledger.sqlite3"
    ledger = create_execution_ledger(
        backend="sqlite", sqlite_path=path, shared_required=False
    )
    try:
        assert isinstance(ledger, ExecutionLedger)
        assert path.exists()
    finally:
        ledger.close()


def test_factory_rejects_sqlite_when_shared_storage_is_required():
    with pytest.raises(ExecutionStorageConfigurationError, match="SQLite cannot coordinate"):
        create_execution_ledger(backend="sqlite", shared_required=True)


def test_factory_rejects_memory_database_when_shared_storage_is_required():
    with pytest.raises(ExecutionStorageConfigurationError, match="SQLite cannot coordinate"):
        create_execution_ledger(
            backend="sqlite", sqlite_path=":memory:", shared_required=True
        )


def test_factory_requires_explicit_postgres_dsn():
    with pytest.raises(ExecutionStorageConfigurationError, match="DSN is missing"):
        create_execution_ledger(backend="postgresql", dsn="", shared_required=True)


def test_factory_rejects_unknown_backend():
    with pytest.raises(ExecutionStorageConfigurationError, match="Unsupported"):
        create_execution_ledger(backend="redis", shared_required=True)


def test_factory_enforces_fencing_when_shared_postgres_is_required(monkeypatch):
    import execution_storage

    captured = {}

    class FakePostgresLedger:
        def __init__(self, dsn, **kwargs):
            captured["dsn"] = dsn
            captured.update(kwargs)

    monkeypatch.setattr(execution_storage, "PostgresExecutionLedger", FakePostgresLedger)
    ledger = create_execution_ledger(
        backend="postgresql",
        dsn="postgresql://redacted.invalid/db",
        shared_required=True,
        require_fenced_writes=False,  # shared_required must override this unsafe opt-out
    )
    assert isinstance(ledger, FakePostgresLedger)
    assert captured["require_fenced_writes"] is True
    assert captured["sslmode"] == "require"


def test_factory_can_enable_fencing_explicitly_for_non_shared_postgres(monkeypatch):
    import execution_storage

    captured = {}

    class FakePostgresLedger:
        def __init__(self, dsn, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(execution_storage, "PostgresExecutionLedger", FakePostgresLedger)
    create_execution_ledger(
        backend="postgresql",
        dsn="postgresql://redacted.invalid/db",
        shared_required=False,
        require_fenced_writes=True,
    )
    assert captured["require_fenced_writes"] is True
