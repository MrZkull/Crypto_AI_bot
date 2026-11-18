from __future__ import annotations

import os
import re
import sys
import threading
import types
import uuid
from concurrent.futures import ThreadPoolExecutor

import pytest

from exchange_reliability import (
    IdempotencyConflict,
    InvalidTransition,
    OrderState,
    ProtectionInvariantError,
)
from postgres_execution_ledger import PostgresExecutionLedger, _SCHEMA_STATEMENTS, apply_schema_migrations


class _FakeCursor:
    def __init__(self, connection):
        self.connection = connection
        self._row = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, sql, params=None):
        self.connection.statements.append((" ".join(sql.split()), params))
        if "SELECT MAX(version)" in sql:
            self._row = {"version": 4}
        elif "RETURNING fencing_token" in sql or "RETURNING run_leases.fencing_token" in sql:
            self._row = {"fencing_token": 7}
        elif "RETURNING run_leases.lease_key" in sql:
            self._row = {"lease_key": "unit-lease"}
        elif "SELECT lease_key FROM run_leases" in sql:
            self._row = {"lease_key": "unit-lease"}
        return self

    def fetchone(self):
        return self._row


class _FakeConnection:
    def __init__(self):
        self.statements = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def cursor(self):
        return _FakeCursor(self)


class _FakePsycopg(types.ModuleType):
    def __init__(self):
        super().__init__("psycopg")
        self.calls = []
        self.connections = []

    def connect(self, dsn, **kwargs):
        self.calls.append((dsn, kwargs))
        connection = _FakeConnection()
        self.connections.append(connection)
        return connection


def test_postgres_runtime_constructor_verifies_schema_without_ddl_and_requires_tls(monkeypatch):
    fake_driver = _FakePsycopg()
    fake_rows = types.ModuleType("psycopg.rows")
    fake_rows.dict_row = object()
    monkeypatch.setitem(sys.modules, "psycopg", fake_driver)
    monkeypatch.setitem(sys.modules, "psycopg.rows", fake_rows)

    ledger = PostgresExecutionLedger("postgresql://user:secret@example.invalid/db")

    assert fake_driver.calls[0][0] == "postgresql://user:secret@example.invalid/db"
    assert fake_driver.calls[0][1]["sslmode"] == "require"
    assert fake_driver.calls[0][1]["connect_timeout"] == 10
    executed = [sql for sql, _ in fake_driver.connections[0].statements]
    assert executed == ["SELECT MAX(version) AS version FROM execution_ledger_schema"]
    assert not any(re.match(r"^(CREATE|ALTER|DROP|INSERT|UPDATE|DELETE)\b", sql, re.I) for sql in executed)
    ledger.close()


def test_explicit_migration_applies_idempotent_ddl_under_migration_connection(monkeypatch):
    fake_driver = _FakePsycopg()
    fake_rows = types.ModuleType("psycopg.rows")
    fake_rows.dict_row = object()
    monkeypatch.setitem(sys.modules, "psycopg", fake_driver)
    monkeypatch.setitem(sys.modules, "psycopg.rows", fake_rows)

    assert apply_schema_migrations("postgresql://migration-role@db.invalid/test", sslmode="require") == 4
    executed = [sql for sql, _ in fake_driver.connections[0].statements]
    assert executed[0].startswith("SELECT pg_advisory_xact_lock")
    assert len(executed) == len(_SCHEMA_STATEMENTS) + 2
    assert any("CREATE TABLE IF NOT EXISTS order_intents" in sql for sql in executed)
    assert any("ALTER TABLE run_leases" in sql for sql in executed)
    assert executed[-1] == "SELECT MAX(version) AS version FROM execution_ledger_schema"


def test_postgres_constructor_rejects_invalid_sslmode():
    with pytest.raises(ValueError, match="sslmode"):
        PostgresExecutionLedger("postgresql://unused", sslmode="off")


def test_postgres_lease_statement_is_single_conditional_upsert():
    source = open(os.path.join(os.path.dirname(__file__), "..", "postgres_execution_ledger.py"), encoding="utf-8").read()
    assert "ON CONFLICT (lease_key) DO UPDATE" in source
    assert "run_leases.expires_at <= EXCLUDED.updated_at" in source
    assert "run_leases.owner_id = EXCLUDED.owner_id" in source
    assert "RETURNING fencing_token" in source
    assert "fencing_token" in source


def test_postgres_lease_acquire_and_renew_use_database_clock():
    source = open(os.path.join(os.path.dirname(__file__), "..", "postgres_execution_ledger.py"), encoding="utf-8").read()
    assert source.count("EXTRACT(EPOCH FROM clock_timestamp())") >= 3
    assert "FROM db_clock" in source
    assert "run_leases.expires_at>db_clock.now_epoch" in source
    assert "finite positive ttl_seconds" in source


POSTGRES_DSN = os.getenv("EXECUTION_LEDGER_TEST_DSN", "").strip()
POSTGRES_SSLMODE = os.getenv("EXECUTION_LEDGER_TEST_SSLMODE", "require").strip().lower()
POSTGRES_MIGRATION_DSN = os.getenv("EXECUTION_LEDGER_TEST_MIGRATION_DSN", "").strip()
EXPECT_DML_ONLY = os.getenv("EXECUTION_LEDGER_TEST_EXPECT_DML_ONLY", "false").strip().lower() in {"1", "true", "yes"}


@pytest.fixture(scope="module", autouse=True)
def _migrate_real_postgres_once():
    if POSTGRES_DSN:
        # Prefer a separate privileged DSN; a local one-role test may use the
        # same DSN, while CI proves that runtime has DML-only permissions.
        apply_schema_migrations(
            POSTGRES_MIGRATION_DSN or POSTGRES_DSN, sslmode=POSTGRES_SSLMODE
        )


def _integration_ledger(*, require_fenced_writes=False):
    return PostgresExecutionLedger(
        POSTGRES_DSN, sslmode=POSTGRES_SSLMODE,
        require_fenced_writes=require_fenced_writes,
    )


@pytest.mark.skipif(
    not POSTGRES_DSN or not EXPECT_DML_ONLY,
    reason="Requires a dedicated DML-only PostgreSQL runtime role",
)
def test_real_postgres_runtime_role_cannot_create_schema_objects():
    import psycopg
    from psycopg import errors

    probe = "cryptobot_runtime_ddl_denied_" + uuid.uuid4().hex[:12]
    with psycopg.connect(POSTGRES_DSN, sslmode=POSTGRES_SSLMODE) as conn:
        with pytest.raises(errors.InsufficientPrivilege):
            with conn.cursor() as cur:
                cur.execute(f'CREATE TABLE public."{probe}" (id INTEGER)')


@pytest.mark.skipif(not POSTGRES_DSN, reason="Set EXECUTION_LEDGER_TEST_DSN to run real PostgreSQL integration tests")
def test_real_postgres_order_ledger_lifecycle_and_restart():
    ledger = _integration_ledger()
    client_id = "pg-it-" + uuid.uuid4().hex[:24]
    account_id = "integration-" + uuid.uuid4().hex[:12]
    try:
        intent = ledger.create_intent(
            exchange="DERIBIT", environment="testnet", account_id=account_id,
            client_order_id=client_id, instrument_id="ETH_USDC-PERPETUAL",
            side="BUY", quantity="1", settlement_currency="USDC",
            payload={"test": "postgres-lifecycle"},
        )
        replay = ledger.create_intent(
            exchange="DERIBIT", environment="testnet", account_id=account_id,
            client_order_id=client_id, instrument_id="ETH_USDC-PERPETUAL",
            side="BUY", quantity="1", settlement_currency="USDC",
            payload={"test": "postgres-lifecycle"},
        )
        assert replay["intent_id"] == intent["intent_id"]
        assert len(ledger.events(intent["intent_id"])) == 1

        ledger.transition(intent["intent_id"], OrderState.SUBMITTING)
        ledger.transition(intent["intent_id"], OrderState.ACKNOWLEDGED, exchange_order_id="pg-order-1")
        filled = ledger.record_fill(
            intent["intent_id"], cumulative_filled_quantity="1",
            average_fill_price="2500", exchange_order_id="pg-order-1",
            fee="0.01", fee_currency="USDC",
        )
        assert filled["state"] == OrderState.FILLED.value
        assert filled["filled_quantity"] == "1"
        assert filled["payload"]["test"] == "postgres-lifecycle"
        with pytest.raises(ProtectionInvariantError):
            ledger.mark_protected(intent["intent_id"], "0.99")
        assert ledger.mark_protected(intent["intent_id"], "1")["protection_state"] == "PROTECTED"
        assert len(ledger.events(intent["intent_id"])) >= 5

        mutation_id = "pg-mutation-" + uuid.uuid4().hex
        ledger.begin_mutation(
            mutation_id=mutation_id, exchange="DERIBIT", environment="testnet",
            account_id=account_id, operation="CANCEL", resource_id="pg-order-1",
            request_hash="a" * 64, payload={"test": "mutation-restart"},
        )
        assert len(ledger.unresolved_mutations(
            exchange="DERIBIT", environment="testnet", account_id=account_id,
        )) == 1
        with pytest.raises(Exception, match="unresolved"):
            ledger.begin_mutation(
                mutation_id=mutation_id + "-replay", exchange="DERIBIT", environment="testnet",
                account_id=account_id, operation="CANCEL", resource_id="pg-order-1",
                request_hash="b" * 64,
            )
        ledger.finish_mutation(mutation_id, state="UNKNOWN", payload={"error_type": "timeout"})
        evidence = {
            "verified": True, "verification_method": "read_only_order_lookup",
            "observed_status": "cancelled", "observed_at": "2026-11-18T12:00:00Z",
            "resource_id": "pg-order-1",
        }
        ledger.resolve_mutation(
            mutation_id, state="ACKNOWLEDGED", evidence=evidence,
            resolved_by="postgres-integration-operator",
        )
        assert ledger.unresolved_mutations(
            exchange="DERIBIT", environment="testnet", account_id=account_id,
        ) == []
        assert [event["event_type"] for event in ledger.mutation_events(mutation_id)] == [
            "SUBMISSION_STARTED", "EXCHANGE_RESPONSE_RECORDED", "READ_ONLY_RECONCILIATION_CONFIRMED",
        ]

        # Simulate a process restart by closing every connection held by the
        # first ledger instance and reopening a fresh runtime instance. Durable
        # order, fill, protection and control-mutation state must survive.
        ledger.close()
        ledger = _integration_ledger()
        reopened_intent = ledger.get_intent(intent["intent_id"])
        assert reopened_intent["state"] == OrderState.FILLED.value
        assert reopened_intent["filled_quantity"] == "1"
        assert reopened_intent["protection_state"] == "PROTECTED"
        assert ledger.get_mutation(mutation_id)["state"] == "ACKNOWLEDGED"
        assert [event["event_type"] for event in ledger.mutation_events(mutation_id)] == [
            "SUBMISSION_STARTED", "EXCHANGE_RESPONSE_RECORDED", "READ_ONLY_RECONCILIATION_CONFIRMED",
        ]
    finally:
        ledger.close()


@pytest.mark.skipif(not POSTGRES_DSN, reason="Set EXECUTION_LEDGER_TEST_DSN to run real PostgreSQL integration tests")
def test_real_postgres_lease_has_one_winner_across_independent_connections():
    key = "integration-lease-" + uuid.uuid4().hex
    owners = ["owner-A-" + uuid.uuid4().hex[:8], "owner-B-" + uuid.uuid4().hex[:8]]
    barrier = threading.Barrier(2)

    def acquire(owner):
        ledger = _integration_ledger()
        try:
            barrier.wait(timeout=10)
            return ledger.acquire_run_lease(key, owner, ttl_seconds=30)
        finally:
            ledger.close()

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(acquire, owners))
    assert sum(bool(value) for value in results) == 1

    ledger = _integration_ledger()
    try:
        winning_owner = owners[results.index(True)]
        losing_owner = owners[results.index(False)]
        assert not ledger.release_run_lease(key, losing_owner)
        assert ledger.release_run_lease(key, winning_owner)
    finally:
        ledger.close()


def test_schema_v2_migrates_existing_run_leases_with_monotonic_generation():
    joined = " ".join(" ".join(statement.split()) for statement in _SCHEMA_STATEMENTS)
    assert "ALTER TABLE run_leases ADD COLUMN IF NOT EXISTS fencing_token BIGINT NOT NULL DEFAULT 0" in joined
    assert "INSERT INTO execution_ledger_schema(version, applied_at) VALUES (2" in joined
    assert "INSERT INTO execution_ledger_schema(version, applied_at) VALUES (3" in joined
    assert "INSERT INTO execution_ledger_schema(version, applied_at) VALUES (4" in joined

def test_schema_v4_persists_control_mutation_events_and_unresolved_index():
    joined = " ".join(" ".join(statement.split()) for statement in _SCHEMA_STATEMENTS)
    assert "CREATE TABLE IF NOT EXISTS execution_mutation_events" in joined
    assert "idx_execution_mutations_unresolved" in joined
    assert "idx_execution_mutation_events_mutation" in joined
    source = open(os.path.join(os.path.dirname(__file__), "..", "postgres_execution_ledger.py"), encoding="utf-8").read()
    assert "def unresolved_mutations(" in source
    assert "def resolve_mutation(" in source
    assert "def mutation_events(" in source
    assert "LOCK TABLE execution_mutations IN SHARE ROW EXCLUSIVE MODE" in source


def test_fenced_lease_acquire_increments_generation_and_release_preserves_row():
    source = open(os.path.join(os.path.dirname(__file__), "..", "postgres_execution_ledger.py"), encoding="utf-8").read()
    assert "def acquire_fenced_run_lease(" in source
    assert "run_leases.fencing_token + 1" in source
    assert "def release_fenced_run_lease(" in source
    assert "SET expires_at=db_clock.now_epoch" in source
    assert "DELETE FROM run_leases" not in source

def test_fenced_write_validation_locks_current_unexpired_generation():
    source = open(os.path.join(os.path.dirname(__file__), "..", "postgres_execution_ledger.py"), encoding="utf-8").read()
    assert "def _assert_fenced_lease(" in source
    assert "fencing_token=%s" in source
    assert "expires_at > EXTRACT(EPOCH FROM clock_timestamp())" in source
    assert "FOR UPDATE" in source

@pytest.mark.skipif(not POSTGRES_DSN, reason="Set EXECUTION_LEDGER_TEST_DSN to run real PostgreSQL integration tests")
def test_real_postgres_fencing_token_increases_after_expiry_and_stale_owner_is_rejected():
    from exchange_reliability import ReliabilityError
    key = "integration-fenced-lease-" + uuid.uuid4().hex
    ledger = _integration_ledger(require_fenced_writes=True)
    owner_a = "owner-a-" + uuid.uuid4().hex[:8]
    owner_b = "owner-b-" + uuid.uuid4().hex[:8]
    try:
        token_a = ledger.acquire_fenced_run_lease(key, owner_a, ttl_seconds=30)
        assert isinstance(token_a, int) and token_a > 0
        assert ledger.release_fenced_run_lease(key, owner_a, token_a)
        token_b = ledger.acquire_fenced_run_lease(key, owner_b, ttl_seconds=30)
        assert isinstance(token_b, int) and token_b > token_a
        with pytest.raises(ReliabilityError, match="expired, replaced or fenced"):
            ledger.create_intent(
                exchange="DERIBIT", environment="testnet", account_id="fencing-it",
                client_order_id="stale-" + uuid.uuid4().hex, instrument_id="ETH_USDC-PERPETUAL",
                side="BUY", quantity="1", settlement_currency="USDC",
                lease_key=key, lease_owner_id=owner_a, fencing_token=token_a,
            )
    finally:
        ledger.release_fenced_run_lease(key, owner_b, token_b if 'token_b' in locals() else 0)
        ledger.close()


@pytest.mark.skipif(not POSTGRES_DSN, reason="Set EXECUTION_LEDGER_TEST_DSN to run real PostgreSQL integration tests")
def test_real_postgres_lease_expires_by_database_clock_without_release():
    import time
    key = "integration-natural-expiry-" + uuid.uuid4().hex
    ledger = _integration_ledger(require_fenced_writes=True)
    owner_a = "expiry-a-" + uuid.uuid4().hex[:8]
    owner_b = "expiry-b-" + uuid.uuid4().hex[:8]
    try:
        token_a = ledger.acquire_fenced_run_lease(key, owner_a, ttl_seconds=0.25)
        assert isinstance(token_a, int) and token_a > 0
        # Do not call release: exercise the actual database-time expiry path.
        time.sleep(0.45)
        token_b = ledger.acquire_fenced_run_lease(key, owner_b, ttl_seconds=10)
        assert isinstance(token_b, int) and token_b > token_a
        from exchange_reliability import ReliabilityError
        with pytest.raises(ReliabilityError, match="expired, replaced or fenced"):
            ledger.create_intent(
                exchange="DERIBIT", environment="testnet", account_id="natural-expiry-it",
                client_order_id="stale-expiry-" + uuid.uuid4().hex,
                instrument_id="ETH_USDC-PERPETUAL", side="BUY", quantity="1",
                settlement_currency="USDC", lease_key=key,
                lease_owner_id=owner_a, fencing_token=token_a,
            )
    finally:
        if "token_b" in locals():
            ledger.release_fenced_run_lease(key, owner_b, token_b)
        ledger.close()


def test_fenced_lease_api_returns_generation_and_renews_and_releases(monkeypatch):
    fake_driver = _FakePsycopg()
    fake_rows = types.ModuleType("psycopg.rows")
    fake_rows.dict_row = object()
    monkeypatch.setitem(sys.modules, "psycopg", fake_driver)
    monkeypatch.setitem(sys.modules, "psycopg.rows", fake_rows)

    ledger = PostgresExecutionLedger(
        "postgresql://redacted.invalid/db", require_fenced_writes=True
    )
    try:
        token = ledger.acquire_fenced_run_lease("unit-lease", "owner-1", ttl_seconds=30)
        assert token == 7
        assert ledger.renew_fenced_run_lease("unit-lease", "owner-1", token, ttl_seconds=30)
        assert ledger.release_fenced_run_lease("unit-lease", "owner-1", token)
        statements = [sql for conn in fake_driver.connections for sql, _ in conn.statements]
        assert any("fencing_token=CASE" in sql for sql in statements)
        connection = _FakeConnection()
        with connection.cursor() as cursor:
            ledger._assert_fenced_lease(
                cursor, lease_key="unit-lease", lease_owner_id="owner-1", fencing_token=token
            )
        assert any("FOR UPDATE" in sql and "fencing_token=%s" in sql for sql, _ in connection.statements)
    finally:
        ledger.close()


def test_fenced_ledger_refuses_owner_only_lease_api(monkeypatch):
    fake_driver = _FakePsycopg()
    fake_rows = types.ModuleType("psycopg.rows")
    fake_rows.dict_row = object()
    monkeypatch.setitem(sys.modules, "psycopg", fake_driver)
    monkeypatch.setitem(sys.modules, "psycopg.rows", fake_rows)
    ledger = PostgresExecutionLedger(
        "postgresql://redacted.invalid/db", require_fenced_writes=True
    )
    try:
        from exchange_reliability import ReliabilityError
        with pytest.raises(ReliabilityError, match="Fenced mode requires"):
            ledger.acquire_run_lease("unit-lease", "owner-1")
        assert not ledger.renew_run_lease("unit-lease", "owner-1")
        assert not ledger.release_run_lease("unit-lease", "owner-1")
    finally:
        ledger.close()


def test_postgres_entry_gate_serializes_account_and_detects_unresolved_or_unprotected_intents():
    source = open(os.path.join(os.path.dirname(__file__), "..", "postgres_execution_ledger.py"), encoding="utf-8").read()
    assert "enforce_unresolved_gate" in source
    assert "pg_advisory_xact_lock" in source
    assert "state IN ('INTENT_CREATED','SUBMITTING','ACKNOWLEDGED','PARTIALLY_FILLED','UNKNOWN','RECONCILING')" in source
    assert "protection_state != 'PROTECTED'" in source


def test_fenced_write_rejects_missing_or_invalid_lease_context():
    from exchange_reliability import ReliabilityError
    ledger = object.__new__(PostgresExecutionLedger)
    ledger.require_fenced_writes = True
    with pytest.raises(ReliabilityError, match="current lease key"):
        ledger._assert_fenced_lease(
            _FakeConnection().cursor(), lease_key=None, lease_owner_id=None, fencing_token=None
        )
