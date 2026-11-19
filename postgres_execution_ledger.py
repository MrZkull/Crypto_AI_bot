"""PostgreSQL-backed execution ledger for multi-host deployments.

This module stores order intents, append-only events, and fenced run leases in
shared PostgreSQL. It does not submit, cancel, or amend exchange orders. Schema
v2 adds monotonic lease generations and database-validated write fencing.

The caller must provide a dedicated database DSN. Connections are opened per
operation so independent GitHub Actions runners and Render processes contend
on the same database transactions rather than process-local locks.
"""
from __future__ import annotations

import importlib
import json
import math
import time
import uuid
from decimal import Decimal
from typing import Any, Mapping, Optional

from exchange_reliability import (
    IdempotencyConflict,
    InvalidTransition,
    OrderState,
    ProtectionInvariantError,
    ReliabilityError,
    UnresolvedExecutionState,
    _ALLOWED_TRANSITIONS,
    _decimal,
)


_SCHEMA_STATEMENTS = (
    """
    CREATE TABLE IF NOT EXISTS order_intents (
        intent_id TEXT PRIMARY KEY,
        exchange TEXT NOT NULL,
        environment TEXT NOT NULL,
        account_id TEXT NOT NULL,
        client_order_id TEXT NOT NULL,
        instrument_id TEXT NOT NULL,
        side TEXT NOT NULL CHECK (side IN ('BUY','SELL')),
        quantity TEXT NOT NULL,
        settlement_currency TEXT NOT NULL,
        reduce_only BOOLEAN NOT NULL,
        state TEXT NOT NULL,
        exchange_order_id TEXT,
        filled_quantity TEXT NOT NULL DEFAULT '0',
        average_fill_price TEXT,
        protected_quantity TEXT NOT NULL DEFAULT '0',
        protection_state TEXT NOT NULL DEFAULT 'NOT_APPLICABLE',
        payload_json TEXT NOT NULL,
        created_at DOUBLE PRECISION NOT NULL,
        updated_at DOUBLE PRECISION NOT NULL,
        UNIQUE(exchange, environment, account_id, client_order_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS order_events (
        event_id BIGSERIAL PRIMARY KEY,
        intent_id TEXT NOT NULL REFERENCES order_intents(intent_id),
        state TEXT NOT NULL,
        event_type TEXT NOT NULL,
        payload_json TEXT NOT NULL,
        created_at DOUBLE PRECISION NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_order_events_intent ON order_events(intent_id, event_id)",
    "CREATE INDEX IF NOT EXISTS idx_order_intents_recovery ON order_intents(exchange, environment, state, updated_at)",
    """
    CREATE TABLE IF NOT EXISTS run_leases (
        lease_key TEXT PRIMARY KEY,
        owner_id TEXT NOT NULL,
        expires_at DOUBLE PRECISION NOT NULL,
        updated_at DOUBLE PRECISION NOT NULL,
        fencing_token BIGINT NOT NULL DEFAULT 0
    )
    """,
    "ALTER TABLE run_leases ADD COLUMN IF NOT EXISTS fencing_token BIGINT NOT NULL DEFAULT 0",
    "UPDATE run_leases SET fencing_token=0 WHERE fencing_token IS NULL",
    "ALTER TABLE run_leases ALTER COLUMN fencing_token SET DEFAULT 0",
    "ALTER TABLE run_leases ALTER COLUMN fencing_token SET NOT NULL",
    """
    CREATE TABLE IF NOT EXISTS execution_ledger_schema (
        version INTEGER PRIMARY KEY,
        applied_at DOUBLE PRECISION NOT NULL
    )
    """,
    "INSERT INTO execution_ledger_schema(version, applied_at) VALUES (1, EXTRACT(EPOCH FROM NOW())) ON CONFLICT (version) DO NOTHING",
    "INSERT INTO execution_ledger_schema(version, applied_at) VALUES (2, EXTRACT(EPOCH FROM NOW())) ON CONFLICT (version) DO NOTHING",
    """
    CREATE TABLE IF NOT EXISTS execution_mutations (
        mutation_id TEXT PRIMARY KEY,
        exchange TEXT NOT NULL,
        environment TEXT NOT NULL,
        account_id TEXT NOT NULL,
        operation TEXT NOT NULL,
        resource_id TEXT NOT NULL,
        request_hash TEXT NOT NULL,
        payload_json TEXT NOT NULL,
        state TEXT NOT NULL,
        response_id TEXT,
        created_at DOUBLE PRECISION NOT NULL,
        updated_at DOUBLE PRECISION NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_execution_mutations_scope ON execution_mutations(exchange, environment, account_id, created_at)",
    "CREATE INDEX IF NOT EXISTS idx_execution_mutations_unresolved ON execution_mutations(exchange, environment, account_id, state) WHERE state IN ('SUBMITTING','UNKNOWN')",
    """CREATE TABLE IF NOT EXISTS execution_mutation_events (
        event_id BIGSERIAL PRIMARY KEY,
        mutation_id TEXT NOT NULL REFERENCES execution_mutations(mutation_id),
        event_type TEXT NOT NULL,
        state TEXT NOT NULL,
        payload_json TEXT NOT NULL,
        created_at DOUBLE PRECISION NOT NULL
    )""",
    "CREATE INDEX IF NOT EXISTS idx_execution_mutation_events_mutation ON execution_mutation_events(mutation_id, event_id)",
    """
    CREATE TABLE IF NOT EXISTS execution_protection_links (
        parent_intent_id TEXT NOT NULL REFERENCES order_intents(intent_id),
        protective_intent_id TEXT NOT NULL REFERENCES order_intents(intent_id),
        active BOOLEAN NOT NULL DEFAULT TRUE,
        payload_json TEXT NOT NULL,
        created_at DOUBLE PRECISION NOT NULL,
        updated_at DOUBLE PRECISION NOT NULL,
        PRIMARY KEY(parent_intent_id, protective_intent_id)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_execution_protection_active ON execution_protection_links(protective_intent_id, active)",
    """CREATE TABLE IF NOT EXISTS account_risk_snapshots (
        exchange TEXT NOT NULL,
        environment TEXT NOT NULL,
        account_id TEXT NOT NULL,
        currency TEXT NOT NULL,
        observed_at DOUBLE PRECISION NOT NULL,
        equity TEXT NOT NULL,
        source TEXT NOT NULL,
        PRIMARY KEY(exchange, environment, account_id, currency, observed_at)
    )""",
    "CREATE INDEX IF NOT EXISTS idx_account_risk_snapshots_lookup ON account_risk_snapshots(exchange, environment, account_id, currency, observed_at DESC)",
    """CREATE TABLE IF NOT EXISTS daily_risk_halts (
        exchange TEXT NOT NULL,
        environment TEXT NOT NULL,
        account_id TEXT NOT NULL,
        utc_day TEXT NOT NULL,
        reason TEXT NOT NULL,
        baseline_equity TEXT NOT NULL,
        current_equity TEXT NOT NULL,
        loss_fraction TEXT NOT NULL,
        observed_at DOUBLE PRECISION NOT NULL,
        PRIMARY KEY(exchange, environment, account_id, utc_day)
    )""",
    "INSERT INTO execution_ledger_schema(version, applied_at) VALUES (3, EXTRACT(EPOCH FROM NOW())) ON CONFLICT (version) DO NOTHING",
    "INSERT INTO execution_ledger_schema(version, applied_at) VALUES (4, EXTRACT(EPOCH FROM NOW())) ON CONFLICT (version) DO NOTHING",
    "INSERT INTO execution_ledger_schema(version, applied_at) VALUES (5, EXTRACT(EPOCH FROM NOW())) ON CONFLICT (version) DO NOTHING",
    """CREATE TABLE IF NOT EXISTS risk_capture_leases (
        lease_key TEXT PRIMARY KEY,
        owner_id TEXT NOT NULL,
        expires_at DOUBLE PRECISION NOT NULL,
        updated_at DOUBLE PRECISION NOT NULL,
        fencing_token BIGINT NOT NULL DEFAULT 0
    )""",
    "INSERT INTO execution_ledger_schema(version, applied_at) VALUES (6, EXTRACT(EPOCH FROM NOW())) ON CONFLICT (version) DO NOTHING",
)


def apply_schema_migrations(
    dsn: str,
    *,
    sslmode: str = "require",
    connect_timeout: int = 10,
) -> int:
    """Apply the idempotent execution-ledger schema using a migration DSN.

    This function performs DDL and must never be called by the scanner or
    dashboard runtime. The supplied database role must own the schema/tables
    or otherwise have the required CREATE/ALTER privileges. Runtime roles
    should be limited to SELECT/INSERT/UPDATE plus sequence access.
    """
    if not isinstance(dsn, str) or not dsn.strip():
        raise ValueError("A privileged PostgreSQL migration DSN is required")
    if sslmode not in {"disable", "allow", "prefer", "require", "verify-ca", "verify-full"}:
        raise ValueError("Unsupported PostgreSQL sslmode")
    if int(connect_timeout) < 1 or int(connect_timeout) > 60:
        raise ValueError("connect_timeout must be between 1 and 60 seconds")
    try:
        psycopg = importlib.import_module("psycopg")
        rows_module = importlib.import_module("psycopg.rows")
        dict_row = rows_module.dict_row
    except ImportError as exc:
        raise RuntimeError(
            "PostgreSQL migrations require psycopg 3; install requirements-postgres.txt"
        ) from exc
    try:
        with psycopg.connect(
            dsn.strip(), connect_timeout=int(connect_timeout), sslmode=sslmode,
            row_factory=dict_row,
        ) as conn:
            with conn.cursor() as cur:
                # Serialize deployment jobs so two migrations cannot interleave.
                cur.execute(
                    "SELECT pg_advisory_xact_lock(hashtext(%s))",
                    ("cryptobot_execution_ledger_schema",),
                )
                for statement in _SCHEMA_STATEMENTS:
                    cur.execute(statement)
                cur.execute("SELECT MAX(version) AS version FROM execution_ledger_schema")
                row = cur.fetchone()
                version = int(row["version"] or 0) if row else 0
                if version != 6:
                    raise ReliabilityError("Unsupported execution-ledger schema version after migration")
                return version
    except ReliabilityError:
        raise
    except Exception as exc:
        # Do not include driver exception text: it can contain host/user details.
        raise ReliabilityError(
            f"Could not apply PostgreSQL execution-ledger migrations ({type(exc).__name__})"
        ) from exc


class PostgresExecutionLedger:
    """Shared transactional ledger implementing the SQLite ledger interface.

    Each method obtains its own connection and transaction. Row transitions use
    SELECT ... FOR UPDATE, and run lease acquisition is one conditional UPSERT.
    When require_fenced_writes is enabled, every ledger mutation must present a
    still-current owner/generation tuple which is checked and row-locked in the
    same transaction as that mutation.
    """

    def __init__(
        self,
        dsn: str,
        *,
        sslmode: str = "require",
        connect_timeout: int = 10,
        require_fenced_writes: bool = False,
    ) -> None:
        if not isinstance(dsn, str) or not dsn.strip():
            raise ValueError("A dedicated PostgreSQL execution-ledger DSN is required")
        if sslmode not in {"disable", "allow", "prefer", "require", "verify-ca", "verify-full"}:
            raise ValueError("Unsupported PostgreSQL sslmode")
        if int(connect_timeout) < 1 or int(connect_timeout) > 60:
            raise ValueError("connect_timeout must be between 1 and 60 seconds")
        self._dsn = dsn.strip()
        self.sslmode = sslmode
        self.connect_timeout = int(connect_timeout)
        self.require_fenced_writes = bool(require_fenced_writes)
        try:
            self._psycopg = importlib.import_module("psycopg")
            rows_module = importlib.import_module("psycopg.rows")
            self._dict_row = rows_module.dict_row
        except ImportError as exc:
            raise RuntimeError(
                "PostgreSQL execution storage requires psycopg 3; install "
                "the project requirements (psycopg[binary])"
            ) from exc
        # Runtime roles are deliberately DML-only. Schema changes run through
        # scripts/migrate_execution_ledger.py using a separate privileged DSN.
        self._verify_schema()

    def _connect(self):
        # Never log or include the DSN in raised errors.
        return self._psycopg.connect(
            self._dsn,
            row_factory=self._dict_row,
            connect_timeout=self.connect_timeout,
            sslmode=self.sslmode,
        )

    def _verify_schema(self) -> None:
        """Verify the deployed schema without attempting DDL or migration."""
        try:
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT MAX(version) AS version FROM execution_ledger_schema")
                    row = cur.fetchone()
                    if not row or int(row["version"] or 0) != 6:
                        raise ReliabilityError("Unsupported execution-ledger schema version")
        except ReliabilityError:
            raise
        except Exception as exc:
            raise ReliabilityError(
                f"Could not verify PostgreSQL execution-ledger schema ({type(exc).__name__})"
            ) from exc

    @staticmethod
    def _json(payload: Optional[Mapping[str, Any]]) -> str:
        return json.dumps(dict(payload or {}), sort_keys=True, separators=(",", ":"), default=str)

    @staticmethod
    def _normalise_intent(row: Mapping[str, Any]) -> dict:
        result = dict(row)
        result["reduce_only"] = bool(result["reduce_only"])
        result["payload"] = json.loads(result.pop("payload_json"))
        return result

    def close(self) -> None:
        # Connections are scoped to individual operations and closed by context managers.
        return None

    def _assert_fenced_lease(
        self,
        cur,
        *,
        lease_key: Optional[str],
        lease_owner_id: Optional[str],
        fencing_token: Optional[int],
    ) -> None:
        """Lock and validate the current lease in the same transaction as a write.

        The SELECT FOR UPDATE prevents a new owner from advancing the generation
        until this short ledger transaction completes. The database clock is the
        sole authority for expiry. This fences stale database writers; it does
        not itself fence an external exchange API call.
        """
        supplied_context = any(value is not None for value in (lease_key, lease_owner_id, fencing_token))
        if not self.require_fenced_writes and not supplied_context:
            return
        key = str(lease_key or "").strip()
        owner = str(lease_owner_id or "").strip()
        try:
            token = int(fencing_token) if fencing_token is not None else 0
        except (TypeError, ValueError, OverflowError):
            token = 0
        if not key or not owner or token < 1:
            raise ReliabilityError("A current lease key, owner ID and positive fencing token are required")
        cur.execute(
            """SELECT lease_key FROM run_leases
            WHERE lease_key=%s AND owner_id=%s AND fencing_token=%s
              AND expires_at > EXTRACT(EPOCH FROM clock_timestamp())
            FOR UPDATE""",
            (key, owner, token),
        )
        if cur.fetchone() is None:
            raise ReliabilityError("Execution lease is expired, replaced or fenced by a newer owner")

    def create_intent(
        self,
        *,
        exchange: str,
        environment: str,
        account_id: str,
        client_order_id: str,
        instrument_id: str,
        side: str,
        quantity: Any,
        settlement_currency: str,
        reduce_only: bool = False,
        payload: Optional[Mapping[str, Any]] = None,
        enforce_unresolved_gate: bool = False,
        lease_key: Optional[str] = None,
        lease_owner_id: Optional[str] = None,
        fencing_token: Optional[int] = None,
    ) -> dict:
        values = {
            "exchange": str(exchange).upper(),
            "environment": str(environment).lower(),
            "account_id": str(account_id),
            "client_order_id": str(client_order_id),
            "instrument_id": str(instrument_id),
            "side": str(side).upper(),
            "quantity": str(_decimal(quantity, "quantity")),
            "settlement_currency": str(settlement_currency).upper(),
            "reduce_only": bool(reduce_only),
        }
        if values["side"] not in {"BUY", "SELL"} or _decimal(values["quantity"], "quantity") <= 0:
            raise ValueError("Intent side or quantity is invalid")
        required = ("exchange", "environment", "account_id", "client_order_id", "instrument_id", "settlement_currency")
        if not all(values[key] for key in required):
            raise ValueError("Intent identity fields must be non-empty")
        identity = ("exchange", "environment", "account_id", "client_order_id")
        intent_id = str(uuid.uuid5(uuid.NAMESPACE_URL, "|".join(values[key] for key in identity)))
        payload_json = self._json(payload)
        now = time.time()
        with self._connect() as conn:
            with conn.cursor() as cur:
                self._assert_fenced_lease(
                    cur, lease_key=lease_key, lease_owner_id=lease_owner_id,
                    fencing_token=fencing_token,
                )
                if enforce_unresolved_gate and not values["reduce_only"]:
                    # Serialize entry admissions for this account across PostgreSQL clients.
                    gate_key = "|".join(values[key] for key in ("exchange", "environment", "account_id"))
                    cur.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (gate_key,))
                    cur.execute(
                        "SELECT intent_id FROM order_intents WHERE exchange=%s AND environment=%s AND account_id=%s AND client_order_id=%s",
                        tuple(values[key] for key in ("exchange", "environment", "account_id", "client_order_id")),
                    )
                    existing_for_key = cur.fetchone()
                    if existing_for_key is None:
                        cur.execute(
                            """SELECT intent_id FROM order_intents
                            WHERE exchange=%s AND environment=%s AND account_id=%s AND reduce_only=FALSE
                              AND (state IN ('INTENT_CREATED','SUBMITTING','ACKNOWLEDGED','PARTIALLY_FILLED','UNKNOWN','RECONCILING')
                                   OR (CAST(filled_quantity AS NUMERIC) > 0
                                       AND (protection_state != 'PROTECTED'
                                            OR CAST(protected_quantity AS NUMERIC) < CAST(filled_quantity AS NUMERIC))))
                            ORDER BY created_at LIMIT 1""",
                            (values["exchange"], values["environment"], values["account_id"]),
                        )
                        unresolved = cur.fetchone()
                        if unresolved is not None:
                            raise UnresolvedExecutionState(
                                "New entry blocked by unresolved order/position intent " + str(unresolved["intent_id"])
                            )
                cur.execute(
                    """INSERT INTO order_intents
                    (intent_id,exchange,environment,account_id,client_order_id,
                     instrument_id,side,quantity,settlement_currency,reduce_only,
                     state,payload_json,created_at,updated_at)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    ON CONFLICT (exchange,environment,account_id,client_order_id) DO NOTHING
                    RETURNING intent_id""",
                    (intent_id, values["exchange"], values["environment"], values["account_id"],
                     values["client_order_id"], values["instrument_id"], values["side"],
                     values["quantity"], values["settlement_currency"], values["reduce_only"],
                     OrderState.INTENT_CREATED.value, payload_json, now, now),
                )
                inserted = cur.fetchone() is not None
                cur.execute(
                    """SELECT * FROM order_intents
                    WHERE exchange=%s AND environment=%s AND account_id=%s AND client_order_id=%s
                    FOR UPDATE""",
                    tuple(values[key] for key in identity),
                )
                row = cur.fetchone()
                if row is None:
                    raise ReliabilityError("PostgreSQL intent insert could not be read back")
                for key in ("instrument_id", "side", "quantity", "settlement_currency", "reduce_only"):
                    if row[key] != values[key]:
                        raise IdempotencyConflict(f"Client order id reused with different {key}")
                if inserted:
                    cur.execute(
                        "INSERT INTO order_events(intent_id,state,event_type,payload_json,created_at) VALUES(%s,%s,%s,%s,%s)",
                        (intent_id, OrderState.INTENT_CREATED.value, "INTENT_CREATED", payload_json, now),
                    )
        return self.get_intent(intent_id)

    def get_intent_by_client_order_id(
        self, *, exchange: str, environment: str, account_id: str, client_order_id: str,
    ) -> Optional[dict]:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """SELECT * FROM order_intents WHERE exchange=%s AND environment=%s
                    AND account_id=%s AND client_order_id=%s""",
                    (str(exchange).upper(), str(environment).lower(), str(account_id), str(client_order_id)),
                )
                row = cur.fetchone()
                return self._normalise_intent(row) if row is not None else None

    def get_intent_by_exchange_order_id(self, exchange_order_id: str) -> Optional[dict]:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT * FROM order_intents WHERE exchange_order_id=%s ORDER BY updated_at DESC LIMIT 1",
                    (str(exchange_order_id),),
                )
                row = cur.fetchone()
                return self._normalise_intent(row) if row is not None else None

    def claim_submission(
        self, intent_id: str, *, lease_key: Optional[str] = None,
        lease_owner_id: Optional[str] = None, fencing_token: Optional[int] = None,
    ) -> bool:
        now = time.time()
        with self._connect() as conn:
            with conn.cursor() as cur:
                self._assert_fenced_lease(
                    cur, lease_key=lease_key, lease_owner_id=lease_owner_id,
                    fencing_token=fencing_token,
                )
                cur.execute(
                    "UPDATE order_intents SET state=%s,updated_at=%s WHERE intent_id=%s AND state=%s RETURNING intent_id",
                    (OrderState.SUBMITTING.value, now, str(intent_id), OrderState.INTENT_CREATED.value),
                )
                row = cur.fetchone()
                if row is None:
                    return False
                cur.execute(
                    "INSERT INTO order_events(intent_id,state,event_type,payload_json,created_at) VALUES(%s,%s,%s,%s,%s)",
                    (str(intent_id), OrderState.SUBMITTING.value, "SUBMISSION_CLAIMED", "{}", now),
                )
                return True

    def link_protection_order(
        self, parent_intent_id: str, protective_intent_id: str,
        *, payload: Optional[Mapping[str, Any]] = None,
        lease_key: Optional[str] = None, lease_owner_id: Optional[str] = None,
        fencing_token: Optional[int] = None,
    ) -> None:
        now = time.time()
        with self._connect() as conn:
            with conn.cursor() as cur:
                self._assert_fenced_lease(
                    cur, lease_key=lease_key, lease_owner_id=lease_owner_id,
                    fencing_token=fencing_token,
                )
                cur.execute("SELECT instrument_id,reduce_only FROM order_intents WHERE intent_id=%s", (str(parent_intent_id),))
                parent = cur.fetchone()
                cur.execute("SELECT instrument_id,reduce_only FROM order_intents WHERE intent_id=%s", (str(protective_intent_id),))
                protective = cur.fetchone()
                if parent is None or protective is None:
                    raise KeyError("parent or protective intent missing")
                if not protective["reduce_only"] or parent["instrument_id"] != protective["instrument_id"]:
                    raise ReliabilityError("Protective link must reference a same-instrument reduce-only order")
                cur.execute(
                    """INSERT INTO execution_protection_links
                    (parent_intent_id,protective_intent_id,active,payload_json,created_at,updated_at)
                    VALUES(%s,%s,TRUE,%s,%s,%s)
                    ON CONFLICT(parent_intent_id,protective_intent_id) DO UPDATE SET
                    active=TRUE,payload_json=EXCLUDED.payload_json,updated_at=EXCLUDED.updated_at""",
                    (str(parent_intent_id), str(protective_intent_id), self._json(payload), now, now),
                )

    def active_protection_parent_ids(
        self, protective_intent_id: Optional[str] = None,
        *, lease_key: Optional[str] = None, lease_owner_id: Optional[str] = None,
        fencing_token: Optional[int] = None,
    ) -> list[str]:
        with self._connect() as conn:
            with conn.cursor() as cur:
                self._assert_fenced_lease(
                    cur, lease_key=lease_key, lease_owner_id=lease_owner_id,
                    fencing_token=fencing_token,
                )
                if protective_intent_id is None:
                    cur.execute("SELECT DISTINCT parent_intent_id FROM execution_protection_links WHERE active=TRUE")
                else:
                    cur.execute("SELECT DISTINCT parent_intent_id FROM execution_protection_links WHERE active=TRUE AND protective_intent_id=%s", (str(protective_intent_id),))
                return [str(row["parent_intent_id"]) for row in cur.fetchall()]

    def deactivate_protection_links(
        self, protective_intent_id: Optional[str] = None,
        *, lease_key: Optional[str] = None, lease_owner_id: Optional[str] = None,
        fencing_token: Optional[int] = None,
    ) -> int:
        with self._connect() as conn:
            with conn.cursor() as cur:
                self._assert_fenced_lease(
                    cur, lease_key=lease_key, lease_owner_id=lease_owner_id,
                    fencing_token=fencing_token,
                )
                if protective_intent_id is None:
                    cur.execute("UPDATE execution_protection_links SET active=FALSE,updated_at=EXTRACT(EPOCH FROM clock_timestamp()) WHERE active=TRUE")
                else:
                    cur.execute("UPDATE execution_protection_links SET active=FALSE,updated_at=EXTRACT(EPOCH FROM clock_timestamp()) WHERE active=TRUE AND protective_intent_id=%s", (str(protective_intent_id),))
                return int(cur.rowcount)

    def begin_mutation(
        self, *, mutation_id: str, exchange: str, environment: str, account_id: str,
        operation: str, resource_id: str, request_hash: str, payload: Optional[Mapping[str, Any]] = None,
        lease_key: Optional[str] = None, lease_owner_id: Optional[str] = None,
        fencing_token: Optional[int] = None,
    ) -> dict:
        now = time.time()
        with self._connect() as conn:
            with conn.cursor() as cur:
                self._assert_fenced_lease(
                    cur, lease_key=lease_key, lease_owner_id=lease_owner_id,
                    fencing_token=fencing_token,
                )
                # Serialize the check+insert across all PostgreSQL writers. The
                # execution lease further fences the authorized owner where enabled.
                cur.execute("LOCK TABLE execution_mutations IN SHARE ROW EXCLUSIVE MODE")
                cur.execute(
                    """SELECT mutation_id,state,operation,resource_id FROM execution_mutations
                    WHERE exchange=%s AND environment=%s AND account_id=%s
                      AND state IN ('SUBMITTING','UNKNOWN')
                    ORDER BY created_at LIMIT 1 FOR UPDATE""",
                    (str(exchange).upper(), str(environment).lower(), str(account_id)),
                )
                unresolved = cur.fetchone()
                if unresolved is not None:
                    raise UnresolvedExecutionState(
                        "Prior exchange control mutation remains unresolved: "
                        + str(unresolved["mutation_id"]) + " (" + str(unresolved["state"]) + ")"
                    )
                cur.execute(
                    """INSERT INTO execution_mutations
                    (mutation_id,exchange,environment,account_id,operation,resource_id,request_hash,
                     payload_json,state,created_at,updated_at)
                    VALUES(%s,%s,%s,%s,%s,%s,%s,%s,'SUBMITTING',%s,%s)""",
                    (str(mutation_id), str(exchange).upper(), str(environment).lower(), str(account_id),
                     str(operation), str(resource_id), str(request_hash), self._json(payload), now, now),
                )
                cur.execute(
                    "INSERT INTO execution_mutation_events(mutation_id,event_type,state,payload_json,created_at) VALUES(%s,%s,%s,%s,%s)",
                    (str(mutation_id), "SUBMISSION_STARTED", "SUBMITTING", self._json(payload), now),
                )
        return self.get_mutation(mutation_id)

    def finish_mutation(
        self, mutation_id: str, *, state: str, response_id: Optional[str] = None,
        payload: Optional[Mapping[str, Any]] = None, lease_key: Optional[str] = None,
        lease_owner_id: Optional[str] = None, fencing_token: Optional[int] = None,
    ) -> dict:
        state = str(state).upper()
        if state not in {"ACKNOWLEDGED", "REJECTED", "UNKNOWN"}:
            raise ValueError("Invalid execution mutation terminal state")
        now = time.time()
        with self._connect() as conn:
            with conn.cursor() as cur:
                self._assert_fenced_lease(
                    cur, lease_key=lease_key, lease_owner_id=lease_owner_id,
                    fencing_token=fencing_token,
                )
                cur.execute(
                    """UPDATE execution_mutations SET state=%s,response_id=%s,payload_json=%s,updated_at=%s
                    WHERE mutation_id=%s AND state='SUBMITTING' RETURNING mutation_id""",
                    (state, response_id, self._json(payload), now, str(mutation_id)),
                )
                if cur.fetchone() is None:
                    raise ReliabilityError("Execution mutation missing or no longer submitting")
                cur.execute(
                    "INSERT INTO execution_mutation_events(mutation_id,event_type,state,payload_json,created_at) VALUES(%s,%s,%s,%s,%s)",
                    (str(mutation_id), "EXCHANGE_RESPONSE_RECORDED", state, self._json(payload), now),
                )
        return self.get_mutation(mutation_id)

    def get_mutation(self, mutation_id: str) -> dict:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM execution_mutations WHERE mutation_id=%s", (str(mutation_id),))
                row = cur.fetchone()
                if row is None:
                    raise KeyError(mutation_id)
                result = dict(row)
                result["payload"] = json.loads(result.pop("payload_json"))
                return result

    def resolve_mutation(
        self, mutation_id: str, *, state: str, evidence: Mapping[str, Any], resolved_by: str,
        lease_key: Optional[str] = None, lease_owner_id: Optional[str] = None,
        fencing_token: Optional[int] = None,
    ) -> dict:
        state = str(state).upper()
        if state not in {"ACKNOWLEDGED", "REJECTED"}:
            raise ValueError("A reconciled mutation must resolve to ACKNOWLEDGED or REJECTED")
        if not isinstance(evidence, Mapping) or evidence.get("verified") is not True:
            raise ValueError("Read-only reconciliation evidence must explicitly set verified=True")
        required = ("verification_method", "observed_status", "observed_at", "resource_id")
        if any(not str(evidence.get(key) or "").strip() for key in required):
            raise ValueError("Reconciliation evidence requires method, status, timestamp and resource identity")
        operator = str(resolved_by or "").strip()
        if not operator:
            raise ValueError("resolved_by operator identity is required")
        now = time.time()
        with self._connect() as conn:
            with conn.cursor() as cur:
                self._assert_fenced_lease(
                    cur, lease_key=lease_key, lease_owner_id=lease_owner_id,
                    fencing_token=fencing_token,
                )
                cur.execute("SELECT * FROM execution_mutations WHERE mutation_id=%s FOR UPDATE", (str(mutation_id),))
                row = cur.fetchone()
                if row is None:
                    raise KeyError(mutation_id)
                if row["state"] != "UNKNOWN":
                    raise ReliabilityError("Only UNKNOWN control mutations require explicit reconciliation")
                if str(evidence.get("resource_id")) != str(row["resource_id"]):
                    raise ReliabilityError("Reconciliation evidence resource does not match the mutation")
                payload = json.loads(row["payload_json"])
                payload["reconciliation"] = dict(evidence)
                payload["reconciled_by"] = operator
                cur.execute(
                    "UPDATE execution_mutations SET state=%s,payload_json=%s,updated_at=%s WHERE mutation_id=%s AND state='UNKNOWN'",
                    (state, self._json(payload), now, str(mutation_id)),
                )
                if cur.rowcount != 1:
                    raise ReliabilityError("Control mutation reconciliation lost its state race")
                cur.execute(
                    "INSERT INTO execution_mutation_events(mutation_id,event_type,state,payload_json,created_at) VALUES(%s,%s,%s,%s,%s)",
                    (str(mutation_id), "READ_ONLY_RECONCILIATION_CONFIRMED", state,
                     self._json({"reconciliation": dict(evidence), "resolved_by": operator}), now),
                )
        return self.get_mutation(mutation_id)

    def mutation_events(self, mutation_id: str) -> list[dict]:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM execution_mutation_events WHERE mutation_id=%s ORDER BY event_id", (str(mutation_id),))
                rows = cur.fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["payload"] = json.loads(item.pop("payload_json"))
            result.append(item)
        return result

    def unresolved_mutations(
        self, *, exchange: Optional[str] = None, environment: Optional[str] = None,
        account_id: Optional[str] = None,
    ) -> list[dict]:
        clauses = ["state IN ('SUBMITTING','UNKNOWN')"]
        params = []
        for column, value, normalizer in (
            ("exchange", exchange, str.upper),
            ("environment", environment, str.lower),
            ("account_id", account_id, str),
        ):
            if value is not None:
                clauses.append(f"{column}=%s")
                params.append(normalizer(str(value)))
        sql = "SELECT * FROM execution_mutations WHERE " + " AND ".join(clauses) + " ORDER BY created_at"
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, tuple(params))
                rows = cur.fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["payload"] = json.loads(item.pop("payload_json"))
            result.append(item)
        return result

    def get_intent(self, intent_id: str) -> dict:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM order_intents WHERE intent_id=%s", (str(intent_id),))
                row = cur.fetchone()
                if row is None:
                    raise KeyError(intent_id)
                return self._normalise_intent(row)

    def transition(
        self,
        intent_id: str,
        new_state: OrderState,
        *,
        event_type: Optional[str] = None,
        exchange_order_id: Optional[str] = None,
        payload: Optional[Mapping[str, Any]] = None,
        lease_key: Optional[str] = None,
        lease_owner_id: Optional[str] = None,
        fencing_token: Optional[int] = None,
    ) -> dict:
        if not isinstance(new_state, OrderState):
            new_state = OrderState(str(new_state))
        now = time.time()
        payload_json = self._json(payload)
        with self._connect() as conn:
            with conn.cursor() as cur:
                self._assert_fenced_lease(
                    cur, lease_key=lease_key, lease_owner_id=lease_owner_id,
                    fencing_token=fencing_token,
                )
                cur.execute("SELECT state FROM order_intents WHERE intent_id=%s FOR UPDATE", (str(intent_id),))
                row = cur.fetchone()
                if row is None:
                    raise KeyError(intent_id)
                current = OrderState(row["state"])
                if new_state != current:
                    if new_state not in _ALLOWED_TRANSITIONS[current]:
                        raise InvalidTransition(f"{current.value} -> {new_state.value} is not allowed")
                    cur.execute(
                        """UPDATE order_intents SET state=%s,
                        exchange_order_id=COALESCE(%s,exchange_order_id), updated_at=%s
                        WHERE intent_id=%s""",
                        (new_state.value, exchange_order_id, now, str(intent_id)),
                    )
                    cur.execute(
                        "INSERT INTO order_events(intent_id,state,event_type,payload_json,created_at) VALUES(%s,%s,%s,%s,%s)",
                        (str(intent_id), new_state.value, event_type or new_state.value, payload_json, now),
                    )
        return self.get_intent(intent_id)

    def record_fill(
        self,
        intent_id: str,
        *,
        cumulative_filled_quantity: Any,
        average_fill_price: Any,
        exchange_order_id: Optional[str] = None,
        fee: Optional[Any] = None,
        fee_currency: Optional[str] = None,
        payload: Optional[Mapping[str, Any]] = None,
        lease_key: Optional[str] = None,
        lease_owner_id: Optional[str] = None,
        fencing_token: Optional[int] = None,
    ) -> dict:
        filled = _decimal(cumulative_filled_quantity, "cumulative_filled_quantity")
        price = _decimal(average_fill_price, "average_fill_price")
        if filled < 0 or (filled > 0 and price <= 0):
            raise ValueError("Invalid fill quantity or price")
        now = time.time()
        with self._connect() as conn:
            with conn.cursor() as cur:
                self._assert_fenced_lease(
                    cur, lease_key=lease_key, lease_owner_id=lease_owner_id,
                    fencing_token=fencing_token,
                )
                cur.execute("SELECT * FROM order_intents WHERE intent_id=%s FOR UPDATE", (str(intent_id),))
                row = cur.fetchone()
                if row is None:
                    raise KeyError(intent_id)
                requested = Decimal(row["quantity"])
                previous = Decimal(row["filled_quantity"])
                state = OrderState(row["state"])
                if filled < previous or filled > requested:
                    raise ValueError("Cumulative filled quantity must be monotonic and <= requested quantity")
                allowed_current = {
                    OrderState.SUBMITTING, OrderState.ACKNOWLEDGED,
                    OrderState.PARTIALLY_FILLED, OrderState.UNKNOWN,
                    OrderState.RECONCILING,
                }
                if state not in allowed_current:
                    raise InvalidTransition(f"Cannot record a fill while order state is {state.value}")
                if filled <= 0:
                    raise ValueError("A fill event must have positive cumulative filled quantity")
                new_state = OrderState.FILLED if filled == requested else OrderState.PARTIALLY_FILLED
                if new_state not in _ALLOWED_TRANSITIONS[state] and new_state != state:
                    raise InvalidTransition(f"{state.value} -> {new_state.value} is not allowed")
                if new_state == state and filled == previous:
                    return self._normalise_intent(row)
                merged = dict(payload or {})
                if fee is not None:
                    if not str(fee_currency or "").strip():
                        raise ValueError("fee_currency is required when a fee is recorded")
                    merged["fee"] = str(_decimal(fee, "fee"))
                    merged["fee_currency"] = str(fee_currency).upper()
                cur.execute(
                    """UPDATE order_intents SET state=%s,
                    exchange_order_id=COALESCE(%s,exchange_order_id), filled_quantity=%s,
                    average_fill_price=%s, protection_state='REQUIRED', updated_at=%s
                    WHERE intent_id=%s""",
                    (new_state.value, exchange_order_id, str(filled), str(price), now, str(intent_id)),
                )
                cur.execute(
                    "INSERT INTO order_events(intent_id,state,event_type,payload_json,created_at) VALUES(%s,%s,%s,%s,%s)",
                    (str(intent_id), new_state.value, "FILL_UPDATE", self._json(merged), now),
                )
        return self.get_intent(intent_id)

    def mark_protected(
        self,
        intent_id: str,
        protected_quantity: Any,
        *,
        payload: Optional[Mapping[str, Any]] = None,
        lease_key: Optional[str] = None,
        lease_owner_id: Optional[str] = None,
        fencing_token: Optional[int] = None,
    ) -> dict:
        protected = _decimal(protected_quantity, "protected_quantity")
        now = time.time()
        with self._connect() as conn:
            with conn.cursor() as cur:
                self._assert_fenced_lease(
                    cur, lease_key=lease_key, lease_owner_id=lease_owner_id,
                    fencing_token=fencing_token,
                )
                cur.execute("SELECT filled_quantity,state FROM order_intents WHERE intent_id=%s FOR UPDATE", (str(intent_id),))
                row = cur.fetchone()
                if row is None:
                    raise KeyError(intent_id)
                filled = Decimal(row["filled_quantity"])
                if filled <= 0 or protected < filled:
                    raise ProtectionInvariantError(
                        f"Protective quantity {protected} does not cover filled quantity {filled}"
                    )
                if OrderState(row["state"]) not in {
                    OrderState.PARTIALLY_FILLED, OrderState.FILLED, OrderState.CANCELLED
                }:
                    raise InvalidTransition("Protection cannot be confirmed before an exchange fill")
                cur.execute(
                    "UPDATE order_intents SET protected_quantity=%s, protection_state='PROTECTED', updated_at=%s WHERE intent_id=%s",
                    (str(protected), now, str(intent_id)),
                )
                cur.execute(
                    "INSERT INTO order_events(intent_id,state,event_type,payload_json,created_at) VALUES(%s,%s,%s,%s,%s)",
                    (str(intent_id), row["state"], "PROTECTION_VERIFIED", self._json(payload), now),
                )
        return self.get_intent(intent_id)

    def mark_unprotected(
        self, intent_id: str, *, payload: Optional[Mapping[str, Any]] = None,
        lease_key: Optional[str] = None, lease_owner_id: Optional[str] = None,
        fencing_token: Optional[int] = None,
    ) -> dict:
        now = time.time()
        with self._connect() as conn:
            with conn.cursor() as cur:
                self._assert_fenced_lease(
                    cur, lease_key=lease_key, lease_owner_id=lease_owner_id,
                    fencing_token=fencing_token,
                )
                cur.execute("SELECT filled_quantity FROM order_intents WHERE intent_id=%s FOR UPDATE", (str(intent_id),))
                row = cur.fetchone()
                if row is None:
                    raise KeyError(intent_id)
                if Decimal(row["filled_quantity"]) <= 0:
                    raise ProtectionInvariantError("No confirmed filled quantity to protect")
                cur.execute(
                    "UPDATE order_intents SET protection_state='UNPROTECTED', updated_at=%s WHERE intent_id=%s",
                    (now, str(intent_id)),
                )
                cur.execute(
                    "INSERT INTO order_events(intent_id,state,event_type,payload_json,created_at) VALUES(%s,%s,%s,%s,%s)",
                    (str(intent_id), "UNPROTECTED", "PROTECTION_FAILURE", self._json(payload), now),
                )
        return self.get_intent(intent_id)

    def unresolved_intents(
        self, *, exchange: str, environment: str, account_id: str
    ) -> list[dict]:
        query = """SELECT * FROM order_intents
            WHERE exchange=%s AND environment=%s AND account_id=%s
              AND (state IN ('INTENT_CREATED','SUBMITTING','ACKNOWLEDGED','PARTIALLY_FILLED','UNKNOWN','RECONCILING')
                   OR (CAST(filled_quantity AS NUMERIC) > 0
                       AND (protection_state != 'PROTECTED'
                            OR CAST(protected_quantity AS NUMERIC) < CAST(filled_quantity AS NUMERIC))))
            ORDER BY created_at"""
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (str(exchange).upper(), str(environment).lower(), str(account_id)))
                return [self._normalise_intent(row) for row in cur.fetchall()]

    def has_unresolved_intents(
        self, *, exchange: str, environment: str, account_id: str
    ) -> bool:
        return bool(self.unresolved_intents(
            exchange=exchange, environment=environment, account_id=account_id
        ))

    def pending_reconciliation(
        self, *, exchange: Optional[str] = None, environment: Optional[str] = None
    ) -> list[dict]:
        where = ["state IN ('UNKNOWN','RECONCILING')"]
        params: list[Any] = []
        if exchange:
            where.append("exchange=%s")
            params.append(str(exchange).upper())
        if environment:
            where.append("environment=%s")
            params.append(str(environment).lower())
        query = "SELECT * FROM order_intents WHERE " + " AND ".join(where) + " ORDER BY created_at"
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(query, tuple(params))
                rows = cur.fetchall()
                return [self._normalise_intent(row) for row in rows]

    def acquire_fenced_run_lease(
        self, lease_key: str, owner_id: str, *, ttl_seconds: float = 120.0
    ) -> Optional[int]:
        """Acquire a lease and return its monotonically increasing generation.

        Re-acquisition by the current owner preserves its generation. A new
        owner can acquire only after expiry/release and receives generation+1.
        Owner IDs must be unique per process/run attempt, not static host names.
        """
        key = str(lease_key or "").strip()
        owner = str(owner_id or "").strip()
        ttl = float(ttl_seconds)
        if not key or not owner or not math.isfinite(ttl) or ttl <= 0:
            raise ValueError("lease_key, owner_id and finite positive ttl_seconds are required")
        query = """WITH db_clock AS (
            SELECT EXTRACT(EPOCH FROM clock_timestamp())::double precision AS now_epoch
        )
        INSERT INTO run_leases(lease_key,owner_id,expires_at,updated_at,fencing_token)
        SELECT %s,%s,db_clock.now_epoch + %s,db_clock.now_epoch,1 FROM db_clock WHERE TRUE
        ON CONFLICT (lease_key) DO UPDATE SET
            owner_id=EXCLUDED.owner_id,
            expires_at=EXCLUDED.expires_at,
            updated_at=EXCLUDED.updated_at,
            fencing_token=CASE
                WHEN run_leases.fencing_token < 1 THEN 1
                WHEN run_leases.expires_at <= EXCLUDED.updated_at THEN run_leases.fencing_token + 1
                ELSE run_leases.fencing_token
            END
        WHERE run_leases.expires_at <= EXCLUDED.updated_at
           OR run_leases.owner_id = EXCLUDED.owner_id
        RETURNING fencing_token"""
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (key, owner, ttl))
                row = cur.fetchone()
                return int(row["fencing_token"]) if row is not None else None

    def renew_fenced_run_lease(
        self, lease_key: str, owner_id: str, fencing_token: int, *, ttl_seconds: float = 120.0
    ) -> bool:
        ttl = float(ttl_seconds)
        try:
            token = int(fencing_token)
        except (TypeError, ValueError, OverflowError):
            token = 0
        if not math.isfinite(ttl) or ttl <= 0 or token < 1:
            raise ValueError("ttl_seconds must be finite and fencing_token must be positive")
        query = """WITH db_clock AS (
            SELECT EXTRACT(EPOCH FROM clock_timestamp())::double precision AS now_epoch
        )
        UPDATE run_leases
        SET expires_at=db_clock.now_epoch + %s, updated_at=db_clock.now_epoch
        FROM db_clock
        WHERE lease_key=%s AND owner_id=%s AND fencing_token=%s
          AND run_leases.expires_at>db_clock.now_epoch
        RETURNING run_leases.fencing_token"""
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (ttl, str(lease_key), str(owner_id), token))
                return cur.fetchone() is not None

    def release_fenced_run_lease(
        self, lease_key: str, owner_id: str, fencing_token: int
    ) -> bool:
        try:
            token = int(fencing_token)
        except (TypeError, ValueError, OverflowError):
            token = 0
        if token < 1:
            return False
        # Keep the row so the next owner cannot reset the generation to 1.
        query = """WITH db_clock AS (
            SELECT EXTRACT(EPOCH FROM clock_timestamp())::double precision AS now_epoch
        )
        UPDATE run_leases
        SET expires_at=db_clock.now_epoch, updated_at=db_clock.now_epoch
        FROM db_clock
        WHERE lease_key=%s AND owner_id=%s AND fencing_token=%s
        RETURNING run_leases.lease_key"""
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (str(lease_key), str(owner_id), token))
                return cur.fetchone() is not None

    def acquire_run_lease(
        self, lease_key: str, owner_id: str, *, ttl_seconds: float = 120.0
    ) -> bool:
        """Compatibility API for non-fenced development callers only."""
        if self.require_fenced_writes:
            raise ReliabilityError("Fenced mode requires acquire_fenced_run_lease() and its token")
        return self.acquire_fenced_run_lease(
            lease_key, owner_id, ttl_seconds=ttl_seconds
        ) is not None

    def renew_run_lease(
        self, lease_key: str, owner_id: str, *, ttl_seconds: float = 120.0,
        fencing_token: Optional[int] = None,
    ) -> bool:
        if self.require_fenced_writes:
            if fencing_token is None:
                return False
            return self.renew_fenced_run_lease(
                lease_key, owner_id, fencing_token, ttl_seconds=ttl_seconds
            )
        # Legacy owner-only renewal remains available only when fencing is not required.
        ttl = float(ttl_seconds)
        if not math.isfinite(ttl) or ttl <= 0:
            raise ValueError("ttl_seconds must be finite and positive")
        query = """WITH db_clock AS (
            SELECT EXTRACT(EPOCH FROM clock_timestamp())::double precision AS now_epoch
        )
        UPDATE run_leases
        SET expires_at=db_clock.now_epoch + %s, updated_at=db_clock.now_epoch
        FROM db_clock
        WHERE lease_key=%s AND owner_id=%s AND run_leases.expires_at>db_clock.now_epoch
        RETURNING run_leases.lease_key"""
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (ttl, str(lease_key), str(owner_id)))
                return cur.fetchone() is not None

    def release_run_lease(
        self, lease_key: str, owner_id: str, *, fencing_token: Optional[int] = None
    ) -> bool:
        if self.require_fenced_writes:
            if fencing_token is None:
                return False
            return self.release_fenced_run_lease(lease_key, owner_id, fencing_token)
        # Retain the lease row and generation; never DELETE it.
        query = """WITH db_clock AS (
            SELECT EXTRACT(EPOCH FROM clock_timestamp())::double precision AS now_epoch
        )
        UPDATE run_leases
        SET expires_at=db_clock.now_epoch, updated_at=db_clock.now_epoch
        FROM db_clock
        WHERE lease_key=%s AND owner_id=%s
        RETURNING run_leases.lease_key"""
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (str(lease_key), str(owner_id)))
                return cur.fetchone() is not None

    def events(self, intent_id: str) -> list[dict]:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT * FROM order_events WHERE intent_id=%s ORDER BY event_id",
                    (str(intent_id),),
                )
                return [dict(row) for row in cur.fetchall()]

    def _assert_fenced_risk_capture_lease(
        self, cur, *, lease_key: str, lease_owner_id: str, fencing_token: int,
    ) -> None:
        """Validate the dedicated read-only baseline writer lease, not run_leases."""
        key = str(lease_key or "").strip()
        owner = str(lease_owner_id or "").strip()
        try:
            token = int(fencing_token)
        except (TypeError, ValueError, OverflowError):
            token = 0
        if not key or not owner or token < 1:
            raise ReliabilityError("A risk-capture lease key, owner and positive fence are required")
        cur.execute(
            """SELECT lease_key FROM risk_capture_leases
            WHERE lease_key=%s AND owner_id=%s AND fencing_token=%s
              AND expires_at > EXTRACT(EPOCH FROM clock_timestamp())
            FOR UPDATE""",
            (key, owner, token),
        )
        if cur.fetchone() is None:
            raise ReliabilityError("Risk-capture lease is expired, replaced or fenced")

    def acquire_fenced_risk_capture_lease(
        self, lease_key: str, owner_id: str, *, ttl_seconds: float = 180.0,
    ) -> Optional[int]:
        """Acquire an independently fenced lease isolated from trading run_leases."""
        key = str(lease_key or "").strip()
        owner = str(owner_id or "").strip()
        ttl = float(ttl_seconds)
        if not key or not owner or not math.isfinite(ttl) or ttl <= 0:
            raise ValueError("risk-capture lease identity and positive finite TTL are required")
        query = """WITH db_clock AS (
            SELECT EXTRACT(EPOCH FROM clock_timestamp())::double precision AS now_epoch
        )
        INSERT INTO risk_capture_leases(lease_key,owner_id,expires_at,updated_at,fencing_token)
        SELECT %s,%s,db_clock.now_epoch + %s,db_clock.now_epoch,1 FROM db_clock WHERE TRUE
        ON CONFLICT (lease_key) DO UPDATE SET
            owner_id=EXCLUDED.owner_id,
            expires_at=EXCLUDED.expires_at,
            updated_at=EXCLUDED.updated_at,
            fencing_token=CASE
                WHEN risk_capture_leases.fencing_token < 1 THEN 1
                WHEN risk_capture_leases.expires_at <= EXCLUDED.updated_at THEN risk_capture_leases.fencing_token + 1
                ELSE risk_capture_leases.fencing_token
            END
        WHERE risk_capture_leases.expires_at <= EXCLUDED.updated_at
           OR risk_capture_leases.owner_id = EXCLUDED.owner_id
        RETURNING fencing_token"""
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (key, owner, ttl))
                row = cur.fetchone()
                return int(row["fencing_token"]) if row is not None else None

    def renew_fenced_risk_capture_lease(
        self, lease_key: str, owner_id: str, fencing_token: int, *, ttl_seconds: float = 180.0,
    ) -> bool:
        ttl = float(ttl_seconds)
        try:
            token = int(fencing_token)
        except (TypeError, ValueError, OverflowError):
            token = 0
        if not math.isfinite(ttl) or ttl <= 0 or token < 1:
            raise ValueError("risk-capture TTL and fencing token must be positive and finite")
        query = """WITH db_clock AS (
            SELECT EXTRACT(EPOCH FROM clock_timestamp())::double precision AS now_epoch
        )
        UPDATE risk_capture_leases
        SET expires_at=db_clock.now_epoch + %s, updated_at=db_clock.now_epoch
        FROM db_clock
        WHERE lease_key=%s AND owner_id=%s AND fencing_token=%s
          AND risk_capture_leases.expires_at>db_clock.now_epoch
        RETURNING risk_capture_leases.fencing_token"""
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (ttl, str(lease_key), str(owner_id), token))
                return cur.fetchone() is not None

    def release_fenced_risk_capture_lease(
        self, lease_key: str, owner_id: str, fencing_token: int,
    ) -> bool:
        try:
            token = int(fencing_token)
        except (TypeError, ValueError, OverflowError):
            token = 0
        if token < 1:
            return False
        query = """WITH db_clock AS (
            SELECT EXTRACT(EPOCH FROM clock_timestamp())::double precision AS now_epoch
        )
        UPDATE risk_capture_leases
        SET expires_at=db_clock.now_epoch, updated_at=db_clock.now_epoch
        FROM db_clock
        WHERE lease_key=%s AND owner_id=%s AND fencing_token=%s
        RETURNING risk_capture_leases.lease_key"""
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (str(lease_key), str(owner_id), token))
                return cur.fetchone() is not None

    def record_account_risk_snapshot_with_capture_lease(
        self, *, exchange: str, environment: str, account_id: str,
        currency: str, equity: Any, observed_at: Optional[float] = None,
        source: str = "readonly_daily_baseline_capture",
        risk_capture_lease_key: str, risk_capture_lease_owner_id: str,
        risk_capture_fencing_token: int,
    ) -> dict:
        """Append a baseline using only the dedicated risk-capture lease table."""
        scope = (
            str(exchange or "").strip().upper(),
            str(environment or "").strip().lower(),
            str(account_id or "").strip(),
            str(currency or "").strip().upper(),
        )
        if not all(scope):
            raise ValueError("Risk snapshot exchange, environment, account and currency are required")
        amount = _decimal(equity, "equity")
        stamp = float(time.time() if observed_at is None else observed_at)
        if amount <= 0 or not math.isfinite(stamp) or stamp <= 0:
            raise ValueError("Risk snapshot equity and timestamp must be positive and finite")
        try:
            with self._connect() as conn:
                with conn.cursor() as cur:
                    self._assert_fenced_risk_capture_lease(
                        cur, lease_key=risk_capture_lease_key,
                        lease_owner_id=risk_capture_lease_owner_id,
                        fencing_token=risk_capture_fencing_token,
                    )
                    cur.execute(
                        """INSERT INTO account_risk_snapshots
                        (exchange,environment,account_id,currency,observed_at,equity,source)
                        VALUES(%s,%s,%s,%s,%s,%s,%s)
                        ON CONFLICT(exchange,environment,account_id,currency,observed_at) DO NOTHING""",
                        (*scope, stamp, str(amount), str(source or "readonly_daily_baseline_capture")),
                    )
                    cur.execute(
                        """SELECT exchange,environment,account_id,currency,observed_at,equity,source
                        FROM account_risk_snapshots
                        WHERE exchange=%s AND environment=%s AND account_id=%s AND currency=%s AND observed_at=%s""",
                        (*scope, stamp),
                    )
                    row = cur.fetchone()
            if row is None:
                raise ReliabilityError("Risk-capture account snapshot could not be persisted")
            normalized = dict(row)
            if _decimal(normalized.get("equity"), "persisted_equity") != amount:
                raise ReliabilityError("Risk snapshot timestamp collided with a different equity observation")
            return normalized
        except ReliabilityError:
            raise
        except Exception as exc:
            raise ReliabilityError(
                f"Could not persist read-only baseline snapshot ({type(exc).__name__})"
            ) from exc

    def record_account_risk_snapshot(
        self, *, exchange: str, environment: str, account_id: str,
        currency: str, equity: Any, observed_at: Optional[float] = None,
        source: str = "exchange_account_snapshot",
        lease_key: Optional[str] = None, lease_owner_id: Optional[str] = None,
        fencing_token: Optional[int] = None,
    ) -> dict:
        """Append an immutable account-equity checkpoint under the current writer fence."""
        scope = (
            str(exchange or "").strip().upper(),
            str(environment or "").strip().lower(),
            str(account_id or "").strip(),
            str(currency or "").strip().upper(),
        )
        if not all(scope):
            raise ValueError("Risk snapshot exchange, environment, account and currency are required")
        amount = _decimal(equity, "equity")
        stamp = float(time.time() if observed_at is None else observed_at)
        if amount <= 0 or not math.isfinite(stamp) or stamp <= 0:
            raise ValueError("Risk snapshot equity and timestamp must be positive and finite")
        try:
            with self._connect() as conn:
                with conn.cursor() as cur:
                    self._assert_fenced_lease(
                        cur, lease_key=lease_key, lease_owner_id=lease_owner_id,
                        fencing_token=fencing_token,
                    )
                    cur.execute(
                        """INSERT INTO account_risk_snapshots
                        (exchange,environment,account_id,currency,observed_at,equity,source)
                        VALUES(%s,%s,%s,%s,%s,%s,%s)
                        ON CONFLICT(exchange,environment,account_id,currency,observed_at) DO NOTHING""",
                        (*scope, stamp, str(amount), str(source or "exchange_account_snapshot")),
                    )
                    cur.execute(
                        """SELECT exchange,environment,account_id,currency,observed_at,equity,source
                        FROM account_risk_snapshots
                        WHERE exchange=%s AND environment=%s AND account_id=%s AND currency=%s AND observed_at=%s""",
                        (*scope, stamp),
                    )
                    row = cur.fetchone()
            if row is None:
                raise ReliabilityError("Account risk snapshot could not be persisted")
            return dict(row)
        except ReliabilityError:
            raise
        except Exception as exc:
            raise ReliabilityError(
                f"Could not persist account risk snapshot ({type(exc).__name__})"
            ) from exc

    def latest_account_risk_snapshot_before(
        self, *, exchange: str, environment: str, account_id: str,
        currency: str, cutoff: float,
    ) -> Optional[dict]:
        stamp = float(cutoff)
        if not math.isfinite(stamp) or stamp <= 0:
            raise ValueError("Risk snapshot cutoff must be positive and finite")
        try:
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """SELECT exchange,environment,account_id,currency,observed_at,equity,source
                        FROM account_risk_snapshots
                        WHERE exchange=%s AND environment=%s AND account_id=%s AND currency=%s AND observed_at<=%s
                        ORDER BY observed_at DESC LIMIT 1""",
                        (str(exchange).upper(), str(environment).lower(), str(account_id), str(currency).upper(), stamp),
                    )
                    row = cur.fetchone()
            return dict(row) if row is not None else None
        except Exception as exc:
            raise ReliabilityError(
                f"Could not read account risk baseline ({type(exc).__name__})"
            ) from exc

    def first_account_risk_snapshot_between(
        self, *, exchange: str, environment: str, account_id: str,
        currency: str, start: float, end: float,
    ) -> Optional[dict]:
        """Return the earliest immutable account snapshot in a bounded UTC window."""
        start_value, end_value = float(start), float(end)
        if (not math.isfinite(start_value) or not math.isfinite(end_value)
                or start_value <= 0 or end_value < start_value):
            raise ValueError("Risk snapshot window is invalid")
        try:
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """SELECT exchange,environment,account_id,currency,observed_at,equity,source
                        FROM account_risk_snapshots
                        WHERE exchange=%s AND environment=%s AND account_id=%s AND currency=%s
                          AND observed_at>=%s AND observed_at<=%s
                        ORDER BY observed_at ASC LIMIT 1""",
                        (str(exchange).upper(), str(environment).lower(), str(account_id),
                         str(currency).upper(), start_value, end_value),
                    )
                    row = cur.fetchone()
            return dict(row) if row is not None else None
        except Exception as exc:
            raise ReliabilityError(
                f"Could not read daily account-risk baseline ({type(exc).__name__})"
            ) from exc

    def get_daily_risk_halt(
        self, *, exchange: str, environment: str, account_id: str, utc_day: str,
    ) -> Optional[dict]:
        from datetime import date
        date.fromisoformat(str(utc_day))
        try:
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """SELECT * FROM daily_risk_halts
                        WHERE exchange=%s AND environment=%s AND account_id=%s AND utc_day=%s""",
                        (str(exchange).upper(), str(environment).lower(), str(account_id), str(utc_day)),
                    )
                    row = cur.fetchone()
            return dict(row) if row is not None else None
        except Exception as exc:
            raise ReliabilityError(
                f"Could not read daily risk halt ({type(exc).__name__})"
            ) from exc

    def latch_daily_risk_halt(
        self, *, exchange: str, environment: str, account_id: str, utc_day: str,
        reason: str, baseline_equity: Any, current_equity: Any,
        loss_fraction: Any, observed_at: Optional[float] = None,
        lease_key: Optional[str] = None, lease_owner_id: Optional[str] = None,
        fencing_token: Optional[int] = None,
    ) -> dict:
        """Persist a sticky per-UTC-day entry halt under the current writer fence."""
        from datetime import date
        date.fromisoformat(str(utc_day))
        amount0 = _decimal(baseline_equity, "baseline_equity")
        amount1 = _decimal(current_equity, "current_equity")
        fraction = _decimal(loss_fraction, "loss_fraction")
        stamp = float(time.time() if observed_at is None else observed_at)
        if amount0 <= 0 or amount1 < 0 or fraction < 0 or not math.isfinite(stamp) or stamp <= 0:
            raise ValueError("Daily risk halt values are invalid")
        scope = (str(exchange).upper(), str(environment).lower(), str(account_id), str(utc_day))
        try:
            with self._connect() as conn:
                with conn.cursor() as cur:
                    self._assert_fenced_lease(
                        cur, lease_key=lease_key, lease_owner_id=lease_owner_id,
                        fencing_token=fencing_token,
                    )
                    cur.execute(
                        """INSERT INTO daily_risk_halts
                        (exchange,environment,account_id,utc_day,reason,baseline_equity,current_equity,loss_fraction,observed_at)
                        VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s)
                        ON CONFLICT(exchange,environment,account_id,utc_day) DO NOTHING""",
                        (*scope, str(reason or "DAILY_RISK_LIMIT"), str(amount0), str(amount1), str(fraction), stamp),
                    )
                    cur.execute(
                        """SELECT * FROM daily_risk_halts
                        WHERE exchange=%s AND environment=%s AND account_id=%s AND utc_day=%s""",
                        scope,
                    )
                    row = cur.fetchone()
            if row is None:
                raise ReliabilityError("Daily risk halt could not be persisted")
            return dict(row)
        except ReliabilityError:
            raise
        except Exception as exc:
            raise ReliabilityError(
                f"Could not persist daily risk halt ({type(exc).__name__})"
            ) from exc
