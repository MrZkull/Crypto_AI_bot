# Neon PostgreSQL — Execution Ledger Operations (Testnet only)

## Scope and safety boundary

Neon stores only the shared execution ledger: order intents, lifecycle events,
control-mutation audit events, protection links and fenced single-writer leases.
It does not store model files, predictions, F08 evidence, Stage 2E state or the
scanner's legacy JSON state. The exchange remains authoritative for actual orders,
positions and fills.

The database is not yet connected to the live scanner or Render dashboard by this
batch. `ORDER_LIFECYCLE_COORDINATOR_INTEGRATED` must remain `False`; the Deribit
mutation gate must remain disabled. A database connection is not approval to trade.

## Roles

Use three distinct identities:

1. **Migration owner** — the Neon project owner/admin DSN, used only for the explicit
   migration command. Store it only as a protected GitHub Environment secret during
   migration. Do not give this DSN to the scanner or Render.
2. **Runtime writer** — a dedicated login used by the future single GitHub Actions
   execution owner. It needs `SELECT`, `INSERT`, `UPDATE` on execution-ledger tables
   and `USAGE`/`SELECT` on the event sequences. It must not own the schema and must
   not have `CREATE`, `ALTER`, `DROP` or `TRUNCATE` privileges.
3. **Dashboard reader** — a separate login for a later dashboard phase. This batch
   does not grant base-table access to it. First create narrow sanitized SQL views
   for the exact dashboard fields; then grant `SELECT` on those views only.

Run the following statements as the same project owner that creates the ledger
schema, after migrations. The roles are created without passwords; set those
separately using Neon role management or secure SQL-console input. Do not paste
passwords into chat, scripts, SQL files or repository files.

```sql
-- Run only in a dedicated database used for this bot's execution ledger.
-- This removes the default schema-creation grant from all PUBLIC members.
REVOKE CREATE ON SCHEMA public FROM PUBLIC;
CREATE ROLE cryptobot_ledger_writer LOGIN;
CREATE ROLE cryptobot_ledger_reader LOGIN;
GRANT CONNECT ON DATABASE neondb TO cryptobot_ledger_writer;
GRANT CONNECT ON DATABASE neondb TO cryptobot_ledger_reader;
GRANT USAGE ON SCHEMA public TO cryptobot_ledger_writer;
GRANT SELECT, INSERT, UPDATE ON TABLE
  order_intents, run_leases, execution_mutations, execution_protection_links
  TO cryptobot_ledger_writer;
GRANT SELECT, INSERT ON TABLE
  order_events, execution_mutation_events TO cryptobot_ledger_writer;
GRANT SELECT ON TABLE execution_ledger_schema TO cryptobot_ledger_writer;
GRANT USAGE, SELECT ON SEQUENCE
  order_events_event_id_seq, execution_mutation_events_event_id_seq
  TO cryptobot_ledger_writer;
```

The dashboard reader intentionally receives no table grants in this batch. The
runtime writer receives no DELETE or schema privileges. When future migrations add
tables or sequences, grant only the required DML/sequence privileges explicitly
before using them. Run this in a dedicated Neon project/database used for the bot's
execution ledger.

## Migration and runtime configuration

Install the optional dependency:

```powershell
python -m pip install -r requirements-postgres.txt
```

The migration command requires `EXECUTION_LEDGER_MIGRATION_DSN` and uses
`EXECUTION_LEDGER_SSLMODE=require` by default:

```powershell
$env:EXECUTION_LEDGER_MIGRATION_DSN = '<privileged-migration-dsn>'
$env:EXECUTION_LEDGER_SSLMODE = 'require'
python scripts/migrate_execution_ledger.py
Remove-Item Env:\EXECUTION_LEDGER_MIGRATION_DSN
```

Run this only against a new/dedicated Testnet ledger database. Do not run routine
test suites against the live Neon ledger; CI uses a disposable PostgreSQL 16 service.
After migrations and permission review, a future runtime can use
`EXECUTION_LEDGER_BACKEND=postgresql`, `EXECUTION_LEDGER_SHARED_REQUIRED=true`,
`EXECUTION_LEDGER_REQUIRE_FENCED_WRITES=true`, `EXECUTION_LEDGER_SSLMODE=require`,
and `EXECUTION_LEDGER_DSN` set to the runtime-writer connection string. Runtime
construction now verifies schema version only; it performs no DDL and cannot silently
downgrade to SQLite when shared storage is required.

Keep DSNs only in protected GitHub Environment secrets. Do not place them in YAML,
repository variables, command output, artifacts or logs. Rotate credentials if they
are ever exposed. Do not configure a Render DSN in this batch.

## CI and test strategy

`.github/workflows/postgres_execution_ledger.yml` starts an ephemeral PostgreSQL 16
service, applies schema as an admin/migration owner, provisions a separate DML-only
runtime role, and runs the real database integration suite for each pull request to
`main` or manual dispatch. It tests that the runtime role cannot create schema objects,
plus lifecycle persistence, idempotency, independent-connection lease contention,
fencing and natural lease expiry. The workflow has no Neon or exchange credentials.

## Outage policy (mandatory before runtime wiring)

If the ledger cannot be reached or a write acknowledgement is ambiguous, block new
entries and all non-protective mutations. Do not fall back to SQLite or JSON as a
second writer. Reconcile any potentially open order/position from Deribit using exact
labels, instrument and order IDs; incomplete or bounded lookup results remain
`UNKNOWN`. Protective actions/close behavior must be separately proven not to depend
on a database fallback. Fault-injection tests are required before any Testnet execution
review.

## Backups

Maintain encrypted database exports in private storage, separate from source releases.
Keep the encryption key outside the backup and the repository. Test restoration to a
separate Neon branch or disposable PostgreSQL service before relying on a backup.
Free-tier history/snapshot features are supplementary, not a complete backup plan.

## Not included in this batch

- No live Neon migration has been run.
- No scanner/runtime DSN has been configured.
- No dashboard database reader/views have been enabled.
- No JSON dual-write or dashboard state-source cutover has occurred.
- No workflow, deployment, Git operation or exchange action has been dispatched.
