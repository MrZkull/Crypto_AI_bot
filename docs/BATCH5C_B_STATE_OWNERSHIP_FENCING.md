# Batch 5C-B — State ownership and PostgreSQL fencing

Status: implementation candidate; not execution approval.

## Included

- PostgreSQL execution-ledger schema v2 with a monotonic `run_leases.fencing_token`.
- Idempotent v1-to-v2 migration (`ADD COLUMN IF NOT EXISTS`) and version verification.
- Database-clock lease acquisition, renewal and release. Release expires the lease row and preserves the generation.
- Fenced write validation (`SELECT ... FOR UPDATE`) in the same database transaction as intent, transition, fill and protection writes.
- Shared-required PostgreSQL mode cannot opt out of fencing and cannot silently downgrade to SQLite.
- Coordinator lease context propagation and fail-closed constructor validation.
- Account-scoped unresolved-intent admission gate. Existing idempotent intents are not treated as a new admission.
- Reconciliation preserves known partial fills when an order is still open, cancelled, or conflicts with a later rejection.
- Scanner and hourly auditor stage exact data mirror paths instead of `git add data/`.

## Workflow mirror allowlists

Scanner: `data/trades.json`, `data/trade_history.json`, `data/signals.json`, `data/scan_mode.json`, `data/balance.json`, `data/orphan_candidates.json`, `data/scan_status.json`, `data/predictions.json`, `data/reliability.json`, and `data/cooldown.json`. The workflow checks that each path exists before staging it, so optional mirrors do not fail the job.

Hourly research: `data/predictions.json`, `data/coin_dossier.json`, `data/trade_history.json`, and `data/adaptation_proposals.json`. Each is checked for existence before staging.

Candidate and paper execution remain separate state domains. F08 ownership and artifacts are untouched.

## Explicit non-goals

- Does not add order-submit/cancel methods to `ExecutionCoordinator`.
- Does not wire the coordinator around `DeribitClient._post` or the legacy executor.
- Does not enable `ORDER_LIFECYCLE_COORDINATOR_INTEGRATED`.
- Does not set `DERIBIT_TESTNET_MUTATIONS_ENABLED=true`.
- Does not configure Neon, read secrets, change dashboard or Render, dispatch workflows, push Git, or place orders.

Fencing protects database writes made by code that presents the current token. It does not make Deribit enforce a fencing token. The next stage must integrate every mutation path behind one controlled gateway and reconcile ambiguous outcomes before replay.

## Validation in the supplied build environment

- Python compilation: passed for all changed modules and tests.
- Full available suite: 574 passed, 3 skipped in 9.49 seconds, excluding three dashboard test modules because Flask is not installed in this build environment.
- The three PostgreSQL integration tests remain skipped without `EXECUTION_LEDGER_TEST_DSN`; no live PostgreSQL or Neon validation has occurred.
- Dashboard test modules and the full Windows regression must be run in the user's project environment before release qualification.
