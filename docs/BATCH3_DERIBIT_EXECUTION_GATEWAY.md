# Batch 3 — Deribit Testnet execution gateway (candidate only)

**Status: integration candidate; Testnet writes remain disabled.** This addendum supplements `TESTNET_RELEASE_RUNBOOK.md`; it does not replace or relax it.

## Scope

This batch adds a persist-first execution gateway between the Deribit client and the existing execution ledger/coordinator. It is restricted to Deribit Testnet and the bot's explicitly mapped USDC perpetual instruments. It does not enable exchange mutations, change the dashboard, change workflows, alter F08 state or validators, or deploy anything.

The release constant `ORDER_LIFECYCLE_COORDINATOR_INTEGRATED` remains `False`. The environment opt-in alone is therefore insufficient to enable writes. The dashboard remains read-only. Mainnet is not supported by this gateway.

## Order lifecycle

1. Validate the method, instrument, label, side, quantity and strict `reduce_only` value.
2. Persist a unique order intent with instrument and request metadata.
3. Atomically claim the intent as `SUBMITTING` before making one exchange request.
4. Never retry an order mutation after a transport or response ambiguity. The intent is marked `UNKNOWN` when that can be durably recorded; the order-path breaker is latched when persistence itself fails.
5. Validate the response and persist acknowledgement, fill, rejection, or unknown outcome.
6. Reconcile only through read-only exchange queries. The label, instrument, side and quantity must agree with the durable intent. Multiple or contradictory matches remain unresolved.

A process crash between the claim and receipt of an exchange order ID is recovered through exact client-label lookup, not by repeating the submission. An empty or incomplete label search is not proof that the order was rejected. The Deribit recent closed-order-by-label window is limited; the runbook documents the relevant history limits. Any evidence outside the bounded/complete search window must remain unresolved until independently reconciled.

## Control mutations

Cancel, cancel-all, edit and leverage mutations are separately journaled in `execution_mutations`, with append-only `execution_mutation_events`. The ledger atomically refuses a new control mutation while an earlier mutation for the same account remains `SUBMITTING` or `UNKNOWN`. This avoids replaying an operation after a process restart just because the caller lost the response.

An ambiguous control mutation can only be resolved with explicit read-only evidence and a named resolver. Resolution does not reset the order-path circuit breaker; operator acknowledgement remains a separate action after review.

## Protection invariant

The gateway does not count an entry as protected based on a local flag. The stop must have its own durable acknowledged intent and exchange order ID, be reduce-only, be a `stop_market` using `mark_price`, be on the same instrument and opposite side, and cover at least the confirmed entry fill. The durable protection link is committed under the ownership lease. Missing or inconsistent proof is a blocker, not a pass.

## Ownership and storage

- SQLite is for one process/one host only. The default local ledger path is `data/execution_ledger.sqlite3`; the path must live on durable storage and be protected by backups before any future write-enabled test.
- Do not use local SQLite as a shared ledger between GitHub Actions, Render, or other independent hosts. The gateway selects a shared PostgreSQL ledger by default in GitHub Actions and requires fenced leases there; it fails closed when that configuration is unavailable rather than falling back to local SQLite.
- The PostgreSQL schema in this candidate adds mutation/event tables and schema marker v4. The code-level tests do not replace a real database integration test. Independent connections, migrations, lease expiry/renewal/fencing, process restart, database outage/recovery and backup/restore still require validation against a disposable test database.
- A free/ephemeral host filesystem is not a safe durable SQLite ledger. Do not turn on Testnet writes from a Render free instance unless durable state/ownership has been demonstrated and the full deployment topology reviewed.

## Release gates still open

This candidate is **not approved for Testnet writes**. Before considering a separate, explicit enablement change, all of the following are required:

1. Apply this package to the matching Testnet worktree and run its focused suite there; then run the complete Windows repository regression suite, including paper execution and dashboard tests with the repository's complete dependencies and source files.
2. Run real PostgreSQL integration tests using a newly provisioned disposable database, including independent clients and fencing/restart/failure cases. No live user database credentials belong in source, logs or test archives.
3. Complete read-only Testnet verification with the intended account and exact supported instruments. Verify account identity, permissions, instrument mapping, minimum amount/precision, margin/risk settings and position snapshot completeness.
4. Fault-inject every ambiguous boundary: immediately before and after intent claim, exchange acceptance before response, response before durable acknowledgement, partial fills, protection submission/verification, cancel/edit/leverage ambiguity, process restart and ledger outage.
5. Verify the full order lifecycle on Testnet with small capped risk only after separate review: entry, partial/full fill, stop protection, TP, cancel/replace, emergency reduce-only close, reconciliation and restart recovery. Every unresolved condition must halt new exposure.
6. Review the exact deployment topology and guarantee a single mutation owner. For multiple hosts/runners, use a shared transactional PostgreSQL ledger with fenced writes; never rely on a local file.
7. Rotate any previously exposed exchange or database credentials and configure least-privilege Testnet-only credentials outside source control. Never use Mainnet credentials.
8. Obtain explicit approval for a separate gate change. This package intentionally does not change the integration constant or set `DERIBIT_TESTNET_MUTATIONS_ENABLED`.

## Validation performed for this candidate

In the isolated source snapshot, Python compilation passed and the focused regression set passed **280 tests, with 3 skipped**. The skips are optional real PostgreSQL integration tests; no PostgreSQL DSN was supplied. No Deribit account was contacted and no exchange mutation was sent.

The complete repository suite was not established in this isolated environment: the source archive did not include `candidate_market_monitor.py`, which is imported by the paper-execution tests, and Flask was not installed for dashboard test collection. This is a limitation of the supplied snapshot/environment, not a claim that the user's Windows checkout fails. The complete Windows regression remains a required gate.

## Out of scope / unchanged

F08 evidence, snapshots, validators and workflows; dashboard files; GitHub workflow files; Render deployment files; secrets; exchange actions; Git commits and pushes. The installer deliberately touches only the listed Python files and this addendum.
