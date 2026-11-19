# Deribit Testnet Read-only Dashboard and Reliability Foundation — Runbook

## Scope

This branch contains the original read-only Testnet dashboard foundation plus the Deribit-first reliability and multi-exchange read-only layer. It is **not** approved for live Testnet or mainnet order execution.

The existing dashboard layout is retained. The dashboard's Deribit panel is a read-only view of account, position, exact-instrument order and selected fill snapshots. Stop-order presence is not proof of full quantity coverage. A failed position read is never interpreted as a flat account.

The added reliability modules provide failure policies, bounded read retries, circuit breaker, order-intent/event ledger, a no-submit execution coordinator, currency-aware risk accounting, capital-feasibility calculation and isolated read-only adapters for Deribit, Delta Demo and Binance USD-M test/demo. The legacy Deribit execution code also contains deterministic pre-submit protection intents, one-shot verified stop-market placement/replacement, reduce-only TP readback, and read-only unknown-outcome reconciliation. These are reliability safeguards, not approval to enable execution; the coordinator remains deliberately disconnected from the live legacy order path.

See [`RELIABILITY_PROGRAMME_STATUS.md`](RELIABILITY_PROGRAMME_STATUS.md) for implementation details, test evidence, known limitations and every outstanding release blocker.

## Branch and isolation

- Target branch: `feat/testnet-readonly-dashboard`
- Expected base commit: `a30c8084445a41e9dd220513bfb089d2cbe47587`
- Keep `feat/hourly-non-f08-research-isolation`, PR #15 and F08 state separate.
- Do not reset/clean the main project working tree to apply this patch.
- Apply only the reviewable Git patch to the Testnet worktree. Verify the branch and exact base commit, review the patch contents, run `git apply --check`, then apply with `git apply`. Do not use a self-extracting installer for this candidate.

## Exchange environment isolation

### Deribit primary

- Use a dedicated Deribit Testnet account/API key.
- Dashboard/preflight credentials must be read-only.
- The Deribit client defaults to `trade:read account:read`.
- Write-scoped Testnet construction requires an explicit `trade:read_write account:read` scope and `allow_mutations=True`.
- Production mutation is rejected by the client.
- `ORDER_LIFECYCLE_COORDINATOR_INTEGRATED` deliberately remains `False`; the scanner mutation gate therefore remains disabled regardless of environment flags.
- Legacy protection code must persist each deterministic label before a mutation and verify the actual exchange order afterward. The entry IOC intent/label is persisted before submission; requested quantity is not recorded as filled quantity, and terminal partial fills are reconciled against the live position before protection sizing. An ambiguous response is never retried. A missing order-by-label result is not proof that the request failed.
- The dashboard is hard-coded monitoring-only in this release. Middleware and the unsafe route handlers reject trade repair, close, kill-switch, scan-dispatch, probation, cooldown, proposal and email mutation requests with 403. Only two in-memory PDF export POSTs are permitted. Dashboard credentials and Testnet preflight credentials must be read-only.

### Delta and Binance

- Delta adapter is restricted to the India Demo/Testnet host used in this package.
- Binance adapter is read-only and restricted to documented USD-M test/demo hosts unless production reads are explicitly opted into.
- Each exchange requires its own key, environment, account identity, instrument mapping, currency ledger and state database.
- Do not share API keys, order IDs, balances or state across exchanges. Do not send the same live signal to multiple exchange accounts.

## Render dashboard variables

Keep these settings in Render:

- `DERIBIT_TESTNET_READONLY_CLIENT_ID`
- `DERIBIT_TESTNET_READONLY_CLIENT_SECRET`
- `DASHBOARD_AUTH_USERNAME`
- `DASHBOARD_AUTH_PASSWORD` (at least 20 characters)
- `DASHBOARD_PUBLIC_ORIGIN` (exact HTTPS origin, no path)
- `GH_READ_TOKEN` (fine-grained GitHub token with Contents:read only)
- `GITHUB_REPO=MrZkull/Crypto_AI_bot`
- `GITHUB_BRANCH=main`

Do not store generic production credentials in the dashboard service. Do not add mainnet execution keys to Render.

## Workflow safety

- `.github/workflows/deribit_testnet_readonly.yml` is manual-only, has `contents: read`, requires `github.ref == 'refs/heads/main'` and checks out `main` explicitly.
- Keep the `deribit-testnet-readonly` GitHub Environment restricted to authorised maintainers and require review where available.
- The preflight must not be dispatched until the branch is reviewed/merged and the read-only environment secrets are configured.
- The existing scanner workflow receives no Testnet write-scoped credentials and hard-sets the mutation flag to `false`. It remains mutation-disabled because the lifecycle coordinator has not been wired around all live order paths. Do not override this code gate.
- Render dashboard GitHub access must use `GH_READ_TOKEN` with Contents:read; remove the legacy write-capable `GH_PAT_TOKEN` from Render before any future deployment. `render.yaml` no longer declares dashboard mutation flags; read-only is enforced in code.
- The scanner, hourly research auditor and manual legacy auditor share `group: cryptobot-state-writers`, `queue: max` and `cancel-in-progress: false`, because they write overlapping prediction/history state. GitHub allows one active run and up to 100 queued runs with `queue: max`; the queue is FIFO by the time a run starts waiting, not by dispatch timestamp. If the queue reaches capacity, additional runs are cancelled. A queued scan must still expire stale signals rather than chase late bars. This lock serialises these GitHub workflows only; it does not coordinate a second external writer.
- Do not automatically dispatch `audit_workflow.yml` from a normal scan. It is manual-only; follow `DECISIONS.md` and keep F08 audit work under its designated owner.

## Validation commands on the Windows worktree

```powershell
$TargetRepo = "D:\ABHI\Ai_Crypto_Bot_brain\Github_clone_Repo\Crypto_AI_bot_testnet"
Push-Location $TargetRepo

python -m compileall -q .
python -m pytest -q

git diff --check
git status --short
git diff --stat

Pop-Location
```

The full test suite must run with the project's declared dependencies installed, including Flask. A partial test run excluding dashboard modules is not the final release gate.

## Hard execution gate

`ORDER_LIFECYCLE_COORDINATOR_INTEGRATED = False` is intentional. The current coordinator records and reconciles lifecycle events but does not submit/cancel orders and is not yet integrated into every legacy execution path. Keep it false. Do not set `DERIBIT_TESTNET_MUTATIONS_ENABLED=true`, do not set a write-scope on any dashboard credential, and do not enable Testnet execution until every release blocker in the reliability status document is closed and separately reviewed.

## Batch 3A/Batch 4 PostgreSQL ledger (selected; not an execution enablement)

The project now selects a managed PostgreSQL execution ledger for durable shared audit and fencing, while retaining one authorized GitHub Actions writer. PostgreSQL is not yet wired to live scanner mutations or the Render dashboard. Use the explicit migration command and a separate runtime writer role; runtime constructors must not perform DDL. CI must run the real PostgreSQL integration tests against an ephemeral service. The exchange remains authoritative for actual orders, positions and fills. See `docs/NEON_POSTGRESQL_OPERATIONS.md`.

Deribit `private/get_order_state_by_label` returns recent closed orders only for the last 30 minutes. Older orders require historical order-history endpoints. Recent user trades by instrument cover 24 hours by default, return at most 1,000 rows per request, and include a user label when one was set. Reconcile exact label, instrument, open orders, positions and recent trades. If any bounded query is empty or incomplete, keep the intent `UNKNOWN`; never assume rejection or resend blindly.

The PostgreSQL implementation sequence is now active, but it remains a separate readiness gate:

1. Apply schema through `scripts/migrate_execution_ledger.py` using the migration-owner DSN; the runtime writer must not have DDL privileges.
2. Configure DSNs/SSL settings only in protected Environment secrets; never put DSNs in source, workflow YAML, logs or artifacts.
3. Run `.github/workflows/postgres_execution_ledger.yml` against its disposable PostgreSQL service; all real database tests must execute, not skip.
4. Test concurrent independent connections, natural lease expiry/fencing, runner restart, database outage/recovery, backup/restore and schema migration.
5. If the database is unavailable while a position may be open, HALT new entries and non-protective mutations, reconcile from Deribit, and never downgrade to SQLite.

`ORDER_LIFECYCLE_COORDINATOR_INTEGRATED = False` remains unchanged.


## Batch 4A — pre-soak controls

- `.github/workflows/deribit_testnet_readonly.yml` is manual-only, `contents: read`, `main`-restricted, and uses only dedicated read-only Testnet credentials. Its evidence artifacts must not contain account balances/equity, position sizes/PnL, private position symbols, credentials, local trade state or proprietary predictions.
- Run the read-only preflight manually every ~4 hours for 24–48 hours after the reviewed workflow is available on `main`. Capture the run ID, commit SHA, completion counts, error types and UTC time. Any authentication, incomplete position snapshot, instrument mismatch, stale quote or unexpected credential/host mismatch fails the soak.
- Run `python scripts/verify_f08_firewall.py --base a30c8084445a41e9dd220513bfb089d2cbe47587` before local commits. It checks tracked and untracked paths. F08 files/workflows remain out of scope and untouched.
- The signal source-lock test hashes `generate_signal`, observation-contract and prediction-record helpers against the `a30c808` source. A runtime golden-vector replay with frozen candles and byte-identical serialized records remains a separate release gate.
- See `docs/DERIBIT_TESTNET_READONLY_SOAK.md` for the 24–48h soak protocol and `docs/F09_MACRO_REGIME_PREREGISTRATION.md` for the research-only F09 proposal.


## Monitoring-only dashboard GitHub token

Render must use `GH_READ_TOKEN`, a fine-grained GitHub token with Contents:read only. Remove the legacy `GH_PAT_TOKEN` from the Render service. The dashboard's `gh_push` function is a hard no-op in this release; write routes are rejected by middleware before route execution. PDF export is an in-memory exception only. Do not reuse scanner write credentials in Render.

## Historical unknown-order reconciliation

After `private/get_order_state_by_label` (recent closed-order window), `private/get_open_orders_by_label`, complete position snapshot and paginated recent `private/get_user_trades_by_instrument`, ambiguous entries also query `private/get_order_history_by_instrument` and `private/get_user_trades_by_order` with `historical=true`. Historical indexing may lag and does not include the recent window. The candidate pages historical order history up to a bounded maximum and performs per-order trade lookups for matching IDs. Query success, pagination completeness and per-order lookup coverage are recorded separately. If the maximum page count is reached, a query fails, or the matching order set exceeds the per-order lookup limit, evidence is incomplete and the intent remains `UNKNOWN`. An empty result from any one source is never treated as proof of rejection or no fill. A match requires exact client label, exact instrument identity and, for fills, exact order ID.


## Read-only soak: timing and evidence requirements

The preflight uses three `public/get_time` samples, selects the lowest-round-trip observation, reports offset and half-round-trip uncertainty, and rejects when `abs(offset) + uncertainty` exceeds the pre-registered 2,000 ms bound. It does not adjust the machine clock. Spread is computed as `(ask - bid) / midpoint` and reported both as a fraction and basis points; the preflight also checks that the client-reported spread agrees with the computed value. The 24–48 hour window is a manual finite soak (6–12 samples approximately four hours apart), not a permanent scheduled job.

F09 macro/crowding/CatBoost research is a separate preregistration draft only. It must not affect the live model, Stage 2E records or F08 evidence unless a separately approved experiment amendment is created.
