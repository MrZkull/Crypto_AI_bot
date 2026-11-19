# CryptoBot AI — Integrated Reliability Programme Status

**Release:** Deribit-first reliability foundation / multi-exchange read-only architecture
**Target branch:** `feat/testnet-readonly-dashboard`
**Expected base commit:** `a30c8084445a41e9dd220513bfb089d2cbe47587`
**Execution status:** Hard-disabled; no exchange order submission is integrated in this release.

## Executive status

This package adds an exchange-independent reliability foundation, a Deribit-first currency-aware risk boundary, and isolated read-only adapters for Deribit, Delta Exchange and Binance USD-M. It is an engineering foundation, not a production-ready multi-exchange execution system.

The package deliberately does **not** connect the persistent coordinator to the legacy order-submit/cancel path. `ORDER_LIFECYCLE_COORDINATOR_INTEGRATED` remains `False`; the scanner's Testnet mutation gate therefore remains false regardless of the `DERIBIT_TESTNET_MUTATIONS_ENABLED` environment value. Deribit production mutations are rejected by the client. The Delta and Binance adapters contain read methods only.

## Implemented in this package

### Failure and resilience primitives

- Named `RETRY`, `HOLD`, `DEGRADE`, `HALT` and `FLATTEN` policies with explicit failure-class mappings.
- Bounded read retry helper with jitter, maximum attempts and total deadline. It is expressly not for order submission.
- Circuit breaker with read/new-entry separation, a manual acknowledgement latch for order-path failures, and an allowlist for protective actions while merely degraded.
- Strict snapshot errors so failed or malformed exchange reads are not converted into plausible empty positions/orders.
- SQLite-backed order-intent and event ledger with deterministic intent IDs, unique `(exchange, environment, account, client_order_id)` identity, state-transition validation, fill accumulation, protection-quantity invariant and same-database run leases.

### Execution lifecycle primitives

- `ExecutionCoordinator` records intents, transitions, fills, protection and unknown outcomes.
- Unknown outcomes can only be reconciled through an injected read-only lookup; the coordinator never blindly resubmits.
- An open-order lookup without exchange order ID and a fill lookup without exchange ID/quantity/price remain `UNKNOWN`.
- Reduce-only lifecycle actions may continue during a transient `OPEN`/`HALF_OPEN` breaker, but not after a manually latched `HALTED` order-path failure.
- The coordinator intentionally has no order-submit, cancel, leverage or margin-mode mutation methods.

### Exchange integration boundary

- `DeribitReadOnlyAdapter`: exact exchange `instrument_name` for positions/orders/fills; host/environment validation.
- `DeltaReadOnlyAdapter`: Demo/Testnet host-pinned reads; cursor pagination with repeated-cursor and page-budget failure; product-ID identity; wallet transaction reads for later cashflow reconciliation.
- `BinanceUSDMReadOnlyAdapter`: HMAC-signed GET-only private reads, allowlisted test/demo hosts, explicit opt-in for production reads.
- A fault-injecting read-only adapter is available for deterministic resilience tests.
- Adapter interfaces do not expose order placement/cancellation.

### Deribit client and accounting safeguards

- Scoped OAuth Bearer authentication; read-only scope by default (`trade:read account:read`).
- Write-scoped Testnet client requires explicit construction (`trade:read_write account:read`, `allow_mutations=True`); production mutation is rejected.
- Deribit `public/auth` client-credential/refresh exchange uses the documented GET query form; JSON-RPC method calls use HTTP POST with a JSON-RPC 2.0 body and Bearer token for private methods. Mutating methods remain locally gated through a strict allowlist and are Testnet-only.
- `_get` refuses mutation endpoints even on a write-scoped client; `_post` is retained as a compatibility wrapper but only allows explicitly listed Testnet mutation methods.
- Transport exception details are sanitised so a failed authentication URL cannot leak the client secret through exception text.
- Order-state reads require matching order ID and a valid state; only explicit order-not-found error codes are terminal. Generic `not_found`, timeout and auth errors remain failures/unknown.
- `get_risk_capital_snapshot()` distinguishes native currency balances, account-wide USD equity, USDC equity and USDC available funds. The active `margin_model` (`segregated_sm`, `segregated_pm`, `cross_sm`, `cross_pm`) is authoritative; `cross_collateral_enabled` alone is a capability/feature flag and must not determine the active account model. The API may return `cross_collateral_enabled=true` while the selected model is `segregated_sm`.
- For the configured USDC-settled linear instruments, new-entry risk capital is available only for a validated segregated USDC account (`segregated_sm` or `segregated_pm`) with positive USDC equity and available funds. Cross-collateral accounts (`cross_sm` or `cross_pm`) are read for reporting but remain blocked for entries pending an explicit USDC allocation/FX margin model. Native currency equity values are never summed as USD.
- Balance persistence uses one account snapshot and does not add floating PnL a second time. `balance.json` legacy USD fields are not populated with native USDC values.
- `execute_trade()` requires a matching, current validated USDC risk snapshot and available margin, and refuses invalid/mismatched risk amounts.

### Risk and capital feasibility

- Explicit `RiskCapital`, `InstrumentSpec` and `RiskLimits` models using `Decimal` arithmetic.
- Linear-contract sizing respects quantity step, minimum/maximum quantity, minimum notional, settlement currency, tier cap, stop-risk cap, notional cap and estimated round-trip taker fees.
- A capital-tier feasibility-matrix builder returns explicit feasible/rejected outcomes. It does not force minimum lots above the risk budget.
- The matrix still requires exchange-verified, current instrument specifications, prices/stops, fee schedules, funding assumptions and order-book depth to become a real per-symbol capital plan.

### Legacy execution and dashboard safeguards

- `ORDER_LIFECYCLE_COORDINATOR_INTEGRATED = False` remains a hard gate; the normal scanner is not approved to mutate the exchange. The new lifecycle markers and exchange verification do not change this gate.
- `execute_trade()` persists a deterministic entry intent and client label before submitting the entry IOC; requested quantity remains separate from confirmed position quantity. Acknowledgement, terminal fill quantity and authoritative fill price are persisted as they are verified. Terminal IOC expiries/cancellations with partial fills continue through position reconciliation and protection sizing; zero-fill terminal orders are recorded as closed without inventing exposure.
- Once a terminal entry fill and matching live position are verified, `execute_trade()` persists the actual exposure and deterministic stop/TP client labels before the first protective-order mutation.
- The initial protective stop is a single reduce-only stop-market with `mark_price` trigger and independent exchange readback for exact instrument, direction, amount, trigger, type, state and reduce-only flag.
- Each TP intent is persisted before its one-shot reduce-only limit submission. A TP is acknowledged locally only after independent exchange readback verifies instrument, direction, amount, price, limit type, reduce-only flag and resting/filled state.
- Unknown TP/stop outcomes retain deterministic labels and block automatic resubmission. Label recovery is read-only, requires exact order identity and contract attributes, and never interprets an empty recent-label result as proof of rejection.
- Stop repair, break-even replacement and trailing-stop replacement cancel once, independently verify the old order is absent, use the fresh exchange position quantity and place one verified stop-market. Unknown results are held for reconciliation; known protection failures use only the separately verified emergency recovery path.
- The dashboard is now hard-coded monitoring-only for this release: all unsafe HTTP methods are rejected server-side (except the in-memory PDF export), execution/repair/scan/close/kill routes return 403, and the browser no longer exposes trade mutation controls. The earlier bracket-repair implementation remains unavailable through the dashboard until a separately reviewed control plane is built.
- Failed ghost-trade fill/position reconciliation does not fabricate a zero-PnL close; local records remain pending reconciliation.
- Dashboard read-only protection is unconditional in code and does not depend on `DASHBOARD_READ_ONLY_MODE` being correctly set in Render. A client-side guard is defense in depth; the server guard is authoritative.
- The legacy `emergency_close.py` direct-HTTP script has been retired because it bypassed the shared client/intent/verification safety path and could clear local state despite an unverified exchange close.
- The scanner workflow no longer receives Testnet write credentials or the mutation acknowledgement. It hard-sets `DERIBIT_TESTNET_MUTATIONS_ENABLED=false`; a separate manual read-only workflow uses dedicated read-only secrets and emits redacted evidence only.
- `exchange_environment_guard.py` requires a pinned HTTPS Testnet host, dedicated Testnet-only credentials, an explicit mutation flag and acknowledgement, rejects legacy generic/mainnet credential variables in the same process, and is additionally protected by the false code-level coordinator gate.
- Render declares only `GH_READ_TOKEN` (Contents:read), dashboard authentication/origin values, and a dedicated Deribit read-only Testnet credential. The dashboard is hard-coded read-only in server code; it does not rely on a Render `DASHBOARD_READ_ONLY_MODE` or mutation environment variable.
- The Deribit read-only GitHub workflow is manual, uses `contents: read`, requires the `main` branch and checks out `main` explicitly.

## Latest candidate validation snapshot (local reconstruction; not a Windows result)

- Candidate non-dashboard suite: **480 passed, 2 skipped**. The two skipped cases require a live PostgreSQL service/DSN.
- Three Flask-dependent modules (`tests/test_dashboard_auth.py`, `tests/test_dashboard_paper_state.py`, `tests/test_testnet_dashboard.py`) could not be run in the isolated Linux build environment because Flask was unavailable and package-index access failed. Run the complete Windows suite before approval.
- Python compilation, YAML parsing for 26 workflow files, inline dashboard JavaScript syntax, `git diff --check` and the F08 path firewall passed in the local reconstructed source. This is not a replacement for verification on the user's actual Windows worktree.
- The last user-provided Windows cumulative result, before this Batch 4A candidate, was **451 passed, 2 skipped, 3 subtests passed**. No claim is made that the new Batch 4A candidate has passed on Windows.
- No 24–48 hour Testnet soak has been performed; no workflow dispatch, deployment, Git push or exchange mutation has occurred.

## Explicitly not completed / release blockers

1. **Coordinator integration:** persistent intent creation, submit acknowledgement, exchange readback, fill lifecycle and protection verification are not yet wrapped around the existing live Deribit order path. No order execution is approved.
2. **Durable cross-run reconciliation:** the SQLite ledger is transactional only for processes sharing one file and does not persist across an ephemeral GitHub-hosted runner. The project has now selected managed PostgreSQL for durable shared audit and lease fencing while retaining one authorized GitHub Actions writer. PostgreSQL remains unwired pending explicit migrations, DML-only runtime privileges, real PostgreSQL CI, outage/recovery and backup/restore validation. Recovery must still reconcile Deribit open orders, positions, recent trades and historical order/trade endpoints; a missing result from a bounded endpoint remains UNKNOWN.
3. **Delta execution:** no Delta order-submit/cancel adapter is implemented or approved. Current Delta integration is read-only and demo-host restricted.
4. **Binance execution:** no Binance order-submit/cancel adapter is implemented or approved. Current integration is read-only.
5. **Cashflow/PnL accounting:** Delta wallet-transaction and related adapter reads are groundwork only. Deposit/withdrawal rebase, fee-currency conversion, funding reconciliation and realised/unrealised PnL reconciliation are not fully integrated into the bot's authoritative ledger.
6. **Risk policy persistence:** daily loss, drawdown ratchets, maximum correlated exposure, pending-order margin reservation and capital-tier promotion/demotion are not yet integrated as enforceable portfolio-wide gates.
7. **Market-data protection:** the full forming-candle, duplicate/out-of-order, price-source-parity, staleness, failure-budget, server-clock and model golden-vector policy set is not yet uniformly enforced end-to-end.
8. **Protection coordination:** the legacy path now has deterministic stop/TP intents, exchange readback, one-shot stop replacement and fail-closed unknown-outcome handling. However, these paths are not yet coordinated through `ExecutionCoordinator`; every legacy/UI/maintenance submit/cancel/replace/reduce-only route still needs a complete coordinator-backed audit and partial-fill lifecycle integration.
9. **Venue feasibility:** no verified 25-symbol live feasibility matrix is included. Static strategy mappings alone are not sufficient; exchange metadata, fee schedules, funding and order-book depth must be freshly verified per venue.
10. **Operational soak:** no completed read-only soak, Testnet execution soak, restart/recovery soak or mainnet shadow run is claimed.
11. **Dashboard test execution in build environment:** the isolated build environment did not have Flask and package-index network access was unavailable, so dashboard pytest modules could not be collected there. Run the full suite on the target Windows checkout with project dependencies installed.

## Required validation before any mutation is enabled

- Full project pytest suite on the target environment, including all dashboard tests.
- Python compilation and dashboard JavaScript syntax checks.
- Workflow/YAML validation plus actionlint and GitHub environment/branch protection review.
- Contract tests for each venue using documented test/demo endpoints.
- Fault-injection tests for read timeouts, rate limits, auth expiry, rejected orders, unknown submit outcomes, duplicate intents, partial fills, protection failure, runner termination and recovery.
- Integrate coordinator into every submit/cancel/replace/reduce-only path, with no bypass path. The current deterministic JSON lifecycle markers are a fail-closed bridge, not a substitute for this requirement.
- Prove restart recovery from exchange-side order, position and trade history, including aged labels and a missing recent lookup result. Keep the local ledger as cache/evidence unless multi-writer requirements justify a shared database.
- Confirm actual filled quantity, average price, fee currency and protection quantity against exchange state before declaring an order lifecycle complete.
- Complete currency-aware capital allocation, available-margin reservation, daily loss/drawdown and portfolio exposure checks.
- Run read-only soak first, followed by separately approved Testnet lifecycle tests.

## Operating rule

Deribit remains the only designated primary exchange. Delta and Binance may be exercised in parallel only through their isolated read-only adapters and independent demo/test accounts. Do not pool account balances or share order state. Do not enable `DERIBIT_TESTNET_MUTATIONS_ENABLED`, dispatch a trading workflow, deploy, or use mainnet until the release blockers above have passed a separate review.

## Batch 3A/Batch 4 — PostgreSQL shared-ledger implementation (selected; not yet wired)

- Added `PostgresExecutionLedger`, implementing the SQLite ledger interface: deterministic intent identity, state transitions, monotonic cumulative fills, protection coverage, unknown/reconciliation queries, append-only event reads and database run leases.
- PostgreSQL transitions use row-level `SELECT ... FOR UPDATE`; lease acquisition is an atomic conditional upsert using the PostgreSQL database clock. The lease tests verify SQL shape, but they do not prove live multi-process behavior.
- `create_execution_ledger()` supports explicit backend selection and fails closed when shared PostgreSQL is required but unavailable. It never silently falls back from a required shared store to a local SQLite file.
- `psycopg[binary]>=3.2,<4` was moved out of core `requirements.txt` into optional `requirements-postgres.txt`. The default bot/dashboard install does not acquire this dependency.
- The original real PostgreSQL lifecycle, independent-connection and fencing tests require a real DSN. A natural-expiry test and a DML-only-role privilege test have now been added. The local candidate environment has no PostgreSQL server/driver, so those real-database tests have not passed here. The new PostgreSQL CI workflow provisions a disposable PostgreSQL 16 service and runs them with a separate runtime role.
- The earlier deferral decision is superseded by the November 18, 2026 decision to use Neon Free for a persistent Testnet execution ledger. Cost remains constrained: the CI test database is disposable and local PostgreSQL is not required. This selection does not approve exchange writes.
- The exchange is the durable order record. Recovery must reconcile deterministic labels against open orders, positions and recent user trades. Deribit documents that `private/get_order_state_by_label` only returns closed orders from the last 30 minutes; older orders require historical order history. `private/get_user_trades_by_instrument` returns recent trades (24 hours by default, maximum 1,000 per request) and includes an optional user label on trade rows. The code now uses exact label matches from recent trade rows as additional evidence, but absence from a bounded recent response is never proof that no order was accepted.
- **Not yet connected:** `ExecutionCoordinator` and legacy submit/cancel/replace/exit paths do not call this backend. `ORDER_LIFECYCLE_COORDINATOR_INTEGRATED = False` remains unchanged.

### PostgreSQL configuration (selected for durable Testnet ledger; not yet wired)

PostgreSQL is now selected for the durable Testnet execution ledger, but it remains unwired until the separate migration, DML-only runtime role, real PostgreSQL CI, outage/recovery, and backup/restore gates pass. Install `requirements-postgres.txt` only where PostgreSQL support is required. Apply schema with `scripts/migrate_execution_ledger.py` using the migration-owner DSN; runtime construction is verification-only and performs no DDL. See `docs/NEON_POSTGRESQL_OPERATIONS.md`. No Neon credential has been configured by this source patch.


## Batch 4A — environment governance and read-only soak design (candidate; not installed on Windows)

- Candidate adds a strict Testnet environment guard with dedicated `DERIBIT_TESTNET_CLIENT_ID/SECRET`, pinned `https://test.deribit.com/api/v2`, exact human acknowledgement and no generic/mainnet credentials in the same process. The coordinator's code-level gate remains false.
- Candidate workflow changes ensure the scheduled/manual scanner receives no write-scoped Deribit credentials and hard-sets the mutation flag to `false`. The scanner, hourly research auditor and manual legacy auditor share the `cryptobot-state-writers` concurrency group with `queue: max` and `cancel-in-progress: false`. This serialises overlapping JSON state writers; it is not a distributed order ledger. The scanner still has no write-scoped Deribit credentials, and `MANAGE_ONLY` cannot construct a write client while the coordinator gate remains false.
- Candidate removes automatic dispatch of the historical manual auditor from every successful scanner run. F08 audit/recovery workflows are not modified; follow the existing `DECISIONS.md` ownership and dispatch rules.
- Candidate makes dashboard server routes and browser controls monitoring-only. The unsafe HTTP method guard is unconditional, and the old direct `emergency_close.py` network mutation script is retired.
- Candidate read-only preflight artifacts contain completeness/count metadata and public instrument/order-book verification, not balances, equity, position quantities, PnL, credential values or private position instrument lists.
- Candidate adds `scripts/verify_f08_firewall.py` to inspect modified and untracked paths against the `a30c808` baseline. It also adds source-lock hashes for the signal, observation-contract and prediction-record helpers. This source-lock is not a substitute for the still-pending runtime golden-vector test using frozen candles and serialized prediction output.
- Candidate adds a 24–48 hour manual read-only soak runbook and an F09 macro-regime preregistration draft. F09 is research-only, not merged into production features or the order path.
- Delta and Binance adapters remain read-only foundations and are parked outside the Deribit Testnet critical path.


## Batch 4A — single-writer governance and read-only soak preparation (candidate; not yet installed on Windows)

- Dashboard unsafe methods are blocked before route handlers and also have route-level refusal bodies. Dashboard HTML removes manual scan/close/kill, probation bench/release, cooldown release, adaptation approve/reject, and email dispatch controls. Only two in-memory PDF exports remain permitted; they do not mutate trading or repository state.
- Render dashboard GitHub access is narrowed to a fine-grained `GH_READ_TOKEN` with Contents:read. Dashboard GitHub `gh_push` is an unconditional refusal/no-op in the candidate patch. Before a future deployment, replace/remove the existing Render `GH_PAT_TOKEN`; do not reuse scanner state-writer credentials.
- `crypto_bot.yml`, `hourly_research_pipeline.yml` and manual `audit_workflow.yml` use `cryptobot-state-writers`, `queue: max`, and `cancel-in-progress: false` because they write overlapping prediction/history state. The scanner receives no Deribit write credentials. `audit_workflow.yml` remains manual-only and is no longer auto-dispatched after every scan. Stale signals must expire; queue order is not permission to chase old signals.
- A dedicated manual-only `deribit_testnet_readonly.yml` workflow reads account/position completeness and public instruments/order books with dedicated read-only Testnet credentials, `contents:read`, pinned Testnet host, three-sample server-clock measurement and redacted evidence. It has not been dispatched.
- Unknown-order reconciliation now uses recent state-by-label, open orders, a complete live-position snapshot, paginated recent trades, paginated historical order history and historical trades-by-order for matched order IDs. Exact label/instrument/order-ID checks are applied. Query success, pagination completeness and per-order lookup coverage are recorded separately. A failed query, capped page window, over-limit match set or absent match remains UNKNOWN; no result is converted into rejection or flatness.
- F08 firewall and signal helper source-lock test are included. The source-lock is not a runtime golden-vector; a frozen-input signal/observation output byte comparison remains pending.
- Delta/Binance order execution remains deferred. PostgreSQL is optional/deferred until multiple writers or an audit requirement justify it; the two live DB tests remain skipped without a PostgreSQL DSN.
- F09 macro/flow/CatBoost research is a pre-registration draft only. Nothing in F09 enters the live model or order path.
