# Deribit Testnet read-only soak — Stage 1

**Status:** Prepared; not dispatched.
**Minimum observation window:** 24 hours; target 48 hours.
**Execution mode:** Read-only only. No orders, cancellations, leverage changes, transfers or state repair.

## Purpose

Prove that the dedicated Deribit Testnet read-only credential, exact Testnet host, instrument mapping, account snapshot and order-book reads are reliable over time before any Testnet mutation is considered.

This is an exchange-connectivity and evidence-quality exercise. It does not measure strategy profitability and it does not approve order execution.

## Preconditions

1. The reviewed branch has passed the Windows full regression and F08 firewall.
2. The workflow is dispatched only from the reviewed `main` ref. The workflow itself checks out `main` and has `contents: read`.
3. The protected `deribit-testnet-readonly` GitHub Environment contains only a dedicated key scoped to `trade:read account:read`.
4. The host is pinned to `https://test.deribit.com/api/v2`.
5. `crypto_bot.yml` remains code-gated by `ORDER_LIFECYCLE_COORDINATOR_INTEGRATED = False`; Testnet mutation variable is absent/false; production mutation remains blocked.
6. Dashboard is monitoring-only. Render has only a dedicated Deribit read-only key and a fine-grained GitHub Contents:read token; dashboard GitHub writes are hard-disabled.
7. No exchange mutation workflow is enabled or dispatched during this soak.

## Cadence and duration

Run manually every four hours for 24–48 hours (6–12 samples). Do not add a permanent scheduled workflow for this finite soak. Record workflow run ID, attempt, commit SHA, UTC timestamps and artifact name for every sample.

## Each sample must verify

- OAuth authentication succeeds with the read-only scope.
- Server clock offset and measurement uncertainty are within the pre-registered 2,000 ms bound.
- The API host is exactly Deribit Testnet.
- Account snapshot is complete (record only `account_snapshot_complete` and asset count in uploaded evidence; do not upload balances or equity).
- Position snapshot is complete (record only completeness and counts; do not upload private instrument lists, quantities, average entry, mark, unrealised PnL or equity).
- Every selected symbol resolves to the expected instrument and passes contract verification.
- Bid and ask are positive, finite and ordered; spread is finite and within the pre-registered operational threshold.
- A failed private read is recorded as a failed sample, never as an empty/flat account.
- `order_submission_performed`, `order_cancellation_performed` and `leverage_change_performed` remain false.

## Evidence handling

The artifact is intentionally redacted. It may contain public instrument identifiers, public top-of-book prices/spreads, counts, error classes, run metadata and timestamps. It must not contain API keys/tokens, account equity, asset balances, margin amounts, position size, average/mark prices for private positions, private PnL or raw exchange response bodies.

Keep each sample's artifact and a small index in the release review. Do not commit evidence artifacts or private account snapshots to the repository. Retain the evidence only for the approved review period.

## Pass/fail rules

A sample passes only if authentication, complete account/position reads and all selected instrument checks pass. Any authentication error, incomplete/ambiguous snapshot, host mismatch, instrument-spec mismatch, invalid book or failed read is a sample failure. Do not convert transient failures to zero/empty values.

The 24–48 hour soak passes only when all planned samples pass, with no unexplained gaps and no mutation calls. Any failed sample restarts the observation window after the cause is fixed and documented.

## Output

Produce a signed-off soak summary with:

- branch/commit SHA reviewed;
- UTC sample timestamps and run IDs;
- pass/fail per sample and reason class;
- instruments selected / verified / rejected;
- account and position snapshot completeness counts only;
- confirmation that no exchange mutation endpoint was called;
- incidents, recovery and follow-up actions;
- reviewer and decision (pass, extend, fail).

## Explicit non-goals

No tiny orders, stop/TP placement, cancellation, leverage changes, kill-switch drill or mainnet shadow are part of Stage 1. Those belong to a separate approved mutation harness after full coordinator integration, environment isolation and risk gates pass.
