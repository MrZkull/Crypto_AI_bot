# Risk Baseline Capture and Verified Dust Policy (R4)

## Daily baseline capture

`daily_risk_baseline_capture.py` is a separate, read-only Deribit Testnet process. It constructs a `DeribitClient` with exactly `trade:read account:read`, `allow_mutations=False`, and the hard-pinned Testnet host. It does not construct `DeribitExecutionGateway` and does not obtain exchange mutation permissions.

The workflow `.github/workflows/deribit_daily_risk_baseline.yml` attempts capture at **00:02, 00:07, 00:12 and 00:17 UTC**. Its concurrency key is unique per run, so GitHub does not coalesce the attempts; the database lease is the single-writer control. Only the first valid checkpoint in the first 20 UTC minutes is used. Repeated runs leave an existing checkpoint unchanged. A missed opening window, clock skew, malformed account data, incomplete position read, missing database privileges or lost fence fails closed; it does not manufacture a baseline.

### Required GitHub Environment configuration

Use the existing `deribit-testnet-readonly` environment and its read-only exchange credentials:

- `DERIBIT_TESTNET_READONLY_CLIENT_ID`
- `DERIBIT_TESTNET_READONLY_CLIENT_SECRET`

Add these separately:

- `DERIBIT_EXECUTION_ACCOUNT_ID` (environment secret or variable; must match the trading gateway account scope)
- `EXECUTION_LEDGER_RISK_CAPTURE_DSN` (dedicated PostgreSQL DML-only DSN)

Do **not** use the migration-owner DSN or the trading runner's `EXECUTION_LEDGER_DSN` as the baseline capture identity.

### Schema and production database role

Apply schema version **6** using the privileged migration process before enabling the scheduled capture workflow. Provision a dedicated role through the database's secret-management process. Replace the placeholder password; never commit it.

```sql
CREATE ROLE cryptobot_risk_capture_writer LOGIN PASSWORD '<managed-secret>';
GRANT CONNECT ON DATABASE <database_name> TO cryptobot_risk_capture_writer;
GRANT USAGE ON SCHEMA public TO cryptobot_risk_capture_writer;
GRANT SELECT ON TABLE execution_ledger_schema TO cryptobot_risk_capture_writer;
GRANT SELECT, INSERT ON TABLE account_risk_snapshots TO cryptobot_risk_capture_writer;
GRANT SELECT, INSERT, UPDATE ON TABLE risk_capture_leases TO cryptobot_risk_capture_writer;
```

Do **not** grant this role access to `run_leases`, `order_intents`, `order_events`, `execution_mutations`, `daily_risk_halts`, or any exchange execution table. The capture process uses a separate monotonically fenced `risk_capture_leases` table. A baseline writer therefore cannot renew, replace, or release the trading coordinator's lease. This role cannot submit or modify exchange orders; its Deribit credentials are read-only.

The PostgreSQL CI workflow provisions the same privilege boundary with CI-only credentials and proves that the capture role cannot read the trading lease or order tables. It also runs the capture-ledger integration without skipped database tests.

## T0 Testnet capital profile

`config.py` defines only `T0_TESTNET` as enabled. It uses the configured Testnet symbol universe, the existing eight-position ceiling, 3% base per-trade risk (subject to the existing 5% effective-risk hard ceiling), 5% maximum per-position notional, 2% daily loss fraction, 2% aggregate stop-risk fraction, and 20% available-funds reserve. No T1–T4 live-money profile is enabled or inferred from account deposits. Mainnet remains blocked by the separate exchange environment guard.

## Verified dust residual policy

A residual is **not** ignored merely because it is below the venue's minimum order quantity. It can be assigned `DUST_RESIDUAL_VERIFIED` only when all of the following are true:

1. The residual is attributable to the persisted bot prediction and exact entry order ID.
2. The local entry-attempt history and exchange entry order both have a verified terminal disposition; the order reports a positive fill at least as large as the residual.
3. Two position reads are non-zero, same-direction, and stable within a strict quantity tolerance.
4. Two complete open-order snapshots are both empty after order cleanup.
5. Current mark notional is at most **0.10 USDC** for that residual.
6. The emergency-close verifier uses a `1e-12` base-quantity flatness tolerance, so a non-zero residual larger than that precision floor is not silently recorded as flat before dust classification.
7. The complete disposition is durably written to the existing trade-state mirrors with no unresolved-recovery flags.

The residual record is closed for normal trade counting but retains its exact quantity, direction, both position-read observations, entry order identity, prediction identity, mark observation and verification evidence. No close price, fee or PnL is fabricated. The same symbol remains blocked while the residual is non-zero. Unrelated symbols may proceed only if the live residual still matches the record, no order is open, its mark notional remains within the cap, and the combined dust reserve is no more than **0.25 USDC**. The portfolio reserves **0.10 USDC per verified dust residual**, not merely the current tiny notional.

An untracked position, including any non-zero sub-minimum amount without complete bot provenance, remains a global admission blocker. If a recorded dust residual grows above its cap, changes direction, becomes unstable, or has an open order, the portfolio gate blocks new entries again. A verified dust residual never authorizes another entry on the same symbol.

The 0.10/0.25 USDC values are conservative **Testnet policy thresholds**, not a mathematical guarantee of maximum short-position loss or protection against exchange insolvency. Unknown provenance, unstable reads, open orders, missing mark price, larger residuals or mismatched local/exchange state block all new entries pending reconciliation.

## Release gates

- Baseline capture integration must pass with zero skips in PostgreSQL CI.
- The risk-capture role must be unable to read/modify `run_leases` and order tables.
- The dust tests must prove that verified dust allows unrelated symbols, but untracked dust blocks admission.
- Keep `EXECUTION_MODE=PREDICT_ONLY` and `DERIBIT_TESTNET_MUTATIONS_ENABLED=false` until a separate reviewed Testnet authorization.
