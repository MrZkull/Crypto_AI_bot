# Risk Integrity and Capital Governance Batch

## Scope

This batch hardens the existing Deribit Testnet entry path. It does not enable
exchange mutations, change the execution lifecycle integration gate, deploy,
commit, or push anything. Existing gateway, idempotency, protection and recovery
components remain authoritative.

## Runtime controls

- Every eligible new entry refreshes the validated USDC account summary and
  complete position snapshot immediately before sizing.
- Snapshot freshness uses a monotonic clock. Deribit server time is used for
  the UTC-day key only after the local/server clock-skew check passes.
- The daily USDC loss guard uses immutable account-equity checkpoints stored in
  the fenced execution ledger. It selects the first checkpoint within 20
  minutes of 00:00 UTC. If the opening checkpoint is missing or late, new entries
  remain blocked for that UTC day; no prior-day mark-to-market value is reused.
- Daily PnL is current exchange equity less the baseline and classified net
  deposits, withdrawals and transfers. Trading/settlement/funding/fees remain
  reflected in exchange equity and are not added a second time. Unknown or
  unclassified transaction types block new entries.
- Transaction-log calls are serialized to respect the exchange's documented
  request interval. Pagination is bounded.
- A breached daily limit is sticky for that UTC day in the execution ledger.
- Portfolio admission requires local records and exchange positions to agree;
  every live position must have a currently resting, exchange-verified,
  reduce-only mark-price stop-market order. Unknown state blocks new entries.
- Open stop risk plus prospective trade risk may not exceed the configured
  portfolio risk cap.
- Sizing uses Decimal arithmetic, adverse directional tick rounding,
  entry-plus-stop exit fee estimation, and a configurable available-funds
  reserve (default 20%). Leverage is not assumed or automatically changed.

## Optional active capital-tier configuration

The repository's `config.py` was not part of the 21-file review bundle, so this
batch does not invent user capital tiers or overwrite local config. The entry
path now accepts these optional values:

- `CAPITAL_TIER_ALLOWED_SYMBOLS`: list/tuple/set of symbols permitted by the
  currently selected tier. When set, all other symbols are rejected.
- `CAPITAL_TIER_CAP_USDC`: fixed USDC equity ceiling; sizing uses the lesser of
  live equity and this cap.
- `CAPITAL_TIER_RISK_FRACTION`: per-trade risk fraction before strategy scaling.
- `CAPITAL_TIER_MAX_POSITIONS`: maximum concurrent positions.
- `CAPITAL_TIER_MAX_NOTIONAL_FRACTION`: per-position notional fraction.
- `CAPITAL_TIER_MAX_DAILY_LOSS_USDC`: absolute daily loss ceiling, combined with
  `MAX_DAILY_LOSS_FRACTION` by taking the stricter limit.
- `CAPITAL_TIER_MAX_PORTFOLIO_RISK_USDC`: absolute aggregate stop-risk ceiling,
  combined with `MAX_PORTFOLIO_RISK_FRACTION` by taking the stricter limit.
- `AVAILABLE_FUNDS_RESERVE_FRACTION`: available collateral haircut (defaults to
  0.20 when absent).
- `MAX_DAILY_LOSS_FRACTION`: fractional daily loss ceiling (defaults to 0.02).
- `MAX_PORTFOLIO_RISK_FRACTION`: aggregate open stop-risk ceiling (defaults to
  0.02 and may not exceed 0.05).
- `MAX_DERIBIT_CLOCK_SKEW_MS`: maximum allowed local/Deribit clock skew
  (defaults to 5000 ms).

The feasibility matrix reports T0-T4-style tier economics only when verified
available funds are supplied. It does not treat a tier's capital cap as proof
that the exchange account has that much free collateral. Actual tier names,
limits, symbols and movement criteria remain operator-reviewed configuration.

## PostgreSQL schema and rollout

The execution ledger schema is version 5. Apply migrations using the privileged
migration role and the existing migration workflow before using the new ledger
methods. The runtime role needs only the existing fencing access plus SELECT and
INSERT on `account_risk_snapshots` and `daily_risk_halts`; the runtime never runs
DDL. Until a valid opening checkpoint is recorded in the UTC opening window,
new entries are blocked for that day. No migration or execution gate is enabled
by this source patch automatically.

## Validation boundary

The isolated source bundle does not include `config.py`,
`exchange_environment_guard.py`, or all full-repository integration dependencies.
The focused tests can validate pure risk controls and ledger interfaces, but
full candidate regression must be run after applying the guarded patch to the
Windows worktree. PostgreSQL integration must run against the existing CI
PostgreSQL service; skipped database tests are not accepted there.
