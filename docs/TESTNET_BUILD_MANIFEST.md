# Build Manifest — Testnet Read-only Dashboard

Target baseline: `main` commit `a30c8084445a41e9dd220513bfb089d2cbe47587` (the Testnet worktree's verified starting HEAD).

Feature branch: `feat/testnet-readonly-dashboard`.

This pack's installer starts from `origin/main`, not PR #15 or any F08 candidate branch.

## Source files modified by installer

- `dashboard.py`: dedicated read-only Testnet credentials, explicit Testnet host check, fail-closed account/position APIs, new `/api/testnet/overview`, no fabricated fee/slippage/volume/duration values, model accuracy sourced from `model_performance.json`, read-only method guard.
- `deribit_client.py`: strict private `get_account_summary()` probe and strict optional error propagation for exchange trade history.
- `dashboard_static/index.html`: additive Testnet read-only panel; honest unavailable states; preserve the rest of the legacy dashboard; hide mutating controls in this monitor build.
- `render.yaml`: separate read-only Deribit Testnet secrets, GitHub data variables and read-only mode.
- `.gitattributes`: explicit line-ending rules.

## New payload files

- `.github/workflows/deribit_testnet_readonly.yml`
- `testnet_readonly_check.py`
- `tests/test_testnet_dashboard.py`
- `docs/TESTNET_RELEASE_RUNBOOK.md`
- `docs/TESTNET_BUILD_MANIFEST.md`

## Safety guarantees intended by this release

- No writes to `main`.
- No edit to PR #15.
- No Testnet or mainnet order submission.
- No order cancellation or leverage change.
- Dashboard uses only separately named read-only Testnet keys.
- Failed/incomplete exchange position reads remain visibly unverified.
- The manual workflow has read-only repository permissions and does not contain order calls.

The installer must still be run against the current local clone, then the generated diff and tests must be reviewed. This manifest is not proof of a successful local application or deployment.
