# Reliability programme — state-writer inventory

**Status:** Review candidate; no workflow has been dispatched or enabled by this document.

## Shared scanner/auditor state

The following workflows write overlapping `predictions.json` and/or `trade_history.json` state and therefore use the same GitHub Actions concurrency group: `cryptobot-state-writers`.

| Workflow | Trigger | Shared outputs | Role |
|---|---|---|---|
| `crypto_bot.yml` | 15-minute schedule; manual `PREDICT_ONLY` / `MANAGE_ONLY` | `predictions.json`, `trade_history.json`, plus scanner state | Prediction scanner and only future candidate order-mutating writer |
| `hourly_research_pipeline.yml` | Every six hours; manual | `predictions.json`, `trade_history.json`, `coin_dossier.json`, `adaptation_proposals.json` | Scheduled prediction auditor and failure taxonomy |
| `audit_workflow.yml` | Manual only | `predictions.json`, `trade_history.json`, `coin_dossier.json`, `adaptation_proposals.json` | Manual legacy auditor; do not dispatch while the hourly ownership model is active except an approved recovery procedure |

GitHub Actions permits one running workflow per concurrency group. `queue: max` permits up to 100 pending runs. Queue order is based on when runs start waiting, not guaranteed dispatch chronology; stale trading signals must expire rather than execute late. The concurrency group is a repository-level coordination aid, not a durable order ledger or a substitute for exchange reconciliation.

## Separate state domains

| Workflow | State | Isolation rule |
|---|---|---|
| `paper_execution.yml` | `paper_state_v2.json`, `paper_events_v2.jsonl` | Stage 2E only; no exchange order path |
| `candidate_scanner.yml` | `candidate_events.jsonl`, `candidate_state.json` | Shadow candidate research only |
| `train_model.yml` | model/candidate metrics | Model-build output; no direct exchange execution |
| `backtest_workflow.yml` | backtest history/results | Research only |
| `Weekly_exit_report.yml` | dated report | Reporting only |
| `run_phase2d.yml` | F08 authoritative artifacts | Sole F08 state owner; reliability changes must not touch F08 paths or artifacts |

## One order writer and dashboard separation

- No second exchange-mutating workflow is approved.
- `crypto_bot.yml` receives no Deribit credentials and hard-sets `DERIBIT_TESTNET_MUTATIONS_ENABLED=false` in this release.
- `deribit_testnet_readonly.yml` has `contents: read`, is manual-only, checks out `main`, and receives only dedicated `trade:read account:read` credentials.
- Render receives only a dedicated Deribit read-only key and `GH_READ_TOKEN` (Contents:read). Dashboard mutation handlers are disabled at both middleware and route levels.
- `emergency_close.py` is retired; do not invoke it as a kill switch.

## PostgreSQL decision

PostgreSQL has now been selected as the durable Testnet execution-ledger backend, but it remains unintegrated pending migrations, a DML-only runtime role, real PostgreSQL CI and outage/restart validation. Shared GitHub concurrency does not by itself solve ephemeral runner crash recovery. The critical recovery case remains: Deribit acknowledges an order, the runner is killed before Git state sync, and the next run reconstructs the order using a deterministic `cb_` label, open-order snapshot, complete position snapshot, recent trade history and historical order/trade endpoints. An empty/incomplete result must remain UNKNOWN. The database will store durable intent and audit evidence; Deribit remains authoritative for actual orders, positions and fills.

## Review and operational rules

- Do not change F08 workflow/artifact ownership in this reliability batch.
- Do not push evidence artifacts or private balance/position data to GitHub.
- Do not use the dashboard to repair, close or submit trades.
- Keep `ORDER_LIFECYCLE_COORDINATOR_INTEGRATED=False`, Testnet mutations disabled and production mutations blocked until a separate signed-off execution approval.
