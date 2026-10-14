# Crypto_AI_bot — Decisions and State Ownership

Last updated: 2026-10-15

## State ownership

| State or output | Owner | Rule |
|---|---|---|
| F08 ledger, state and snapshots | `.github/workflows/run_phase2d.yml` only | No other workflow may restore, mutate, stage or upload authoritative F08 state. |
| `predictions.json` | Production scanner appends; one designated auditor reconciles | Avoid multiple concurrently running auditor writers. |
| Auditor outputs, including `coin_dossier.json` | `.github/workflows/hourly_research_pipeline.yml` as the sole scheduled auditor | Six-hour schedule; the separate manual audit workflow must not be dispatched while this ownership model is in effect. |
| `paper_state` and `paper_events` | Stage 2E | Remain isolated from F08 state and the auxiliary research sync. |

## Hourly pipeline isolation

- The hourly research pipeline must not restore Phase 2D artifacts, download or verify locked Phase 2D models, run the Phase 2D prospective validator, restore Phase 2D snapshots, stage `research_outputs/`, or upload `phase2d-state`.
- The auxiliary state-sync shell step must exit before any reset, rebase, staging or push when `GITHUB_REF` is not `refs/heads/main`.
- The scheduled auditor cadence is every six hours at minute 10 UTC.
- The pipeline remains disabled until the isolation fix is reviewed and merged. Re-enable only after confirming that exactly one auditor workflow is enabled for routine execution.

## Artifact provenance investigation

The inventory was checked on 2026-10-15 using the GitHub Actions artifact API and workflow-run index.

- The query returned 157 `phase2d-state` artifact records in the requested period.
- All 157 mapped to `.github/workflows/run_phase2d.yml` (CryptoBot AI — Phase 2D Prospective Validation).
- No returned artifact mapped to `.github/workflows/hourly_research_pipeline.yml`.
- The earliest returned artifact was created at `2026-09-30 22:03:24Z`; the latest was created at `2026-10-14 20:08:14Z`.

**Conclusion:** the queried inventory does not demonstrate that the hourly workflow actually uploaded a `phase2d-state` artifact. The competing-writer risk remains confirmed by the old workflow code, but this inventory does not prove that it caused the historical observation-count regressions. The returned range does not establish what may have happened before the earliest returned artifact.

Do not attribute historical count regressions to the hourly workflow without further evidence. Preserve this distinction in any incident or forensic report.

The canonical `.github/workflows/run_phase2d.yml` historical artifact collection is scoped to its own workflow and the `main` branch. Keep that ownership restriction.

## F08 regime-unit finding

The canonical candidate feature path stores `atr_pct` as a fraction (`atr / close`), while the legacy production path stores it in percentage points (`atr / close * 100`). The frozen regime cutoffs are in percentage points, and the gate classifies each model's `regime_atr_pct` using those cutoffs.

Treat this as an outcome-independent eligibility defect to be logged as an `F08_GATE_AMENDMENT` before C25, with the old/new code hashes and rationale. Preserve the already-recorded C15 and C20 checkpoint events and intervals. The post-hoc common-unit diagnostic still does not satisfy the two-regime coverage requirement at either checkpoint. Do not reinterpret those historical checkpoint results as passes.

If required coverage remains unmet at C30, apply the pre-registered inconclusive outcome and reject the candidate as a production replacement. No promotion is authorized by this amendment.

## Deferred architecture work

Moving auxiliary state synchronization from `main` to a dedicated `bot-state` branch is deliberately outside this workflow-isolation patch. It should be designed, tested and reviewed separately, including dashboard state-source behavior, before adoption.