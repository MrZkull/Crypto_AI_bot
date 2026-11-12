# F08 Schema-7 Not-Scored Disposition — Release Gates

**Status: implementation candidate only; not authorized for execution.**

This document describes the isolated schema-7 implementation and the conditions that must be met before either protected workflow may be run. No approval, official ledger append, state migration, C25 evaluation, F08 re-enable, promotion, or live trading authorization is implied.

## Frozen source and data bindings

- Repository: `MrZkull/Crypto_AI_bot`
- Candidate's original source-ancestry/audit base: `ff821b60dbade382431149e86ff9b5ce13a3243f`
- Original schema-7 source worktree base: `944744f632da260a8d874d7faacc84820164a30a`
- Historical integration source branch: `integration/f08-schema7-disposition-main-20261110` (integration snapshot based on `7f294c2e98dddb99d3ba60d8056b6dc59d559101`).
- Release-candidate branch: `feature/f08-schema7-release-candidate-20261111`.
- Release-candidate base: `dbbfd478bc3dcc57494d7a7eaf65e7986748a59e` (`origin/main` fetched on 2026-11-11).
- Initial release-candidate commit: `e1507a4b7bb9233d8e35d4ef9c37142839b18412`; review is tracked in draft PR [#15](https://github.com/MrZkull/Crypto_AI_bot/pull/15).
- Do not silently rebase the candidate or overwrite the six runtime/research files identified during integration.
- Target branch for the controlled workflow: `main`
- Experiment: `PHASE2D-HARDENED-20260924-F08`
- Authoritative source run: `41720710702`
- Authoritative source artifact ID: `13272461237` (`phase2d-state`)
- Source state schema: 6
- Source observations: 16,208 (candidate 2,959; production 13,249)
- Pending cohort: 618 (candidate 74; production 544)
- Reconciliation ID: `F08-HIST-BF-c8a8330915c85db8fe923615e2df2a58`. It remains the deterministic ID produced from source hashes, resolver identity, pending-ID digest, and normalized outcome-row digest; resolver timestamps and candle-time provenance are held in the separately hashed sidecar and do not change deterministic outcome-row identity.
- Superseded pre-provenance manifest SHA256: `a32917de06548c3d4d0bcc6ec09b7c6974c00235dc90ef86db0df057b836c691`.
- Current provenance-inclusive manifest SHA256: `1aae5f75292241298977378b81ac6297e24c3270e1afb5957776a1f8b1767ab3`.
- Deterministic `observations.jsonl` SHA256: `3f05c466810fc86a911d4105575c951a59a73e78a3b6271c9c01cb0042386878` (the normalized observation rows remain byte-identical).
- Separate `resolution_provenance.jsonl` SHA256: `d679494c5d58292f12227b95492ec3d47daf4b5b5f2a38b21f962ca465b994c8` (618 records; exact resolver timestamps and independently derived outcome-candle times).
- Pending cohort ID digest: `a1751b84e4fedd5dad40348392bac0ccf8c1be00a419118d152ee561a9d1c644`
- Frozen promotion gate canonical SHA256: `f787bc3caa810c9f82f56b44d2d8624637693e47eb4613e86a4c30f461c2faab`

The immutable backfill manifest must remain `READY_FOR_OWNER_REVIEW` / `PENDING_OWNER_APPROVAL`. The approval records are distinct attestations and must never be inserted into or used to rewrite the manifest.

## Required state transition

The only approved semantic transition is:

1. Preserve the exact schema-6 source state, observation ledger, and experiment journal as byte-identical baseline files.
2. Append exactly one `F08_VALIDATOR_AMENDMENT` event to a working copy of the experiment journal after its own independently approved protected-environment gate. Do not change the state or official observation ledger in this stage.
3. Append exactly one `STALE_PENDING_NOT_SCORED` event to a working copy of the official observation ledger after a second independently approved protected-environment gate. This event must report `outcome_rows_appended: 0` and bind the reconciliation ID, source heads, manifest/bundle hashes, pending-ID digest, amendment event ID/hash, validator hash, and frozen promotion-gate hash.
4. Atomically migrate schema 6 to schema 7, moving the exact 74 candidate and 544 production payloads from `pending` to `dispositioned`. Preserve each original payload, the rest of `pending`, `resolved`, and `seen`, and bind each item to the event ID/hash.
5. Verify the event-first / state-save-second retry. If the state save is interrupted after the event append, a later attempt must accept only that exact event and finish the same deterministic migration.

The 618 historical TP/SL/EXPIRED records remain in the separate `HISTORICAL_BACKFILL` evidence bundle. They must never be appended as official observation rows or scored into the primary F08 cohort.

### Per-prediction resolution provenance requirement

The immutable bundle must include a separately hashed `resolution_provenance.jsonl` sidecar with exactly 618 records, one per `(model, prediction_id)`. The sidecar must preserve the exact UTC `resolved_at` string returned by the pinned production resolver and independently derive the historical outcome candle's open/close timestamps from the byte-hashed raw 15-minute OHLC window. These are separate concepts: resolver execution time is not market outcome time. The close boundary follows the pinned Deribit fetcher convention (`selected_candle_open_time + 900000 ms`). Every sidecar record is bound to the full raw-window canonical SHA256. Independent verification must prove the 74 candidate / 544 production mapping is complete, unique, sorted, and matches each bundle row's status and `outcome_bar`; a re-hashed but semantically altered sidecar must fail. The deterministic `observations.jsonl` rows must continue to omit `resolved_at`, and the sidecar must not be imported into the official observation ledger.

## Protected GitHub environments

The manifest remains `READY_FOR_OWNER_REVIEW` / `PENDING_OWNER_APPROVAL` and is
never rewritten to store an approval. The two environment approvals are
captured separately by `tools/collect_f08_approval_evidence.py`: first for the
`F08_VALIDATOR_AMENDMENT` journal event, then for the
`STALE_PENDING_NOT_SCORED` official observation-ledger event. The full captured
GitHub environment and approval-history evidence snapshot is embedded in the
corresponding event as `owner_approval` or `approval_provenance`; it is not
stored in the immutable manifest. See the approval trust-boundary section in
`docs/phase2d_evidence_policy.md`.

Create these environments in repository Settings → Environments:

- `f08-validator-amendment`
- `f08-disposition`

Each environment must have exactly one designated required reviewer: the repository owner account `MrZkull` (numeric user ID `166250462`). Enable **Prevent self-review**. The environment deployment branch policy must be **Protected branches only** (`protected_branches: true`, `custom_branch_policies: false`). Do not add another reviewer, team, custom branch policy, or alternate approval path.

The workflow initiator must be a different GitHub user from the owner. Only the first attempt of a newly dispatched workflow run is accepted; reruns are rejected so the triggering actor context cannot be confused with the original initiator. The code independently reads the environment configuration and the approval history for the exact workflow-run ID, then verifies the reviewer account ID and login. It does not trust a workflow input such as `approved-by` or the initiating actor's display name.

Because the environment requires protected branches, `main` must first have a GitHub branch protection rule/ruleset. The repository metadata read during implementation review reported `main` as unprotected. The current GitHub connection has repository push permission but not repository administration permission, so the branch protection rule and environment configuration must be applied by an authorized repository administrator in GitHub Settings. The two workflows are intentionally not runnable until those settings are correct.

The workflow token requires repository `Actions: read` to read environment configuration and run approval history. The collector stores the exact JSON responses and never writes the token. No separate personal access token secret is required. See the GitHub REST documentation:

- Get an environment: https://docs.github.com/en/rest/deployments/environments#get-an-environment
- Get review history for a workflow run: https://docs.github.com/en/rest/actions/workflow-runs#get-the-review-history-for-a-workflow-run

## Workflow sequence

1. Run `Phase2D F08 Validator Amendment` from `main`, with a non-owner initiator. The job must wait for the required reviewer. It verifies the pinned source artifact ID and full historical-backfill evidence package, then appends only the validator-amendment event to an artifact copy of the baseline journal. Its output artifact is an immutable handoff, not an official source-state mutation.
2. After the first workflow succeeds, run `Phase2D F08 Controlled Disposition` from the same `main` commit, again with a non-owner initiator and a separate protected-environment approval. Supply the successful first workflow run ID. It verifies the handoff, applies the not-scored disposition to an artifact working copy, seals a V2 artifact, and uploads it as `f08-schema7-disposition-quarantined`.
3. If event append succeeds but state save is interrupted and the checkpoint artifact was uploaded, provide that failed attempt's run ID as `resume_attempt_run_id`. The retry still verifies the same baseline, exact existing event, and source bindings before completing migration.

The attempt checkpoint is an artifact for recovery. It is not a new source-of-truth and must not be manually edited.

## V1 immutability and V2 quarantine

The V1 restore guard, V1 snapshot restore, and V1 archive restore remain frozen. The `main` branch remains the production authority; this source-review candidate does not change `main` or authorize a workflow dispatch.

The V2 restore module creates and verifies a separate manifest with `artifact_status: QUARANTINED_PENDING_PIPELINE_HANDOFF`. It verifies the exact baseline heads, the one-event observation-ledger extension, the one-event journal extension, the entire migrated state against the deterministic migration, event and approval evidence, the preserved C15/C20 history, and the absence of C25-or-later evaluations. Its restore command is a verify-before-write operation and refuses to overwrite a non-empty destination.

The candidate branch now contains a separate V2 bootstrap and continuation handoff in `.github/workflows/run_phase2d.yml`, including an immutable Release bootstrap and a dedicated `phase2d-state-f08` continuation artifact referenced by a `bot-state` pointer. This is an implementation candidate only. It has passed local regression and workflow lint in the candidate worktree, but it has not been merged, dispatched, or validated against the live GitHub environment configuration. The handoff code must receive independent source review and the exact source-commit check before a Release bootstrap is eligible.

**Do not enable F08 or dispatch the candidate handoff yet.** The production `main` workflow remains unchanged until a reviewed PR is approved and merged. After merge, the controlled amendment and disposition approvals, immutable Release publication, exact-commit bootstrap, GitHub environment protections, and runtime-state verification remain separate mandatory gates. None is implied by local tests.

### Continuation artifact retention and recovery

The candidate currently sets the dedicated `phase2d-state-f08` GitHub Actions artifact retention to 90 days. The `bot-state` pointer identifies that artifact by run and artifact ID. If the artifact expires or is deleted before a successor pointer is published, continuation deliberately fails closed. The current candidate does not provide an automatic recovery path for an expired continuation artifact: the one-time bootstrap refuses to replace an existing pointer, and the immutable initial Release is not a substitute for later prospective state. Before relying on unattended operation across a period in which no successful continuation is published, the owner must approve and test a durable recovery/checkpoint procedure. This is an operational-readiness blocker, not a reason to weaken the pointer or silently restore an older run.

## Mandatory pre-run checks

- [ ] Confirm the pinned artifact ID `13272461237` is still available and not expired.
- [ ] Make the exact immutable backfill manifest, bundle, provenance sidecar, and all referenced `evidence/` files available under `research_outputs/f08_historical_backfill/` on the approved `main` commit. The complete package is included in draft PR #15 but is not yet on `main`; neither protected workflow may run before the reviewed package is present on the exact approved `main` commit and all hashes pass.
- [ ] Confirm `resolution_provenance.jsonl` is included and its manifest SHA256, 618-record count, raw-window SHA256, one-to-one ID mapping, exact resolver timestamps, and candle-derived outcome close times all verify independently.
- [ ] Confirm the main branch is protected by a branch protection rule/ruleset.
- [ ] Configure both environments with only the designated owner as required reviewer, prevent self-review enabled, and protected-branches-only deployment policy.
- [ ] Ensure the workflow initiator is not the owner/reviewer.
- [ ] Run unit tests, full regression, workflow YAML/action lint, source hash checks, and V2 seal/verify tests against the exact commit that will run.
- [ ] Independently inspect the source commit and workflow changes before approving either environment.

## Explicit non-goals and prohibitions

- Do not score the 618 stale pending predictions.
- Do not append 618 observation rows or any stale TP/SL/EXPIRED rows to the official ledger.
- Do not alter the historical-backfill manifest to embed approval.
- Do not append a second `F08_GATE_AMENDMENT`.
- Do not rewrite historical observations or re-evaluate prior decisions.
- Do not evaluate C25 during amendment or disposition.
- Do not re-enable F08, authorize promotion, switch to FULL/live execution, or modify the trading kill switch as part of this work.
- Do not reset, clean, merge, rebase, or otherwise mutate the user's main project or recovery worktrees as part of this implementation handoff.

## Differential audit checkpoint — 2026-11-08

The pinned-vs-candidate persisted-ledger and deterministic-fixture differential
is PASS. The full result and its source hashes are recorded in
`docs/F08_VALIDATOR_DIFFERENTIAL_AUDIT_20261108.md` and
`research_outputs/f08_validator_differential_audit.json`.

The full historical scoring replay remains BLOCKED until the original candidate
and production model binaries and exact historical 15m/1h candle windows are
recovered and hash-verified. Synthetic fixtures and the persisted observation
ledger do not replace that requirement. No official F08 state or ledger change
is authorized by this audit.
