# Phase 2D F08 evidence policy and stale-pending disposition

## Scope

This patch adds evidence enforcement around the existing Phase 2D statistical promotion gate. It does **not** edit `phase2d_promotion_gate.py`; that source remains unchanged because the recorded F08 gate-amendment provenance pins its contents. `phase2d_evidence_policy.py` validates the observations first, then delegates to the existing `evaluate_promotion()` implementation.

The only legacy compatibility exception is the exact 16,208-observation cohort recovered in workflow run `41720710702`:

| Model | Count | Canonical full-payload SHA-256 |
|---|---:|---|
| Candidate | 2,959 | `adaa9db710328bda2153a484f2bf9554e42e491ecbfe8e6becd1dfb2801ef888` |
| Production | 13,249 | `3a586eed79660ec32b716fd4c4362002b24f9d4feaacefabef7ae5b00fa7ea23` |
| **Total** | **16,208** | |

The source ledger SHA-256 is `4ba3daaa2e4f25eae895699e01a4cd1d0a4fd9e7b64a2632bf4fcac7d56274e9`; the recovered state SHA-256 is `4dd39ee1eefb0866dc5060719d69babf2b31f0d73ee814be5418322ac1d262e5`.

The legacy cutoff is `2026-10-16T14:08:36.946361+00:00`. The cohort fingerprint is computed from the complete observation payload, canonical JSON (`sort_keys=True`, compact separators, UTF-8), sorted by `observation_key`. The `evidence_class` key must be absent on these rows. A present-but-null value is not legacy.

## Evidence classes admitted to primary F08 scoring

- The exact fingerprinted legacy rows above are admitted as the fixed baseline.
- New rows must have `evidence_class="PROSPECTIVE"`, with both `created_at` and `resolved_at` strictly after the legacy cutoff.
- `HISTORICAL_BACKFILL` is rejected from the primary scoring population. It belongs only in the separate research bundle and an explicitly labelled sensitivity analysis.
- Missing, null, unknown or unsupported classes outside the exact legacy cohorts fail closed.

The shared `resolve_prediction()` function remains unchanged and neutral. `PROSPECTIVE` is assigned only at the official prospective observation-writer boundary, after the resolver has returned a result and before the row is stored in state/ledger.

## `STALE_PENDING_NOT_SCORED` manifest contract

The validator looks for a single approved manifest at:

`research_outputs/f08_historical_backfill/manifest.json`

The manifest is not created by this patch, and this patch does not append the event. The revised historical-bundle utility must produce the manifest and separate 618-record bundle after its own preflight and independent verification.

The manifest must include:

- `manifest_schema_version: 1`, `operation: "STALE_PENDING_NOT_SCORED"`, the F08 experiment ID, `evidence_class: "HISTORICAL_BACKFILL"`, and the exact legacy cutoff.
- A non-empty `reconciliation_id`.
- `source_hashes` for `state`, `ledger`, `journal`, `results`, `snapshot`, `proposal`, `raw_windows`, `dry_run`, `independent`, and `actual_resolver`. The state and original ledger hashes must equal the pinned values above. `raw_ohlc_windows_sha256` and `actual_resolver_report_sha256` must match the corresponding source hashes.
- `resolver_identity` matching the pinned review HEAD, validator blob, dependency blobs, and PR #14 merge ancestry. Include the exact PR #14 merge commit as `pr14_merge_commit`.
- `actual_resolver_verification` with `verification="PASS"`, type `actual_production_resolve_prediction`, 618 verified predictions (74 candidate, 544 production), zero mismatches, and before/after state and ledger hashes identical to the original pinned inputs.
- `historical_bundle` with `evidence_class="HISTORICAL_BACKFILL"`, total count 618, candidate count 74, production count 544, and a 64-character SHA-256 for the separately stored bundle.
- `pending_population.ids_by_model` containing the exact 74 candidate IDs and 544 production IDs, `pending_count: 618`, and `pending_ids_sha256` computed over canonical JSON of the sorted per-model ID lists.
- Owner approval is **not stored in the immutable manifest**. The manifest must remain `READY_FOR_OWNER_REVIEW` / `PENDING_OWNER_APPROVAL` and must not contain `approval_provenance` or a mutable approval-attestation file. Protected-environment approval is captured separately and bound into the official ledger event.

The official observation ledger event must be exactly one `STALE_PENDING_NOT_SCORED` event for this experiment. It must reference the exact manifest path and raw manifest SHA-256, reconciliation ID, pending count, per-model counts, pending-ID digest, and matching approval actor/reference/run ID.

The validator compares the approved manifest ID sets with the exact current stale pending IDs. Stale pending is identified by `created_at <= legacy cutoff`. It also compares the complete stale pending record payloads in the live state with the recovered source-state copy, not only the IDs. A missing stale cohort is a hard failure; it is never treated as `NOT_REQUIRED` for canonical F08. No row is resolved, deleted, removed from `pending`, or appended to the official observation ledger by the disposition consumer. Those IDs are returned as a skip set only.

## Expected first-run behaviour

For the canonical F08 experiment, `--allow-fresh-start` is prohibited. The validator checks the existing state, validates the full legacy evidence population, and consumes the disposition event before the ordinary state-migration path or market-data work. A missing state/ledger, invalid baseline, absent event, or event/manifest mismatch therefore fails before the official state or ledger can be initialized or modified by that run.

Until the separately verified manifest and owner-approved event exist, the current state has 618 stale pending predictions, so a manual validator invocation must fail closed before processing pending predictions. That is intentional. Do not re-enable or dispatch F08 while the event is absent.

After the approved event exists, each normal run will:

1. Verify the event, manifest hash, reconciliation ID, approval provenance, pending count and exact pending-ID sets.
2. Keep the approved 618 predictions in `state.pending` and skip them before market-data resolution.
3. Resolve only non-dispositioned pending predictions.
4. Tag newly resolved official observations as `PROSPECTIVE` at the writer boundary.
5. Validate the complete official observation population before summary statistics and gate evaluation.

## Tests and execution safety

Focused tests cover exact legacy cohort matching, rejection of arbitrary untagged records, rejection of `HISTORICAL_BACKFILL`, prospective timestamp requirements, manifest/event mismatch cases, exact pending-ID reconciliation, read-only behaviour, and resolver neutrality.

This patch does not create a disposition event, run the old reconciliation utility with `--apply`, modify the official state or ledger, evaluate C25, change C15/C20, enable the F08 workflow, promote a model, or enable live trading.

## Byte-verifiable historical-backfill artifacts

The manifest is accepted only when the ten named source artifacts are present as byte-for-byte evidence copies under `evidence/` and each copied file's calculated SHA-256 equals both its `source_artifacts[name].sha256` and `source_hashes[name]`. The state and original observation-ledger source hashes remain pinned to the recovered run-41720710702 baseline. The copied state is re-read and its 618 stale pending IDs are compared with the current state; the copied source ledger must still contain exactly 16,208 observations and three events, with no disposition event.

The manifest must contain exactly the ten required `source_hashes` and `source_artifacts` entries, and must name `historical_bundle.bundle_path` as `observations.jsonl`. The validator reads the file, calculates its actual byte SHA-256, recomputes the canonical result-row digest and deterministic reconciliation ID, parses every JSONL row, and verifies exactly 618 unique IDs (74 candidate, 544 production). Each row must be `HISTORICAL_BACKFILL`, refer to the exact experiment and reconciliation ID, match the copied source prediction and proposal outcome, and be deterministically sorted by model/ID. The proposal, actual-resolver report, journal, raw OHLC windows, results, snapshot, dry-run report, and independent report are all checked from their evidence copies, not merely accepted as hash-shaped strings.

`resolved_at` is omitted from canonical backfill rows. It is a resolver-process wall-clock field and changes on each execution, while the outcome-bearing fields are compared against the independently hashed proposal and actual resolver report. The report itself is copied and SHA-256 verified. This omission makes bundle bytes, reconciliation ID, and manifest deterministic for identical inputs. The manifest explicitly documents this normalization.

## Safe bundle builder (no official mutations)

`phase2d_f08_historical_backfill.py` replaces the old reconciliation utility's output design for this stage. It has read-only preflight by default; `--write-bundle` creates only:

- `research_outputs/f08_historical_backfill/observations.jsonl`
- `research_outputs/f08_historical_backfill/manifest.json`
- `research_outputs/f08_historical_backfill/evidence/` with the ten immutable evidence copies

It does not write a projected state, copy or append to the official observation ledger, change pending predictions, append any event, alter the experiment journal/results/snapshot, or evaluate the promotion gate. The first manifest is deliberately marked `READY_FOR_OWNER_REVIEW` / `PENDING_OWNER_APPROVAL` and cannot be consumed by the F08 validator.

Repeated runs with identical inputs return `ALREADY_PRESENT_IDENTICAL` after comparing every output file hash. If an output directory already exists with different bytes, the utility refuses to overwrite it. The output path is fixed to `<repo_root>/research_outputs/f08_historical_backfill`; arbitrary destinations are rejected.

## Protected-environment approval and disposition-event trust boundary

The historical-backfill manifest is immutable after bundle creation. It must
remain `READY_FOR_OWNER_REVIEW` / `PENDING_OWNER_APPROVAL`. It must not be
rewritten to `APPROVED_FOR_DISPOSITION` / `APPROVED`, and it must not contain
an `approval_provenance` object or an `approval_workflow_context.json` file.
The bundle hashes and evidence copies are reviewed as fixed inputs; approval is
captured separately and bound into the corresponding official event.

Approval is a two-stage, protected GitHub-environment process implemented by
the two checked-in workflows:

1. `Phase2D F08 Validator Amendment` runs by `workflow_dispatch` on
   `refs/heads/main`, with an initiator other than the designated owner. Its job
   uses the `f08-validator-amendment` environment. The workflow collects the
   actual environment configuration and workflow-run approval history through
   the GitHub API using `GITHUB_TOKEN`, then passes the resulting
   `validator_amendment_approval_evidence.json` to
   `phase2d_f08_disposition.py append-validator-amendment`. The command verifies
   the evidence and appends exactly one `F08_VALIDATOR_AMENDMENT` event to the
   working experiment journal. The event carries `owner_approval`, including
   the normalized environment/run/reviewer fields and the exact API evidence
   snapshot. State and the official observation ledger remain unchanged in
   this stage. The workflow publishes the approved source handoff artifact.

2. `Phase2D F08 Controlled Disposition` runs by a separate
   `workflow_dispatch` on the same `main` commit, again with a non-owner
   initiator and the successful first-stage run ID. It verifies that the prior
   run used the same source SHA, succeeded, was its first attempt, and came from
   the pinned repository/workflow. Its job uses the `f08-disposition`
   environment. The workflow collects the actual environment configuration
   and approval-history responses into
   `working/disposition_approval_evidence.json`. The
   `apply-disposition` command validates this evidence and appends exactly one
   `STALE_PENDING_NOT_SCORED` event to the working copy of the official
   observation ledger. The event carries `approval_provenance`, including the
   environment and workflow identity, repository, initiator and reviewer IDs,
   run ID/attempt, source head SHA, event/ref, hashes of the environment and
   approval-history responses, and the complete captured API evidence
   snapshot. The event does not append any TP/SL/EXPIRED observation rows.

The read-only validator and disposition consumer require the manifest to stay
in its draft statuses and reject `approval_provenance` inside the manifest.
They verify the approval snapshot embedded in the official event against the
pinned environment, workflow, repository, main ref, first-attempt, designated
reviewer, protected-branch policy, and self-review-prevention requirements.
The Python code validates the captured API response contents; the workflow
collector is the component that obtains those responses from GitHub using the
run's token. This is not a separately hand-authored
`approval_workflow_context.json` attestation and it is not a standalone
`Phase2D F08 Disposition Approval` workflow.

Each environment must have exactly one designated required reviewer:
`MrZkull` (numeric GitHub user ID `166250462`), with **Prevent self-review**
enabled and deployment restricted to protected branches. `main` must be
protected before either environment can deploy. The workflow initiator must
not be the designated reviewer, and only run attempt 1 is accepted. These
settings must be verified in GitHub before the workflows are enabled.

Do not rewrite the immutable manifest, hand-author approval data, append the
disposition event manually, or bypass either protected environment. Until the
source artifacts, owner review, both environment configurations, and both
workflow approvals are independently verified, the manifest remains a draft,
the disposition event remains absent, and F08 must remain disabled.

## Per-prediction historical resolution provenance (schema 1)

The 618-row `HISTORICAL_BACKFILL` observation bundle remains deterministic and
does not contain the volatile resolver `resolved_at` timestamp. The bundle
now carries a separate, SHA-256-bound `resolution_provenance.jsonl` sidecar
with exactly one record for every `(model, prediction_id)` in the outcome
bundle. The sidecar preserves the exact UTC `resolved_at` string returned by
the pinned resolver invocation. This is the resolver execution timestamp; it
is not the historical market outcome time.

The sidecar separately records `outcome_bar_open_time_ms`,
`outcome_bar_close_time_ms`, and `outcome_bar_close_time`. These are derived
from the selected outcome candle in the byte-hashed raw 15-minute OHLC window
using the pinned interval convention: candle open timestamp plus 900,000 ms
(the 15-minute interval close boundary). `raw_window_sha256` binds the complete
canonical raw window for that prediction. The verifier requires an exact
one-to-one mapping with the 618 outcome rows, checks the model-specific
74/544 counts, verifies the raw-window digest and outcome bar, and recomputes
the close timestamp from the raw candle sequence. Missing, duplicate, extra,
unsorted, altered, or non-UTC provenance records fail closed.

The sidecar SHA-256 and its raw-window SHA-256 are recorded in the immutable
bundle manifest. Resolver timestamps remain outside the deterministic
observation-row hash and reconciliation ID; changing a resolver execution
timestamp changes the sidecar/manifest bytes and therefore cannot be treated
as an identical existing bundle. This is a historical research artifact only:
these 618 outcomes remain excluded from primary prospective scoring and are
not appended as TP/SL/EXPIRED observations to the official ledger.
