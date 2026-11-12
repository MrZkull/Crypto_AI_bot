# F08 Validator Differential Audit — 8 November 2026

## Result

**PASS for the available differential scope. Full historical model replay remains blocked.**

The pinned F08 validator and the Schema 7 candidate were run in separate temporary workspaces against identical copies of the authoritative schema-6 state, observation ledger, experiment journal, feature source, and promotion gate. The candidate and pinned outputs were byte-for-byte equivalent after canonical JSON serialization.

This is not approval to perform the F08 disposition, append the amendment/disposition events, migrate the official state, or restart prospective execution.

## Immutable inputs verified

The authoritative artifact used was workflow run `41720710702`, artifact `phase2d-state`, archive SHA256:

`a7d8224651e036aa4ee3c3626fe296b81f6bc35c91e20074c382e7628141ebac`

| Input | SHA256 |
|---|---|
| `phase2d_state.json` | `4dd39ee1eefb0866dc5060719d69babf2b31f0d73ee814be5418322ac1d262e5` |
| `phase2d_observations.jsonl` | `4ba3daaa2e4f25eae895699e01a4cd1d0a4fd9e7b64a2632bf4fcac7d56274e9` |
| `experiment_ledger.jsonl` | `cf12ac74b39e44d8487951bb1abde203f12140a626d7c78fc388ebd91e80e267` |

The authoritative feature source used in both runs matches the recorded feature-code SHA256 `f60a2508e19892baeeead20d4aa41d8210f559c827affd5cd50ff541973d4683`. The frozen promotion gate canonical SHA256 remains `f787bc3caa810c9f82f56b44d2d8624637693e47eb4613e86a4c30f461c2faab`.

The separate pinned resolver copy had a different feature-file representation, so it was deliberately **not** used as the feature source. Both validator processes imported the exact feature file whose hash is recorded in the authoritative F08 run. This prevents a source-copy/line-ending mismatch from contaminating the comparison.

## Differential checks passed

1. Schema-6 state integrity passed under both validator versions.
2. Candidate and production summaries over all authoritative observation rows matched exactly between versions and matched the persisted `last_run` summaries.
3. Model-scoring API output matched exactly using the same deterministic fixed-probability model fixture.
4. Eight outcome fixtures matched: BUY and SELL, each covering TP, SL, AMBIGUOUS, and EXPIRED.
5. Current-feature and legacy-production feature frames matched by row count, column order, dtype sequence, and complete-frame digest.
6. Frozen promotion-gate output matched exactly between the two runs and the persisted F08 result fields checked by the harness.
7. The promotion gate remained `NOT_READY`, with 2,948 candidate and 13,215 production gate-eligible observations across 23 paired UTC day-blocks. Only C15 and C20 are recorded as evaluated. No C25 checkpoint was evaluated.
8. The copied experiment journal was byte-identical before and after gate evaluation. No new checkpoint event was appended.

The full machine-readable result is `research_outputs/f08_validator_differential_audit.json`.

## Important source changes in the candidate

The following core routines are AST-identical between the pinned and candidate validator: feature construction (`current_features`, `build_production_features`, `_legacy_production_indicators`, `_align_htf_point_in_time`, `_add_btc_cross_features`, `aggregate_1h_to_4h`), model feature selection/scoring (`model_features`, `score_model`), prediction resolution (`resolve_prediction`), summary calculations (`_side_summary`, `summarize`), and the remaining unchanged shared calculation helpers.

Intentional validator changes are concentrated in:

- `validate_state_integrity`: accepts the controlled schema-7 structure only with the additional F08 disposition integrity checks.
- `load_compatible_state`: refuses ordinary schema-6 F08 execution and requires an independently verified amendment for schema 7.
- `append_resolved_unique`: prevents a dispositioned prediction from being resolved a second time.
- `build_experiment_identity` / hash helpers: use canonical source hashing for validator identity.
- `main`: adds the F08 fail-closed execution and evidence-policy preflight.
- New historical range-fetch and prediction-window helpers: support complete bounded historical resolution; these are additive and were not invoked as a substitute for missing historical source data in this audit.

Pinned validator canonical SHA256: `2b9f16f87f8222fbb00cfd0e92a0f149dfefdd107d0bfa3079a9a97040e299c5`.

Candidate Schema 7 validator canonical SHA256: `50ebe8079674c06648832b7fd0c85632e33e1963efffd826de815b69c191d371`.

The prior candidate validator hash `6dc8d71ea81732e9b62ddabce34574e3be39bcd0b1f11f0bd555c6a9c537cd3d` is **not** the identity of this current candidate and must not be reused in an amendment.

## Model binaries recovered by immutable release identity

GitHub release metadata for the original repository lists both model assets with SHA256 digests matching the model locks in the authoritative F08 state:

| Model | Release | Asset size | SHA256 |
|---|---|---:|---|
| Candidate | [`phase2d-htf-removal-candidate-20260921`](https://github.com/MrZkull/Crypto_AI_bot/releases/tag/phase2d-htf-removal-candidate-20260921) | 5,831,177 bytes | `7213987874543e5d6e18b0bac32556647d40080b3e4222532e40a225a1bf08a3` |
| Production | [`v2.0`](https://github.com/MrZkull/Crypto_AI_bot/releases/tag/v2.0) | 240,931,315 bytes | `f55b887c7f624179b3d9fee56d792c29424a589e3d8734edcd78ce1be71f2c21` |

The release metadata digest matches the locked model SHA256 for each asset. The binary bytes were not materialized and independently hashed in this audit runtime. The bundle therefore includes `tools/download_f08_model_assets.ps1`, which downloads the exact release assets to `research_inputs/f08_model_replay/`, verifies each SHA256, and refuses to overwrite an existing file whose hash does not match. This script does not unpickle or execute either model.

## Remaining hard blocker: exact historical candle windows

The authoritative archive contains the state, observation ledger, experiment journal, results, latest snapshot, and validator log. It does not contain the exact raw 15-minute and 1-hour candle windows used by the original run. A local search of the available mounted workspace also found no copies of those raw candle datasets.

Therefore this audit proves equivalence of the preserved calculation code and the outputs that can be independently recomputed from the persisted ledger, plus deterministic synthetic candle/scoring regression fixtures. It does **not** prove that a full raw-market-data replay of the original model scores is equivalent.

Do not close that blocker with newly fetched candles, reconstructed historical windows, synthetic market data, or a backfill result generated after the original run. The exact original candles must be recovered from their original immutable artifact/run source and hash-verified. The model downloader's successful local SHA verification is a prerequisite, but it does not replace the missing candle inputs.

## Test status and next gate

- Full candidate test suite: **158 passed**.
- Python compilation check: passed for the modified validator, disposition, V2 restore, historical backfill, approval collector, and differential audit runner.
- No main repository, local project worktree, pinned resolver worktree, authoritative recovery state, official state file, or official ledger was modified by this audit.
- No owner approval was collected or consumed.
- No official amendment/disposition event was appended.
- No schema migration, C25 evaluation, F08 restart, promotion, or trading-mode change occurred.

Next required step: recover and verify the exact model artifacts and historical candle windows, then run a full deterministic old/new validator replay using those immutable inputs. After that, review the workflow approval/resume edge cases and complete the separate GitHub protected-environment/admin configuration checks. Only then can the project be considered for an independently approved controlled F08 transition.
