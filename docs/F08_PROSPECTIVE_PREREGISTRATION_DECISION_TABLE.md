# F08 Prospective Evaluation Pre-Registration and Fixed Decision Table

**Policy ID:** `F08-PRE-REGISTRATION-2026-11-11-v1`
**Status:** Prepared for commit and merge before any prospective F08 restart. It is not effective merely because it exists in a local worktree.
**Scope:** Prospective Phase 2D/F08 evaluation only. This document records the existing frozen promotion-gate policy; it does not amend or modify that gate.

## 1. Frozen source and policy identity

The policy is defined by the frozen `phase2d_promotion_gate.py` source:

- Canonical LF SHA-256: `f787bc3caa810c9f82f56b44d2d8624637693e47eb4613e86a4c30f461c2faab`
- Integration base used for this record: `7f294c2e98dddb99d3ba60d8056b6dc59d559101`
- The promotion-gate source is not part of the schema-7 changes and must remain byte-for-byte/canonical-hash identical.

The canonical LF SHA-256 of this pre-registration document must be recorded in the final integration manifest and the approved F08 release evidence. The document and its hash must be committed to `main` before any prospective F08 restart or protected F08 workflow dispatch.

## 2. Primary estimand and analysis unit

1. Include only eligible prospective, economically resolved observations in the primary F08 evaluation.
2. Key blocks by UTC calendar day derived from prediction `open_time`.
3. Calculate one daily mean Net-R per model for each UTC day.
4. Pair candidate and production daily means on common UTC days. The comparison is candidate minus production on those paired daily blocks.
5. The bootstrap unit is the paired UTC-day block. Do not treat individual predictions within one day as independent bootstrap units.
6. Use the same locked eligibility, resolution, cost, and evidence rules throughout the experiment. A checkpoint is evaluated once and is never re-evaluated.

The 618 stale pending predictions described by the outcome-blind cohort report are not members of this prospective primary population. They remain a separate `HISTORICAL_BACKFILL` research artifact and must not be appended as historical TP/SL/EXPIRED observations to the official observation ledger or included in primary prospective scoring.

## 3. Fixed statistical design

| Parameter | Pre-registered value |
|---|---:|
| Minimum resolved observations, candidate | 50 |
| Minimum resolved observations, production | 50 |
| Minimum unique UTC daily blocks, candidate | 15 |
| Minimum unique UTC daily blocks, production | 15 |
| Minimum common paired UTC daily blocks | 15 |
| Fixed checkpoint looks | C15, C20, C25, C30 |
| Checkpoint paired-block counts | 15, 20, 25, 30 |
| Bootstrap method | Paired UTC-day block bootstrap of daily-mean Net-R |
| Bootstrap rounds | 10,000 |
| Bootstrap seed | 42 |
| Total family-wise alpha budget | 0.05 |
| Alpha per fixed look | 0.0125 |
| Two-sided confidence interval at each look | 98.75% |
| Maximum experiment age | 90 calendar days |

The four fixed looks use equal alpha spending (`0.05 / 4 = 0.0125` per look), with the Bonferroni/union-bound family-wise alpha ceiling of 0.05. The 98.75% two-sided interval at each look corresponds to that per-look alpha.

The gate must not add looks, change the seed, change the number of bootstrap rounds, change the block definition, or re-evaluate a checkpoint after it has been persisted.

## 4. Coverage gates

Coverage is conjunctive: every requirement below must pass.

### 4.1 Calendar coverage in the final 15 paired UTC-day blocks

The gate uses the final 15 common paired UTC-day blocks for this coverage assessment:

- At least **3 weekend blocks**.
- At least **10 weekday blocks**.

These are counts of paired UTC calendar-day blocks, not counts of individual predictions.

### 4.2 Frozen volatility-regime coverage

A frozen `phase2d_regime_reference.json` for the same experiment must exist, validate, and be ready. The gate does not derive or invent new cutoffs from the prospective test period.

Using that frozen reference, candidate and production are assessed separately over the same final 15 paired UTC-day blocks. For **each model**:

- The LOW, MID, and HIGH ATR-volatility terciles are evaluated using the frozen cutoffs.
- At least **2 of the 3 terciles** must be represented.
- A tercile counts as represented only when it contains at least `ceil(0.15 × 15) = 3` daily observations in that model's final-15-day coverage slice.

Candidate ATR values are normalized to the reference's percentage-point convention for this temporary coverage comparison only; source observations and model features are not rewritten.

If the frozen regime reference is absent, invalid, belongs to a different experiment, or either model does not represent at least two qualifying terciles, regime coverage is not ready and the overall coverage gate fails closed.

## 5. Efficacy, harm, and checkpoint decision table

At each fixed look, use the gate's 98.75% two-sided confidence intervals:

- **Candidate profitability:** lower confidence bound of candidate daily mean Net-R must be strictly greater than `0.0`.
- **Candidate superiority:** lower confidence bound of candidate-minus-production daily mean Net-R must be strictly greater than `0.0`.
- Both conditions are required. The production stream's own profitability lower bound is not an additional promotion condition in this frozen gate.
- **Coverage:** all calendar and regime coverage requirements in Section 4 must be ready.

| Priority | Condition at an evaluated checkpoint | Required decision |
|---:|---|---|
| 1 | Upper confidence bound of candidate-minus-production is strictly below `0.0` | `EARLY_HARM_STOP`; fail and stop immediately |
| 2 | Candidate profitability lower bound `> 0.0`, candidate-minus-production lower bound `> 0.0`, and coverage is ready | `EFFICACY_PASS`; record the passing checkpoint and stop evaluation |
| 3 | Neither harm nor efficacy is established at C15, C20, or C25 | `CHECKPOINT_CONTINUE`; continue only to the next already registered look |
| 4 | C30 is evaluated and neither early harm nor all efficacy/coverage conditions pass | `FINAL_CHECKPOINT_NO_EFFICACY`; fail and stop |

The ordering is intentional: early harm is evaluated before efficacy. A positive candidate result without superiority is not a pass. Superiority without candidate profitability is not a pass. Positive statistical results without the required coverage are not a pass.

### 5.1 C30 is terminal

C30 is the final and terminal efficacy look.

- An inconclusive C30 result is **rejection/no promotion**, recorded as `FINAL_CHECKPOINT_NO_EFFICACY`.
- There is no C35, C40, ad hoc extension, extra bootstrap look, seed change, reset, or retrospective checkpoint re-evaluation to obtain a pass.
- Once C30 has been recorded as `FINAL_CHECKPOINT_NO_EFFICACY`, the experiment is stopped.
- A previously recorded checkpoint is never evaluated again.

### 5.2 Eligibility shortfall and hard calendar ceiling

Before a statistical look can be evaluated, the gate requires at least 50 resolved observations in each model, at least 15 unique daily blocks in each model, and at least 15 paired common blocks. Until these eligibility conditions are met, the gate returns `NOT_READY`; that is not an efficacy pass.

The experiment has a hard ceiling of **90 calendar days**. If no qualifying efficacy decision has been reached by that ceiling, the gate fails and stops the experiment. The 90-day ceiling is not permission to extend beyond C30 or to create additional statistical looks.

## 6. Historical cohort disclosure and exclusion firewall

The 618 stale pending outcomes had already been resolved/observed before this disposition and exclusion design was finalized. This timing must be disclosed in any research presentation or mentor report.

The 618 records are one historical contiguous slice, not 618 independent observations. The outcome-blind cohort reconciliation identifies:

- 618 prediction records: 74 candidate and 544 production.
- 573 unique symbol/open-time windows.
- 45 common candidate/production windows; all 45 are opposite-side pairs.
- 528 unpaired windows.

The detailed outcome-blind cohort report is:

`docs/F08_COHORT_BREAKDOWN_OUTCOME_BLIND_20261111.json`

Its recorded SHA-256 is:

`3aae700aa5bf4fea57d71f8d8db611eac22d43fececb4b9d6cc27b5f98a7061b`

The cohort report was produced from the exact recovered state and observation-ledger source hashes and did not read outcome fields. The exclusion rule is based on the pre-existing stale-pending population and is applied irrespective of the eventual TP/SL/EXPIRED labels. The 618 historical results must remain outside the primary prospective score and outside the official outcome ledger.

## 7. Freeze and change control

Before resuming F08:

1. Commit this decision table to `main` through the reviewed integration PR.
2. Record this document's canonical LF SHA-256 in the final integration manifest and approved release evidence.
3. Verify the frozen promotion-gate SHA-256 in Section 1.
4. Verify the frozen regime-reference artifact and its SHA-256 for the exact experiment.
5. Confirm the prospective eligibility and cohort exclusion firewalls.
6. Confirm the checkpoint history contains no conflicting C25/C30 event and no duplicate checkpoint evaluation.
7. Obtain the required protected-environment approvals through the two distinct F08 approval workflows.

Any proposed change to the statistical method, checkpoint schedule, alpha spending, confidence interval, coverage, stopping rule, or final-decision rule requires a new pre-registration and mentor approval **before** the affected prospective evidence is collected. It must not be changed mid-experiment.

This document records the policy. It does not approve a workflow dispatch, official ledger mutation, F08 restart, model promotion, or live trading.

