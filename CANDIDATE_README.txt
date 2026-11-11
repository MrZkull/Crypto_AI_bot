Crypto_AI_bot — F08 Schema 7 Disposition Candidate
Checkpoint: 2026-11-08

PURPOSE
This is an isolated implementation candidate for the F08 controlled schema-7
stale-pending disposition. It is not merged into the user's main project and
is not authorized for production use or official state mutation.

LATEST VALIDATION
- Candidate source-only suite, independently rerun before this workflow-hardening patch: 157 passed, 1 skipped (the skip requires the authoritative recovery artifact to be mounted).
- Isolated Windows integration full regression after schema-5 contract-test alignment and before this workflow-hardening patch: 367 passed, 5 skipped, and 3 subtests passed on 2026-11-08. Four skips were due to unavailable symlink creation; one required the authoritative recovery artifact.
- Python compile check: passed for modified modules and audit runner.
- Pinned-vs-candidate differential audit: PASS for persisted observation
  summaries, model scoring fixture, BUY/SELL TP/SL/AMBIGUOUS/EXPIRED fixtures,
  feature-frame outputs, and frozen promotion-gate output.
- Promotion gate replay stayed NOT_READY at 23 paired UTC blocks; only C15/C20
  are evaluated; C25 remains unevaluated; experiment journal remained unchanged.

HARD LIMITATION
Full historical model replay is still BLOCKED because the authoritative
phase2d-state artifact does not include the model binaries or exact historical
15m/1h candle windows. Synthetic fixtures validate code paths only; they are
not substitutes for original historical inputs. See:
  docs/F08_VALIDATOR_DIFFERENTIAL_AUDIT_20261108.md

HISTORICAL RESOLUTION PROVENANCE
The bundle writer now requires a separately hashed resolution_provenance.jsonl
sidecar with one record per stale prediction (74 candidate / 544 production).
It preserves the exact resolver-returned UTC resolved_at value separately from
the market outcome time, which is derived from the selected raw 15-minute OHLC
candle's open timestamp plus 900,000 ms. The sidecar is verified one-to-one
against every historical outcome row and the exact raw-window SHA256. The
deterministic observations.jsonl continues to omit resolved_at. This artifact
remains HISTORICAL_BACKFILL and is excluded from primary prospective scoring.
The existing historical bundle must be regenerated under the reviewed code
only after its source, resolver, and owner-approval release gates are met.

IMPORTANT HASHES
- Pinned validator canonical SHA256:
  2b9f16f87f8222fbb00cfd0e92a0f149dfefdd107d0bfa3079a9a97040e299c5
- Current candidate validator canonical SHA256:
  50ebe8079674c06648832b7fd0c85632e33e1963efffd826de815b69c191d371
- Frozen promotion gate canonical SHA256:
  f787bc3caa810c9f82f56b44d2d8624637693e47eb4613e86a4c30f461c2faab
- Authoritative state SHA256:
  4dd39ee1eefb0866dc5060719d69babf2b31f0d73ee814be5418322ac1d262e5

RUN TESTS
  python -m pytest -q

DOWNLOAD AND VERIFY THE LOCKED MODEL BINARIES (Windows PowerShell)
  powershell -ExecutionPolicy Bypass -File .\tools\download_f08_model_assets.ps1

The verified files are stored under research_inputs\f08_model_replay\ and
are kept separate from the live scanner model path.

RUN DIFFERENTIAL AUDIT
The complete audit bundle should be extracted preserving its folder layout:
  patch_repo/       candidate source tree (this project)
  pinned_resolver/  pinned historical resolver source
  evidence/         authoritative run archive

From patch_repo/ run:
  python tools/f08_validator_differential_audit.py

The audit runner creates research_outputs/f08_validator_differential_audit.json.
It uses temporary copies and does not modify the supplied source/evidence files.

SAFETY / RELEASE STATE
- Do not run F08 disposition with --apply.
- Do not change the authoritative schema-6 state or official ledgers manually.
- Do not append F08_VALIDATOR_AMENDMENT or STALE_PENDING_NOT_SCORED manually.
- Do not evaluate C25, restart F08, promote a model, or enable live/full trading.
- GitHub environment protection, independent owner approvals, complete
  historical replay, V2 restore-selector handoff, and admin checks remain gates.
