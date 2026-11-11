Crypto_AI_bot — F08 Schema 7 Disposition Candidate
Checkpoint: 2026-11-11 — release candidate tracked in draft PR #15

PURPOSE
This is an isolated implementation candidate for the F08 controlled schema-7
stale-pending disposition. It is not merged into the user's main project and
is not authorized for production use or official state mutation.

LATEST VALIDATION — RELEASE CANDIDATE PR #15
- Windows full regression before the follow-up pin/CI patch: 412 passed, 5 skipped, and 3 subtests passed. Four skips were Windows symlink limitations; one required the external authoritative recovery artifact path.
- Latest Windows full regression after consolidated hardening (2026-11-11): 417 passed, 4 skipped, and 3 subtests passed in 34.38s. Focused regression: 70 passed in 9.90s.
- GitHub Actions PR regression run #15 on Ubuntu 26.04 / Python 3.12, initial candidate commit e1507a4b: 416 passed, 1 skipped, 14 warnings, and 3 subtests passed. All four symlink tests ran and passed on Linux.
- The sole Linux skip was the real-authoritative-state test, which expected an external /mnt/data mount. The release candidate includes a byte-identical evidence/state.json copy; the follow-up test change uses it only after matching the pinned SHA256.
- Latest GitHub Actions PR regression run #17 on commit `773adfb6c97e73ca84d2015c69eaec44a7fcd57c`: **418 passed, 0 skipped**, 14 warnings, and 3 subtests passed in 9.77s. The pinned actionlint 1.7.12 checksum, workflow lint, core-module compilation, and full Linux regression passed.
- Pinned-vs-candidate differential audit: PASS for persisted observation summaries, model scoring fixture, BUY/SELL TP/SL/AMBIGUOUS/EXPIRED fixtures, feature-frame outputs, and frozen promotion-gate output.
- Promotion gate replay stayed NOT_READY at 23 paired UTC blocks; only C15/C20 are evaluated; C25 remains unevaluated; experiment journal remained unchanged.

HARD LIMITATION
Full historical model replay is still BLOCKED because the authoritative
phase2d-state artifact does not include the model binaries or exact historical
15m/1h candle windows. Synthetic fixtures validate code paths only; they are
not substitutes for original historical inputs. See:
  docs/F08_VALIDATOR_DIFFERENTIAL_AUDIT_20261108.md

HISTORICAL RESOLUTION PROVENANCE
The provenance-inclusive historical bundle is included in draft PR #15, but is
not yet on main. Its immutable evidence hashes are:

- manifest.json SHA256: 1aae5f75292241298977378b81ac6297e24c3270e1afb5957776a1f8b1767ab3
- observations.jsonl SHA256: 3f05c466810fc86a911d4105575c951a59a73e78a3b6271c9c01cb0042386878
- resolution_provenance.jsonl SHA256: d679494c5d58292f12227b95492ec3d47daf4b5b5f2a38b21f962ca465b994c8

The deterministic observations.jsonl continues to omit resolved_at. Resolution
timestamps and outcome-candle timing remain in the separately hashed provenance
sidecar. The bundle remains HISTORICAL_BACKFILL and is excluded from primary
prospective scoring.

The disposition validator must pin the current manifest hash. Owner approval
remains pending and separate. Approval must not rewrite the immutable bundle.

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
- PR #15 remains a draft and is not merged.
- The candidate is not authorized for production use or official state mutation.
- GitHub main-branch protection and both protected environments remain admin gates.
- Independent owner approvals remain required for the two separate protected workflows.
- Complete historical model replay and V2 restore-selector handoff remain separate release gates.
- Do not dispatch either protected workflow from this candidate branch.
- Do not run F08 disposition with --apply.
- Do not change the authoritative schema-6 state or official ledgers manually.
- Do not append F08_VALIDATOR_AMENDMENT or STALE_PENDING_NOT_SCORED manually.
- Do not evaluate C25, restart F08, promote a model, or enable live/full trading.
- F08 remains disabled.
