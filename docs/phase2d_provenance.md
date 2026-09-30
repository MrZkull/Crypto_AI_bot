# Phase 2D Provenance Record

## F08 historical observation recovery

The 1,169 observations added during F08 reconciliation were recovered exclusively
from preserved historical Phase 2D `phase2d-state` artifacts produced by
successful GitHub Actions runs.

Reconciliation evidence:

- Historical F08 state artifacts examined: 60
- Historical unique observations: 2,229
- Existing ledger observations before reconciliation: 1,060
- Added during reconciliation: 1,169
- Candidate added: 32
- Production added: 1,137
- Conflicts: 0
- Remaining missing: 0

The reconciliation implementation reads the saved `resolved` prediction dictionaries
from those historical state artifacts and does not re-fetch historical market data
or recompute historical `entry`, `atr`, outcomes, or `net_r` using current feature
engineering code.

Reconciliation workflow run:
- Run: 36338546341
- Artifact: 10937758986

## F08 known feature confound: taker_buy_ratio

The frozen F08 Deribit research fetcher sets
`taker_buy_base_vol = volume * 0.5` for every research candle.

Consequently, canonical feature engineering produces a constant
`taker_buy_ratio = 0.5` for F08 observations.

The Phase 2C HTF-removal candidate retains `taker_buy_ratio` in its
25-feature candidate feature set. The locked production feature list does
not contain this feature.

This is a known F08 confound and is recorded for auditability only.
It does not alter the frozen F08 methodology, thresholds, research data
source, feature construction, observation ledger, statistical gate, or
promotion criteria.

## Production regime-reference provenance

Production model:
- Model SHA256: f55b887c7f624179b3d9fee56d792c29424a589e3d8734edcd78ce1be71f2c21
- Training commit: 893a79e554029d4a62a0cbeafa17dac2d8ca6257
- Training workflow run: 32660128586
- trained_at: 2026-08-23T19:30:12.136077+00:00

Reconstructed reference:
- Source: Binance public 15m klines
- Recent-data anchor: production model trained_at
- Reference basis: reconstructed production training dataset
- Reconstructed train rows: 266,803
- Reconstructed calibration rows: 63,428
- Reconstructed test rows: 84,614
- ATR% tercile cutoffs:
  - Q33: 0.456749443111475
  - Q66: 0.7998676866544822
- ADX tercile cutoffs:
  - Q33: 26.02955652599114
  - Q66: 37.3693860528105

## Exact-data-hash limitation

The original raw Binance training responses from production training were not
preserved with the production model.

The production training workflow's artifact list does not contain the original
training dataset. The production `model_performance.json` records the training
split counts but does not contain a training-input content hash.

The production training code itself fetches Binance market data directly during
training rather than consuming a byte-identifiable archived training dataset.

Therefore the reconstructed regime reference cannot be verified byte-for-byte
against the exact raw data seen during the original production training run.

This limitation is intentional and explicit. The reconstructed reference must not
be represented as an exact historical-data hash match.
