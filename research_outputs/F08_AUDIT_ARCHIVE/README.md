# F08 Reconciliation Audit Archive

Experiment ID: PHASE2D-HARDENED-20260924-F08

Source GitHub Actions Run: 36338546341
Reconciliation Baseline Run: 36331652191
Historical F08 State Runs Examined: 60

Historical Unique Observations: 2229
Existing Before Reconciliation: 1060
Recovered/Added: 1169
Candidate Added: 32
Production Added: 1137
Conflicts: 0
Remaining Missing: 0

Final Candidate Ledger Count: 545
Final Production Ledger Count: 1684

The reconciliation recovered historical F08 observations exclusively from
preserved successful GitHub Actions Phase 2D state artifacts. Historical
entry/ATR/outcome/net-R values were not re-fetched or recomputed using
current feature code.

The archive contains:
- Reconciled Phase 2D state
- Reconciled observation ledger
- Experiment ledger
- Reconciliation report
- Reconciliation manifest listing all 60 source runs
- Pre-reconciliation baseline archive from Run 36331652191

GitHub artifact:
Run 36338546341
Artifact ID: 10937758986
Artifact SHA256:
a69f5f35e0bdd6e1a113281eaa10f3928414f4520ba6eb081b6c8af9f07c09df

Important limitation:
The original raw Binance training responses used to construct the
production model were not preserved, so byte-identical reconstruction of
the original training dataset cannot be established. The frozen regime
reference is therefore a reconstructed reference, not an exact raw-data
hash match.
