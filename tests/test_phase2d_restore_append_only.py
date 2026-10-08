import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from phase2d_restore_guard import (
    RestoreGuardError,
    inspect_artifact,
)


EXPERIMENT = "PHASE2D-HARDENED-20260924-F08"


EVENTS = [
    {
        "record_type": "event",
        "event": "F08_LEDGER_INITIALIZED_FROM_STATE",
        "experiment_id": EXPERIMENT,
    },
    {
        "record_type": "event",
        "event": "F08_LEDGER_MIGRATION_METADATA_CORRECTED",
        "experiment_id": EXPERIMENT,
    },
    {
        "record_type": "event",
        "event": "F08_LEDGER_HISTORICAL_RECONCILIATION",
        "experiment_id": EXPERIMENT,
        "remaining_missing": 0,
    },
]


def observation(model, symbol, ts):
    return {
        "record_type": "observation",
        "experiment_id": EXPERIMENT,
        "model": model,
        "symbol": symbol,
        "open_time": ts,
        "status": "TP",
        "net_r": 1.0,
    }


def make_ledger(path, extra=None):
    rows = list(EVENTS)

    rows += [
        observation("candidate", "BTCUSDT", 1000),
        observation("candidate", "BTCUSDT", 86_401_000),
        observation("production", "ETHUSDT", 1000),
        observation("production", "ETHUSDT", 86_401_000),
        observation("candidate", "SOLUSDT", 172_801_000),
        observation("production", "SOLUSDT", 172_801_000),
    ]

    if extra:
        rows.extend(extra)

    path.write_text(
        "\n".join(
            json.dumps(row)
            for row in rows
        ) + "\n",
        encoding="utf-8",
    )


def baseline_manifest(path, ledger_path):
    digest = hashlib.sha256(
        ledger_path.read_bytes()
    ).hexdigest()

    path.write_text(
        json.dumps(
            {
                "experiment_id": EXPERIMENT,
                "ledger": {
                    "observation_count": 6,
                    "candidate_observation_count": 3,
                    "production_observation_count": 3,
                    "unique_utc_blocks": 3,
                    "sha256": digest,
                },
                "restore_policy": {
                    "observation_count_must_be_gte": 6,
                    "candidate_observation_count_must_be_gte": 3,
                    "production_observation_count_must_be_gte": 3,
                    "unique_utc_blocks_must_be_gte": 3,
                },
            }
        ),
        encoding="utf-8",
    )


def state_file(path):
    path.write_text(
        json.dumps(
            {
                "observation_ledger": {
                    "migration": {
                        "historical_reconciliation": {
                            "historical_unique_observations": 0,
                            "remaining_missing": 0,
                        }
                    }
                },
                "locked_models": {
                    "experiment_id": EXPERIMENT
                }
            }
        ),
        encoding="utf-8",
    )


class TestPhase2DRestoreAppendOnly(unittest.TestCase):

    def test_exact_baseline_is_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            candidate = root / "candidate.jsonl"
            baseline = root / "baseline.jsonl"
            manifest = root / "manifest.json"
            state = root / "state.json"

            make_ledger(baseline)
            candidate.write_bytes(
                baseline.read_bytes()
            )

            baseline_manifest(
                manifest,
                baseline,
            )
            state_file(state)

            stats = inspect_artifact(
                state,
                candidate,
                manifest,
                EXPERIMENT,
                baseline,
            )

            self.assertEqual(
                stats["observation_count"],
                6,
            )

    def test_richer_artifact_preserving_baseline_is_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            baseline = root / "baseline.jsonl"
            candidate = root / "candidate.jsonl"
            manifest = root / "manifest.json"
            state = root / "state.json"

            make_ledger(baseline)
            candidate.write_bytes(
                baseline.read_bytes()
            )

            with candidate.open(
                "a",
                encoding="utf-8",
            ) as handle:
                handle.write(
                    json.dumps(
                        observation(
                            "candidate",
                            "XRPUSDT",
                            259_201_000,
                        )
                    )
                    + "\n"
                )

            baseline_manifest(
                manifest,
                baseline,
            )
            state_file(state)

            stats = inspect_artifact(
                state,
                candidate,
                manifest,
                EXPERIMENT,
                baseline,
            )

            self.assertEqual(
                stats["observation_count"],
                7,
            )

    def test_richer_count_cannot_replace_canonical_observation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            baseline = root / "baseline.jsonl"
            candidate = root / "candidate.jsonl"
            manifest = root / "manifest.json"
            state = root / "state.json"

            make_ledger(baseline)

            rows = baseline.read_text(
                encoding="utf-8"
            ).splitlines()

            rows.pop(3)

            rows.append(
                json.dumps(
                    observation(
                        "candidate",
                        "NEWUSDT",
                        259_201_000,
                    )
                )
            )

            candidate.write_text(
                "\n".join(rows) + "\n",
                encoding="utf-8",
            )

            baseline_manifest(
                manifest,
                baseline,
            )
            state_file(state)

            with self.assertRaises(
                RestoreGuardError
            ):
                inspect_artifact(
                    state,
                    candidate,
                    manifest,
                    EXPERIMENT,
                    baseline,
                )

    def test_equal_observation_count_with_changed_content_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            baseline = root / "baseline.jsonl"
            candidate = root / "candidate.jsonl"
            manifest = root / "manifest.json"
            state = root / "state.json"

            make_ledger(baseline)

            candidate.write_text(
                baseline.read_text(
                    encoding="utf-8"
                ).replace(
                    '"net_r": 1.0',
                    '"net_r": 0.5',
                    1,
                ),
                encoding="utf-8",
            )

            baseline_manifest(
                manifest,
                baseline,
            )
            state_file(state)

            with self.assertRaises(
                RestoreGuardError
            ):
                inspect_artifact(
                    state,
                    candidate,
                    manifest,
                    EXPERIMENT,
                    baseline,
                )

    def test_richer_artifact_regression_is_distinguished_from_floor_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            baseline = root / "baseline.jsonl"
            candidate = root / "candidate.jsonl"
            manifest = root / "manifest.json"
            state = root / "state.json"

            make_ledger(baseline)

            # Remove one canonical record and add two new records:
            # total becomes richer, but canonical history is no longer
            # preserved.
            rows = baseline.read_text(
                encoding="utf-8"
            ).splitlines()

            removed = False
            output = []

            for line in rows:
                record = json.loads(line)

                if (
                    not removed
                    and record.get("record_type") == "observation"
                ):
                    removed = True
                    continue

                output.append(line)

            output.extend(
                [
                    json.dumps(
                        observation(
                            "candidate",
                            "NEW1USDT",
                            259_201_000,
                        )
                    ),
                    json.dumps(
                        observation(
                            "candidate",
                            "NEW2USDT",
                            259_202_000,
                        )
                    ),
                ]
            )

            candidate.write_text(
                "\n".join(output) + "\n",
                encoding="utf-8",
            )

            baseline_manifest(
                manifest,
                baseline,
            )
            state_file(state)

            with self.assertRaises(
                RestoreGuardError
            ) as ctx:
                inspect_artifact(
                    state,
                    candidate,
                    manifest,
                    EXPERIMENT,
                    baseline,
                )

            self.assertIn(
                "FATAL_RICHER_ARTIFACT_REGRESSION",
                str(ctx.exception),
            )

    def test_missing_f08_event_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            baseline = root / "baseline.jsonl"
            candidate = root / "candidate.jsonl"
            manifest = root / "manifest.json"
            state = root / "state.json"

            make_ledger(baseline)

            candidate.write_text(
                "\n".join(
                    line
                    for line in baseline.read_text(
                        encoding="utf-8"
                    ).splitlines()
                    if "F08_LEDGER_MIGRATION_METADATA_CORRECTED"
                    not in line
                ) + "\n",
                encoding="utf-8",
            )

            baseline_manifest(
                manifest,
                baseline,
            )
            state_file(state)

            with self.assertRaises(
                RestoreGuardError
            ):
                inspect_artifact(
                    state,
                    candidate,
                    manifest,
                    EXPERIMENT,
                    baseline,
                )

    def test_nonzero_reconciliation_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            baseline = root / "baseline.jsonl"
            candidate = root / "candidate.jsonl"
            manifest = root / "manifest.json"
            state = root / "state.json"

            make_ledger(baseline)

            candidate.write_text(
                baseline.read_text(
                    encoding="utf-8"
                ).replace(
                    '"remaining_missing": 0',
                    '"remaining_missing": 1',
                    1,
                ),
                encoding="utf-8",
            )

            baseline_manifest(
                manifest,
                baseline,
            )
            state_file(state)

            with self.assertRaises(
                RestoreGuardError
            ):
                inspect_artifact(
                    state,
                    candidate,
                    manifest,
                    EXPERIMENT,
                    baseline,
                )


if __name__ == "__main__":
    unittest.main()
