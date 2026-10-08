import json
import tempfile
import unittest
from pathlib import Path

from phase2d_restore_guard import (
    RestoreGuardError,
    inspect_artifact,
)


EXPERIMENT = "TEST-F08"


def write_artifact(
    root: Path,
    candidate_count: int,
    production_count: int,
    blocks: int,
):
    state = root / "phase2d_state.json"
    ledger = root / "phase2d_observations.jsonl"
    baseline = root / "baseline.json"

    state.write_text(
        json.dumps(
            {
                "locked_models": {
                    "experiment_id": EXPERIMENT
                }
            }
        ),
        encoding="utf-8",
    )

    rows = []

    # Spread observations across the requested UTC blocks.
    for model, count in (
        ("candidate", candidate_count),
        ("production", production_count),
    ):
        for i in range(count):
            block = i % blocks
            open_time = block * 86_400_000 + i
            rows.append(
                {
                    "record_type": "observation",
                    "experiment_id": EXPERIMENT,
                    "model": model,
                    "symbol": f"{model[:3].upper()}USDT",
                    "open_time": open_time,
                    "status": "TP",
                    "net_r": 1.0,
                }
            )

    events = [
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

    ledger.write_text(
        "\n".join(
            json.dumps(r)
            for r in events + rows
        ) + "\n",
        encoding="utf-8",
    )

    # Baseline will be written by caller because its SHA
    # depends on the generated ledger.
    return state, ledger, baseline


def make_baseline(
    path: Path,
    ledger: Path,
    total: int,
    candidate: int,
    production: int,
    blocks: int,
):
    import hashlib

    h = hashlib.sha256()
    h.update(ledger.read_bytes())

    path.write_text(
        json.dumps(
            {
                "experiment_id": EXPERIMENT,
                "ledger": {
                    "observation_count": total,
                    "candidate_observation_count": candidate,
                    "production_observation_count": production,
                    "unique_utc_blocks": blocks,
                    "sha256": h.hexdigest(),
                },
                "restore_policy": {
                    "observation_count_must_be_gte": total,
                    "candidate_observation_count_must_be_gte": candidate,
                    "production_observation_count_must_be_gte": production,
                    "unique_utc_blocks_must_be_gte": blocks,
                },
            }
        ),
        encoding="utf-8",
    )


class TestPhase2DRestoreGuard(unittest.TestCase):

    def test_equal_baseline_is_compatible(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            state, ledger, baseline = write_artifact(
                root,
                3,
                3,
                3,
            )

            make_baseline(
                baseline,
                ledger,
                6,
                3,
                3,
                3,
            )

            stats = inspect_artifact(
                state,
                ledger,
                baseline,
                EXPERIMENT,
            )

            self.assertEqual(
                stats["observation_count"],
                6,
            )

    def test_below_baseline_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            state, ledger, baseline = write_artifact(
                root,
                2,
                2,
                3,
            )

            make_baseline(
                baseline,
                ledger,
                5,
                2,
                3,
                3,
            )

            with self.assertRaises(
                RestoreGuardError
            ):
                inspect_artifact(
                    state,
                    ledger,
                    baseline,
                    EXPERIMENT,
                )

    def test_equal_count_with_different_ledger_hash_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            state, ledger, baseline = write_artifact(
                root,
                3,
                3,
                3,
            )

            make_baseline(
                baseline,
                ledger,
                6,
                3,
                3,
                3,
            )

            data = ledger.read_text(
                encoding="utf-8"
            )

            ledger.write_text(
                data.replace(
                    '"net_r": 1.0',
                    '"net_r": 0.5',
                    1,
                ),
                encoding="utf-8",
            )

            with self.assertRaises(
                RestoreGuardError
            ):
                inspect_artifact(
                    state,
                    ledger,
                    baseline,
                    EXPERIMENT,
                )

    def test_richer_artifact_is_compatible(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            state, ledger, baseline = write_artifact(
                root,
                4,
                4,
                4,
            )

            make_baseline(
                baseline,
                ledger,
                6,
                3,
                3,
                3,
            )

            stats = inspect_artifact(
                state,
                ledger,
                baseline,
                EXPERIMENT,
            )

            self.assertEqual(
                stats["observation_count"],
                8,
            )
            self.assertEqual(
                stats["unique_utc_blocks"],
                4,
            )

    def test_missing_required_f08_event_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            state, ledger, baseline = write_artifact(
                root,
                3,
                3,
                3,
            )

            lines = ledger.read_text(
                encoding="utf-8"
            ).splitlines()

            lines = [
                line
                for line in lines
                if "F08_LEDGER_MIGRATION_METADATA_CORRECTED"
                not in line
            ]

            ledger.write_text(
                "\n".join(lines) + "\n",
                encoding="utf-8",
            )

            make_baseline(
                baseline,
                ledger,
                6,
                3,
                3,
                3,
            )

            with self.assertRaises(
                RestoreGuardError
            ):
                inspect_artifact(
                    state,
                    ledger,
                    baseline,
                    EXPERIMENT,
                )

    def test_nonzero_f08_reconciliation_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            state, ledger, baseline = write_artifact(
                root,
                3,
                3,
                3,
            )

            data = ledger.read_text(
                encoding="utf-8"
            ).replace(
                '"remaining_missing": 0',
                '"remaining_missing": 1',
                1,
            )

            ledger.write_text(
                data,
                encoding="utf-8",
            )

            make_baseline(
                baseline,
                ledger,
                6,
                3,
                3,
                3,
            )

            with self.assertRaises(
                RestoreGuardError
            ):
                inspect_artifact(
                    state,
                    ledger,
                    baseline,
                    EXPERIMENT,
                )

    def test_duplicate_observation_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            state, ledger, baseline = write_artifact(
                root,
                3,
                3,
                3,
            )

            # Use identical identity for a duplicated record.
            lines = ledger.read_text(
                encoding="utf-8"
            ).splitlines()

            lines.append(lines[0])

            ledger.write_text(
                "\n".join(lines) + "\n",
                encoding="utf-8",
            )

            make_baseline(
                baseline,
                ledger,
                7,
                3,
                3,
                3,
            )

            with self.assertRaises(
                RestoreGuardError
            ):
                inspect_artifact(
                    state,
                    ledger,
                    baseline,
                    EXPERIMENT,
                )


if __name__ == "__main__":
    unittest.main()
