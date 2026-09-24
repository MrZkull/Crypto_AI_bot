import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import phase2d_prospective_validator as validator


class TestPhase2DExperimentIntegrity(unittest.TestCase):

    def _identity(self):
        return {
            "experiment_id": "TEST-PHASE2D",
            "candidate_sha256": "candidate",
            "production_sha256": "production",
            "feature_code_hash": "feature-code",
            "feature_schema_hash": "feature-schema",
            "validator_code_hash": "validator-code",
        }

    def _definition(self, identity):
        return {
            "experiment_id": identity[
                "experiment_id"
            ],
            "lookahead_bars": 24,
            "buy_tp_r": 3.5,
            "buy_sl_r": 2.5,
            "sell_tp_r": 3.5,
            "sell_sl_r": 2.5,
            "friction_r": 0.12,
            "candidate_thresholds": {
                "buy": 0.40,
                "sell": 1.01,
            },
            "production_thresholds": {
                "buy": 0.40,
                "sell": 0.45,
            },
        }

    def test_fresh_start_requires_explicit_flag(self):
        identity = self._identity()
        definition = self._definition(
            identity
        )

        with tempfile.TemporaryDirectory() as tmp:
            state_path = (
                Path(tmp)
                / "phase2d_state.json"
            )

            with patch.object(
                validator,
                "STATE_FILE",
                state_path,
            ):
                with self.assertRaises(
                    validator.StateCompatibilityError
                ):
                    validator.load_compatible_state(
                        identity,
                        definition,
                        allow_fresh_start=False,
                    )

    def test_explicit_fresh_start_creates_state(self):
        identity = self._identity()
        definition = self._definition(
            identity
        )

        with tempfile.TemporaryDirectory() as tmp:
            state_path = (
                Path(tmp)
                / "phase2d_state.json"
            )

            ledger_path = (
                Path(tmp)
                / "experiment_ledger.jsonl"
            )

            with patch.object(
                validator,
                "STATE_FILE",
                state_path,
            ), patch.object(
                validator,
                "LEDGER_FILE",
                ledger_path,
            ):
                state, fresh, reason = (
                    validator.load_compatible_state(
                        identity,
                        definition,
                        allow_fresh_start=True,
                    )
                )

                self.assertTrue(
                    fresh
                )

                self.assertEqual(
                    state[
                        "schema_version"
                    ],
                    validator.STATE_SCHEMA,
                )

                self.assertEqual(
                    state[
                        "locked_models"
                    ],
                    identity,
                )

                self.assertTrue(
                    ledger_path.exists()
                )

    def test_duplicate_resolved_id_is_rejected(self):
        identity = self._identity()
        definition = self._definition(
            identity
        )

        state = validator.empty_state(
            identity,
            definition,
        )

        item = {
            "id": "candidate:ETHUSDT:123",
            "symbol": "ETHUSDT",
            "open_time": 123,
            "status": "TP",
        }

        state["pending"][
            "candidate"
        ].append(dict(item))

        state["seen"][
            "candidate"
        ].append(
            "ETHUSDT:123"
        )

        validator.append_resolved_unique(
            state,
            "candidate",
            dict(item),
        )

        with self.assertRaises(
            RuntimeError
        ):
            validator.append_resolved_unique(
                state,
                "candidate",
                dict(item),
            )

    def test_state_integrity_rejects_pending_resolved_overlap(self):
        identity = self._identity()
        definition = self._definition(
            identity
        )

        state = validator.empty_state(
            identity,
            definition,
        )

        item = {
            "id": "candidate:ETHUSDT:123",
            "symbol": "ETHUSDT",
            "open_time": 123,
            "status": "TP",
        }

        state["pending"][
            "candidate"
        ].append(dict(item))

        state["resolved"][
            "candidate"
        ].append(dict(item))

        state["seen"][
            "candidate"
        ].append(
            "ETHUSDT:123"
        )

        with self.assertRaises(
            validator.StateCompatibilityError
        ):
            validator.validate_state_integrity(
                state
            )


if __name__ == "__main__":
    unittest.main()
