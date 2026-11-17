import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import phase2d_observation_ledger as ledger
import phase2d_prospective_validator as validator


class TestPhase2DObservationLedger(unittest.TestCase):

    def test_append_is_idempotent_and_detects_conflict(self):
        record = {
            "id": "candidate:ETHUSDT:123",
            "model": "candidate",
            "symbol": "ETHUSDT",
            "open_time": 123,
            "signal": "BUY",
            "status": "TP",
            "net_r": 3.38,
            "entry": 100.0,
            "atr": 1.0,
            "resolved_at": "2026-09-27T00:00:00+00:00",
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "phase2d_observations.jsonl"
            self.assertTrue(ledger.append_observation(path, "TEST-PHASE2D", record))
            self.assertFalse(ledger.append_observation(path, "TEST-PHASE2D", dict(record)))
            self.assertEqual(len(ledger.load_observations(path, "TEST-PHASE2D")), 1)

            conflict = dict(record)
            conflict["net_r"] = 0.5
            with self.assertRaises(ValueError):
                ledger.append_observation(path, "TEST-PHASE2D", conflict)

    def test_f08_schema5_migration_is_blocked_pending_schema7_authorization(self):
        identity = {
            "experiment_id": "PHASE2D-HARDENED-20260924-F08",
            "candidate_sha256": "candidate",
            "production_sha256": "production",
            "feature_code_hash": "feature-code",
            "feature_schema_hash": "feature-schema",
            "validator_code_hash": "validator-code-new",
        }
        definition = {"experiment_id": identity["experiment_id"]}
        state = validator.empty_state(identity, definition)
        state["schema_version"] = 5

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state_path = root / "phase2d_state.json"
            observation_path = root / "phase2d_observations.jsonl"
            audit_path = root / "experiment_ledger.jsonl"
            original = json.dumps(state, sort_keys=True)
            state_path.write_text(original, encoding="utf-8")

            with patch.object(validator, "STATE_FILE", state_path), \
                 patch.object(validator, "OBSERVATION_LEDGER_FILE", observation_path), \
                 patch.object(validator, "LEDGER_FILE", audit_path):
                with self.assertRaisesRegex(validator.StateCompatibilityError, "schema must be 7"):
                    validator.load_compatible_state(identity, definition, allow_fresh_start=True)

            self.assertEqual(state_path.read_text(encoding="utf-8"), original)
            self.assertFalse(observation_path.exists())
            self.assertFalse(audit_path.exists())


    def test_f08_missing_state_cannot_be_recreated_with_fresh_start(self):
        identity = {
            "experiment_id": "PHASE2D-HARDENED-20260924-F08",
            "candidate_sha256": "candidate",
            "production_sha256": "production",
            "feature_code_hash": "feature-code",
            "feature_schema_hash": "feature-schema",
            "validator_code_hash": "validator-code-new",
        }
        definition = {"experiment_id": identity["experiment_id"]}

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state_path = root / "missing_phase2d_state.json"
            observation_path = root / "phase2d_observations.jsonl"
            audit_path = root / "experiment_ledger.jsonl"
            with patch.object(validator, "STATE_FILE", state_path), \
                 patch.object(validator, "OBSERVATION_LEDGER_FILE", observation_path), \
                 patch.object(validator, "LEDGER_FILE", audit_path):
                with self.assertRaisesRegex(validator.StateCompatibilityError, "fresh start is prohibited"):
                    validator.load_compatible_state(identity, definition, allow_fresh_start=True)
            self.assertFalse(state_path.exists())
            self.assertFalse(observation_path.exists())
            self.assertFalse(audit_path.exists())

    def test_f08_unreadable_state_cannot_be_replaced_with_fresh_start(self):
        identity = {
            "experiment_id": "PHASE2D-HARDENED-20260924-F08",
            "candidate_sha256": "candidate",
            "production_sha256": "production",
            "feature_code_hash": "feature-code",
            "feature_schema_hash": "feature-schema",
            "validator_code_hash": "validator-code-new",
        }
        definition = {"experiment_id": identity["experiment_id"]}

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state_path = root / "phase2d_state.json"
            observation_path = root / "phase2d_observations.jsonl"
            audit_path = root / "experiment_ledger.jsonl"
            state_path.write_text("{not-valid-json", encoding="utf-8")
            before = state_path.read_bytes()
            with patch.object(validator, "STATE_FILE", state_path), \
                 patch.object(validator, "OBSERVATION_LEDGER_FILE", observation_path), \
                 patch.object(validator, "LEDGER_FILE", audit_path):
                with self.assertRaisesRegex(validator.StateCompatibilityError, "cannot be reset"):
                    validator.load_compatible_state(identity, definition, allow_fresh_start=True)
            self.assertEqual(state_path.read_bytes(), before)
            self.assertFalse(observation_path.exists())
            self.assertFalse(audit_path.exists())

    def test_load_reads_more_than_500_cache_entries(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "phase2d_observations.jsonl"
            for i in range(601):
                ledger.append_observation(path, "TEST-PHASE2D", {
                    "id": f"candidate:ETHUSDT:{i}",
                    "model": "candidate", "symbol": "ETHUSDT", "open_time": i,
                    "signal": "BUY", "status": "TP", "net_r": 1.0,
                })
            self.assertEqual(len(ledger.load_observations(path, "TEST-PHASE2D")), 601)

    def test_summarize_drawdown_is_chronological(self):
        records = [
            {"status": "SL", "net_r": -3.0, "open_time": 200},
            {"status": "TP", "net_r": 2.0, "open_time": 100},
        ]
        summary = validator.summarize(records)
        self.assertEqual(summary["cum_net_r"], -1.0)
        self.assertEqual(summary["max_drawdown_r"], 3.0)


    def test_reconcile_appends_missing_observations_without_mutation(self):
        base = {
            "id": "candidate:ETHUSDT:123",
            "model": "candidate",
            "symbol": "ETHUSDT",
            "open_time": 123,
            "signal": "BUY",
            "status": "TP",
            "net_r": 3.38,
        }

        recovered = {
            "id": "production:BTCUSDT:456",
            "model": "production",
            "symbol": "BTCUSDT",
            "open_time": 456,
            "signal": "BUY",
            "status": "SL",
            "net_r": -2.62,
        }

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "phase2d_observations.jsonl"

            self.assertTrue(
                ledger.append_observation(
                    path,
                    "TEST-PHASE2D",
                    base,
                )
            )

            result = ledger.reconcile_resolved_observations(
                path,
                "TEST-PHASE2D",
                [base, recovered, recovered],
            )

            self.assertEqual(result["added"], 1)
            self.assertEqual(
                result["added_by_model"]["production"],
                1,
            )
            self.assertEqual(result["conflicts"], 0)

            observations = ledger.load_observations(
                path,
                "TEST-PHASE2D",
            )

            self.assertEqual(len(observations), 2)

            second = ledger.reconcile_resolved_observations(
                path,
                "TEST-PHASE2D",
                [base, recovered, recovered],
            )

            self.assertEqual(second["added"], 0)
            self.assertEqual(second["existing_before"], 2)
            self.assertEqual(second["historical_unique"], 2)

    def test_reconcile_fails_closed_on_conflicting_historical_duplicate(self):
        record = {
            "id": "candidate:ETHUSDT:123",
            "model": "candidate",
            "symbol": "ETHUSDT",
            "open_time": 123,
            "signal": "BUY",
            "status": "TP",
            "net_r": 3.38,
        }

        conflict = dict(record)
        conflict["net_r"] = 0.5

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "phase2d_observations.jsonl"

            with self.assertRaises(ValueError):
                ledger.reconcile_resolved_observations(
                    path,
                    "TEST-PHASE2D",
                    [record, conflict],
                )

    def test_f08_schema6_cannot_auto_migrate_or_bypass_hash_mismatch(self):
        identity = {
            "experiment_id": "PHASE2D-HARDENED-20260924-F08",
            "candidate_sha256": "candidate",
            "production_sha256": "production",
            "feature_code_hash": "feature-code",
            "feature_schema_hash": "feature-schema",
            "validator_code_hash": "validator-code-new",
        }
        definition = {"experiment_id": identity["experiment_id"]}
        state = validator.empty_state(identity, definition)
        state["schema_version"] = 6
        state["locked_models"]["validator_code_hash"] = "old-validator-code"

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state_path = root / "phase2d_state.json"
            observation_path = root / "phase2d_observations.jsonl"
            audit_path = root / "experiment_ledger.jsonl"
            original = json.dumps(state, sort_keys=True)
            state_path.write_text(original, encoding="utf-8")

            with patch.object(validator, "STATE_FILE", state_path), \
                 patch.object(validator, "OBSERVATION_LEDGER_FILE", observation_path), \
                 patch.object(validator, "LEDGER_FILE", audit_path):
                with self.assertRaisesRegex(validator.StateCompatibilityError, "cannot migrate or reset"):
                    validator.load_compatible_state(identity, definition, allow_fresh_start=True)

            self.assertEqual(state_path.read_text(encoding="utf-8"), original)
            self.assertFalse(observation_path.exists())
            self.assertFalse(audit_path.exists())


if __name__ == "__main__":
    unittest.main()
