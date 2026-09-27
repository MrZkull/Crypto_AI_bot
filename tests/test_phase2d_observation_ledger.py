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

    def test_f08_schema5_migrates_and_records_unrecoverable_count(self):
        identity = {
            "experiment_id": "PHASE2D-HARDENED-20260924-F08",
            "candidate_sha256": "candidate",
            "production_sha256": "production",
            "feature_code_hash": "feature-code",
            "feature_schema_hash": "feature-schema",
            "validator_code_hash": "validator-code-old",
        }
        definition = {
            "experiment_id": identity["experiment_id"],
            "lookahead_bars": 24,
            "buy_tp_r": 3.5,
            "buy_sl_r": 2.5,
            "sell_tp_r": 3.5,
            "sell_sl_r": 2.5,
            "friction_r": 0.12,
            "candidate_thresholds": {"buy": 0.40, "sell": 1.01},
            "production_thresholds": {"buy": 0.40, "sell": 0.45},
        }
        state = validator.empty_state(identity, definition)
        state["schema_version"] = 5
        state["resolved"]["candidate"] = [{
            "id": "candidate:ETHUSDT:123", "model": "candidate",
            "symbol": "ETHUSDT", "open_time": 123, "signal": "BUY",
            "status": "TP", "net_r": 3.38,
        }]
        state["resolved"]["production"] = [{
            "id": "production:ETHUSDT:123", "model": "production",
            "symbol": "ETHUSDT", "open_time": 123, "signal": "BUY",
            "status": "SL", "net_r": -2.62,
        }]
        state["pending"]["candidate"] = [{
            "id": "candidate:BTCUSDT:456", "model": "candidate",
            "symbol": "BTCUSDT", "open_time": 456, "signal": "BUY",
        }]
        state["seen"]["candidate"] = ["ETHUSDT:123", "BTCUSDT:456", "SOLUSDT:789"]
        state["seen"]["production"] = ["ETHUSDT:123"]

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state_path = root / "phase2d_state.json"
            observation_path = root / "phase2d_observations.jsonl"
            audit_path = root / "experiment_ledger.jsonl"
            state_path.write_text(json.dumps(state), encoding="utf-8")

            with patch.object(validator, "STATE_FILE", state_path), \
                 patch.object(validator, "OBSERVATION_LEDGER_FILE", observation_path), \
                 patch.object(validator, "LEDGER_FILE", audit_path):
                restored, fresh, reason = validator.load_compatible_state(
                    identity, definition, allow_fresh_start=False
                )

            self.assertFalse(fresh)
            self.assertEqual(restored["schema_version"], 6)
            self.assertIn("schema upgraded from 5 to 6", reason)
            migration = restored["observation_ledger"]["migration"]
            self.assertTrue(migration["initialized"])
            self.assertEqual(migration["migrated_resolved"], 2)
            self.assertEqual(migration["pending_at_migration"], 1)
            self.assertEqual(migration["unrecoverable_seen"], 1)
            self.assertEqual(len(ledger.load_observations(observation_path, identity["experiment_id"])), 2)
            events = ledger.load_events(observation_path, identity["experiment_id"])
            self.assertEqual(sum(e.get("event") == "F08_LEDGER_INITIALIZED_FROM_STATE" for e in events), 1)

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


if __name__ == "__main__":
    unittest.main()
