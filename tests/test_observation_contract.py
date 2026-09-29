import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from market_data_integrity import (
    build_observation_contract,
    validate_observation_contract,
    observation_identity,
)


class TestObservationContract(unittest.TestCase):
    SOURCE = {
        "exchange": "binance",
        "market_type": "spot",
    }

    @staticmethod
    def valid_kwargs():
        return {
            "logical_symbol": "ETHUSDT",
            "observation_market": "BINANCE_SPOT",
            "observation_source": dict(TestObservationContract.SOURCE),
            "execution_instrument": None,
            "interval": "15m",
            "open_time": 1_700_000_000_000,
            "close_time": 1_700_000_899_999,
            "retrieved_at_ms": 1_700_001_000_000,
            "htf_sources": {
                "1h": {
                    "logical_symbol": "ETHUSDT",
                    "observation_market": "BINANCE_SPOT",
                    "observation_source": dict(TestObservationContract.SOURCE),
                    "interval": "1h",
                    "open_time": 1_699_996_800_000,
                    "close_time": 1_700_000_399_999,
                },
            },
        }

    def test_build_and_validate_valid_contract(self):
        contract = build_observation_contract(**self.valid_kwargs())
        self.assertTrue(validate_observation_contract(contract))
        self.assertEqual(contract["contract_version"], "1.0")
        self.assertEqual(
            observation_identity(contract),
            (
                "ETHUSDT",
                1_700_000_000_000,
                "15m",
                "binance",
                "spot",
            ),
        )

    def test_validator_rejects_partial_nested_provenance(self):
        contract = build_observation_contract(**self.valid_kwargs())
        contract["htf_sources"]["1h"].pop("observation_source")
        with self.assertRaises(ValueError):
            validate_observation_contract(contract)

    def test_validator_rejects_future_htf_source(self):
        kwargs = self.valid_kwargs()
        kwargs["htf_sources"]["1h"]["open_time"] = 1_700_000_040_000
        kwargs["htf_sources"]["1h"]["close_time"] = 1_700_003_639_999
        contract = build_observation_contract(
            logical_symbol=kwargs["logical_symbol"],
            observation_market=kwargs["observation_market"],
            observation_source=kwargs["observation_source"],
            execution_instrument=kwargs["execution_instrument"],
            interval=kwargs["interval"],
            open_time=kwargs["open_time"],
            close_time=kwargs["close_time"],
            retrieved_at_ms=kwargs["retrieved_at_ms"],
        )
        contract["htf_sources"] = dict(kwargs["htf_sources"])
        with self.assertRaises(ValueError):
            validate_observation_contract(contract)

    def test_identity_ignores_retrieval_and_execution_metadata(self):
        contract_a = build_observation_contract(**self.valid_kwargs())
        kwargs = self.valid_kwargs()
        kwargs["retrieved_at_ms"] += 12345
        kwargs["execution_instrument"] = "ETH-PERPETUAL"
        contract_b = build_observation_contract(**kwargs)
        self.assertEqual(
            observation_identity(contract_a),
            observation_identity(contract_b),
        )

    def test_validator_rejects_unknown_htf_key(self):
        contract = build_observation_contract(
            logical_symbol="ETHUSDT",
            observation_market="BINANCE_SPOT",
            observation_source=dict(self.SOURCE),
            execution_instrument=None,
            interval="15m",
            open_time=1_700_000_000_000,
            close_time=1_700_000_899_999,
            retrieved_at_ms=1_700_001_000_000,
        )
        contract["htf_sources"]["2h"] = {
            "logical_symbol": "ETHUSDT",
            "observation_market": "BINANCE_SPOT",
            "observation_source": dict(self.SOURCE),
            "interval": "2h",
            "open_time": 1_699_992_000_000,
            "close_time": 1_699_999_199_999,
        }
        with self.assertRaises(ValueError):
            validate_observation_contract(contract)

    def test_prediction_record_contains_valid_contract(self):
        import trade_executor

        row = {
            "open_time": 1_700_000_000_000,
            "close_time": 1_700_000_899_999,
            "close": 2000.0,
            "atr": 20.0,
            "rsi": 55.0,
            "adx": 30.0,
        }
        record = trade_executor._pred_record(
            "ETHUSDT",
            "BUY",
            81.0,
            {"trained_at": "test"},
            row,
            {"ensemble_disagreement": 0.0, "high_disagreement": False},
            pred_id="stage2c-test",
        )
        self.assertTrue(validate_observation_contract(record["observation_contract"]))
        self.assertEqual(
            record["observation_identity"],
            ["ETHUSDT", 1_700_000_000_000, "15m", "binance", "spot"],
        )

    def test_candidate_setup_contains_valid_contract(self):
        import json
        import candidate_scanner

        class DummyLock:
            def __enter__(self):
                return self
            def __exit__(self, exc_type, exc, tb):
                return False

        with tempfile.TemporaryDirectory() as td:
            events_path = Path(td) / "events.jsonl"
            state_path = Path(td) / "state.json"

            with patch.object(candidate_scanner, "EVENTS_LEDGER_FILE", events_path),                  patch.object(candidate_scanner, "STATE_FILE", state_path),                  patch.object(candidate_scanner, "FileLock", return_value=DummyLock()):

                manifest = {
                    "candidate_id": "stage2c-test-candidate",
                    "model_sha256": "m",
                    "feature_schema_hash": "s",
                    "feature_code_hash": "f",
                    "execution_policy_hash": "e",
                    "decision_policy_hash": "d",
                    "config_hash": "c",
                }
                config = {
                    "evaluation_balance_usd": 10000.0,
                    "ATR_STOP_MULT": 2.5,
                    "ATR_TARGET1_MULT": 3.5,
                    "ATR_TARGET2_MULT": 7.5,
                    "RISK_PER_TRADE": 0.015,
                }
                row = {
                    "open_time": 1_700_000_000_000,
                    "close_time": 1_700_000_899_999,
                    "atr": 20.0,
                }

                candidate_scanner.record_candidate_setup(
                    manifest=manifest,
                    manifest_config=config,
                    pred_id="stage2c-candidate-test",
                    symbol="ETHUSDT",
                    side="BUY",
                    primary_selected=True,
                    meta_selected=False,
                    row=row,
                    simulated_entry=2000.0,
                    observation_time_ms=1_700_000_000_000,
                )

                event = json.loads(events_path.read_text(encoding="utf-8").strip())

        self.assertTrue(validate_observation_contract(event["observation_contract"]))
        self.assertEqual(
            event["observation_identity"],
            ["ETHUSDT", 1_700_000_000_000, "15m", "binance", "spot"],
        )


if __name__ == "__main__":
    unittest.main()
