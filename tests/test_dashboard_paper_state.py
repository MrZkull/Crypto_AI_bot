import json
import unittest
from unittest.mock import patch

import dashboard


class TestPaperStateEndpoint(unittest.TestCase):
    def setUp(self):
        self.client = dashboard.app.test_client()

    def test_returns_isolated_paper_state_with_prediction_metadata(self):
        state = {
            "schema_version": 1,
            "policy_version": "stage2e-v1",
            "execution_mode": "PAPER_ONLY",
            "initial_balance_usd": 10000.0,
            "risk_per_trade": 0.03,
            "positions": {
                "p-open": {
                    "symbol": "ETHUSDT", "side": "BUY", "status": "ACTIVE", "state": "OPEN",
                    "opened_at_ms": 1700000000000, "entry_price": 100.0, "remaining_qty": 2.0,
                    "qty": 2.0, "initial_risk_usd": 300.0,
                    "observation_contract": {"interval": "15m", "observation_market": "BINANCE_SPOT",
                        "observation_source": {"exchange": "binance", "market_type": "spot"},
                        "open_time": 1700000000000, "close_time": 1700000899999},
                }
            },
            "decisions": {
                "p-open": {"status": "OPENED", "timestamp_ms": 1700000000000},
                "p-skip": {"status": "MISSED_ENTRY_WINDOW", "timestamp_ms": 1700000100000},
            },
        }
        predictions = [
            {"pred_id": "p-open", "symbol": "ETHUSDT", "predicted_signal": "BUY", "confidence": 73.8,
             "rsi_15m": 51.2, "adx_15m": 34.0, "ensemble_disagreement": 0.02,
             "generated_at": "2026-10-09T11:55:00+00:00"},
            {"pred_id": "p-skip", "symbol": "SOLUSDT", "predicted_signal": "SELL",
             "generated_at": "2026-10-09T11:50:00+00:00"},
        ]
        events = "\n".join([
            json.dumps({"event": "PAPER_POSITION_OPENED", "event_id": "OPEN:p-open", "pred_id": "p-open", "symbol": "ETHUSDT", "side": "BUY"}),
            json.dumps({"event": "PAPER_DECISION", "event_id": "DECISION:p-skip:MISSED_ENTRY_WINDOW", "pred_id": "p-skip", "symbol": "SOLUSDT", "side": "SELL"}),
        ])

        def fake_get(name, default):
            return {"paper_state.json": state, "predictions.json": predictions,
                    "paper_events.jsonl": events}.get(name, default)

        with patch.object(dashboard, "get", side_effect=fake_get):
            response = self.client.get("/api/paper/state")

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertTrue(payload["ok"])
        self.assertTrue(payload["paper_only"])
        self.assertEqual(payload["execution_mode"], "PAPER_ONLY")
        self.assertEqual(payload["portfolio"]["active_positions"], 1)
        self.assertEqual(payload["portfolio"]["closed_positions"], 0)
        self.assertEqual(payload["portfolio"]["decision_count"], 2)
        self.assertEqual(payload["portfolio"]["event_count"], 2)
        position = payload["positions"][0]
        self.assertEqual(position["pred_id"], "p-open")
        self.assertEqual(position["confidence"], 73.8)
        self.assertEqual(position["adx_15m"], 34.0)
        self.assertEqual(payload["decisions"][0]["symbol"], "SOLUSDT")
        self.assertEqual(len(payload["events"]), 2)

    def test_empty_or_missing_paper_files_return_an_empty_paper_view(self):
        def fake_get(name, default):
            return default

        with patch.object(dashboard, "get", side_effect=fake_get):
            response = self.client.get("/api/paper/state")

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertTrue(payload["ok"])
        self.assertTrue(payload["paper_only"])
        self.assertEqual(payload["positions"], [])
        self.assertEqual(payload["decisions"], [])
        self.assertEqual(payload["events"], [])
        self.assertEqual(payload["portfolio"]["active_positions"], 0)


if __name__ == "__main__":
    unittest.main()
