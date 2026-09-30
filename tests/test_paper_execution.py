import json
import unittest
from pathlib import Path
from unittest.mock import patch

import paper_execution as paper


SOURCE = {"exchange": "binance", "market_type": "spot"}


def contract(
    open_time=1_700_000_000_000,
    close_time=1_700_000_899_999,
):
    return {
        "contract_version": "1.0",
        "logical_symbol": "ETHUSDT",
        "observation_market": "BINANCE_SPOT",
        "observation_source": dict(SOURCE),
        "execution_instrument": None,
        "interval": "15m",
        "open_time": open_time,
        "close_time": close_time,
        "retrieved_at_ms": 1_700_001_000_000,
        "htf_sources": {},
        "btc_source": None,
    }


def prediction(pred_id="ETHUSDT_1700000000000", **overrides):
    value = {
        "pred_id": pred_id,
        "logical_symbol": "ETHUSDT",
        "symbol": "ETHUSDT",
        "predicted_signal": "BUY",
        "reject_reason": None,
        "open_time": 1_700_000_000_000,
        "interval": "15m",
        "entry_ref": 100.0,
        "atr_ref": 2.0,
        "was_executed": False,
        "observation_contract": contract(),
        "model_version": "test-model",
        "generated_at": "2026-09-30T00:00:00+00:00",
    }
    value.update(overrides)
    return value


def candle(
    open_time=1_700_000_900_000,
    close_time=1_700_001_799_999,
    close=107.0,
    high=108.0,
    low=100.5,
):
    return {
        "open_time": open_time,
        "close_time": close_time,
        "open": 100.0,
        "high": high,
        "low": low,
        "close": close,
    }


class TestPaperExecution(unittest.TestCase):
    def setUp(self):
        self.tmp = __import__("tempfile").TemporaryDirectory()
        self.events = Path(self.tmp.name) / "paper_events.jsonl"
        self.state = Path(self.tmp.name) / "paper_state.json"
        self.predictions = Path(self.tmp.name) / "predictions.json"

    def tearDown(self):
        self.tmp.cleanup()

    def test_paper_position_is_compatible_with_shadow_lifecycle(self):
        pred = paper._normalize_prediction(prediction())
        position = paper._build_position(pred)

        self.assertEqual(position["simulated_entry"], pred["entry_ref"])

        lifecycle_candle = candle(
            close=107.2,
            high=107.5,
            low=106.5,
        )

        outcome = paper.monitor.process_trade_candle(
            position,
            lifecycle_candle,
        )

        self.assertTrue(outcome["modified"])
        self.assertFalse(outcome["closed"])
        self.assertEqual(position["state"], "PARTIAL_TP1")
        self.assertEqual(len(outcome["events"]), 1)
        self.assertEqual(
            outcome["events"][0]["event"],
            "PARTIAL_TP1_FILLED",
        )

    def test_build_position_is_paper_only_and_preserves_identity(self):
        pred = paper._normalize_prediction(prediction())
        position = paper._build_position(pred)

        self.assertEqual(position["execution_mode"], "PAPER_ONLY")
        self.assertEqual(
            position["observation_identity"],
            ["ETHUSDT", 1_700_000_000_000, "15m", "binance", "spot"],
        )
        self.assertAlmostEqual(position["entry_price"], 100.0)
        self.assertAlmostEqual(position["stop"], 95.0)
        self.assertAlmostEqual(position["tp1"], 107.0)
        self.assertAlmostEqual(position["tp2"], 115.0)
        self.assertGreater(position["qty"], 0.0)

    def test_rejected_predictions_are_not_paper_executed(self):
        state = paper._new_state()
        pred = prediction(reject_reason="ADX 10<15")

        with patch.object(
            paper,
            "PAPER_EVENTS_FILE",
            self.events,
        ):
            dirty, actions = paper._open_eligible_predictions(
                state,
                [pred],
            )

        self.assertFalse(dirty)
        self.assertEqual(actions, [])
        self.assertEqual(state["positions"], {})

    def test_later_candle_causes_missed_entry_window(self):
        state = paper._new_state()
        pred = prediction()

        with patch.object(
            paper,
            "PAPER_EVENTS_FILE",
            self.events,
        ), patch.object(
            paper.monitor,
            "fetch_latest_candle",
            return_value=candle(),
        ):
            dirty, actions = paper._open_eligible_predictions(
                state,
                [pred],
            )

        self.assertTrue(dirty)
        self.assertEqual(actions, [])
        self.assertEqual(
            state["decisions"][pred["pred_id"]]["status"],
            "MISSED_ENTRY_WINDOW",
        )

    def test_exact_signal_candle_opens_position(self):
        state = paper._new_state()
        pred = prediction()
        signal_candle = {
            **candle(
                open_time=pred["open_time"],
                close_time=pred["observation_contract"]["close_time"],
                close=100.0,
                high=101.0,
                low=99.5,
            )
        }

        with patch.object(
            paper,
            "PAPER_EVENTS_FILE",
            self.events,
        ), patch.object(
            paper.monitor,
            "fetch_latest_candle",
            return_value=signal_candle,
        ):
            dirty, actions = paper._open_eligible_predictions(
                state,
                [pred],
            )

        self.assertTrue(dirty)
        self.assertEqual(actions[0]["action"], "OPENED")
        self.assertIn(pred["pred_id"], state["positions"])
        self.assertEqual(
            state["positions"][pred["pred_id"]]["status"],
            "ACTIVE",
        )

    def test_signal_candle_is_not_reprocessed_as_exit_candle(self):
        state = paper._new_state()
        pred = paper._normalize_prediction(prediction())

        signal_candle = {
            "open_time": pred["open_time"],
            "close_time": pred["observation_contract"]["close_time"],
            "open": 100.0,
            "high": 101.0,
            "low": 99.5,
            "close": 100.0,
        }

        later_candle = candle()

        with patch.object(
            paper,
            "PAPER_EVENTS_FILE",
            self.events,
        ), patch.object(
            paper.monitor,
            "fetch_latest_candle",
            return_value=signal_candle,
        ):
            dirty, actions = paper._open_eligible_predictions(
                state,
                [prediction()],
            )

        self.assertTrue(dirty)
        self.assertEqual(actions[0]["action"], "OPENED")
        position = state["positions"][pred["pred_id"]]
        self.assertEqual(
            position["last_processed_candle_close_time"],
            pred["observation_contract"]["close_time"],
        )
        self.assertEqual(position["bars_processed"], 0)

        with patch.object(
            paper,
            "PAPER_EVENTS_FILE",
            self.events,
        ), patch.object(
            paper.monitor,
            "fetch_latest_candle",
            return_value=signal_candle,
        ), patch.object(
            paper.monitor,
            "process_trade_candle",
        ) as lifecycle:
            dirty_same, actions_same = paper._monitor_open_positions(state)

        self.assertFalse(dirty_same)
        self.assertEqual(actions_same, [])
        lifecycle.assert_not_called()
        self.assertEqual(position["bars_processed"], 0)

        with patch.object(
            paper,
            "PAPER_EVENTS_FILE",
            self.events,
        ), patch.object(
            paper.monitor,
            "fetch_latest_candle",
            return_value=later_candle,
        ), patch.object(
            paper.monitor,
            "process_trade_candle",
            return_value={
                "modified": False,
                "closed": False,
                "events": [],
            },
        ) as lifecycle:
            dirty_later, actions_later = paper._monitor_open_positions(state)

        self.assertTrue(dirty_later)
        self.assertEqual(actions_later, [])
        lifecycle.assert_called_once()
        self.assertEqual(position["bars_processed"], 1)
        self.assertEqual(
            position["last_processed_candle_close_time"],
            later_candle["close_time"],
        )
    def test_open_and_lifecycle_events_record_observation_provenance(self):
        state = paper._new_state()
        pred = paper._normalize_prediction(prediction())

        signal_candle = {
            "open_time": pred["open_time"],
            "close_time": pred["observation_contract"]["close_time"],
            "open": 100.0,
            "high": 101.0,
            "low": 99.5,
            "close": 100.0,
        }
        next_candle = candle()

        with patch.object(
            paper,
            "PAPER_EVENTS_FILE",
            self.events,
        ), patch.object(
            paper.monitor,
            "fetch_latest_candle",
            return_value=signal_candle,
        ):
            dirty, actions = paper._open_eligible_predictions(
                state,
                [prediction()],
            )

        self.assertTrue(dirty)
        self.assertEqual(actions[0]["action"], "OPENED")
        position = state["positions"][pred["pred_id"]]
        self.assertEqual(
            position["entry_observation_open_time"],
            pred["open_time"],
        )
        self.assertEqual(
            position["entry_observation_close_time"],
            pred["observation_contract"]["close_time"],
        )
        self.assertEqual(
            position["entry_observation_identity"],
            pred["observation_identity"],
        )

        with patch.object(
            paper,
            "PAPER_EVENTS_FILE",
            self.events,
        ), patch.object(
            paper.monitor,
            "fetch_latest_candle",
            return_value=next_candle,
        ), patch.object(
            paper.monitor,
            "process_trade_candle",
            return_value={
                "modified": False,
                "closed": False,
                "events": [
                    {
                        "event": "CANDLE_OBSERVED",
                    }
                ],
            },
        ):
            paper._monitor_open_positions(state)

        self.assertEqual(
            position["last_observation_open_time"],
            next_candle["open_time"],
        )
        self.assertEqual(
            position["last_observation_close_time"],
            next_candle["close_time"],
        )
        self.assertEqual(
            position["last_observation_identity"],
            [
                "ETHUSDT",
                next_candle["open_time"],
                "15m",
                "binance",
                "spot",
            ],
        )

        events = [
            json.loads(line)
            for line in self.events.read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        ]
        lifecycle = [
            item
            for item in events
            if item.get("event") == "CANDLE_OBSERVED"
        ][0]
        self.assertEqual(
            lifecycle["candle_observation_identity"],
            position["last_observation_identity"],
        )
    def test_active_position_is_processed_once_per_closed_candle(self):
        state = paper._new_state()
        pred = paper._normalize_prediction(prediction())
        state["positions"][pred["pred_id"]] = paper._build_position(pred)

        next_candle = candle()

        with patch.object(
            paper,
            "PAPER_EVENTS_FILE",
            self.events,
        ), patch.object(
            paper.monitor,
            "fetch_latest_candle",
            return_value=next_candle,
        ), patch.object(
            paper.monitor,
            "process_trade_candle",
            return_value={
                "modified": True,
                "closed": False,
                "events": [
                    {
                        "event": "PARTIAL_TP1_FILLED",
                        "partial_qty": 1.0,
                    }
                ],
            },
        ) as lifecycle:
            dirty1, actions1 = paper._monitor_open_positions(state)
            dirty2, actions2 = paper._monitor_open_positions(state)

        self.assertTrue(dirty1)
        self.assertFalse(dirty2)
        self.assertEqual(len(actions1), 1)
        self.assertEqual(actions1[0]["action"], "UPDATED")
        self.assertEqual(actions1[0]["state"], "OPEN")
        self.assertEqual(actions2, [])
        lifecycle.assert_called_once()

    def test_horizon_expires_remaining_position(self):
        state = paper._new_state()
        pred = paper._normalize_prediction(prediction())
        position = paper._build_position(pred)
        position["bars_processed"] = paper.PAPER_HORIZON_BARS - 1
        state["positions"][pred["pred_id"]] = position

        horizon_candle = candle(close=103.0, high=104.0, low=102.0)

        with patch.object(
            paper,
            "PAPER_EVENTS_FILE",
            self.events,
        ), patch.object(
            paper.monitor,
            "fetch_latest_candle",
            return_value=horizon_candle,
        ), patch.object(
            paper.monitor,
            "process_trade_candle",
            return_value={
                "modified": False,
                "closed": False,
                "events": [],
            },
        ):
            dirty, actions = paper._monitor_open_positions(state)

        self.assertTrue(dirty)
        self.assertEqual(actions[0]["reason"], "EXPIRED")
        self.assertEqual(
            position["exit_reason"],
            "EXPIRED",
        )
        self.assertEqual(position["status"], "CLOSED")
        self.assertEqual(position["remaining_qty"], 0.0)
        self.assertIsNotNone(position["net_r"])

    def test_reconciliation_fails_on_identity_tamper(self):
        state = paper._new_state()
        pred = paper._normalize_prediction(prediction())
        position = paper._build_position(pred)
        position["observation_identity"][1] += 1
        state["positions"][pred["pred_id"]] = position

        with patch.object(
            paper,
            "PAPER_EVENTS_FILE",
            self.events,
        ):
            result = paper.reconcile_state(state)

        self.assertFalse(result["ok"])
        self.assertTrue(
            any(
                "OBSERVATION_IDENTITY_MISMATCH" in item
                for item in result["failures"]
            )
        )

    def test_rejects_non_spot_observation_source(self):
        bad_contract = contract()
        bad_contract["observation_source"] = {
            "exchange": "other-exchange",
            "market_type": "perpetual",
        }

        with self.assertRaises(ValueError):
            paper._normalize_prediction(
                prediction(observation_contract=bad_contract)
            )

    def test_paper_module_has_no_live_execution_symbols(self):
        source = Path(paper.__file__).read_text(encoding="utf-8").lower()
        for forbidden in (
            "deribitclient",
            "execute_trade",
            "place_market_order",
            "place_limit_order",
        ):
            self.assertNotIn(forbidden, source)

    def test_run_once_reports_operational_template_without_predictions(self):
        self.predictions.write_text("[]", encoding="utf-8")
        with patch.object(
            paper,
            "PAPER_STATE_FILE",
            self.state,
        ), patch.object(
            paper,
            "PAPER_EVENTS_FILE",
            self.events,
        ), patch.object(
            paper,
            "PREDICTIONS_FILE",
            self.predictions,
        ):
            result = paper.run_once()

        self.assertEqual(result["report_version"], 1)
        self.assertEqual(result["run"]["execution_mode"], "PAPER_ONLY")
        self.assertEqual(result["input"]["source_status"], "PRESENT_EMPTY")
        self.assertEqual(result["input"]["records_loaded"], 0)
        self.assertEqual(result["input"]["valid_predictions"], 0)
        self.assertEqual(result["input"]["rejected_predictions"], 0)
        self.assertIn("candles_processed", result["lifecycle"])
        self.assertIn("reconciliation", result)

    def test_run_once_reports_prediction_handoff_and_open(self):
        pred = prediction()
        signal_candle = {
            "open_time": pred["open_time"],
            "close_time": pred["observation_contract"]["close_time"],
            "open": 100.0,
            "high": 101.0,
            "low": 99.5,
            "close": 100.0,
        }
        self.predictions.write_text(
            json.dumps([pred]),
            encoding="utf-8",
        )

        with patch.object(
            paper,
            "PAPER_STATE_FILE",
            self.state,
        ), patch.object(
            paper,
            "PAPER_EVENTS_FILE",
            self.events,
        ), patch.object(
            paper,
            "PREDICTIONS_FILE",
            self.predictions,
        ), patch.object(
            paper.monitor,
            "fetch_latest_candle",
            return_value=signal_candle,
        ):
            result = paper.run_once()

        self.assertEqual(result["input"]["source_status"], "PRESENT")
        self.assertEqual(result["input"]["records_loaded"], 1)
        self.assertEqual(result["input"]["valid_predictions"], 1)
        self.assertEqual(result["decisions"]["opened"], 1)
        self.assertEqual(result["portfolio"]["active_positions"], 1)

    def test_run_once_is_noop_without_predictions(self):
        self.predictions.write_text("[]", encoding="utf-8")
        with patch.object(
            paper,
            "PAPER_STATE_FILE",
            self.state,
        ), patch.object(
            paper,
            "PAPER_EVENTS_FILE",
            self.events,
        ), patch.object(
            paper,
            "PREDICTIONS_FILE",
            self.predictions,
        ):
            result = paper.run_once()
            self.assertFalse(result["state_changed"])
            self.assertTrue(result["reconciliation"]["ok"])


if __name__ == "__main__":
    unittest.main()
