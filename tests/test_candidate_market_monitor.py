import json
import unittest
from unittest.mock import patch

import candidate_market_monitor as monitor


class FakeResponse:
    def __init__(self, payload, status=200):
        self.payload = payload
        self.status = status

    def read(self):
        return json.dumps(self.payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def trade(**overrides):
    value = {
        "symbol": "ETHUSDT",
        "side": "BUY",
        "simulated_entry": 100.0,
        "stop": 90.0,
        "tp1": 110.0,
        "tp2": 120.0,
        "qty": 10.0,
        "initial_risk_usd": 100.0,
        "entry_fee_usd": 0.60,
        "state": "OPEN",
        "status": "ACTIVE",
    }
    value.update(overrides)
    return value


class TestCandidateMarketMonitor(unittest.TestCase):

    def test_latest_closed_candle_is_selected(self):
        now_ms = 2_000_000_000_000
        closed = [now_ms - 1000, 100, 101, 99, 100.5, 10, now_ms - 1000]
        live = [now_ms + 10000, 101, 103, 100, 102, 11, now_ms + 10000]

        with patch(
            "candidate_market_monitor.time.time",
            return_value=now_ms / 1000.0,
        ), patch(
            "candidate_market_monitor.urllib.request.urlopen",
            return_value=FakeResponse([closed, live]),
        ):
            candle = monitor.fetch_latest_candle("ETHUSDT")

        self.assertEqual(candle["close"], 100.5)
        self.assertEqual(candle["close_time"], now_ms - 1000)

    def test_tp1_reduces_quantity_and_records_partial_pnl(self):
        t = trade()
        result = monitor.process_trade_candle(
            t,
            {"high": 111.0, "low": 105.0, "close": 110.5, "close_time": 0},
        )

        self.assertTrue(result["modified"])
        self.assertFalse(result["closed"])
        self.assertEqual(t["state"], "PARTIAL_TP1")
        self.assertEqual(t["remaining_qty"], 5.0)
        self.assertAlmostEqual(t["tp1_realized_gross_pnl"], 50.0)
        self.assertAlmostEqual(t["tp1_fee_usd"], 0.33, places=10)

    def test_final_exit_uses_remaining_quantity_and_total_pnl(self):
        t = trade()

        monitor.process_trade_candle(
            t,
            {"high": 111.0, "low": 105.0, "close": 110.5, "close_time": 0},
        )

        result = monitor.process_trade_candle(
            t,
            {"high": 121.0, "low": 115.0, "close": 120.0, "close_time": 0},
        )

        self.assertTrue(result["closed"])
        self.assertEqual(t["final_exit_qty"], 5.0)
        self.assertAlmostEqual(t["realized_gross_pnl"], 150.0)
        self.assertAlmostEqual(t["final_exit_fee_usd"], 0.36, places=10)
        self.assertAlmostEqual(t["net_r"], 1.4871, places=10)
        self.assertEqual(t["exit_reason"], "TAKE_PROFIT")

    def test_same_candle_tp1_tp2_stop_is_ambiguous(self):
        t = trade()

        result = monitor.process_trade_candle(
            t,
            {"high": 121.0, "low": 89.0, "close": 105.0, "close_time": 0},
        )

        self.assertTrue(result["closed"])
        self.assertEqual(t["exit_reason"], "AMBIGUOUS_BARRIER")
        self.assertIsNone(t["exit_price"])
        self.assertIsNone(t["net_r"])
        self.assertNotIn("tp1_realized_gross_pnl", t)

    def test_partial_position_tp2_and_stop_same_candle_is_ambiguous(self):
        t = trade(
            state="PARTIAL_TP1",
            remaining_qty=5.0,
            tp1_realized_gross_pnl=50.0,
            tp1_fee_usd=0.33,
        )

        result = monitor.process_trade_candle(
            t,
            {"high": 121.0, "low": 89.0, "close": 105.0, "close_time": 0},
        )

        self.assertTrue(result["closed"])
        self.assertEqual(t["exit_reason"], "AMBIGUOUS_BARRIER")
        self.assertIsNone(t["net_r"])


if __name__ == "__main__":
    unittest.main()
