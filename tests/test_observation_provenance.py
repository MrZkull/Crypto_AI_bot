import unittest

import trade_executor


class TestObservationProvenance(unittest.TestCase):
    def test_production_prediction_record_emits_canonical_provenance(self):
        record = trade_executor._pred_record(
            "ETHUSDT",
            "BUY",
            0.81,
            {"trained_at": "test"},
            {
                "open_time": 1_700_000_000_000,
                "close": 2000.0,
                "atr": 20.0,
                "rsi": 55.0,
                "adx": 30.0,
            },
            {"ensemble_disagreement": 0.0, "high_disagreement": False},
            pred_id="stage2b-test",
        )

        self.assertEqual(record["logical_symbol"], "ETHUSDT")
        self.assertEqual(record["open_time"], 1_700_000_000_000)
        self.assertEqual(record["interval"], "15m")
        self.assertEqual(
            record["observation_source"],
            {"exchange": "binance", "market_type": "spot"},
        )
        self.assertEqual(record["observation_market"], "BINANCE_SPOT")


if __name__ == "__main__":
    unittest.main()
