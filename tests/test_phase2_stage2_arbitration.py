import unittest

from phase2_stage2_arbitration import arbitrate_observations, arbitrate_signals


class TestStage2Arbitration(unittest.TestCase):
    BASE = {
        "candidate_symbol": "ETHUSDT",
        "candidate_open_time": 1_700_000_000_000,
        "production_symbol": "ETHUSDT",
        "production_open_time": 1_700_000_000_000,
    }

    PROVENANCE = {
        "logical_symbol": "ETHUSDT",
        "interval": "15m",
        "observation_source": {
            "exchange": "binance",
            "market_type": "spot",
        },
    }

    def call(self, candidate_signal, production_signal, **overrides):
        args = {
            **self.BASE,
            "candidate_signal": candidate_signal,
            "production_signal": production_signal,
            **overrides,
        }
        return arbitrate_signals(**args)

    def test_production_sell_wins_candidate_buy_conflict(self):
        result = self.call("BUY", "SELL")
        self.assertEqual(result.final_signal, "SELL")
        self.assertEqual(result.selected_policy, "production")
        self.assertEqual(result.reason, "PRODUCTION_SELL_PRECEDENCE")
        self.assertTrue(result.comparable)

    def test_both_buy_produce_same_policy_without_substitution(self):
        result = self.call("BUY", "BUY")
        self.assertEqual(result.final_signal, "BUY")
        self.assertEqual(result.selected_policy, "production")
        self.assertEqual(result.reason, "BOTH_BUY_NO_POLICY_CHANGE")

    def test_candidate_buy_substitutes_production_no_trade(self):
        result = self.call("BUY", "NO_TRADE")
        self.assertEqual(result.final_signal, "BUY")
        self.assertEqual(result.selected_policy, "candidate")
        self.assertEqual(result.reason, "CANDIDATE_BUY_SUBSTITUTION")

    def test_candidate_no_trade_does_not_replace_production_buy(self):
        result = self.call("NO_TRADE", "BUY")
        self.assertEqual(result.final_signal, "BUY")
        self.assertEqual(result.selected_policy, "production")
        self.assertEqual(result.reason, "CANDIDATE_NOT_ELIGIBLE_FOR_SUBSTITUTION")

    def test_candidate_sell_does_not_replace_production_buy(self):
        result = self.call("SELL", "BUY")
        self.assertEqual(result.final_signal, "BUY")
        self.assertEqual(result.selected_policy, "production")
        self.assertEqual(result.reason, "CANDIDATE_NOT_ELIGIBLE_FOR_SUBSTITUTION")

    def test_candidate_sell_does_not_replace_production_no_trade(self):
        result = self.call("SELL", "NO_TRADE")
        self.assertEqual(result.final_signal, "NO_TRADE")
        self.assertEqual(result.selected_policy, "production")
        self.assertEqual(result.reason, "CANDIDATE_NOT_ELIGIBLE_FOR_SUBSTITUTION")

    def test_candidate_no_trade_preserves_production_sell(self):
        result = self.call("NO_TRADE", "SELL")
        self.assertEqual(result.final_signal, "SELL")
        self.assertEqual(result.selected_policy, "production")
        self.assertEqual(result.reason, "PRODUCTION_SELL_PRECEDENCE")

    def test_same_symbol_different_timestamp_is_not_comparable(self):
        result = self.call(
            "BUY",
            "NO_TRADE",
            production_open_time=self.BASE["production_open_time"] + 15 * 60 * 1000,
        )
        self.assertEqual(result.final_signal, "NO_TRADE")
        self.assertEqual(result.reason, "TIMESTAMP_MISMATCH")
        self.assertFalse(result.comparable)

    def test_different_symbol_same_timestamp_is_not_comparable(self):
        result = self.call("BUY", "NO_TRADE", production_symbol="BTCUSDT")
        self.assertEqual(result.final_signal, "NO_TRADE")
        self.assertEqual(result.reason, "SYMBOL_MISMATCH")
        self.assertFalse(result.comparable)

    def test_production_sell_invariant_holds_for_all_candidate_signals(self):
        for candidate_signal in ("BUY", "SELL", "NO_TRADE"):
            result = self.call(candidate_signal, "SELL")
            self.assertEqual(result.final_signal, "SELL")
            self.assertEqual(result.selected_policy, "production")
            self.assertEqual(result.reason, "PRODUCTION_SELL_PRECEDENCE")

    @staticmethod
    def _observation(signal: str, **overrides) -> dict:
        data = {
            **TestStage2Arbitration.PROVENANCE,
            "symbol": "ETHUSDT",
            "open_time": 1_700_000_000_000,
            "signal": signal,
        }
        data.update(overrides)
        return data

    def test_real_observation_shape_same_timestamp_is_comparable(self):
        result = arbitrate_observations(
            self._observation("BUY"),
            self._observation("NO_TRADE"),
        )
        self.assertTrue(result.comparable)
        self.assertEqual(result.final_signal, "BUY")
        self.assertEqual(result.selected_policy, "candidate")
        self.assertEqual(result.reason, "CANDIDATE_BUY_SUBSTITUTION")

    def test_real_observation_shape_timestamp_mismatch_blocks_substitution(self):
        result = arbitrate_observations(
            self._observation("BUY"),
            self._observation(
                "NO_TRADE",
                open_time=1_700_000_000_000 + 15 * 60 * 1000,
            ),
        )
        self.assertFalse(result.comparable)
        self.assertEqual(result.final_signal, "NO_TRADE")
        self.assertEqual(result.reason, "TIMESTAMP_MISMATCH")

    def test_real_observation_shape_different_symbol_blocks_substitution(self):
        result = arbitrate_observations(
            self._observation("BUY"),
            self._observation(
                "NO_TRADE",
                symbol="BTCUSDT",
                logical_symbol="BTCUSDT",
            ),
        )
        self.assertFalse(result.comparable)
        self.assertEqual(result.final_signal, "NO_TRADE")
        self.assertEqual(result.reason, "SYMBOL_MISMATCH")

    def test_real_observation_shape_production_sell_still_has_precedence(self):
        result = arbitrate_observations(
            self._observation("BUY"),
            self._observation("SELL"),
        )
        self.assertTrue(result.comparable)
        self.assertEqual(result.final_signal, "SELL")
        self.assertEqual(result.reason, "PRODUCTION_SELL_PRECEDENCE")

    def test_real_observation_shape_accepts_real_record_field_names(self):
        candidate = {
            **self._observation("BUY"),
            "signal": None,
            "side": "BUY",
        }
        production = {
            **self._observation("NO_TRADE"),
            "signal": None,
            "predicted_signal": "NO_TRADE",
        }
        result = arbitrate_observations(candidate, production)
        self.assertTrue(result.comparable)
        self.assertEqual(result.final_signal, "BUY")
        self.assertEqual(result.reason, "CANDIDATE_BUY_SUBSTITUTION")

    def test_real_observation_shape_source_mismatch_blocks_substitution(self):
        result = arbitrate_observations(
            self._observation("BUY"),
            self._observation(
                "NO_TRADE",
                observation_source={
                    "exchange": "deribit",
                    "market_type": "linear_usdc_perpetual",
                },
            ),
        )
        self.assertFalse(result.comparable)
        self.assertEqual(result.final_signal, "NO_TRADE")
        self.assertEqual(result.reason, "OBSERVATION_SOURCE_MISMATCH")

    def test_real_observation_shape_interval_mismatch_blocks_substitution(self):
        result = arbitrate_observations(
            self._observation("BUY"),
            self._observation("NO_TRADE", interval="1h"),
        )
        self.assertFalse(result.comparable)
        self.assertEqual(result.final_signal, "NO_TRADE")
        self.assertEqual(result.reason, "INTERVAL_MISMATCH")

    def test_real_observation_shape_missing_provenance_blocks_substitution(self):
        candidate = {
            "symbol": "ETHUSDT",
            "open_time": 1_700_000_000_000,
            "signal": "BUY",
        }
        production = self._observation("NO_TRADE")
        result = arbitrate_observations(candidate, production)
        self.assertFalse(result.comparable)
        self.assertEqual(result.final_signal, "NO_TRADE")
        self.assertEqual(result.selected_policy, "production")
        self.assertEqual(result.reason, "MISSING_OBSERVATION_PROVENANCE")


if __name__ == "__main__":
    unittest.main()
