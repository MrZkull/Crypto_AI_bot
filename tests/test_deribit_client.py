import unittest
from unittest.mock import MagicMock
from deribit_client import DeribitClient


class DeribitClientUnitTests(unittest.TestCase):

    def setUp(self):
        self.client = DeribitClient.__new__(DeribitClient)
        self.client.get_instrument_info = MagicMock(return_value={
            "tick_size": 0.0005,
            "min_trade_amount": 0.001,
            "tick_size_steps": [
                {"above_price": 100.0, "tick_size": 0.001},
                {"above_price": 200.0, "tick_size": 0.005}
            ]
        })

    def test_tick_ladder_exact_boundaries(self):
        eps = 1e-6
        # Tier 0 -> Tier 1 boundary around 100.0
        self.assertEqual(self.client.get_tick_size("TEST", price=100.0 - eps), 0.0005)
        self.assertEqual(self.client.get_tick_size("TEST", price=100.0), 0.001)
        self.assertEqual(self.client.get_tick_size("TEST", price=100.0 + eps), 0.001)

        # Tier 1 -> Tier 2 boundary around 200.0
        self.assertEqual(self.client.get_tick_size("TEST", price=200.0 - eps), 0.001)
        self.assertEqual(self.client.get_tick_size("TEST", price=200.0), 0.005)
        self.assertEqual(self.client.get_tick_size("TEST", price=200.0 + eps), 0.005)

    def test_round_amount_floor_and_dust_elimination(self):
        # min_trade_amount = 0.001
        self.assertEqual(self.client.round_amount("TEST", 0.0029), 0.002)
        self.assertEqual(self.client.round_amount("TEST", 0.0010), 0.001)
        # Sizing below legal minimum lot returns 0.0
        self.assertEqual(self.client.round_amount("TEST", 0.0009), 0.0)

    def test_split_amount_single_vs_dual(self):
        # total 0.002 divides into 0.001 and 0.001
        self.assertEqual(self.client.split_amount("TEST", 0.002), (0.001, 0.001))
        # total 0.001 cannot be halved without dropping below min_lot 0.001 -> Single TP mode
        self.assertEqual(self.client.split_amount("TEST", 0.001), (0.001, 0.0))


if __name__ == "__main__":
    unittest.main()
