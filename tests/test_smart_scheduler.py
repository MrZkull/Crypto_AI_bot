import unittest
from datetime import datetime, timezone
from unittest.mock import patch

import smart_scheduler as scheduler


class TestDrawdownRatchet(unittest.TestCase):
    def _today(self):
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def _history(self, pnl):
        return [{
            "pnl": pnl,
            "closed_at": f"{self._today()}T12:00:00+00:00",
            "close_reason": "STOP_LOSS",
        }]

    def test_under_two_percent_loss_uses_day_start_balance(self):
        # Day-start = 100.00, current = 98.01, PnL = -1.99%.
        with patch.object(
            scheduler,
            "_read_balance_and_history",
            return_value=({"usdt": 98.01}, self._history(-1.99)),
        ):
            self.assertEqual(scheduler.get_drawdown_ratchet(), 1.0)

    def test_under_five_percent_loss_does_not_halt_from_shrinking_denominator(self):
        # Day-start = 100.00, current = 95.10, PnL = -4.90%.
        # The previous current-balance denominator would incorrectly cross -5%.
        with patch.object(
            scheduler,
            "_read_balance_and_history",
            return_value=({"usdt": 95.10}, self._history(-4.90)),
        ):
            self.assertEqual(scheduler.get_drawdown_ratchet(), 0.5)

    def test_five_percent_loss_halts(self):
        # Day-start = 100.00, current = 95.00, PnL = -5.00%.
        with patch.object(
            scheduler,
            "_read_balance_and_history",
            return_value=({"usdt": 95.00}, self._history(-5.00)),
        ):
            self.assertEqual(scheduler.get_drawdown_ratchet(), 0.0)

    def test_positive_daily_pnl_does_not_reduce_risk(self):
        with patch.object(
            scheduler,
            "_read_balance_and_history",
            return_value=({"usdt": 105.00}, self._history(5.00)),
        ):
            self.assertEqual(scheduler.get_drawdown_ratchet(), 1.0)

    def test_invalid_reconstructed_day_start_fails_closed(self):
        # current_balance - today's PnL must remain positive.
        with patch.object(
            scheduler,
            "_read_balance_and_history",
            return_value=({"usdt": 5.00}, self._history(-10.00)),
        ):
            self.assertEqual(scheduler.get_drawdown_ratchet(), 0.0)


if __name__ == "__main__":
    unittest.main()
