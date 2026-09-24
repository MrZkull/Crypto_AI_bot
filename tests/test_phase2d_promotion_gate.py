import sys
import unittest
from pathlib import Path

sys.path.insert(
    0,
    str(Path(__file__).parents[1]),
)

from phase2d_promotion_gate import (
    MIN_UNIQUE_BLOCKS,
    evaluate_promotion,
)


DAY_MS = 24 * 60 * 60 * 1000
FIFTEEN_MIN_MS = 15 * 60 * 1000


class TestPhase2DPromotionGate(unittest.TestCase):

    def _records(
        self,
        n,
        net_r,
        status="TP",
        offset=0,
        spread_days=True,
        signal="BUY",
    ):
        records = []

        for i in range(n):
            if spread_days:
                open_time = (
                    (offset + i)
                    * DAY_MS
                    + i * FIFTEEN_MIN_MS
                )
            else:
                open_time = (
                    (offset + i)
                    * FIFTEEN_MIN_MS
                )

            records.append(
                {
                    "id": f"{status}:{offset}:{i}",
                    "open_time": open_time,
                    "net_r": net_r,
                    "status": status,
                    "signal": signal,
                }
            )

        return records

    def test_insufficient_sample_is_fail_closed(self):
        result = evaluate_promotion(
            self._records(
                49,
                1.0,
            ),
            self._records(
                50,
                0.1,
                offset=1000,
            ),
        )

        self.assertFalse(
            result["promotion_ready"]
        )

        self.assertEqual(
            result["gate_status"],
            "NOT_READY",
        )

        self.assertIn(
            "candidate N=49",
            result["reason"],
        )

    def test_unique_block_requirement_is_symmetric(self):
        result = evaluate_promotion(
            self._records(
                50,
                0.5,
                spread_days=False,
            ),
            self._records(
                50,
                0.1,
                offset=1000,
                spread_days=True,
            ),
        )

        self.assertFalse(
            result["promotion_ready"]
        )

        self.assertEqual(
            result[
                "unique_blocks_candidate"
            ],
            1,
        )

        self.assertLess(
            result[
                "unique_blocks_candidate"
            ],
            MIN_UNIQUE_BLOCKS,
        )

        self.assertFalse(
            result["diversity_gate"]
        )

    def test_positive_candidate_not_enough_if_not_better(self):
        result = evaluate_promotion(
            self._records(
                60,
                0.20,
                signal="BUY",
            ),
            self._records(
                60,
                0.30,
                offset=1000,
                signal="BUY",
            ),
        )

        self.assertFalse(
            result["promotion_ready"]
        )

        self.assertFalse(
            result["candidate_beats_production"]
        )

    def test_ambiguous_and_invalid_are_excluded(self):
        candidate = (
            self._records(
                50,
                0.5,
            )
            + self._records(
                20,
                99.0,
                status="AMBIGUOUS",
                offset=1000,
            )
        )

        production = self._records(
            50,
            0.1,
            offset=2000,
        )

        result = evaluate_promotion(
            candidate,
            production,
        )

        self.assertEqual(
            result["n_candidate"],
            50,
        )

        self.assertEqual(
            result["n_production"],
            50,
        )


if __name__ == "__main__":
    unittest.main()
