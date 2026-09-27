from pathlib import Path
from unittest.mock import patch
import tempfile
import unittest

import pandas as pd
import phase2d_promotion_gate as gate

DAY_MS = 24 * 60 * 60 * 1000


class TestPhase2DHardeningEvidence(unittest.TestCase):

    def test_regime_reference_reconstruction_fixture(self):
        atr_values = pd.Series([
            0.5, 0.6, 0.7,
            1.0, 1.1, 1.2,
            1.5, 1.6, 1.7,
        ])

        low_cutoff = float(atr_values.quantile(0.33))
        high_cutoff = float(atr_values.quantile(0.66))

        self.assertAlmostEqual(low_cutoff, 0.892, places=12)
        self.assertAlmostEqual(high_cutoff, 1.284, places=12)

        cutoffs = [low_cutoff, high_cutoff]

        self.assertEqual(
            gate._assign_volatility_tercile(0.80, cutoffs),
            "LOW",
        )
        self.assertEqual(
            gate._assign_volatility_tercile(1.00, cutoffs),
            "MID",
        )
        self.assertEqual(
            gate._assign_volatility_tercile(1.50, cutoffs),
            "HIGH",
        )

    def test_paired_day_intersection_hand_checked(self):
        candidate = pd.DataFrame({
            "open_time": [
                0 * DAY_MS,
                1 * DAY_MS,
                2 * DAY_MS,
                3 * DAY_MS,
            ],
            "net_r": [1.0, 2.0, 3.0, 4.0],
            "regime_atr_pct": [1.0] * 4,
        })

        production = pd.DataFrame({
            "open_time": [
                1 * DAY_MS,
                2 * DAY_MS,
                3 * DAY_MS,
                4 * DAY_MS,
            ],
            "net_r": [10.0, 20.0, 30.0, 40.0],
            "regime_atr_pct": [1.0] * 4,
        })

        paired = gate._paired_daily_means(
            candidate,
            production,
        )

        self.assertEqual(
            paired["day_id"].tolist(),
            [1, 2, 3],
        )
        self.assertEqual(
            paired["candidate_daily_mean_net_r"].tolist(),
            [2.0, 3.0, 4.0],
        )
        self.assertEqual(
            paired["production_daily_mean_net_r"].tolist(),
            [10.0, 20.0, 30.0],
        )
        self.assertEqual(
            paired["difference_daily_mean_net_r"].tolist(),
            [-8.0, -17.0, -26.0],
        )

    def test_early_harm_uses_same_ci_object(self):
        paired = pd.DataFrame({
            "candidate_daily_mean_net_r": [0.1] * 15,
            "production_daily_mean_net_r": [0.5] * 15,
            "difference_daily_mean_net_r": [-0.4] * 15,
        })

        candidate = pd.DataFrame(index=range(15))
        production = pd.DataFrame(index=range(15))

        fake_bootstrap = {
            "candidate_ci": [0.01, 0.50],
            "production_ci": [0.20, 0.80],
            "difference_ci": [-0.80, -0.20],
        }

        with tempfile.TemporaryDirectory() as tmp:
            audit_path = Path(tmp) / "experiment_ledger.jsonl"
            reference_path = (
                Path(tmp) / "phase2d_regime_reference.json"
            )

            with patch.object(
                gate,
                "_paired_bootstrap",
                return_value=fake_bootstrap,
            ) as bootstrap_mock, patch.object(
                gate,
                "_coverage_gate",
                return_value={"ready": False},
            ):
                result = gate._decision_from_checkpoint(
                    candidate,
                    production,
                    paired,
                    checkpoint_blocks=15,
                    cutoff_iso="2026-09-28T00:00:00+00:00",
                    cutoff_ms=0,
                    experiment_id="PHASE2D-HARDENING-TEST",
                    audit_ledger_path=audit_path,
                    regime_reference_path=reference_path,
                )

        self.assertEqual(bootstrap_mock.call_count, 1)
        self.assertEqual(
            result["difference_ci"],
            [-0.80, -0.20],
        )
        self.assertTrue(result["early_harm"])
        self.assertEqual(
            result["decision"],
            "EARLY_HARM_STOP",
        )
        self.assertEqual(
            result["confidence_level"],
            gate.CONF,
        )


if __name__ == "__main__":
    unittest.main()
