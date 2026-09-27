import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(
    0,
    str(Path(__file__).parents[1]),
)

import phase2d_promotion_gate as gate

from phase2d_promotion_gate import (
    MIN_UNIQUE_BLOCKS,
    CHECKPOINT_BLOCKS,
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
        model="candidate",
        regime_atr_pct=1.0,
        resolved_at_base="2026-09-27T18:00:00+00:00",
        observations_per_day=1,
    ):
        records = []

        base = (
            __import__("datetime")
            .datetime.fromisoformat(
                resolved_at_base
            )
        )

        for i in range(n):
            if spread_days:
                day_index = i // observations_per_day
                intra_day_index = i % observations_per_day

                open_time = (
                    (offset + day_index)
                    * DAY_MS
                    + intra_day_index * FIFTEEN_MIN_MS
                )
            else:
                open_time = (
                    offset * DAY_MS
                    + i * FIFTEEN_MIN_MS
                )

            resolved_at = (
                base
                + __import__("datetime").timedelta(
                    minutes=i
                )
            ).isoformat()

            records.append(
                {
                    "id": (
                        f"{model}:"
                        f"{status}:"
                        f"{offset}:"
                        f"{i}"
                    ),
                    "model": model,
                    "open_time": open_time,
                    "resolved_at": resolved_at,
                    "net_r": net_r,
                    "status": status,
                    "signal": signal,
                    "regime_atr_pct": regime_atr_pct,
                }
            )

        return records

    def _paths(self):
        temp = tempfile.TemporaryDirectory()
        root = Path(temp.name)

        audit = root / "experiment_ledger.jsonl"
        state = root / "phase2d_state.json"
        reference = root / "phase2d_regime_reference.json"

        return temp, audit, state, reference

    def _write_reference(self, path):
        path.write_text(
            json.dumps(
                {
                    "experiment_id": "PHASE2D-HARDENED-20260924-F08",
                    "reference_metric": "atr_pct",
                    "atr_pct_tercile_cutoffs": [
                        0.75,
                        1.25,
                    ],
                    "source": "TEST_REFERENCE_ONLY",
                }
            ),
            encoding="utf-8",
        )

    def test_insufficient_sample_is_fail_closed(self):
        temp, audit, state, reference = self._paths()

        try:
            result = evaluate_promotion(
                self._records(
                    49,
                    1.0,
                    model="candidate",
                ),
                self._records(
                    50,
                    0.1,
                    offset=1000,
                    model="production",
                ),
                audit_ledger_path=audit,
                state_path=state,
                regime_reference_path=reference,
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
        finally:
            temp.cleanup()

    def test_pairing_requires_fifteen_common_days(self):
        temp, audit, state, reference = self._paths()

        try:
            candidate = self._records(
                15,
                0.5,
                offset=0,
                model="candidate",
            )

            production = self._records(
                14,
                0.1,
                offset=0,
                model="production",
            )

            result = evaluate_promotion(
                candidate,
                production,
                audit_ledger_path=audit,
                state_path=state,
                regime_reference_path=reference,
            )

            self.assertEqual(
                result["unique_blocks_candidate"],
                15,
            )

            self.assertEqual(
                result["unique_blocks_production"],
                14,
            )

            self.assertEqual(
                result["paired_unique_blocks"],
                14,
            )

            self.assertFalse(
                result["diversity_gate"]
            )

            self.assertLess(
                result["paired_unique_blocks"],
                MIN_UNIQUE_BLOCKS,
            )
        finally:
            temp.cleanup()

    def test_paired_bootstrap_pass_requires_frozen_regime_reference_and_weekend_mix(self):
        temp, audit, state, reference = self._paths()

        try:
            # Start on Monday so the 15-day window naturally contains
            # 3+ weekend days and 10+ weekdays.
            start_offset = 25921
            candidate = self._records(
                60,
                0.60,
                offset=start_offset,
                model="candidate",
                regime_atr_pct=1.0,
                observations_per_day=4,
            )
            production = self._records(
                60,
                0.10,
                offset=start_offset,
                model="production",
                regime_atr_pct=1.0,
                observations_per_day=4,
            )

            self._write_reference(
                reference
            )

            with patch.object(
                gate,
                "BOOTSTRAP_ROUNDS",
                500,
            ):
                result = evaluate_promotion(
                    candidate,
                    production,
                    audit_ledger_path=audit,
                    state_path=state,
                    regime_reference_path=reference,
                )

            # Because the synthetic data has only one volatility tercile,
            # the checkpoint must remain blocked rather than promoting.
            self.assertEqual(
                result["paired_unique_blocks"],
                15,
            )

            self.assertFalse(
                result["promotion_ready"]
            )

            self.assertEqual(
                len(
                    result[
                        "new_checkpoint_evaluations"
                    ]
                ),
                1,
            )

            checkpoint = result[
                "new_checkpoint_evaluations"
            ][0]

            self.assertEqual(
                checkpoint["checkpoint_blocks"],
                15,
            )

            self.assertEqual(
                checkpoint["decision"],
                "CHECKPOINT_CONTINUE",
            )

            self.assertFalse(
                checkpoint["coverage"]["ready"]
            )
        finally:
            temp.cleanup()

    def test_checkpoint_is_not_re_evaluated(self):
        temp, audit, state, reference = self._paths()

        try:
            candidate = self._records(
                60,
                0.20,
                offset=25921,
                model="candidate",
                regime_atr_pct=1.0,
                observations_per_day=4,
            )
            production = self._records(
                60,
                0.20,
                offset=25921,
                model="production",
                regime_atr_pct=1.0,
                observations_per_day=4,
            )

            self._write_reference(
                reference
            )

            with patch.object(
                gate,
                "BOOTSTRAP_ROUNDS",
                200,
            ):
                first = evaluate_promotion(
                    candidate,
                    production,
                    audit_ledger_path=audit,
                    state_path=state,
                    regime_reference_path=reference,
                )

                second = evaluate_promotion(
                    candidate,
                    production,
                    audit_ledger_path=audit,
                    state_path=state,
                    regime_reference_path=reference,
                )

            self.assertEqual(
                len(
                    first[
                        "new_checkpoint_evaluations"
                    ]
                ),
                1,
            )

            self.assertEqual(
                second[
                    "new_checkpoint_evaluations"
                ],
                [],
            )

            self.assertEqual(
                second[
                    "latest_evaluated_checkpoint"
                ],
                15,
            )
        finally:
            temp.cleanup()

    def test_early_relative_harm_stops_at_checkpoint(self):
        temp, audit, state, reference = self._paths()

        try:
            candidate = self._records(
                60,
                -0.50,
                offset=25921,
                model="candidate",
                regime_atr_pct=1.0,
                observations_per_day=4,
            )
            production = self._records(
                60,
                0.50,
                offset=25921,
                model="production",
                regime_atr_pct=1.0,
                observations_per_day=4,
            )

            self._write_reference(
                reference
            )

            with patch.object(
                gate,
                "BOOTSTRAP_ROUNDS",
                200,
            ):
                result = evaluate_promotion(
                    candidate,
                    production,
                    audit_ledger_path=audit,
                    state_path=state,
                    regime_reference_path=reference,
                )

            self.assertEqual(
                result["gate_status"],
                "FAIL",
            )

            self.assertTrue(
                result[
                    "experiment_stopped"
                ]
            )

            self.assertIn(
                "harm",
                result["reason"].lower(),
            )

            again = evaluate_promotion(
                candidate,
                production,
                audit_ledger_path=audit,
                state_path=state,
                regime_reference_path=reference,
            )

            self.assertEqual(
                again["gate_status"],
                "FAIL",
            )

            self.assertTrue(
                again[
                    "experiment_stopped"
                ]
            )
        finally:
            temp.cleanup()

    def test_final_30_block_checkpoint_can_fail_closed(self):
        temp, audit, state, reference = self._paths()

        try:
            candidate = self._records(
                120,
                0.10,
                offset=25921,
                model="candidate",
                regime_atr_pct=1.0,
                observations_per_day=4,
            )
            production = self._records(
                120,
                0.10,
                offset=25921,
                model="production",
                regime_atr_pct=1.0,
                observations_per_day=4,
            )

            self._write_reference(
                reference
            )

            # Use exact floating-point equality for candidate/prod,
            # so the paired difference is approximately zero and does
            # not trigger relative-harm stopping.
            with patch.object(
                gate,
                "BOOTSTRAP_ROUNDS",
                200,
            ):
                result = evaluate_promotion(
                    candidate,
                    production,
                    audit_ledger_path=audit,
                    state_path=state,
                    regime_reference_path=reference,
                )

            self.assertEqual(
                result["gate_status"],
                "FAIL",
            )

            self.assertFalse(
                result["promotion_ready"],
            )
        finally:
            temp.cleanup()


if __name__ == "__main__":
    unittest.main()
