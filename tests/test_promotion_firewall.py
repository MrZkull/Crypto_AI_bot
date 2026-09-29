import unittest
from pathlib import Path
from unittest.mock import patch

import promote_candidate


ROOT = Path(__file__).parents[1]


class TestPromotionFirewall(unittest.TestCase):
    def test_legacy_promotion_is_disabled(self):
        self.assertFalse(promote_candidate.LEGACY_PROMOTION_ENABLED)
        self.assertEqual(
            promote_candidate.PROMOTION_STATE,
            "PAPER_STABILITY_REQUIRED",
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "Legacy candidate promotion is retired",
        ):
            promote_candidate.promote()

    def test_legacy_workflow_has_no_schedule_or_promote_invocation(self):
        workflow = (
            ROOT / ".github" / "workflows" / "promote_candidate.yml"
        ).read_text(encoding="utf-8")

        self.assertNotIn("schedule:", workflow)
        self.assertNotIn("python promote_candidate.py", workflow)
        self.assertIn("Legacy Candidate Promotion (Retired)", workflow)
        self.assertIn("No model cutover is permitted.", workflow)

    def test_promote_guard_has_no_filesystem_or_release_side_effects(self):
        with patch.object(promote_candidate, "subprocess") as subprocess_mock:
            with self.assertRaises(RuntimeError):
                promote_candidate.promote()
            subprocess_mock.run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
