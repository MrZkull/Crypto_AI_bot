from pathlib import Path
import unittest


ROOT = Path(__file__).parents[1]


class TestPaperExecutionWorkflow(unittest.TestCase):
    def test_paper_workflow_is_explicitly_paper_only(self):
        workflow = (
            ROOT / ".github" / "workflows" / "paper_execution.yml"
        ).read_text(encoding="utf-8")

        self.assertIn("cron: '9 * * * *'", workflow)
        self.assertNotIn("cron: '9,24,39,54 * * * *'", workflow)
        self.assertIn("python paper_execution.py", workflow)
        self.assertIn("permissions:\n  contents: write", workflow)
        self.assertNotIn("deribit", workflow.lower())
        self.assertNotIn("execute_trade", workflow)
        self.assertNotIn("place_market_order", workflow)
        self.assertNotIn("DERIBIT_", workflow)


if __name__ == "__main__":
    unittest.main()

