import subprocess
import sys
from pathlib import Path


def test_emergency_close_script_is_retired_and_does_not_mutate(tmp_path):
    source = Path(__file__).resolve().parents[1] / "emergency_close.py"
    result = subprocess.run([sys.executable, str(source)], capture_output=True, text=True)
    assert result.returncode != 0
    assert "retired" in (result.stdout + result.stderr).lower()
    code = source.read_text(encoding="utf-8").lower()
    assert "requests.post" not in code
    assert "cancel_all_by_currency" not in code
