from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"


def read(name):
    return (WORKFLOWS / name).read_text(encoding="utf-8")


def test_scanner_uses_shared_state_writer_group_and_explicit_mirror_allowlist():
    text = read("crypto_bot.yml")
    assert "group: cryptobot-state-writers" in text
    assert "queue: max" in text
    assert "cancel-in-progress: false" in text
    assert not re.search(r"(?m)^\s*git add data/\s*(?:\|\| true)?$", text)
    for name in (
        "trades.json", "trade_history.json", "signals.json", "scan_mode.json",
        "balance.json", "orphan_candidates.json", "scan_status.json",
        "predictions.json", "reliability.json", "cooldown.json",
    ):
        assert f"data/{name}" in text
    assert 'if [ -f "$path" ]; then git add -- "$path"; fi' in text
    assert "DERIBIT_TESTNET_MUTATIONS_ENABLED: 'false'" in text


def test_hourly_research_uses_shared_group_and_explicit_mirror_allowlist():
    text = read("hourly_research_pipeline.yml")
    assert "group: cryptobot-state-writers" in text
    assert "queue: max" in text
    assert "cancel-in-progress: false" in text
    assert not re.search(r"(?m)^\s*git add data/\s*$", text)
    for name in ("predictions.json", "coin_dossier.json", "trade_history.json", "adaptation_proposals.json"):
        assert f"data/{name}" in text
    assert 'if [ -f "$path" ]; then git add -- "$path"; fi' in text


def test_candidate_and_paper_state_domains_remain_separate():
    candidate = read("candidate_scanner.yml")
    paper = read("paper_execution.yml")
    assert "group: crypto-bot-candidate-scanner" in candidate
    assert "group: cryptobot-stage2e-paper" in paper


def test_readonly_testnet_preflight_remains_readonly():
    text = read("deribit_testnet_readonly.yml")
    lower = text.lower()
    assert "contents: read" in lower
    assert "workflow_dispatch" in lower
    readonly_source = (ROOT / "testnet_readonly_check.py").read_text(encoding="utf-8")
    assert 'auth_scope="trade:read account:read"' in readonly_source
    assert "allow_mutations=False" in readonly_source
    assert "trade:read_write" not in lower
