"""Contract tests binding the F08 pre-registration to the frozen promotion gate."""
from __future__ import annotations

import ast
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "F08_PROSPECTIVE_PREREGISTRATION_DECISION_TABLE.md"
GATE = ROOT / "phase2d_promotion_gate.py"

EXPECTED_GATE_CANONICAL_SHA256 = (
    "f787bc3caa810c9f82f56b44d2d8624637693e47eb4613e86a4c30f461c2faab"
)


def _read_lf(path: Path) -> str:
    return path.read_text(encoding="utf-8").replace("\r\n", "\n")


def _top_level_assignments(tree: ast.Module) -> dict[str, ast.AST]:
    result: dict[str, ast.AST] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                result[target.id] = node.value
    return result


def _literal_assignments(source: str) -> dict[str, object]:
    tree = ast.parse(source)
    result = {}
    for name, value in _top_level_assignments(tree).items():
        try:
            result[name] = ast.literal_eval(value)
        except (ValueError, TypeError):
            continue
    return result


def _function_source(tree: ast.Module, source: str, name: str) -> str:
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            segment = ast.get_source_segment(source, node)
            assert segment is not None
            return segment
    raise AssertionError(f"Frozen gate function not found: {name}")


def test_preregistration_document_exists_and_is_marked_not_yet_effective() -> None:
    assert DOC.is_file()
    doc = _read_lf(DOC)
    assert "F08-PRE-REGISTRATION-2026-11-11-v1" in doc
    assert "not effective merely because it exists in a local worktree" in doc
    assert "committed to `main` before any prospective F08 restart" in doc


def test_frozen_promotion_gate_hash_matches_pregistered_identity() -> None:
    source = _read_lf(GATE)
    actual = hashlib.sha256(source.encode("utf-8")).hexdigest()
    assert actual == EXPECTED_GATE_CANONICAL_SHA256


def test_fixed_sample_checkpoints_bootstrap_and_alpha_match_frozen_gate() -> None:
    source = _read_lf(GATE)
    values = _literal_assignments(source)

    assert values["MIN_RESOLVED"] == 50
    assert values["MIN_UNIQUE_BLOCKS"] == 15
    assert values["MIN_INDIVIDUAL_BLOCKS"] == 15
    assert values["MIN_PAIRED_BLOCKS"] == 15
    assert values["CHECKPOINT_BLOCKS"] == (15, 20, 25, 30)
    assert values["FINAL_CHECKPOINT"] == 30
    assert values["BOOTSTRAP_ROUNDS"] == 10_000
    assert values["BOOTSTRAP_SEED"] == 42
    assert values["ALPHA_TOTAL"] == 0.05
    assert values["MAX_CALENDAR_DAYS"] == 90
    assert "ALPHA_PER_CHECKPOINT = ALPHA_TOTAL / len(CHECKPOINT_BLOCKS)" in source
    assert "CONF = 1.0 - ALPHA_PER_CHECKPOINT" in source


def test_coverage_requirements_match_frozen_gate() -> None:
    source = _read_lf(GATE)
    tree = ast.parse(source)
    coverage = _function_source(tree, source, "_coverage_gate")

    assert "final_days = paired.tail(" in coverage
    assert "weekend_ok = weekend >= 3" in coverage
    assert "weekday_ok = weekday >= 10" in coverage
    assert '"weekend_requirement": 3' in coverage
    assert '"weekday_requirement": 10' in coverage
    assert "math.ceil(" in coverage
    assert "0.15" in coverage
    assert "len(final_days)" in coverage
    assert '"required_terciles": 2' in coverage
    assert "represented >= 2" in coverage
    assert "regime_ready" in coverage
    assert "REGIME_REFERENCE_FILE" in source


def test_efficacy_harm_and_c30_terminal_decisions_match_frozen_gate() -> None:
    source = _read_lf(GATE)
    tree = ast.parse(source)
    decision = _function_source(tree, source, "_decision_from_checkpoint")

    assert "candidate_ci[0] > 0.0" in decision
    assert "difference_ci[0] > 0.0" in decision
    assert "difference_ci[1] < 0.0" in decision
    assert '"EARLY_HARM_STOP"' in decision
    assert '"EFFICACY_PASS"' in decision
    assert "checkpoint_blocks == FINAL_CHECKPOINT" in decision
    assert '"FINAL_CHECKPOINT_NO_EFFICACY"' in decision

    promotion = _function_source(tree, source, "evaluate_promotion")
    assert '"FINAL_CHECKPOINT_NO_EFFICACY"' in promotion
    assert '"experiment_stopped": True' in promotion
    assert "90-calendar-day hard ceiling reached" in promotion


def test_preregistration_records_618_cohort_exclusion_and_non_independence() -> None:
    doc = _read_lf(DOC)
    assert "618 prediction records: 74 candidate and 544 production" in doc
    assert "573 unique symbol/open-time windows" in doc
    assert "45 common candidate/production windows" in doc
    assert "528 unpaired windows" in doc
    assert "not 618 independent observations" in doc
    assert "must remain outside the primary prospective score" in doc


def test_no_extra_checkpoint_or_mid_experiment_policy_change() -> None:
    doc = _read_lf(DOC)
    assert "There is no C35, C40" in doc
    assert "new pre-registration and mentor approval" in doc
    assert "must not be changed mid-experiment" in doc
