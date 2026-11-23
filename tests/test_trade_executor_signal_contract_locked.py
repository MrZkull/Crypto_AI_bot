from __future__ import annotations

import ast
import hashlib
from pathlib import Path

BASELINE_COMMIT = "a30c8084445a41e9dd220513bfb089d2cbe47587"
EXPECTED_FUNCTION_SHA256 = {
    'generate_signal': 'c0f3bdea1270651817f26970bde57f64ee174e67f2e02ee18b924ccbcd95bc80',
    '_build_signal_observation_contract': '80e3293ff2ba0448c6145f003c030273d2b4f426b7a4e990d28c780a918184ca',
    'save_signal': '03eedf06a6d23bcaa3ff860a3fb77ef23e09184680afb075f78f2e21a46b1cca',
    'save_prediction': '3b88eded6387bd0c4405d3b16f0d277eff8b902ef9e66ae80dd902833156b7b1',
    'mark_prediction_executed': 'b4e357fa7a4cdb5270e783b4a527b616ea34f376c2bdf45a76c0a7c4f7735c2a',
}


def _top_level_function_source(source: str, function_name: str) -> str:
    module = ast.parse(source)
    matches = [
        node
        for node in module.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == function_name
    ]
    if len(matches) != 1:
        raise AssertionError(
            f"Expected exactly one top-level {function_name} definition; found {len(matches)}"
        )
    segment = ast.get_source_segment(source, matches[0])
    if not isinstance(segment, str):
        raise AssertionError(f"Could not extract source for {function_name}")
    return segment


def test_prediction_and_observation_contract_helpers_match_a30c808_baseline():
    """Keep reliability changes from silently changing Stage 2E signal records.

    This is a source-lock regression guard, not a substitute for a runtime
    golden-vector test using frozen candles and serialized prediction records.
    Intentional signal changes must update the baseline only through a reviewed
    experiment/change-control decision.
    """
    project_root = Path(__file__).resolve().parents[1]
    source = (project_root / "trade_executor.py").read_text(encoding="utf-8")
    for function_name, expected_hash in EXPECTED_FUNCTION_SHA256.items():
        segment = _top_level_function_source(source, function_name)
        actual_hash = hashlib.sha256(segment.encode("utf-8")).hexdigest()
        assert actual_hash == expected_hash, (
            f"{function_name} differs from baseline {BASELINE_COMMIT}: "
            f"expected {expected_hash}, got {actual_hash}"
        )
