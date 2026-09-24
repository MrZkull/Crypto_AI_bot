#test_train_model_static.py

import ast
from pathlib import Path

p = Path("train_model.py")
src = p.read_text(encoding="utf-8")
tree = ast.parse(src)

funcs = {
    n.name for n in tree.body
    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
}

required = {
    "_write_json_atomic",
    "_dump_model_atomic",
    "audit_anti_leakage",
    "_align_1h_to_15m",
    "_align_4h_to_15m",
    "_align_btc_to_15m",
    "load_parquet_segment",
    "_add_extra_features",
    "_merge_funding_to_15m",
    "_build_features",
    "preload_symbol_features",
    "make_targets",
    "_apply_targets",
    "build_dataset_from_local_parquet",
    "temporal_symbol_split",
    "undersample_no_trade",
    "train",
    "_parse_args",
}
missing = required - funcs
assert not missing, f"function regression: {sorted(missing)}"

assert any("".join(line.split()) == "N_FEATURES=35" for line in src.splitlines())
assert 'joblib.dump(pipeline, MODEL_FILE)' not in src
assert 'joblib.dump(pipeline, CANDIDATE_MODEL_FILE' not in src
assert "_dump_model_atomic(" in src
assert 'compress=3' in src
assert 'MAX_MODEL_BYTES = 95 * 1024 * 1024' in src
assert '"config": candidate_cfg' in src
assert 'action="store_true"' in src
assert '--production' in src
assert 'is_candidate = not bool(args.production)' in src

print("train_model static integrity test: PASS")
