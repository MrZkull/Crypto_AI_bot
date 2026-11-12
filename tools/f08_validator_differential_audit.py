#!/usr/bin/env python3
"""Run a controlled pinned-vs-candidate Phase 2D differential audit.

This runner intentionally does not claim a full historical scoring replay. It
uses the authoritative schema-6 state/observation ledger/journal, compares the
unchanged validator calculation functions, runs identical deterministic candle
fixtures, and re-evaluates the frozen statistical gate without allowing a new
checkpoint event. Raw production model binaries and raw historical candle
windows are required for the separate full replay and are not synthesized here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PINNED = ROOT.parent / "pinned_resolver"
EVIDENCE = ROOT.parent / "evidence" / "phase2d-state-run-41720710702.zip"
EXPECTED = {
    "state": "4dd39ee1eefb0866dc5060719d69babf2b31f0d73ee814be5418322ac1d262e5",
    "ledger": "4ba3daaa2e4f25eae895699e01a4cd1d0a4fd9e7b64a2632bf4fcac7d56274e9",
    "journal": "cf12ac74b39e44d8487951bb1abde203f12140a626d7c78fc388ebd91e80e267",
}

CHILD = r'''\
import hashlib, json, math, sys
from pathlib import Path
import numpy as np
import pandas as pd
import phase2d_prospective_validator as v
from phase2d_promotion_gate import evaluate_promotion

root = Path.cwd()
state_path = root / "research_outputs/phase2d_state.json"
obs_path = root / "research_outputs/phase2d_observations.jsonl"
journal_path = root / "research_outputs/experiment_ledger.jsonl"
state = json.loads(state_path.read_text(encoding="utf-8"))
rows = [json.loads(line) for line in obs_path.read_text(encoding="utf-8").splitlines() if line.strip()]

# Make resolver timestamps deterministic for byte-level result comparison.
v.utc_now = lambda: "2026-11-08T00:00:00+00:00"
v.validate_state_integrity(state)
by_model = {m: [r for r in rows if r.get("model") == m] for m in ("candidate", "production")}
summaries = {m: v.summarize(by_model[m]) for m in by_model}

# Deterministic fake model with fixed probabilities, preserving the real scorer API.
class FixedEnsemble:
    def predict_proba(self, X):
        return np.array([[0.22, 0.61, 0.17]], dtype=float)
    def predict(self, X):
        return np.array([1], dtype=int)
model = {
    "best_features": ["f_b", "f_a"],
    "ensemble": FixedEnsemble(),
    "label_map": {"0": "NO_TRADE", "1": "BUY", "2": "SELL"},
    "recommended_threshold_buy": 0.55,
    "recommended_threshold_sell": 0.60,
    "trained_at": "fixture",
}
score = v.score_model(model, pd.Series({"f_a": 1.25, "f_b": -0.5, "unused": 99.0}))

# Candle outcome cases cover BUY/SELL x TP/SL/AMBIGUOUS/EXPIRED. These are
# deterministic regression fixtures, not historical market evidence.
ENTRY_MS = 15 * 60 * 1000
base = 1_800_000_000_000

def frame_for(side, mode):
    data = []
    for i in range(26):
        o = base + i * ENTRY_MS
        high, low = 101.0, 99.0
        if i == 1:
            if mode == "TP": high, low = ((104.0, 99.0) if side == "BUY" else (101.0, 96.0))
            elif mode == "SL": high, low = ((101.0, 97.0) if side == "BUY" else (104.0, 99.0))
            elif mode == "AMBIGUOUS": high, low = ((104.0, 97.0) if side == "BUY" else (104.0, 96.0))
        data.append({"open_time": o, "close_time": o + ENTRY_MS - 1,
                     "open": 100.0, "high": high, "low": low, "close": 100.0, "volume": 10.0})
    return pd.DataFrame(data)

resolution_cases = {}
for side in ("BUY", "SELL"):
    for mode in ("TP", "SL", "AMBIGUOUS", "EXPIRED"):
        pred = {"id": f"fixture:{side}:{mode}", "model": "candidate", "symbol": "BTCUSDT",
                "open_time": base, "entry": 100.0, "atr": 1.0, "signal": side}
        resolution_cases[f"{side}_{mode}"] = v.resolve_prediction(pred, frame_for(side, mode))

# Identical synthetic OHLCV data exercises current + legacy feature transforms.
def candle_frame(n, interval_ms, offset=0, price_offset=0.0):
    recs = []
    for i in range(n):
        o = base + offset + i * interval_ms
        close = 100.0 + price_offset + 0.015 * i + 0.35 * math.sin(i / 7.0)
        recs.append({"open_time": o, "close_time": o + interval_ms - 1,
                     "open": close - 0.03, "high": close + 0.22,
                     "low": close - 0.24, "close": close,
                     "volume": 1000.0 + (i % 11) * 13.0,
                     "taker_buy_base_vol": 500.0 + (i % 7) * 5.0})
    return pd.DataFrame(recs)

raw15 = candle_frame(320, ENTRY_MS)
raw1h = candle_frame(120, 60 * 60 * 1000)
btc15 = candle_frame(320, ENTRY_MS, price_offset=20.0)
v.fetch_deribit_1h = lambda symbol: raw1h.copy()
current = v.current_features(raw15, "BTCUSDT")
production = v.build_production_features(raw15, "BTCUSDT", btc15)

def frame_digest(df):
    # Stable full-frame digest, including column order, null placement, dtypes, and values.
    payload = {
        "columns": [str(x) for x in df.columns],
        "dtypes": [str(x) for x in df.dtypes],
        "records": df.replace({np.nan: None}).to_dict(orient="records"),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str, allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()

# The promotion gate is frozen and shared. Re-evaluate the same resolved
# observation population using copied journal/state inputs; assert no journal
# mutation and no C25 event is created.
c_rows = [r for r in by_model["candidate"] if r.get("status") in {"TP", "SL", "EXPIRED"}]
p_rows = [r for r in by_model["production"] if r.get("status") in {"TP", "SL", "EXPIRED"}]
journal_before = journal_path.read_bytes()
gate = evaluate_promotion(c_rows, p_rows, audit_ledger_path=journal_path,
                          state_path=state_path,
                          regime_reference_path=root / "phase2d_regime_reference.json")
journal_after = journal_path.read_bytes()
assert journal_before == journal_after, "gate unexpectedly mutated the frozen experiment journal"
assert not gate.get("new_checkpoint_evaluations"), "gate attempted a new checkpoint evaluation"
assert 25 not in gate.get("checkpoints_evaluated", []), "C25 appears evaluated"

# Tie the recomputed outputs to the authoritative persisted run, not merely
# to the fact that old and new code agree with one another.
stored = state.get("last_run", {})
stored_match = True
for model_name in ("candidate", "production"):
    saved = stored.get(model_name, {})
    calculated = summaries[model_name]
    for key in ("resolved", "tp", "sl", "expired", "ambiguous", "invalid"):
        stored_match = stored_match and saved.get(key) == calculated.get(key)
    for key in ("precision_excluding_ambiguous", "mean_net_r", "cum_net_r", "max_drawdown_r"):
        if saved.get(key) is None or calculated.get(key) is None:
            stored_match = stored_match and saved.get(key) == calculated.get(key)
        else:
            stored_match = stored_match and math.isclose(float(saved[key]), float(calculated[key]), rel_tol=1e-12, abs_tol=1e-12)
stored_gate = stored.get("promotion_gate", {})
for key in ("gate_status", "promotion_ready", "n_candidate", "n_production",
            "paired_unique_blocks", "checkpoints_evaluated", "latest_evaluated_checkpoint"):
    stored_match = stored_match and stored_gate.get(key) == gate.get(key)
assert stored_match, "recomputed summaries/gate do not match authoritative last_run"

out = {
    "state_schema": state.get("schema_version"),
    "state_integrity": "PASS",
    "observation_rows": len(rows),
    "candidate_observation_rows": len(by_model["candidate"]),
    "production_observation_rows": len(by_model["production"]),
    "candidate_summary": summaries["candidate"],
    "production_summary": summaries["production"],
    "score_model": score,
    "resolution_cases": resolution_cases,
    "current_features": {"rows": len(current), "columns": list(current.columns), "sha256": frame_digest(current)},
    "production_features": {"rows": len(production), "columns": list(production.columns), "sha256": frame_digest(production)},
    "promotion_gate": gate,
    "journal_unchanged": True,
    "matches_authoritative_last_run": stored_match,
}
print(json.dumps(out, sort_keys=True, separators=(",", ":"), allow_nan=False, default=str))
'''


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def json_canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def run_case(label: str, source_root: Path, feature_source: Path, files: dict[str, bytes], tmp: Path) -> dict:
    case = tmp / label
    (case / "research_outputs").mkdir(parents=True)
    for name, raw in files.items():
        (case / name).parent.mkdir(parents=True, exist_ok=True)
        (case / name).write_bytes(raw)
    for name in ("phase2d_prospective_validator.py", "phase2d_promotion_gate.py", "phase2d_observation_ledger.py"):
        shutil.copy2(source_root / name, case / name)
    if label == "candidate":
        shutil.copy2(ROOT / "phase2d_evidence_policy.py", case / "phase2d_evidence_policy.py")
    # Use the exact feature source hash bound into the authoritative F08 state
    # for BOTH validators. The separately captured resolver worktree contains
    # a line-ending-normalized/different historical copy of this module.
    shutil.copy2(feature_source, case / "feature_engineering.py")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(case)
    proc = subprocess.run([sys.executable, "-c", CHILD], cwd=case, env=env,
                          text=True, capture_output=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"{label} differential child failed ({proc.returncode}):\n{proc.stdout}\n{proc.stderr}")
    try:
        return json.loads(proc.stdout.strip().splitlines()[-1])
    except Exception as exc:
        raise RuntimeError(f"{label} emitted invalid JSON: {proc.stdout}\n{proc.stderr}") from exc


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "research_outputs" / "f08_validator_differential_audit.json")
    args = parser.parse_args()
    if not EVIDENCE.is_file():
        raise SystemExit(f"Missing authoritative evidence ZIP: {EVIDENCE}")

    with zipfile.ZipFile(EVIDENCE) as z:
        state = z.read("research_outputs/phase2d_state.json")
        journal = z.read("research_outputs/experiment_ledger.jsonl")
        observations = z.read("research_outputs/phase2d_observations.jsonl")
    actual = {"state": sha(state), "journal": sha(journal), "ledger": sha(observations)}
    if actual != EXPECTED:
        raise SystemExit(f"Authoritative input hashes do not match pinned checkpoint: {actual}")

    # Bind both runs to the exact feature code hash recorded by the original
    # authoritative F08 run, and confirm the promotion gate source is identical.
    source_hashes = {}
    feature_hash = sha((ROOT / "feature_engineering.py").read_bytes())
    if feature_hash != "f60a2508e19892baeeead20d4aa41d8210f559c827affd5cd50ff541973d4683":
        raise SystemExit(f"Authoritative feature source hash mismatch: {feature_hash}")
    old_gate_hash = sha((PINNED / "phase2d_promotion_gate.py").read_bytes())
    new_gate_hash = sha((ROOT / "phase2d_promotion_gate.py").read_bytes())
    if old_gate_hash != new_gate_hash:
        raise SystemExit(f"Frozen promotion gate source mismatch: {old_gate_hash} != {new_gate_hash}")
    source_hashes["feature_engineering.py (same authoritative source for both runs)"] = feature_hash
    source_hashes["phase2d_promotion_gate.py"] = old_gate_hash

    files = {
        "research_outputs/phase2d_state.json": state,
        "research_outputs/experiment_ledger.jsonl": journal,
        "research_outputs/phase2d_observations.jsonl": observations,
    }
    with tempfile.TemporaryDirectory(prefix="f08_validator_diff_") as td:
        tmp = Path(td)
        feature_source = ROOT / "feature_engineering.py"
        old = run_case("pinned", PINNED, feature_source, files, tmp)
        new = run_case("candidate", ROOT, feature_source, files, tmp)

    comparison = {
        "pinned": old,
        "candidate": new,
        "identical": json_canonical(old) == json_canonical(new),
    }
    if not comparison["identical"]:
        # Expose stable top-level field differences in the report; fail closed.
        keys = sorted(set(old) | set(new))
        comparison["differences"] = {k: {"pinned": old.get(k), "candidate": new.get(k)}
                                     for k in keys if json_canonical(old.get(k)) != json_canonical(new.get(k))}
    report = {
        "audit_type": "PINNED_VS_SCHEMA7_CANDIDATE_DIFFERENTIAL",
        "status": "PASS" if comparison["identical"] else "FAIL",
        "scope": "authoritative persisted-observation replay plus deterministic candle/scoring fixtures",
        "full_historical_model_replay": "BLOCKED_MISSING_EXACT_HISTORICAL_CANDLE_WINDOWS",
        "model_asset_recovery": {
            "candidate": {
                "release_tag": "phase2d-htf-removal-candidate-20260921",
                "asset": "candidate_model.pkl",
                "sha256": "7213987874543e5d6e18b0bac32556647d40080b3e4222532e40a225a1bf08a3",
                "download_url": "https://github.com/MrZkull/Crypto_AI_bot/releases/download/phase2d-htf-removal-candidate-20260921/candidate_model.pkl"
            },
            "production": {
                "release_tag": "v2.0",
                "asset": "pro_crypto_ai_model.pkl",
                "sha256": "f55b887c7f624179b3d9fee56d792c29424a589e3d8734edcd78ce1be71f2c21",
                "download_url": "https://github.com/MrZkull/Crypto_AI_bot/releases/download/v2.0/pro_crypto_ai_model.pkl"
            },
            "metadata_digest_matches_locked_identity": True,
            "downloaded_and_byte_verified_in_audit_runtime": False
        },
        "authoritative_evidence_zip_sha256": sha(EVIDENCE.read_bytes()),
        "authoritative_input_sha256": actual,
        "frozen_source_sha256": source_hashes,
        "comparison": comparison,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "comparison"}, indent=2, sort_keys=True))
    print("COMPARISON_IDENTICAL:", comparison["identical"])
    if not comparison["identical"]:
        print("DIFFERENCES:", json.dumps(comparison.get("differences", {}), indent=2, sort_keys=True))
        return 2
    print("REPORT:", args.output)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
