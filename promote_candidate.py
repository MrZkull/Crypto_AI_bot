#!/usr/bin/env python3
# promote_candidate.py — Empirical Gate 10 Prospective Evaluator & Atomic Release Cutover

import os
import sys
import json
import shutil
import math
import subprocess
import logging
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from feature_engineering import ALL_FEATURES
from execution_policy import (
    get_policy_hash,
    get_file_hash,
    get_feature_code_hash,
    get_feature_schema_hash,
    build_candidate_config,
    get_config_hash,
)
from gate10 import evaluate_gate_10

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

MANIFEST_FILE = Path("candidate_manifest.json")
MODEL_FILE = Path("candidate_model.pkl")
EVENTS_LEDGER = Path("candidate_events.jsonl")
PROD_MODEL_FILE = Path("pro_crypto_ai_model.pkl")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(f"CRITICAL: {message}")


def evaluate_thesis_outcome(reason: str) -> int:
    """Classifies terminal exit reason into positive (1) or negative (0) thesis."""
    r = str(reason).upper()
    if "TP" in r or "TAKE_PROFIT" in r:
        return 1
    return 0


def ensure_candidate_model_present():
    """Downloads candidate binary from release if not locally present."""
    if not MODEL_FILE.exists():
        log.info("candidate_model.pkl not found locally. Attempting download from GitHub release 'candidate-latest'...")
        try:
            subprocess.run(
                ["gh", "release", "download", "candidate-latest", "--pattern", "candidate_model.pkl", "--clobber"],
                check=True
            )
            log.info("✓ Downloaded candidate_model.pkl successfully.")
        except Exception as e:
            require(False, f"Could not acquire candidate_model.pkl: {e}")


def reconstruct_and_verify_ledger(manifest: dict) -> list[dict]:
    require(EVENTS_LEDGER.exists(), "candidate_events.jsonl ledger not found. No shadow trades logged.")

    events = []
    with open(EVENTS_LEDGER, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))

    require(len(events) > 0, "candidate_events.jsonl is empty.")
    trades = {}

    VALID_TRANSITIONS = {
        "OPEN": ["PARTIAL_TP1", "CLOSED"],
        "COUNTERFACTUAL_OPEN": ["PARTIAL_TP1", "CLOSED"],
        "PARTIAL_TP1": ["TRAILING_TP1", "CLOSED"],
        "TRAILING_TP1": ["CLOSED"]
    }

    for e in events:
        if e.get("candidate_id") != manifest["candidate_id"]:
            continue
        pid = e["pred_id"]

        if e["event"] == "SETUP_OBSERVED":
            require(pid not in trades, f"Duplicate SETUP_OBSERVED for {pid}")
            require(e["model_sha256"] == manifest["model_sha256"], f"Foreign model hash in {pid}")
            require(e["feature_schema_hash"] == manifest["feature_schema_hash"], f"Foreign schema in {pid}")
            require(e["feature_code_hash"] == manifest["feature_code_hash"], f"Foreign feature code in {pid}")
            require(e["execution_policy_hash"] == manifest["execution_policy_hash"], f"Foreign execution policy in {pid}")
            require(e["config_hash"] == manifest["config_hash"], f"Foreign config in {pid}")

            require("entry_fee_usd" in e, f"Missing entry_fee_usd in {pid}")
            require("tp1_fee_usd" in e, f"Missing tp1_fee_usd in {pid}")

            trades[pid] = e.copy()
            trades[pid]["state"] = "OPEN" if e["meta_selected"] else "COUNTERFACTUAL_OPEN"
            trades[pid]["status"] = "ACTIVE"

        elif e["event"] == "STATE_CHANGE":
            require(pid in trades, f"Orphaned state change for {pid}")
            current_state = trades[pid]["state"]
            require(e["new_state"] in VALID_TRANSITIONS.get(current_state, []),
                    f"Illegal transition {current_state} -> {e['new_state']} in {pid}")
            trades[pid]["state"] = e["new_state"]

        elif e["event"] == "PARTIAL_TP1_FILLED":
            require(pid in trades, f"Orphaned TP1 fill for {pid}")
            current_state = trades[pid]["state"]
            require("PARTIAL_TP1" in VALID_TRANSITIONS.get(current_state, []),
                    f"Illegal TP1 fill from state {current_state} in {pid}")
            trades[pid]["state"] = "PARTIAL_TP1"
            trades[pid]["tp1_fee_usd"] = trades[pid].get("tp1_fee_usd", 0.0) + float(e.get("tp1_fee", 0.0))

        elif e["event"] == "OUTCOME_OBSERVED":
            require(pid in trades, f"Orphaned outcome for {pid}")
            t = trades[pid]
            require(t.get("status") != "CLOSED", f"Duplicate terminal outcome for {pid}")
            require("CLOSED" in VALID_TRANSITIONS.get(t["state"], []),
                    f"Illegal close transition from {t['state']} in {pid}")

            for key in ("realized_gross_pnl", "final_exit_fee_usd", "funding_usd"):
                value = float(e.get(key, 0.0))
                require(math.isfinite(value), f"Non-finite {key} in {pid}")

            net_pnl = (
                float(e["realized_gross_pnl"])
                - float(t.get("entry_fee_usd", 0.0))
                - float(t.get("tp1_fee_usd", 0.0))
                - float(e.get("final_exit_fee_usd", 0.0))
                - float(e.get("funding_usd", 0.0))
            )

            require(float(t["initial_risk_usd"]) > 0, f"Zero initial risk in {pid}")
            recomputed_net_r = net_pnl / float(t["initial_risk_usd"])

            t["net_r"] = recomputed_net_r
            t["thesis_label"] = evaluate_thesis_outcome(e.get("reason", ""))
            t["exit_reason"] = e.get("reason", "")
            t["status"] = "CLOSED"

    closed = [t for t in trades.values() if t.get("status") == "CLOSED"]
    log.info(f"Reconstructed {len(closed)} closed candidate trade setups from ledger.")
    return closed


def promote():
    log.info("Starting candidate promotion audit...")
    require(MANIFEST_FILE.exists(), "candidate_manifest.json not found.")

    with open(MANIFEST_FILE, "r") as f:
        manifest = json.load(f)

    ensure_candidate_model_present()

    # 1. Cryptographic Lineage Verification
    require(manifest["model_sha256"] == get_file_hash(MODEL_FILE), "Candidate binary mutated or invalid SHA256.")
    require(manifest["execution_policy_hash"] == get_policy_hash(), "Execution policy mutated.")
    require(manifest["feature_code_hash"] == get_feature_code_hash(), "Feature code mutated.")
    require(manifest["feature_schema_hash"] == get_feature_schema_hash(ALL_FEATURES), "Feature schema mutated.")

    current_config_hash = get_config_hash(build_candidate_config())
    require(manifest["config_hash"] == current_config_hash, "Deployment configuration diverged from candidate manifest.")

    expiry = datetime.fromisoformat(manifest["candidate_expiry_at"])
    require(datetime.now(timezone.utc) < expiry, "Candidate evidence window expired.")

    # 2. Empirical Ledger Reconstruction & Gate 10 Evaluation
    closed_setups = reconstruct_and_verify_ledger(manifest)
    require(len(closed_setups) >= 50, f"Insufficient empirical evidence: {len(closed_setups)}/50 trades closed.")

    df = pd.DataFrame(closed_setups)

    required_cols = {"pred_id", "primary_selected", "meta_selected", "thesis_label", "net_r", "evidence_type"}
    missing = required_cols - set(df.columns)
    require(not missing, f"Gate-10 evidence missing columns: {sorted(missing)}")

    require(not df["pred_id"].duplicated().any(), "Duplicate pred_id in empirical evidence.")
    require(df["primary_selected"].astype(bool).all(), "Evidence dataframe contains non-primary-eligible setups.")

    gate_result = evaluate_gate_10(df)
    log.info(f"Empirical Gate 10 Results:\n{json.dumps(gate_result, indent=2)}")

    require(gate_result.get("passed", False), f"GATE 10 FAILED: {gate_result.get('reason')}")
    require(gate_result.get("production_eligible", False), "Gate 10 passed for research, but NOT eligible for live production.")

    # 3. Atomic File Promotion
    staged = Path("staged_model.pkl")
    shutil.copy(MODEL_FILE, staged)
    require(get_file_hash(staged) == manifest["model_sha256"], "Staging corruption detected.")

    staged.replace(PROD_MODEL_FILE)
    require(get_file_hash(PROD_MODEL_FILE) == manifest["model_sha256"], "Atomic write failed.")

    promotion_record = {
        "candidate_id": manifest["candidate_id"],
        "model_sha256": manifest["model_sha256"],
        "promoted_at": datetime.now(timezone.utc).isoformat(),
        "gate10_metrics": gate_result,
    }

    with open("promotion_record.json", "w") as f:
        json.dump(promotion_record, f, indent=2)

    manifest["status"] = "PROMOTED_TO_PRODUCTION"
    with open(MANIFEST_FILE, "w") as f:
        json.dump(manifest, f, indent=2)

    # 4. Upload Promoted Binary to GitHub Production Release (v2.0)
    log.info("Uploading promoted model to GitHub release 'v2.0'...")
    try:
        subprocess.run(
            ["gh", "release", "upload", "v2.0", str(PROD_MODEL_FILE), "promotion_record.json", "--clobber"],
            check=True
        )
        log.info("✓ Successfully published promoted binary to production release 'v2.0'.")
    except Exception as e:
        log.warning(f"Could not automatically upload to GitHub release: {e}. Please upload pro_crypto_ai_model.pkl manually.")

    log.info("🚀 SUCCESS: Candidate explicitly verified and promoted to live production model!")


if __name__ == "__main__":
    promote()
