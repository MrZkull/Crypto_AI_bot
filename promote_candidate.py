#promote_candidate.py

import json, sys, shutil, math
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd

from feature_engineering import ALL_FEATURES
from execution_policy import (
    get_policy_hash, get_file_hash, get_feature_code_hash, 
    get_feature_schema_hash, build_candidate_config, get_config_hash, evaluate_thesis_outcome
)
from gate10 import evaluate_gate_10

def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(f"CRITICAL: {message}")

def reconstruct_and_verify_ledger(manifest: dict) -> list:
    events = [json.loads(line) for line in open("candidate_events.jsonl")]
    trades = {}
    
    VALID_TRANSITIONS = {
        "OPEN": ["PARTIAL_TP1", "CLOSED"],
        "COUNTERFACTUAL_OPEN": ["PARTIAL_TP1", "CLOSED"],
        "PARTIAL_TP1": ["TRAILING_TP1", "CLOSED"],
        "TRAILING_TP1": ["CLOSED"]
    }
    
    for e in events:
        if e.get("candidate_id") != manifest["candidate_id"]: continue
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

            trades[pid] = e
            trades[pid]["state"] = "OPEN" if e["meta_selected"] else "COUNTERFACTUAL_OPEN"
            
        elif e["event"] == "STATE_CHANGE":
            require(pid in trades, f"Orphaned state change for {pid}")
            require(e["new_state"] in VALID_TRANSITIONS[trades[pid]["state"]], f"Illegal transition {trades[pid]['state']} -> {e['new_state']} in {pid}")
            trades[pid]["state"] = e["new_state"]
            
        elif e["event"] == "PARTIAL_TP1_FILLED":
            require(pid in trades, f"Orphaned TP1 fill for {pid}")
            require("PARTIAL_TP1" in VALID_TRANSITIONS[trades[pid]["state"]], f"Illegal TP1 fill from state {trades[pid]['state']} in {pid}")
            trades[pid]["state"] = "PARTIAL_TP1"
            trades[pid]["tp1_fee_usd"] = trades[pid].get("tp1_fee_usd", 0.0) + e["tp1_fee"]
            
        elif e["event"] == "OUTCOME_OBSERVED":
            require(pid in trades, f"Orphaned outcome for {pid}")
            t = trades[pid]
            require(t.get("status") != "CLOSED", f"Duplicate terminal outcome for {pid}")
            require("CLOSED" in VALID_TRANSITIONS[t["state"]], f"Illegal close transition from {t['state']} in {pid}")
            
            for key in ("realized_gross_pnl", "final_exit_fee_usd", "funding_usd"):
                value = float(e.get(key, 0.0))
                require(math.isfinite(value), f"Non-finite {key} in {pid}")

            net_pnl = (
                e["realized_gross_pnl"]
                - t.get("entry_fee_usd", 0.0)
                - t.get("tp1_fee_usd", 0.0)
                - e["final_exit_fee_usd"]
                - e["funding_usd"]
            )
            
            require(t["initial_risk_usd"] > 0, f"Zero initial risk in {pid}")
            recomputed_net_r = net_pnl / t["initial_risk_usd"]
            
            t["net_r"] = recomputed_net_r
            t["thesis_label"] = evaluate_thesis_outcome(e["reason"])
            t["exit_reason"] = e["reason"]
            t["status"] = "CLOSED"

    return [t for t in trades.values() if t["status"] == "CLOSED"]

def promote():
    require(Path("candidate_manifest.json").exists(), "Candidate manifest not found.")
    manifest = json.load(open("candidate_manifest.json"))
    
    require(manifest["model_sha256"] == get_file_hash("candidate_model.pkl"), "Candidate binary mutated.")
    require(manifest["execution_policy_hash"] == get_policy_hash(), "Execution policy mutated.")
    require(manifest["feature_code_hash"] == get_feature_code_hash(), "Feature code mutated.")
    require(manifest["feature_schema_hash"] == get_feature_schema_hash(ALL_FEATURES), "Feature schema mutated.")
    
    current_config_hash = get_config_hash(build_candidate_config())
    require(manifest["config_hash"] == current_config_hash, "Deployment configuration diverged from candidate test environment.")

    expiry = datetime.fromisoformat(manifest["candidate_expiry_at"])
    require(datetime.now(timezone.utc) < expiry, "Candidate evidence window expired.")

    closed_setups = reconstruct_and_verify_ledger(manifest)
    df = pd.DataFrame(closed_setups)
    
    required_cols = {"pred_id", "primary_selected", "meta_selected", "thesis_label", "net_r", "evidence_type"}
    missing = required_cols - set(df.columns)
    require(not missing, f"Gate-10 evidence missing columns: {sorted(missing)}")
    
    require(not df["pred_id"].duplicated().any(), "Duplicate pred_id in empirical evidence.")
    require(df["primary_selected"].astype(bool).all(), "Evidence dataframe contains non-primary-eligible setups.")

    gate_result = evaluate_gate_10(df)
    print(json.dumps(gate_result, indent=2))
    require(gate_result["passed"], f"GATE 10 FAILED: {gate_result['reason']}")

    staged = "staged_model.pkl"
    shutil.copy("candidate_model.pkl", staged)
    require(get_file_hash(staged) == manifest["model_sha256"], "Staging corruption.")
    
    Path(staged).replace("pro_crypto_ai_model.pkl")
    require(get_file_hash("pro_crypto_ai_model.pkl") == manifest["model_sha256"], "Atomic write failed.")

    with open("promotion_record.json", "w") as f:
        json.dump({"candidate_id": manifest["candidate_id"], "model_sha256": manifest["model_sha256"], "promoted_at": datetime.now(timezone.utc).isoformat(), "gate10_metrics": gate_result}, f, indent=2)

    manifest["status"] = "PROMOTED_TO_V2"
    with open("candidate_manifest.json", "w") as f: json.dump(manifest, f)
    
    print("🚀 SUCCESS: Candidate explicitly verified and safely promoted.")

if __name__ == "__main__":
    promote()
