#candidate_scanner.py

import json, time, math
from filelock import FileLock
from execution_policy import calculate_trade_brackets

def log_event(event_type: str, candidate_id: str, pred_id: str, data: dict):
    with FileLock("candidate_events.jsonl.lock"):
        with open("candidate_events.jsonl", "a") as f:
            f.write(json.dumps({"event": event_type, "candidate_id": candidate_id, "pred_id": pred_id, **data}) + "\n")

def record_candidate_setup(
    manifest: dict,
    manifest_config: dict,
    pred_id: str,
    symbol: str,
    side: str,
    primary_selected: bool,
    meta_selected: bool,
    row: dict,
    simulated_entry: float,
    observation_time_ms: int
):
    if not primary_selected and meta_selected:
        raise ValueError(f"Meta-selected setup is not primary-selected: {pred_id}")
    
    if int(observation_time_ms) <= 0:
        raise ValueError(f"Invalid observation timestamp for {pred_id}")

    balance = float(manifest_config["evaluation_balance_usd"])
    if not (math.isfinite(balance) and balance > 0):
        raise ValueError(f"Invalid candidate evaluation balance: {balance}")

    atr = float(row["atr"])
    brackets = calculate_trade_brackets(simulated_entry, atr, side, "NORMAL", 
                                        manifest_config["atr_stop_mult"], 
                                        manifest_config["atr_target1_mult"], 
                                        manifest_config["atr_target2_mult"])
    
    initial_risk_usd = balance * manifest_config["risk_mult"] * 0.01
    stop_dist = abs(simulated_entry - brackets["stop"])
    qty = initial_risk_usd / stop_dist if stop_dist > 0 else 0.0

    event = {
        "candidate_id": manifest["candidate_id"],
        "pred_id": pred_id,
        "event": "SETUP_OBSERVED",
        
        "model_sha256": manifest["model_sha256"],
        "feature_schema_hash": manifest["feature_schema_hash"],
        "feature_code_hash": manifest["feature_code_hash"],
        "execution_policy_hash": manifest["execution_policy_hash"],
        "decision_policy_hash": manifest.get("decision_policy_hash", ""),
        "config_hash": manifest["config_hash"],
        
        "symbol": symbol,
        "side": side,
        "primary_selected": bool(primary_selected),
        "meta_selected": bool(meta_selected),
        
        "atr": atr,
        "simulated_entry": simulated_entry,
        "stop": brackets["stop"],
        "tp1": brackets["tp1"],
        "tp2": brackets["tp2"],
        "qty": qty,
        "initial_risk_usd": initial_risk_usd,
        
        "entry_time_ms": int(observation_time_ms),
        "recorded_at_ms": int(time.time() * 1000),
        
        "entry_fee_usd": 0.0,
        "tp1_fee_usd": 0.0,
        "thesis_label": None,
        
        "evidence_type": "PROSPECTIVE_SHADOW_EXECUTION" if meta_selected else "PROSPECTIVE_COUNTERFACTUAL"
    }

    log_event("SETUP_OBSERVED", manifest["candidate_id"], pred_id, event)

    # Initialize into active state projection for the stream monitor
    with FileLock("candidate_state.json.lock"):
        state = json.load(open("candidate_state.json")) if Path("candidate_state.json").exists() else {}
        if pred_id in state:
            raise ValueError(f"Duplicate POSITION_OPENED/SETUP_OBSERVED for {pred_id}")
        
        # We project the event into mutable state for active monitoring.
        event["state"] = "OPEN" if meta_selected else "COUNTERFACTUAL_OPEN"
        event["status"] = "ACTIVE" if meta_selected else "COUNTERFACTUAL_ACTIVE"
        event["last_funding_ts"] = int(observation_time_ms)
        event["funding_usd"] = 0.0
        event["realized_gross_pnl"] = 0.0
        
        state[pred_id] = event
        with open("candidate_state.json.tmp", "w") as f: json.dump(state, f, indent=2)
        import os; os.replace("candidate_state.json.tmp", "candidate_state.json")
