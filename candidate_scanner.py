#!/usr/bin/env python3
# candidate_scanner.py — Prospective Shadow Candidate Scanner & Ledger Logger

import os
import sys
import json
import time
import math
import logging
from pathlib import Path
from filelock import FileLock

import numpy as np
import pandas as pd
import joblib

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

CANDIDATE_MODEL_FILE = Path("candidate_model.pkl")
CANDIDATE_META_FILE  = Path("candidate_meta_model.pkl")
CANDIDATE_MANIFEST   = Path("candidate_manifest.json")
EVENTS_LEDGER_FILE   = Path("candidate_events.jsonl")
STATE_FILE           = Path("candidate_state.json")

try:
    from execution_policy import calculate_trade_brackets
except ImportError:
    def calculate_trade_brackets(entry: float, atr: float, side: str, vol_state: str,
                                 stop_mult: float, tp1_mult: float, tp2_mult: float) -> dict:
        side_norm = side.upper().strip()
        if side_norm == "BUY":
            return {
                "stop": entry - atr * stop_mult,
                "tp1":  entry + atr * tp1_mult,
                "tp2":  entry + atr * tp2_mult,
            }
        else:
            return {
                "stop": entry + atr * stop_mult,
                "tp1":  entry - atr * tp1_mult,
                "tp2":  entry - atr * tp2_mult,
            }


def log_event(event_type: str, candidate_id: str, pred_id: str, data: dict):
    with FileLock("candidate_events.jsonl.lock"):
        with open(EVENTS_LEDGER_FILE, "a") as f:
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

    balance = float(manifest_config.get("evaluation_balance_usd", 10_000.0))
    if not (math.isfinite(balance) and balance > 0):
        raise ValueError(f"Invalid candidate evaluation balance: {balance}")

    atr = float(row.get("atr", 0.0))
    if atr <= 0.0:
        log.warning(f"Skipping setup {pred_id}: ATR <= 0")
        return

    brackets = calculate_trade_brackets(
        simulated_entry, atr, side, "NORMAL",
        float(manifest_config.get("ATR_STOP_MULT", 2.5)),
        float(manifest_config.get("ATR_TARGET1_MULT", 3.5)),
        float(manifest_config.get("ATR_TARGET2_MULT", 7.5))
    )
    
    risk_pct = float(manifest_config.get("RISK_PER_TRADE", 0.015))
    initial_risk_usd = balance * risk_pct
    stop_dist = abs(simulated_entry - brackets["stop"])
    qty = (initial_risk_usd / stop_dist) if stop_dist > 0 else 0.0

    # Invariant: evidence_type MUST be "EMPIRICAL_PROSPECTIVE" for all live prospective observations
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
        
        "open_time": int(observation_time_ms),
        "entry_time_ms": int(observation_time_ms),
        "recorded_at_ms": int(time.time() * 1000),
        
        "entry_fee_usd": 0.0,
        "tp1_fee_usd": 0.0,
        "thesis_label": None,
        "net_r": None,
        
        "evidence_type": "EMPIRICAL_PROSPECTIVE"
    }

    log_event("SETUP_OBSERVED", manifest["candidate_id"], pred_id, event)

    # Initialize into active state projection for the stream monitor
    with FileLock("candidate_state.json.lock"):
        state = {}
        if STATE_FILE.exists():
            try:
                with open(STATE_FILE, "r") as sf:
                    state = json.load(sf)
            except Exception:
                state = {}

        if pred_id in state:
            raise ValueError(f"Duplicate POSITION_OPENED/SETUP_OBSERVED for {pred_id}")
        
        event["state"] = "OPEN" if meta_selected else "COUNTERFACTUAL_OPEN"
        event["status"] = "ACTIVE" if meta_selected else "COUNTERFACTUAL_ACTIVE"
        event["last_funding_ts"] = int(observation_time_ms)
        event["funding_usd"] = 0.0
        event["realized_gross_pnl"] = 0.0
        
        state[pred_id] = event
        tmp_file = str(STATE_FILE) + ".tmp"
        with open(tmp_file, "w") as f:
            json.dump(state, f, indent=2)
        os.replace(tmp_file, STATE_FILE)


def scan_and_evaluate_candidate():
    """Evaluates the latest closed 15m candle across symbols using the candidate pipeline."""
    if not (CANDIDATE_MODEL_FILE.exists() and CANDIDATE_MANIFEST.exists()):
        return

    with open(CANDIDATE_MANIFEST, "r") as f:
        manifest = json.load(f)

    if manifest.get("status") != "AWAITING_PROSPECTIVE_EVIDENCE":
        return

    primary_pipeline = joblib.load(CANDIDATE_MODEL_FILE)
    meta_pipeline = joblib.load(CANDIDATE_META_FILE) if CANDIDATE_META_FILE.exists() else None

    from trade_executor import get_data, generate_features_for_symbol
    symbols = primary_pipeline.get("symbols", [])
    
    for symbol in symbols:
        try:
            df = get_data(symbol, "15m", limit=100)
            if df is None or len(df) < 50:
                continue

            last_candle = df.iloc[-2].to_dict()  # Most recent fully closed candle
            obs_time = int(last_candle.get("close_time", last_candle.get("open_time", 0) + 15*60*1000 - 1))
            pred_id = f"{symbol}_{int(last_candle['open_time'])}"

            # Avoid re-evaluating candles already tracked in state
            if STATE_FILE.exists():
                with open(STATE_FILE) as sf:
                    if pred_id in json.load(sf):
                        continue

            feats_df = generate_features_for_symbol(symbol, df)
            if feats_df.empty:
                continue

            row_feats = feats_df.iloc[-2]
            X = row_feats[primary_pipeline["all_features"]].values.reshape(1, -1)
            Xs = primary_pipeline["selector"].transform(X)

            probas = primary_pipeline["ensemble"].predict_proba(Xs)[0]
            classes = primary_pipeline["label_encoder"].classes_
            
            buy_idx = list(classes).index("BUY") if "BUY" in classes else 0
            sell_idx = list(classes).index("SELL") if "SELL" in classes else 2
            
            p_buy = probas[buy_idx]
            p_sell = probas[sell_idx]

            thresh_buy = primary_pipeline.get("recommended_threshold_buy", 0.40)
            thresh_sell = primary_pipeline.get("recommended_threshold_sell", 0.45)

            side = None
            conf = 0.0
            if p_buy >= thresh_buy and p_buy > p_sell:
                side = "BUY"
                conf = p_buy
            elif p_sell >= thresh_sell and p_sell > p_buy:
                side = "SELL"
                conf = p_sell

            primary_selected = side is not None
            if not primary_selected:
                continue

            # Meta-Model evaluation
            meta_selected = False
            if meta_pipeline and primary_selected:
                meta_thresh = meta_pipeline.get("recommended_meta_threshold", 0.50)
                # Augment system meta-features
                meta_input = row_feats.to_dict()
                meta_input["meta_primary_conf"] = conf
                meta_input["meta_primary_side_code"] = 1.0 if side == "BUY" else -1.0
                meta_input["meta_rsi_directional"] = (float(row_feats.get("rsi", 50.0)) - 50.0) * meta_input["meta_primary_side_code"]
                meta_input["meta_macd_directional"] = float(row_feats.get("macd_hist", 0.0)) * meta_input["meta_primary_side_code"]
                meta_input["meta_trend_aligned"] = float(row_feats.get("trend", 0.0)) * meta_input["meta_primary_side_code"]
                
                X_meta = pd.DataFrame([meta_input])[meta_pipeline["meta_features"]].fillna(0).values
                meta_proba = meta_pipeline["meta_ensemble"].predict_proba(X_meta)[0, meta_pipeline["meta_positive_idx"]]
                meta_selected = bool(meta_proba >= meta_thresh)

            record_candidate_setup(
                manifest=manifest,
                manifest_config=manifest.get("config", {}),
                pred_id=pred_id,
                symbol=symbol,
                side=side,
                primary_selected=primary_selected,
                meta_selected=meta_selected,
                row=last_candle,
                simulated_entry=float(last_candle["close"]),
                observation_time_ms=obs_time
            )
            log.info(f"Recorded prospective setup: {pred_id} | Side: {side} | Meta Approved: {meta_selected}")

        except Exception as e:
            log.error(f"Error evaluating candidate on {symbol}: {e}")


if __name__ == "__main__":
    scan_and_evaluate_candidate()
    
