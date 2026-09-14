#execution_policy.py

import hashlib
import json
import math
from pathlib import Path

def get_file_hash(filepath: str) -> str:
    path = Path(filepath).resolve()
    if not path.exists():
        return ""
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

def get_policy_hash() -> str:
    return get_file_hash(__file__)

def get_feature_code_hash() -> str:
    return get_file_hash("feature_engineering.py")

def get_feature_schema_hash(features: list[str]) -> str:
    """Canonical hash of the exact ordered feature schema."""
    payload = json.dumps(list(features), separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

def build_candidate_config() -> dict:
    import config
    return {
        "evaluation_balance_usd": float(getattr(config, "CANDIDATE_EVALUATION_BALANCE_USD", 1000.0)),
        "threshold_buy": float(getattr(config, "THRESHOLD_BUY", 0.40)),
        "threshold_sell": float(getattr(config, "THRESHOLD_SELL", 0.45)),
        "risk_mult": float(getattr(config, "RISK_MULT", 1.0)),
        "atr_stop_mult": float(getattr(config, "ATR_STOP_MULT", 2.5)),
        "atr_target1_mult": float(getattr(config, "ATR_TARGET1_MULT", 3.5)),
        "atr_target2_mult": float(getattr(config, "ATR_TARGET2_MULT", 7.5)),
        "max_open_trades": int(getattr(config, "MAX_OPEN_TRADES", 8)),
        "max_same_direction": int(getattr(config, "MAX_SAME_DIRECTION", 4)),
        "entry_max_spread_pct": float(getattr(config, "ENTRY_MAX_SPREAD_PCT", 0.005))
    }

def get_config_hash(config_dict: dict) -> str:
    cfg = json.dumps(config_dict, sort_keys=True)
    return hashlib.sha256(cfg.encode("utf-8")).hexdigest()

def calculate_continuous_funding(
    side: str,
    qty_base: float,
    mark_price: float,
    funding_rate: float,
    elapsed_ms: int,
) -> float:
    side = str(side).upper()
    if side not in {"BUY", "SELL"}:
        raise ValueError(f"Invalid side: {side}")

    if elapsed_ms < 0:
        raise ValueError(f"elapsed_ms cannot be negative: {elapsed_ms}")

    for name, value in (("qty_base", qty_base), ("mark_price", mark_price), ("funding_rate", funding_rate)):
        if not math.isfinite(float(value)):
            raise ValueError(f"Non-finite {name}: {value}")

    if qty_base <= 0:
        raise ValueError(f"qty_base must be > 0: {qty_base}")
    if mark_price <= 0:
        raise ValueError(f"mark_price must be > 0: {mark_price}")

    if elapsed_ms == 0 or funding_rate == 0:
        return 0.0

    fraction_8h = elapsed_ms / (8.0 * 3600.0 * 1000.0)
    cost = qty_base * mark_price * funding_rate * fraction_8h

    if not math.isfinite(cost):
        raise ValueError(f"Non-finite funding result: {cost}")

    return cost if side == "BUY" else -cost

def calculate_trade_brackets(entry: float, atr: float, side: str, vol_state: str, 
                             stop_mult: float, tp1_mult: float, tp2_mult: float) -> dict:
    dyn_tp1 = tp1_mult * (1.5 if vol_state == "VERY_HIGH" else 0.8 if vol_state in ("DEAD", "UNKNOWN") else 1.0)
    dyn_tp2 = tp2_mult * (1.5 if vol_state == "VERY_HIGH" else 0.8 if vol_state in ("DEAD", "UNKNOWN") else 1.0)
    if side == "BUY":
        return {"stop": entry - (atr * stop_mult), "tp1": entry + (atr * dyn_tp1), "tp2": entry + (atr * dyn_tp2)}
    else:
        return {"stop": entry + (atr * stop_mult), "tp1": entry - (atr * dyn_tp1), "tp2": entry - (atr * dyn_tp2)}

def evaluate_state_transition(side: str, current_state: str, executable_price: float, 
                              entry: float, stop: float, tp1: float, tp2: float) -> tuple[str, str]:
    is_buy = (side == "BUY")
    halfway_tp2 = (entry + tp2) / 2.0

    hit_sl = (executable_price <= stop) if is_buy else (executable_price >= stop)
    hit_tp1 = (executable_price >= tp1) if is_buy else (executable_price <= tp1)
    hit_tp2 = (executable_price >= tp2) if is_buy else (executable_price <= tp2)

    if hit_sl and (hit_tp1 or hit_tp2):
        return "CLOSED", "AMBIGUOUS_GAP_ASSUME_SL"
    if hit_sl:
        return "CLOSED", "HIT_SL"
    
    if current_state in ("OPEN", "COUNTERFACTUAL_OPEN"):
        if hit_tp1: return "PARTIAL_TP1", "HIT_TP1"
    
    if current_state == "PARTIAL_TP1":
        hit_halfway = (executable_price >= halfway_tp2) if is_buy else (executable_price <= halfway_tp2)
        if hit_halfway: return "TRAILING_TP1", "NONE"
    
    if current_state in ("PARTIAL_TP1", "TRAILING_TP1"):
        if hit_tp2: return "CLOSED", "HIT_TP2"

    return current_state, "NONE"

VALID_EXIT_REASONS = {
    "HIT_TP1", "HIT_TP2", "HIT_SL", 
    "AMBIGUOUS_GAP_ASSUME_SL", "TIMEOUT", "DATA_INVALIDATION"
}

def evaluate_thesis_outcome(exit_reason: str) -> int:
    if exit_reason not in VALID_EXIT_REASONS:
        raise ValueError(f"Unknown exit reason: {exit_reason}")
    return 1 if exit_reason in {"HIT_TP1", "HIT_TP2"} else 0

def execute_tp1_partial(trade: dict, fill_price: float, fee_rate: float, partial_pct: float = 0.5):
    is_buy = trade["side"] == "BUY"
    exit_qty = trade["qty"] * partial_pct
    
    gross_pnl = (fill_price - trade["simulated_entry"]) * exit_qty
    if not is_buy: gross_pnl *= -1
    
    fee_usd = (exit_qty * fill_price) * fee_rate
    
    trade["qty"] -= exit_qty
    trade["realized_gross_pnl"] = trade.get("realized_gross_pnl", 0.0) + gross_pnl
    trade["tp1_fee_usd"] = trade.get("tp1_fee_usd", 0.0) + fee_usd
    return trade
