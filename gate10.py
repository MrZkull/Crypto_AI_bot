#gate10.py

import numpy as np
import pandas as pd

ALLOWED_EMPIRICAL = {"PROSPECTIVE_SHADOW_EXECUTION", "PROSPECTIVE_COUNTERFACTUAL"}
ALLOWED_SYNTHETIC = {"SYNTHETIC_BARRIER"}

class Gate10Policy:
    MIN_META_TRADES = 50
    MIN_NET_EV_R = 0.05
    MIN_PRECISION_LIFT_PCT = 2.5
    MIN_RETENTION_RATE = 0.35
    CONFIDENCE_LEVEL = 0.95
    BOOTSTRAP_ROUNDS = 10_000

def evaluate_gate_10(df: pd.DataFrame) -> dict:
    required_cols = {"pred_id", "evidence_type", "primary_selected", "meta_selected", "thesis_label", "net_r"}
    missing = required_cols - set(df.columns)
    if missing:
        return {"passed": False, "reason": f"Missing columns: {sorted(missing)}"}

    evidence_types = set(df["evidence_type"].dropna().astype(str).unique())
    if not evidence_types:
        return {"passed": False, "reason": "No evidence_type values present."}

    if evidence_types <= ALLOWED_SYNTHETIC:
        eval_mode = "RESEARCH_SYNTHETIC"
    elif evidence_types <= ALLOWED_EMPIRICAL:
        eval_mode = "PROSPECTIVE_EMPIRICAL"
    else:
        return {"passed": False, "reason": f"Unsupported/mixed evidence types: {sorted(evidence_types)}"}

    if eval_mode == "PROSPECTIVE_EMPIRICAL":
        if "PROSPECTIVE_SHADOW_EXECUTION" not in evidence_types:
            return {"passed": False, "reason": "No meta-selected prospective population present (Pop C)."}
        if "PROSPECTIVE_COUNTERFACTUAL" not in evidence_types:
            return {"passed": False, "reason": "No primary-only counterfactual population present (Pop B)."}

    primary_sample = df[df["primary_selected"]]
    meta_sample = df[df["meta_selected"]]

    primary_ids = set(primary_sample["pred_id"])
    meta_ids = set(meta_sample["pred_id"])
    if not meta_ids.issubset(primary_ids):
        return {"passed": False, "reason": "Meta population is not a strict subset of primary population."}

    n_primary = len(primary_sample)
    n_meta = len(meta_sample)

    if n_meta < Gate10Policy.MIN_META_TRADES:
        return {"passed": False, "reason": f"Insufficient meta-selected evidence ({n_meta}/{Gate10Policy.MIN_META_TRADES})", "mode": eval_mode}

    mean_net_r = float(meta_sample["net_r"].mean())
    if mean_net_r < Gate10Policy.MIN_NET_EV_R:
        return {"passed": False, "reason": f"Mean Net-R {mean_net_r:.4f} < hurdle", "mode": eval_mode}

    rng = np.random.default_rng(42)
    boot_means = np.empty(Gate10Policy.BOOTSTRAP_ROUNDS)
    
    if "entry_time_ms" in meta_sample.columns and eval_mode == "PROSPECTIVE_EMPIRICAL":
        meta_df = meta_sample.copy()
        meta_df["block_id"] = meta_df["entry_time_ms"] // (24 * 60 * 60 * 1000)
        unique_blocks = meta_df["block_id"].unique()
        block_returns = [meta_df.loc[meta_df["block_id"] == b, "net_r"].values for b in unique_blocks]
        
        for b in range(Gate10Policy.BOOTSTRAP_ROUNDS):
            sampled = rng.choice(len(unique_blocks), size=len(unique_blocks), replace=True)
            resampled = np.concatenate([block_returns[i] for i in sampled])
            boot_means[b] = np.mean(resampled)
    else:
        arr = meta_sample["net_r"].values
        for b in range(Gate10Policy.BOOTSTRAP_ROUNDS):
            boot_means[b] = np.mean(rng.choice(arr, size=n_meta, replace=True))

    ci_lower = float(np.percentile(boot_means, 2.5))
    if ci_lower <= 0.0:
        return {"passed": False, "reason": f"95% CI Lower {ci_lower:.4f} <= 0", "mode": eval_mode}

    meta_prec = float(meta_sample["thesis_label"].mean())
    prim_prec = float(primary_sample["thesis_label"].mean())
    prim_mean_r = float(primary_sample["net_r"].mean())
    
    prec_lift = (meta_prec - prim_prec) * 100.0
    r_lift = mean_net_r - prim_mean_r
    retention = n_meta / n_primary

    if prec_lift < Gate10Policy.MIN_PRECISION_LIFT_PCT or r_lift <= 0.0:
        return {"passed": False, "reason": f"Insufficient lift (Prec={prec_lift:.2f}%, R={r_lift:.4f})", "mode": eval_mode}
        
    if retention < Gate10Policy.MIN_RETENTION_RATE:
        return {"passed": False, "reason": f"Retention {retention*100:.1f}% < 35%", "mode": eval_mode}

    return {
        "passed": True, "mode": eval_mode,
        "n_primary": n_primary, "n_meta": n_meta,
        "mean_net_r": round(mean_net_r, 4), "ci_lower": round(ci_lower, 4),
        "primary_precision_pct": round(prim_prec * 100, 2), "meta_precision_pct": round(meta_prec * 100, 2),
        "precision_lift_pct": round(prec_lift, 2), "r_lift": round(r_lift, 4), "retention_pct": round(retention * 100, 2)
    }
