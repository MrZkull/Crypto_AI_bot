"""Phase 2D promotion gate.

This is deliberately separate from Gate 10: Phase 2D compares two locked
models on the same prospective stream; Gate 10 evaluates meta-labeling.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

BLOCK_MS = 24 * 60 * 60 * 1000
MIN_RESOLVED = 50
BOOTSTRAP_ROUNDS = 10_000
CONF = 0.95


def _clean(records: list[dict]) -> pd.DataFrame:
    rows = [r for r in records if r.get("status") in {"TP", "SL", "EXPIRED"}]
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["open_time", "net_r", "status"])
    df["open_time"] = pd.to_numeric(df["open_time"], errors="coerce")
    df["net_r"] = pd.to_numeric(df["net_r"], errors="coerce")
    df = df[np.isfinite(df["open_time"]) & np.isfinite(df["net_r"])].copy()
    return df.reset_index(drop=True)


def _block_bootstrap_mean(df: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    blocks = (df["open_time"].astype("int64") // BLOCK_MS).to_numpy()
    unique_blocks = np.unique(blocks)
    by_block = [df.loc[blocks == b, "net_r"].to_numpy(dtype=float) for b in unique_blocks]
    n = len(by_block)
    if n == 0:
        raise ValueError("cannot bootstrap an empty sample")
    out = np.empty(BOOTSTRAP_ROUNDS, dtype=float)
    for i in range(BOOTSTRAP_ROUNDS):
        idx = rng.integers(0, n, size=n)
        out[i] = float(np.mean(np.concatenate([by_block[j] for j in idx])))
    return out


def evaluate_promotion(candidate_resolved: list[dict], production_resolved: list[dict]) -> dict:
    candidate = _clean(candidate_resolved)
    production = _clean(production_resolved)

    if len(candidate) < MIN_RESOLVED or len(production) < MIN_RESOLVED:
        return {
            "promotion_ready": False,
            "reason": (
                f"insufficient sample: candidate={len(candidate)}, "
                f"production={len(production)} (need {MIN_RESOLVED} each)"
            ),
            "n_candidate": len(candidate),
            "n_production": len(production),
        }

    rng = np.random.default_rng(42)
    c_boot = _block_bootstrap_mean(candidate, rng)
    p_boot = _block_bootstrap_mean(production, rng)
    diff_boot = c_boot - p_boot

    tail = (1.0 - CONF) / 2.0
    c_ci = np.percentile(c_boot, [tail * 100.0, (1.0 - tail) * 100.0])
    p_ci = np.percentile(p_boot, [tail * 100.0, (1.0 - tail) * 100.0])
    diff_ci = np.percentile(diff_boot, [tail * 100.0, (1.0 - tail) * 100.0])

    candidate_profitable = bool(c_ci[0] > 0.0)
    candidate_beats_production = bool(diff_ci[0] > 0.0)

    return {
        "promotion_ready": bool(candidate_profitable and candidate_beats_production),
        "candidate_profitable": candidate_profitable,
        "candidate_beats_production": candidate_beats_production,
        "candidate_mean_net_r": round(float(candidate["net_r"].mean()), 4),
        "production_mean_net_r": round(float(production["net_r"].mean()), 4),
        "candidate_ci95": [round(float(x), 4) for x in c_ci],
        "production_ci95": [round(float(x), 4) for x in p_ci],
        "diff_ci95": [round(float(x), 4) for x in diff_ci],
        "n_candidate": len(candidate),
        "n_production": len(production),
        "bootstrap_rounds": BOOTSTRAP_ROUNDS,
        "block_ms": BLOCK_MS,
    }
