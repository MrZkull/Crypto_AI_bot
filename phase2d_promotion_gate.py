"""Phase 2D promotion gate.

Phase 2D compares two locked model + threshold units on the same
prospective market stream.

This gate is deliberately separate from:
  - training-time Gate 10,
  - future Policy Parity Validation,
  - production promotion itself.

Current statistical method:
  - independent block bootstrap
  - UTC calendar day blocks
  - block keyed from prediction open_time
  - 10,000 bootstrap rounds
  - 95% confidence intervals

The paired-bootstrap method is intentionally deferred.

TODO(paired-bootstrap):
    Revisit once either model exceeds 150 clean resolved observations
    AND that model spans at least 20 unique UTC day blocks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


BLOCK_MS = 24 * 60 * 60 * 1000

MIN_RESOLVED = 50
MIN_UNIQUE_BLOCKS = 10

BOOTSTRAP_ROUNDS = 10_000
BOOTSTRAP_SEED = 42
CONF = 0.95


def _clean(records: list[dict]) -> pd.DataFrame:
    """Keep only economically resolved observations."""

    rows = [
        r for r in records
        if r.get("status") in {"TP", "SL", "EXPIRED"}
    ]

    df = pd.DataFrame(rows)

    if df.empty:
        return pd.DataFrame(
            columns=["open_time", "net_r", "status", "id", "signal"]
        )

    df["open_time"] = pd.to_numeric(
        df["open_time"],
        errors="coerce",
    )

    df["net_r"] = pd.to_numeric(
        df["net_r"],
        errors="coerce",
    )

    df = df[
        np.isfinite(df["open_time"])
        & np.isfinite(df["net_r"])
    ].copy()

    if "id" not in df.columns:
        df["id"] = ""

    if "signal" not in df.columns:
        df["signal"] = ""

    return df.reset_index(drop=True)


def _unique_block_count(df: pd.DataFrame) -> int:
    if df.empty:
        return 0

    blocks = (
        df["open_time"]
        .astype("int64")
        // BLOCK_MS
    )

    return int(blocks.nunique())


def _block_bootstrap_mean(
    df: pd.DataFrame,
    rng: np.random.Generator,
) -> np.ndarray:
    """Independent block bootstrap of mean net-R."""

    blocks = (
        df["open_time"]
        .astype("int64")
        // BLOCK_MS
    ).to_numpy()

    unique_blocks = np.unique(blocks)

    by_block = [
        df.loc[
            blocks == block_id,
            "net_r",
        ].to_numpy(dtype=float)
        for block_id in unique_blocks
    ]

    n_blocks = len(by_block)

    if n_blocks == 0:
        raise ValueError(
            "cannot bootstrap an empty sample"
        )

    out = np.empty(
        BOOTSTRAP_ROUNDS,
        dtype=float,
    )

    for i in range(BOOTSTRAP_ROUNDS):
        sampled = rng.integers(
            0,
            n_blocks,
            size=n_blocks,
        )

        out[i] = float(
            np.mean(
                np.concatenate(
                    [by_block[j] for j in sampled]
                )
            )
        )

    return out


def _side_summary(
    df: pd.DataFrame,
    side: str,
) -> dict:
    subset = df[df["signal"] == side].copy()

    if subset.empty:
        return {
            "n": 0,
            "tp": 0,
            "sl": 0,
            "expired": 0,
            "precision": None,
            "mean_net_r": None,
            "cum_net_r": 0.0,
        }

    wins = int(
        (subset["status"] == "TP").sum()
    )

    precision = wins / len(subset)

    return {
        "n": int(len(subset)),
        "tp": int((subset["status"] == "TP").sum()),
        "sl": int((subset["status"] == "SL").sum()),
        "expired": int((subset["status"] == "EXPIRED").sum()),
        "precision": round(float(precision), 6),
        "mean_net_r": round(
            float(subset["net_r"].mean()),
            6,
        ),
        "cum_net_r": round(
            float(subset["net_r"].sum()),
            6,
        ),
    }


def evaluate_promotion(
    candidate_resolved: list[dict],
    production_resolved: list[dict],
) -> dict:
    candidate = _clean(candidate_resolved)
    production = _clean(production_resolved)

    candidate_blocks = _unique_block_count(candidate)
    production_blocks = _unique_block_count(production)

    candidate_sample_ok = (
        len(candidate) >= MIN_RESOLVED
    )

    production_sample_ok = (
        len(production) >= MIN_RESOLVED
    )

    candidate_diversity_ok = (
        candidate_blocks >= MIN_UNIQUE_BLOCKS
    )

    production_diversity_ok = (
        production_blocks >= MIN_UNIQUE_BLOCKS
    )

    sample_gate_ok = (
        candidate_sample_ok
        and production_sample_ok
    )

    diversity_gate_ok = (
        candidate_diversity_ok
        and production_diversity_ok
    )

    common_gate_ok = (
        sample_gate_ok
        and diversity_gate_ok
    )

    if not common_gate_ok:
        reasons = []

        if not candidate_sample_ok:
            reasons.append(
                f"candidate N={len(candidate)} "
                f"< {MIN_RESOLVED}"
            )

        if not production_sample_ok:
            reasons.append(
                f"production N={len(production)} "
                f"< {MIN_RESOLVED}"
            )

        if not candidate_diversity_ok:
            reasons.append(
                f"candidate unique_blocks={candidate_blocks} "
                f"< {MIN_UNIQUE_BLOCKS}"
            )

        if not production_diversity_ok:
            reasons.append(
                f"production unique_blocks={production_blocks} "
                f"< {MIN_UNIQUE_BLOCKS}"
            )

        return {
            "promotion_ready": False,
            "gate_status": "NOT_READY",
            "reason": "; ".join(reasons),
            "sample_gate": sample_gate_ok,
            "diversity_gate": diversity_gate_ok,
            "candidate_sample_ok": candidate_sample_ok,
            "production_sample_ok": production_sample_ok,
            "candidate_diversity_ok": candidate_diversity_ok,
            "production_diversity_ok": production_diversity_ok,
            "n_candidate": len(candidate),
            "n_production": len(production),
            "unique_blocks_candidate": candidate_blocks,
            "unique_blocks_production": production_blocks,
            "min_resolved": MIN_RESOLVED,
            "min_unique_blocks": MIN_UNIQUE_BLOCKS,
            "bootstrap_method": "independent_block_bootstrap",
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_rounds": BOOTSTRAP_ROUNDS,
            "confidence_level": CONF,
            "block_ms": BLOCK_MS,
            "block_definition": (
                "UTC calendar day keyed by "
                "prediction open_time"
            ),
            "paired_bootstrap_revisit_trigger": (
                len(candidate) > 150
                and candidate_blocks >= 20
            )
            or (
                len(production) > 150
                and production_blocks >= 20
            ),
        }

    rng = np.random.default_rng(
        BOOTSTRAP_SEED
    )

    candidate_boot = _block_bootstrap_mean(
        candidate,
        rng,
    )

    production_boot = _block_bootstrap_mean(
        production,
        rng,
    )

    difference_boot = (
        candidate_boot
        - production_boot
    )

    tail = (1.0 - CONF) / 2.0

    candidate_ci = np.percentile(
        candidate_boot,
        [
            tail * 100.0,
            (1.0 - tail) * 100.0,
        ],
    )

    production_ci = np.percentile(
        production_boot,
        [
            tail * 100.0,
            (1.0 - tail) * 100.0,
        ],
    )

    difference_ci = np.percentile(
        difference_boot,
        [
            tail * 100.0,
            (1.0 - tail) * 100.0,
        ],
    )

    candidate_profitable = bool(
        candidate_ci[0] > 0.0
    )

    candidate_beats_production = bool(
        difference_ci[0] > 0.0
    )

    promotion_ready = bool(
        candidate_profitable
        and candidate_beats_production
    )

    return {
        "promotion_ready": promotion_ready,
        "gate_status": (
            "PASS"
            if promotion_ready
            else "FAIL"
        ),
        "sample_gate": True,
        "diversity_gate": True,
        "candidate_sample_ok": True,
        "production_sample_ok": True,
        "candidate_diversity_ok": True,
        "production_diversity_ok": True,
        "candidate_profitable": candidate_profitable,
        "candidate_beats_production": candidate_beats_production,
        "candidate_mean_net_r": round(
            float(candidate["net_r"].mean()),
            4,
        ),
        "production_mean_net_r": round(
            float(production["net_r"].mean()),
            4,
        ),
        "candidate_ci95": [
            round(float(x), 4)
            for x in candidate_ci
        ],
        "production_ci95": [
            round(float(x), 4)
            for x in production_ci
        ],
        "diff_ci95": [
            round(float(x), 4)
            for x in difference_ci
        ],
        "n_candidate": len(candidate),
        "n_production": len(production),
        "unique_blocks_candidate": candidate_blocks,
        "unique_blocks_production": production_blocks,
        "min_resolved": MIN_RESOLVED,
        "min_unique_blocks": MIN_UNIQUE_BLOCKS,
        "bootstrap_method": "independent_block_bootstrap",
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_rounds": BOOTSTRAP_ROUNDS,
        "confidence_level": CONF,
        "block_ms": BLOCK_MS,
        "block_definition": (
            "UTC calendar day keyed by "
            "prediction open_time"
        ),
        "paired_bootstrap_revisit_trigger": (
            len(candidate) > 150
            and candidate_blocks >= 20
        )
        or (
            len(production) > 150
            and production_blocks >= 20
        ),
    }
