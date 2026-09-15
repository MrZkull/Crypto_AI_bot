# gate10.py — 6-Tier Statistical Barrier Gate for Dual-Population Prospective Validation

import numpy as np
import pandas as pd


class Gate10Policy:
    MIN_TEST_TRADES: int = 50
    MIN_NET_EV_R: float = 0.05
    MIN_PRECISION_LIFT_PCT: float = 2.5
    MIN_RETENTION_RATE: float = 0.35
    CONFIDENCE_LEVEL: float = 0.95
    BOOTSTRAP_ROUNDS: int = 10_000
    BLOCK_SIZE_MS: int = 24 * 60 * 60 * 1000  # 24-hour calendar clusters


def evaluate_gate_10(df: pd.DataFrame) -> dict:
    """Evaluates 6-tier Gate 10 criteria on either synthetic research or empirical event ledgers."""
    required_cols = {"pred_id", "evidence_type", "primary_selected", "meta_selected", "thesis_label", "net_r"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CRITICAL: Gate 10 dataframe missing required columns: {missing}")

    if df["pred_id"].duplicated().any():
        raise ValueError("CRITICAL: Duplicate pred_id detected in Gate 10 evaluation dataframe.")

    # Prevent mixed evidence pollution
    evidence_types = df["evidence_type"].unique()
    if len(evidence_types) > 1:
        raise ValueError(f"CRITICAL: Corrupted Gate 10 ledger contains mixed evidence types: {evidence_types}")
    evidence_mode = evidence_types[0]

    is_empirical = (evidence_mode == "EMPIRICAL_PROSPECTIVE")
    primary_sample = df[df["primary_selected"].astype(bool)]
    meta_sample = df[df["meta_selected"].astype(bool)]

    n_primary = len(primary_sample)
    n_meta = len(meta_sample)

    # ── Tier 1: Sample Validity ──
    if n_primary == 0:
        return {
            "passed": False,
            "production_eligible": False,
            "passed_production_gate": False,
            "passed_research_gate": False,
            "tier_failed": "SAMPLE_VALIDITY",
            "evidence_mode": evidence_mode,
            "mode": evidence_mode,
            "reason": "Primary policy generated 0 eligible trades."
        }

    if n_meta < Gate10Policy.MIN_TEST_TRADES:
        return {
            "passed": False,
            "production_eligible": False,
            "passed_production_gate": False,
            "passed_research_gate": False,
            "tier_failed": "SAMPLE_VALIDITY",
            "evidence_mode": evidence_mode,
            "mode": evidence_mode,
            "reason": f"Filtered trades {n_meta} < minimum requirement {Gate10Policy.MIN_TEST_TRADES}"
        }

    # ── Tier 2: Economic Viability ──
    mean_net_r = float(meta_sample["net_r"].mean())
    if mean_net_r < Gate10Policy.MIN_NET_EV_R:
        return {
            "passed": False,
            "production_eligible": False,
            "passed_production_gate": False,
            "passed_research_gate": False,
            "tier_failed": "ECONOMIC_VIABILITY",
            "evidence_mode": evidence_mode,
            "mode": evidence_mode,
            "reason": f"Mean Net-R {mean_net_r:.4f} < hurdle {Gate10Policy.MIN_NET_EV_R}R"
        }

    # ── Tier 3: Statistical Significance (Block-Bootstrap) ──
    rng = np.random.default_rng(42)
    boot_means = np.empty(Gate10Policy.BOOTSTRAP_ROUNDS)

    if is_empirical and "open_time" in meta_sample.columns:
        blocks = (meta_sample["open_time"] // Gate10Policy.BLOCK_SIZE_MS).values
        unique_blocks = np.unique(blocks)
        n_blocks = len(unique_blocks)
        block_returns = [meta_sample.loc[blocks == b, "net_r"].values for b in unique_blocks]

        for b in range(Gate10Policy.BOOTSTRAP_ROUNDS):
            sampled_idx = rng.choice(n_blocks, size=n_blocks, replace=True)
            resampled = np.concatenate([block_returns[i] for i in sampled_idx])
            boot_means[b] = np.mean(resampled)
    else:
        returns_arr = meta_sample["net_r"].values
        for b in range(Gate10Policy.BOOTSTRAP_ROUNDS):
            boot_means[b] = np.mean(rng.choice(returns_arr, size=n_meta, replace=True))

    tail = (1.0 - Gate10Policy.CONFIDENCE_LEVEL) / 2.0
    ci_lower = float(np.percentile(boot_means, tail * 100))
    ci_upper = float(np.percentile(boot_means, (1.0 - tail) * 100))

    if ci_lower <= 0.0:
        return {
            "passed": False,
            "production_eligible": False,
            "passed_production_gate": False,
            "passed_research_gate": False,
            "tier_failed": "STATISTICAL_SIGNIFICANCE",
            "evidence_mode": evidence_mode,
            "mode": evidence_mode,
            "reason": f"Bootstrap 95% CI lower bound {ci_lower:.4f} <= 0.0R"
        }

    # ── Tier 4: Incremental Lift ──
    meta_precision = float(meta_sample["thesis_label"].mean())
    primary_precision = float(primary_sample["thesis_label"].mean())
    primary_mean_r = float(primary_sample["net_r"].mean())

    precision_lift_pct = (meta_precision - primary_precision) * 100.0
    r_lift = mean_net_r - primary_mean_r

    if precision_lift_pct < Gate10Policy.MIN_PRECISION_LIFT_PCT or r_lift <= 0.0:
        return {
            "passed": False,
            "production_eligible": False,
            "passed_production_gate": False,
            "passed_research_gate": False,
            "tier_failed": "INCREMENTAL_PERFORMANCE",
            "evidence_mode": evidence_mode,
            "mode": evidence_mode,
            "reason": f"Precision lift {precision_lift_pct:.2f}% < {Gate10Policy.MIN_PRECISION_LIFT_PCT}% or R-lift {r_lift:.4f} <= 0.0R"
        }

    # ── Tier 5: Capacity & Retention ──
    retention_rate = n_meta / n_primary
    if retention_rate < Gate10Policy.MIN_RETENTION_RATE:
        return {
            "passed": False,
            "production_eligible": False,
            "passed_production_gate": False,
            "passed_research_gate": False,
            "tier_failed": "CAPACITY",
            "evidence_mode": evidence_mode,
            "mode": evidence_mode,
            "reason": f"Trade retention {retention_rate*100:.1f}% < minimum {Gate10Policy.MIN_RETENTION_RATE*100:.1f}%"
        }

    # ── Tier 6: Production Eligibility (Strict Mode Enforced) ──
    passed_production = is_empirical and (ci_lower > 0.0)
    passed_research = True

    return {
        "passed": passed_production if is_empirical else passed_research,
        "production_eligible": passed_production,
        "passed_production_gate": passed_production,
        "passed_research_gate": passed_research,
        "evidence_mode": evidence_mode,
        "mode": evidence_mode,
        "n_primary": n_primary,
        "n_meta": n_meta,
        "retention_rate_pct": round(retention_rate * 100, 2),
        "primary_precision_pct": round(primary_precision * 100, 2),
        "meta_precision_pct": round(meta_precision * 100, 2),
        "precision_lift_pct": round(precision_lift_pct, 2),
        "primary_mean_net_r": round(primary_mean_r, 4),
        "meta_mean_net_r": round(mean_net_r, 4),
        "r_lift": round(r_lift, 4),
        "bootstrap_ci_95": [round(ci_lower, 4), round(ci_upper, 4)],
        "production_status": "APPROVED_FOR_PROMOTION" if passed_production else "RESEARCH_BENCHMARK_ONLY"
    }
