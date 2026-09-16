#!/usr/bin/env python3
"""
phase2b_feature_group_ablation.py

Phase 2B — Canonical Feature Group Ablation

Research-only script. It keeps the Phase 2A evaluation path frozen and
removes one canonical baseline feature group at a time.
"""

import json
import logging
import time
from pathlib import Path

import pandas as pd

from feature_engineering import ALL_FEATURES
from train_model import build_dataset_from_local_parquet
from phase2_feature_ablation import (
    BASE_FEATURES,
    run_single_ablation,
    validate_funding_integrity,
    validate_experiment_features,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

BTC_RELATIONAL_FEATURES = {
    "btc_corr_20",
    "btc_beta_20",
    "btc_rel_strength",
}

FEATURE_GROUPS = {
    "Momentum & Trend": [
        "rsi", "macd", "macd_signal", "macd_hist", "adx",
        "plus_di", "minus_di", "trend", "ema20_vs_ema50",
        "price_vs_ema200", "regime_uptrend",
    ],
    "Volatility & Bands": [
        "atr", "atr_pct", "bb_width", "bb_pos", "volatility",
    ],
    "Volume Dynamics": [
        "volume_ratio", "volume_spike", "obv_slope", "vwap_dev",
        "taker_buy_ratio",
    ],
    "Time Cycles": [
        "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    ],
    "Higher Timeframe Context": [
        "rsi_1h", "adx_1h", "trend_1h", "rsi_4h", "trend_4h",
    ],
}

COMBINED_NAME = "Momentum & Trend + Volume Dynamics"
COMBINED_FEATURES = FEATURE_GROUPS["Momentum & Trend"] + FEATURE_GROUPS["Volume Dynamics"]


def validate_group_schema() -> None:
    canonical = set(ALL_FEATURES)
    baseline = set(BASE_FEATURES)
    expected_baseline = canonical - BTC_RELATIONAL_FEATURES

    if baseline != expected_baseline:
        raise ValueError(
            "FATAL: Phase 2B baseline does not equal ALL_FEATURES minus "
            f"BTC relational features. Missing={sorted(expected_baseline - baseline)} "
            f"Extra={sorted(baseline - expected_baseline)}"
        )

    assigned = []
    for name, features in FEATURE_GROUPS.items():
        if len(features) != len(set(features)):
            raise ValueError(f"FATAL: Duplicate feature in group '{name}'.")
        missing = sorted(set(features) - baseline)
        if missing:
            raise ValueError(f"FATAL: Group '{name}' contains non-baseline features: {missing}")
        assigned.extend(features)

    if set(assigned) != baseline or len(assigned) != len(baseline):
        raise ValueError(
            "FATAL: Phase 2B groups do not form an exact partition of the 30-feature baseline."
        )

    if set(COMBINED_FEATURES) - baseline:
        raise ValueError("FATAL: Combined ablation contains non-baseline features.")

    log.info(
        "Phase 2B schema OK | baseline=%d | groups=%d",
        len(BASE_FEATURES),
        len(FEATURE_GROUPS),
    )


def main() -> None:
    t0 = time.time()

    log.info("=" * 120)
    log.info("PHASE 2B — CANONICAL FEATURE GROUP ABLATION")
    log.info("=" * 120)

    validate_group_schema()

    log.info("Loading canonical Parquet dataset...")
    ds = build_dataset_from_local_parquet(sell_tp_mult=3.5, sell_sl_mult=2.5)

    if ds is None or ds.empty:
        raise ValueError("FATAL: Canonical dataset is empty.")

    log.info(
        "Canonical dataset loaded: rows=%d | columns=%d | symbols=%d",
        len(ds), len(ds.columns), ds["symbol"].nunique(),
    )

    validate_funding_integrity(ds)

    grid = [{"name": "Baseline", "features": list(BASE_FEATURES)}]

    for group_name, group_features in FEATURE_GROUPS.items():
        grid.append({
            "name": f"Minus {group_name}",
            "features": [f for f in BASE_FEATURES if f not in set(group_features)],
            "removed_group": group_name,
            "removed_features": list(group_features),
        })

    grid.append({
        "name": f"Minus {COMBINED_NAME}",
        "features": [f for f in BASE_FEATURES if f not in set(COMBINED_FEATURES)],
        "removed_group": COMBINED_NAME,
        "removed_features": list(COMBINED_FEATURES),
    })

    validate_experiment_features(ds, [{"name": x["name"], "features": x["features"]} for x in grid])

    results = []

    for i, cfg in enumerate(grid, 1):
        log.info("")
        log.info("=" * 120)
        log.info("TRAINING VARIANT %d/%d: %s", i, len(grid), cfg["name"])
        log.info("=" * 120)

        metrics = run_single_ablation(
            ds=ds,
            feature_set=cfg["features"],
            reserved_features=[],
        )
        results.append({
            "Config": cfg["name"],
            "Removed_Group": cfg.get("removed_group", ""),
            "Removed_Features": cfg.get("removed_features", []),
            **metrics,
        })

    log.info("")
    log.info("=" * 170)
    log.info("PHASE 2B EXPERIMENT RESULTS")
    log.info("=" * 170)
    log.info(
        "%s | %7s | %5s | %5s | %7s | %7s | %8s | %5s | %5s | %7s | %7s | %8s",
        "Variant".ljust(52), "Acc", "B-Th", "B-N", "B-Prec", "B-EV", "B-Score",
        "S-Th", "S-N", "S-Prec", "S-EV", "S-Score",
    )
    log.info("-" * 170)

    for r in results:
        log.info(
            "%s | %6.1f%% | %5.2f | %5d | %6.1f%% | %6.2fR | %8.3f | %5.2f | %5d | %6.1f%% | %6.2fR | %8.3f",
            r["Config"].ljust(52),
            r["Accuracy"] * 100,
            r["BUY_Thresh"], r["BUY_N"], r["BUY_Prec"] * 100, r["BUY_EV"], r["BUY_Score"],
            r["SELL_Thresh"], r["SELL_N"], r["SELL_Prec"] * 100, r["SELL_EV"], r["SELL_Score"],
        )

    csv_rows = []
    for r in results:
        row = dict(r)
        row["Removed_Features"] = ",".join(row["Removed_Features"])
        row["Selected_Feature_Names"] = ",".join(row["Selected_Feature_Names"])
        csv_rows.append(row)

    csv_path = ARTIFACTS_DIR / "phase2b_feature_group_ablation.csv"
    json_path = ARTIFACTS_DIR / "phase2b_feature_group_ablation.json"
    manifest_path = ARTIFACTS_DIR / "phase2b_feature_group_ablation_manifest.json"

    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    manifest = {
        "experiment": "Phase 2B Canonical Feature Group Ablation",
        "baseline_feature_count": len(BASE_FEATURES),
        "baseline_features": list(BASE_FEATURES),
        "btc_relational_features_excluded": sorted(BTC_RELATIONAL_FEATURES),
        "feature_groups": FEATURE_GROUPS,
        "combined_ablation": {
            "name": COMBINED_NAME,
            "features": COMBINED_FEATURES,
        },
        "dataset_rows": len(ds),
        "dataset_symbols": int(ds["symbol"].nunique()),
        "production_export": False,
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    log.info("=")
    log.info("PHASE 2B COMPLETE in %.1f minutes", (time.time() - t0) / 60.0)
    log.info("CSV artifact     : %s", csv_path)
    log.info("JSON artifact    : %s", json_path)
    log.info("Manifest artifact: %s", manifest_path)
    log.info("=" * 120)


if __name__ == "__main__":
    main()
