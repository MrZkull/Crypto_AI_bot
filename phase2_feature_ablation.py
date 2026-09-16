#!/usr/bin/env python3
# phase2_feature_ablation.py — Isolated Phase 2 Feature Ablation Study

import time
import logging
import pandas as pd
from train_model import (
    preload_symbol_features, _apply_targets, train, BASE_FEATURES
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

def run_feature_ablation():
    t0 = time.time()
    log.info("Loading preloaded symbol features for ablation study...")
    preloaded = preload_symbol_features()
    ds = pd.concat([_apply_targets(df, 3.5, 2.5) for df in preloaded], ignore_index=True)

    # Invariant check: In ablation, fundingRate must be populated
    if "fundingRate" in ds.columns:
        fr = ds["fundingRate"].dropna()
        if len(fr) == 0 or (fr == 0).all() or fr.std() == 0:
            raise ValueError(
                "FATAL: Funding archive empty, all-NaN, or all-zero. "
                "Phase 2 variants would be numerically identical to BASELINE, "
                "producing a false negative result. Aborting ablation study."
            )
    else:
        raise ValueError("FATAL: 'fundingRate' column missing from historical archive.")

    experiment_grid = [
        {"name": "Baseline (No Funding/BTC)", "features": BASE_FEATURES},
        {"name": "Baseline + Funding Rate",   "features": BASE_FEATURES + ["fundingRate"]},
        {"name": "Baseline + BTC Metrics",    "features": BASE_FEATURES + ["btc_corr_20", "btc_beta_20", "btc_rel_strength"]},
        {"name": "Full Kitchen Sink",         "features": BASE_FEATURES + ["fundingRate", "btc_corr_20", "btc_beta_20", "btc_rel_strength"]},
    ]

    results = []
    for config in experiment_grid:
        log.info(f"🚀 Running Feature Ablation: {config['name']}")
        reserved = [f for f in config["features"] if f not in BASE_FEATURES]
        
        acc, score_b, score_s = train(
            ds,
            active_features=config["features"],
            sell_tp_mult=3.5,
            sell_sl_mult=2.5,
            export_artifact=False,
            reserved_features=reserved
        )
        results.append({
            "Config": config["name"],
            "BUY Score": f"{score_b:.2f}",
            "SELL Score": f"{score_s:.2f}",
            "Accuracy": f"{acc*100:.1f}%",
        })

    log.info(f"\n{'='*80}")
    log.info("🎯 FEATURE ABLATION EXPERIMENT SUMMARY (Chronological Lock, Target=3.5/2.5)")
    log.info(f"{'='*80}")
    log.info(f"{'Config':<32} | {'BUY Score':<10} | {'SELL Score':<10} | {'Accuracy':<10}")
    log.info("-" * 80)
    for res in results:
        log.info(f"{res['Config']:<32} | {res['BUY Score']:<10} | {res['SELL Score']:<10} | {res['Accuracy']:<10}")
    log.info("=" * 80)
    log.info(f"Ablation completed in {(time.time()-t0)/60:.1f} min")

if __name__ == "__main__":
    run_feature_ablation()
