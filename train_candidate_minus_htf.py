#!/usr/bin/env python3
"""
train_candidate_minus_htf.py

Builds the Phase 2C HTF-removal hypothesis as a real canonical candidate,
without allowing the current train_model.py to overwrite the production
model file.
"""

import json
import hashlib
from pathlib import Path

import train_model
from feature_engineering import ALL_FEATURES

HTF_FEATURES = {
    "rsi_1h",
    "adx_1h",
    "trend_1h",
    "rsi_4h",
    "trend_4h",
}

BTC_FEATURES = {
    "btc_corr_20",
    "btc_beta_20",
    "btc_rel_strength",
}

BASELINE_FEATURES = [
    f for f in ALL_FEATURES
    if f not in BTC_FEATURES
]

ACTIVE_FEATURES = [
    f for f in BASELINE_FEATURES
    if f not in HTF_FEATURES
]

EXPECTED_ACTIVE_COUNT = 25

RESEARCH_DIR = Path("research_outputs")
PRODUCTION_SINK = RESEARCH_DIR / "production_model_sink.pkl"
RESEARCH_RECORD = RESEARCH_DIR / "candidate_minus_htf_record.json"


def validate_feature_lock():
    canonical = set(ALL_FEATURES)

    if len(ALL_FEATURES) != 33:
        raise ValueError(
            f"FATAL: Expected 33 canonical features, got {len(ALL_FEATURES)}"
        )

    if not BTC_FEATURES <= canonical:
        raise ValueError("FATAL: BTC feature definition drift detected.")

    if not HTF_FEATURES <= canonical:
        raise ValueError("FATAL: HTF feature definition drift detected.")

    if len(BASELINE_FEATURES) != 30:
        raise ValueError(
            f"FATAL: Expected 30-feature Phase 2A baseline, "
            f"got {len(BASELINE_FEATURES)}"
        )

    if len(ACTIVE_FEATURES) != EXPECTED_ACTIVE_COUNT:
        raise ValueError(
            f"FATAL: Expected {EXPECTED_ACTIVE_COUNT} active features, "
            f"got {len(ACTIVE_FEATURES)}"
        )

    if set(ACTIVE_FEATURES) & HTF_FEATURES:
        raise ValueError("FATAL: HTF feature survived removal.")

    if not set(ACTIVE_FEATURES) <= set(BASELINE_FEATURES):
        raise ValueError("FATAL: Candidate contains non-baseline features.")

    print("=" * 90)
    print("HTF-REMOVAL CANDIDATE FEATURE LOCK")
    print("=" * 90)
    print(f"Baseline features : {len(BASELINE_FEATURES)}")
    print(f"Removed HTF       : {sorted(HTF_FEATURES)}")
    print(f"Active features   : {len(ACTIVE_FEATURES)}")
    print("=" * 90)


def main():
    validate_feature_lock()

    RESEARCH_DIR.mkdir(parents=True, exist_ok=True)

    # Protect production: current train() writes MODEL_FILE and
    # CANDIDATE_MODEL_FILE. Redirect only MODEL_FILE to a research sink.
    original_model_file = train_model.MODEL_FILE
    train_model.MODEL_FILE = str(PRODUCTION_SINK)

    try:
        print("Building canonical dataset...")
        dataset = train_model.build_dataset_from_local_parquet(
            sell_tp_mult=3.5,
            sell_sl_mult=2.5,
        )

        if dataset is None or dataset.empty:
            raise ValueError("FATAL: Canonical dataset is empty.")

        print(
            f"Dataset rows={len(dataset)} | "
            f"symbols={dataset['symbol'].nunique()}"
        )

        print("")
        print("=" * 90)
        print("TRAINING 25-FEATURE HTF-REMOVAL CANDIDATE")
        print("=" * 90)

        acc, score_buy, score_sell = train_model.train(
            dataset,
            active_features=ACTIVE_FEATURES,
            sell_tp_mult=3.5,
            sell_sl_mult=2.5,
            export_artifact=True,
            reserved_features=None,
        )

        candidate_path = Path(train_model.CANDIDATE_MODEL_FILE)
        manifest_path = Path(train_model.CANDIDATE_MANIFEST)

        if not candidate_path.exists():
            raise FileNotFoundError(
                "FATAL: candidate_model.pkl was not generated."
            )

        if not manifest_path.exists():
            raise FileNotFoundError(
                "FATAL: candidate_manifest.json was not generated."
            )

        model_sha256 = hashlib.sha256(
            candidate_path.read_bytes()
        ).hexdigest()

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        if manifest.get("model_sha256") != model_sha256:
            raise ValueError(
                "FATAL: Candidate manifest SHA256 does not match model."
            )

        if manifest.get("status") != "AWAITING_PROSPECTIVE_EVIDENCE":
            raise ValueError(
                "FATAL: Unexpected candidate status: "
                f"{manifest.get('status')}"
            )

        record = {
            "experiment": "Phase 2C HTF-removal candidate",
            "baseline_feature_count": len(BASELINE_FEATURES),
            "active_feature_count": len(ACTIVE_FEATURES),
            "removed_features": sorted(HTF_FEATURES),
            "active_features": ACTIVE_FEATURES,
            "tp_r": 3.5,
            "sl_r": 2.5,
            "test_accuracy": float(acc),
            "buy_score": float(score_buy),
            "sell_score": float(score_sell),
            "candidate_id": manifest.get("candidate_id"),
            "candidate_status": manifest.get("status"),
            "candidate_model_sha256": model_sha256,
            "recommended_threshold_buy": manifest.get(
                "recommended_threshold_buy"
            ),
            "recommended_threshold_sell": manifest.get(
                "recommended_threshold_sell"
            ),
        }

        with open(
            RESEARCH_RECORD,
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(record, f, indent=2)

        print("")
        print("=" * 90)
        print("CANDIDATE BUILD COMPLETE")
        print("=" * 90)
        print(f"Test Accuracy : {acc * 100:.2f}%")
        print(f"BUY Score     : {score_buy:.4f}")
        print(f"SELL Score    : {score_sell:.4f}")
        print(f"Candidate     : {candidate_path}")
        print(f"Manifest      : {manifest_path}")
        print(f"Research      : {RESEARCH_RECORD}")
        print("=" * 90)

    finally:
        train_model.MODEL_FILE = original_model_file


if __name__ == "__main__":
    main()
