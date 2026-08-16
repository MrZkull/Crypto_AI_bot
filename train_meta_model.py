# train_meta_model.py — P1: Meta-labeling (López de Prado) Pipeline

import json, logging, time
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, accuracy_score
from sklearn.frozen import FrozenEstimator
from xgboost import XGBClassifier

from train_model import (
    build_dataset, FULL_FEATURES, MODEL_FILE, EMBARGO_BARS, TEST_SPLIT, CALIB_SPLIT
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

META_MODEL_FILE = "meta_pipeline.pkl"
N_META_FEATURES = 25  # Meta-model uses a focused subset of the most decisive features


def get_primary_predictions(ds: pd.DataFrame, primary_pipeline: dict) -> pd.DataFrame:
    """Run the existing trained primary model over the dataset to get directional signals."""
    af = primary_pipeline["all_features"]
    for f in af:
        if f not in ds.columns:
            ds[f] = 0.0

    X  = ds[af].replace([np.inf, -np.inf], np.nan).fillna(0)
    Xs = primary_pipeline["selector"].transform(X)

    preds  = primary_pipeline["ensemble"].predict(Xs)
    probas = primary_pipeline["ensemble"].predict_proba(Xs)
    label_map = primary_pipeline["label_map"]

    ds = ds.copy()
    ds["primary_side"] = [label_map[int(p)] for p in preds]
    ds["primary_conf"] = probas.max(axis=1)
    return ds


def build_meta_labels(ds: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only rows where the primary model called BUY or SELL.
    Meta-label is binary: Did the primary call match the true forward outcome? (1 = Win, 0 = Loss)
    """
    directional = ds[ds["primary_side"] != "NO_TRADE"].copy()
    directional["meta_label"] = (directional["primary_side"] == directional["target"]).astype(int)
    return directional


def per_regime_split(ds: pd.DataFrame, test_split: float, calib_split: float, embargo: int):
    """Embargoed per-regime split to prevent temporal data leakage."""
    train_parts, calib_parts, test_parts = [], [], []
    if "regime" not in ds.columns:
        ds["regime"] = "unknown"

    for regime, grp in ds.groupby("regime", sort=False):
        grp = grp.sort_values("open_time").reset_index(drop=True)
        n_r = len(grp)
        test_size_r  = int(n_r * test_split)
        calib_size_r = int(n_r * calib_split)
        test_start_r  = n_r - test_size_r
        calib_end_r   = test_start_r - embargo
        calib_start_r = calib_end_r - calib_size_r
        train_end_r   = calib_start_r - embargo

        if train_end_r <= 0:
            train_parts.append(grp)
            continue

        train_parts.append(grp.iloc[:train_end_r])
        calib_parts.append(grp.iloc[calib_start_r:calib_end_r])
        test_parts.append(grp.iloc[test_start_r:])

    return (
        pd.concat(train_parts, ignore_index=True),
        pd.concat(calib_parts, ignore_index=True) if calib_parts else train_parts[0].iloc[:0],
        pd.concat(test_parts, ignore_index=True) if test_parts else train_parts[0].iloc[:0]
    )


def train_meta_model():
    log.info("Loading primary model (pro_crypto_ai_model.pkl)...")
    try:
        primary_pipeline = joblib.load(MODEL_FILE)
    except Exception as e:
        log.error(f"Failed to load {MODEL_FILE}. Run train_model.py first! Error: {e}")
        return

    log.info("Building dataset (same data & regimes as primary)...")
    ds = build_dataset()
    ds = ds.sort_values("open_time").reset_index(drop=True)

    # Reconstruct the exact held-out test split from the primary model
    log.info("Isolating primary model's held-out test split to ensure leak-free evaluation...")
    _, _, primary_test = per_regime_split(ds, TEST_SPLIT, CALIB_SPLIT, EMBARGO_BARS)
    log.info(f"Primary held-out test set: {len(primary_test):,} rows")

    log.info("Generating primary model calls on held-out test rows...")
    primary_test = get_primary_predictions(primary_test, primary_pipeline)

    directional = build_meta_labels(primary_test)
    n_dir = len(directional)
    n_correct = directional["meta_label"].sum()
    base_rate = (n_correct / n_dir * 100) if n_dir > 0 else 0.0

    log.info(f"Primary called a direction on {n_dir:,} held-out rows ({n_dir/len(primary_test)*100:.1f}% frequency)")
    log.info(f"True out-of-sample base hit rate: {base_rate:.1f}% ({n_correct:,}/{n_dir:,} correct)")

    if n_dir < 300:
        log.error(f"Sample size too small ({n_dir} directional calls). Need at least 300 to train a meta-model.")
        return

    train_df, calib_df, test_df = per_regime_split(directional, TEST_SPLIT, CALIB_SPLIT, EMBARGO_BARS)
    log.info(f"Meta Split: train={len(train_df):,} | calib={len(calib_df):,} | test={len(test_df):,}")

    for f in FULL_FEATURES:
        for part in (train_df, calib_df, test_df):
            if f not in part.columns:
                part[f] = 0.0

    log.info("Running feature importance scan for meta-model...")
    X_train_full = train_df[FULL_FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0)
    y_train = train_df["meta_label"].values

    scanner = XGBClassifier(n_estimators=100, random_state=42, n_jobs=-1, eval_metric="logloss")
    scanner.fit(X_train_full, y_train)
    top_idx = np.argsort(scanner.feature_importances_)[::-1][:N_META_FEATURES]
    meta_features = [FULL_FEATURES[i] for i in top_idx]
    log.info(f"Top {len(meta_features)} meta-features selected: {meta_features[:8]}...")

    X_train = train_df[meta_features].replace([np.inf, -np.inf], np.nan).fillna(0).values
    X_calib = calib_df[meta_features].replace([np.inf, -np.inf], np.nan).fillna(0).values
    X_test  = test_df[meta_features].replace([np.inf, -np.inf], np.nan).fillna(0).values
    y_calib = calib_df["meta_label"].values
    y_test  = test_df["meta_label"].values

    log.info("Training Meta Ensemble (XGBoost + RandomForest)...")
    meta_xgb = XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.03,
        subsample=0.85, colsample_bytree=0.85, min_child_weight=3,
        eval_metric="logloss", random_state=42, n_jobs=-1,
    )
    meta_xgb.fit(X_train, y_train)

    meta_rf = RandomForestClassifier(
        n_estimators=300, max_depth=10, min_samples_leaf=5,
        random_state=42, n_jobs=-1,
    )
    meta_rf.fit(X_train, y_train)

    meta_ensemble = VotingClassifier(
        estimators=[("xgb", meta_xgb), ("rf", meta_rf)], voting="soft", weights=[2, 1],
    )
    meta_ensemble.fit(X_train, y_train)

    log.info("Calibrating meta-model probabilities on held-out calib set...")
    calibrated_meta = CalibratedClassifierCV(estimator=FrozenEstimator(meta_ensemble), method="isotonic")
    calibrated_meta.fit(X_calib, y_calib)

    y_pred = calibrated_meta.predict(X_test)
    y_proba = calibrated_meta.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    log.info(f"\n{'='*60}")
    log.info(f"META-MODEL TEST ACCURACY: {acc*100:.1f}%  (Base Rate: {base_rate:.1f}%)")
    log.info(f"{'='*60}")
    log.info(f"  Precision (Call is Correct): {report.get('1', {}).get('precision', 0):.1%}")
    log.info(f"  Recall    (Call is Correct): {report.get('1', {}).get('recall', 0):.1%}")

    # Threshold optimization sweep
    log.info("\n── Meta-confidence threshold sweep ──────────────────────")
    best_thresh, best_score = 0.50, 0.0
    for thresh in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
        mask = y_proba >= thresh
        n_sel = mask.sum()
        if n_sel < 10:
            continue
        hit_rate = y_test[mask].mean()
        score = hit_rate * np.sqrt(n_sel)
        log.info(f"  {thresh:.2f}   n={n_sel:>6}   hit_rate={hit_rate*100:.1f}%   score={score:.1f}")
        if score > best_score:
            best_score, best_thresh = score, thresh

    log.info(f"\n  → Best Recommended Meta-Threshold: {best_thresh:.2f}")

    meta_pipeline = {
        "meta_ensemble":              calibrated_meta,
        "meta_features":              meta_features,
        "recommended_meta_threshold": best_thresh,
        "base_rate":                  float(n_correct / n_dir) if n_dir > 0 else 0.0,
        "test_accuracy":              float(acc),
        "trained_at":                 datetime.now(timezone.utc).isoformat(),
        "primary_model_trained_at":   primary_pipeline.get("trained_at", "unknown"),
    }
    joblib.dump(meta_pipeline, META_MODEL_FILE)
    log.info(f"\n✅ Saved: {META_MODEL_FILE}")

    with open("meta_model_performance.json", "w") as f:
        json.dump({
            "test_accuracy":              round(acc * 100, 1),
            "base_rate":                  round(base_rate, 1),
            "n_directional_calls":        int(n_dir),
            "recommended_meta_threshold": best_thresh,
            "meta_features":              meta_features,
        }, f, indent=2)


if __name__ == "__main__":
    t0 = time.time()
    train_meta_model()
    log.info(f"\nDone in {(time.time()-t0)/60:.1f} min")
