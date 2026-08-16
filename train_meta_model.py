# train_meta_model.py — P1: Meta-labeling (López de Prado)

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
N_META_FEATURES = 25

def get_primary_predictions(ds: pd.DataFrame, primary_pipeline: dict) -> pd.DataFrame:
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
    directional = ds[ds["primary_side"] != "NO_TRADE"].copy()
    directional["meta_label"] = (directional["primary_side"] == directional["target"]).astype(int)
    return directional

def per_symbol_regime_split(ds: pd.DataFrame, test_split: float, calib_split: float, embargo: int):
    train_parts, calib_parts, test_parts = [], [], []
    if "regime" not in ds.columns: ds["regime"] = "unknown"
    if "symbol" not in ds.columns: ds["symbol"] = "unknown"

    for (sym, regime), grp in ds.groupby(["symbol", "regime"], sort=False):
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
    primary_pipeline = joblib.load(MODEL_FILE)

    log.info("Building dataset (same data & regimes as primary)...")
    ds = build_dataset()

    log.info("Isolating primary model's held-out test split via per-(symbol, regime) embargo...")
    _, _, primary_test = per_symbol_regime_split(ds, TEST_SPLIT, CALIB_SPLIT, EMBARGO_BARS)

    log.info("Generating primary model calls on held-out test rows...")
    primary_test = get_primary_predictions(primary_test, primary_pipeline)

    directional = build_meta_labels(primary_test)
    n_dir = len(directional)
    n_correct = directional["meta_label"].sum()
    base_rate = (n_correct / n_dir * 100) if n_dir > 0 else 0.0

    log.info(f"Primary called a direction on {n_dir:,} held-out rows ({n_dir/len(primary_test)*100:.1f}% frequency)")
    log.info(f"True out-of-sample base hit rate: {base_rate:.1f}% ({n_correct:,}/{n_dir:,} correct)")

    train_df, calib_df, test_df = per_symbol_regime_split(directional, TEST_SPLIT, CALIB_SPLIT, EMBARGO_BARS)

    for f in FULL_FEATURES:
        for part in (train_df, calib_df, test_df):
            if f not in part.columns:
                part[f] = 0.0

    X_train_full = train_df[FULL_FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0)
    y_train = train_df["meta_label"].values

    scanner = XGBClassifier(n_estimators=100, random_state=42, n_jobs=-1, eval_metric="logloss")
    scanner.fit(X_train_full, y_train)
    top_idx = np.argsort(scanner.feature_importances_)[::-1][:N_META_FEATURES]
    meta_features = [FULL_FEATURES[i] for i in top_idx]

    X_train = train_df[meta_features].replace([np.inf, -np.inf], np.nan).fillna(0).values
    X_calib = calib_df[meta_features].replace([np.inf, -np.inf], np.nan).fillna(0).values
    X_test  = test_df[meta_features].replace([np.inf, -np.inf], np.nan).fillna(0).values
    y_calib = calib_df["meta_label"].values
    y_test  = test_df["meta_label"].values

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

    calibrated_meta = CalibratedClassifierCV(estimator=FrozenEstimator(meta_ensemble), method="isotonic")
    calibrated_meta.fit(X_calib, y_calib)

    calib_proba = calibrated_meta.predict_proba(X_calib)[:, 1]
    best_thresh, best_score = 0.50, 0.0

    for thresh in [0.50, 0.55, 0.60, 0.65, 0.70]:
        mask = calib_proba >= thresh
        if mask.sum() < 10: continue
        hit_rate = y_calib[mask].mean()
        score = hit_rate * np.sqrt(mask.sum())
        if score > best_score:
            best_score, best_thresh = score, thresh

    y_proba = calibrated_meta.predict_proba(X_test)[:, 1]
    test_mask = y_proba >= best_thresh
    meta_prec = y_test[test_mask].mean() if test_mask.sum() > 0 else 0.0

    log.info(f"\n{'='*60}")
    log.info(f"META-MODEL HELD-OUT EVALUATION (Threshold = {best_thresh:.2f}):")
    log.info(f"  Trades Taken: {test_mask.sum()} | Out-of-Sample Precision: {meta_prec*100:.1f}% (Base: {base_rate:.1f}%)")
    log.info(f"{'='*60}")

    meta_pipeline = {
        "meta_ensemble":              calibrated_meta,
        "meta_features":              meta_features,
        "recommended_meta_threshold": best_thresh,
        "base_rate":                  float(base_rate),
        "test_accuracy":              float(meta_prec * 100),
        "trained_at":                 datetime.now(timezone.utc).isoformat(),
    }
    joblib.dump(meta_pipeline, META_MODEL_FILE)
    log.info(f"\n✅ Saved: {META_MODEL_FILE}")

    # ── Write meta performance JSON for GitHub Action step ──
    with open("meta_model_performance.json", "w") as f:
        json.dump({
            "test_accuracy":              round(float(meta_prec) * 100, 1),
            "base_rate":                  round(float(base_rate), 1),
            "n_directional_calls":        int(n_dir),
            "recommended_meta_threshold": float(best_thresh),
            "meta_features":              meta_features,
        }, f, indent=2)

    log.info("✅ Saved: meta_model_performance.json")

if __name__ == "__main__":
    t0 = time.time()
    train_meta_model()
    log.info(f"\nDone in {(time.time()-t0)/60:.1f} min")
