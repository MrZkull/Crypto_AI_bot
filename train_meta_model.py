#!/usr/bin/env python3
# train_meta_model.py — Research Pipeline: Unwrapped OOF Meta-Training & Canonical Candidate Appending

import json
import logging
import time
import hashlib
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import clone
from sklearn.frozen import FrozenEstimator
from xgboost import XGBClassifier

# ── Scikit-Learn 1.6+ Compatibility Patch for XGBoost in VotingClassifier ──
XGBClassifier._estimator_type = "classifier"

try:
    from sklearn.utils._tags import ClassifierTags
    def _xgb_sklearn_tags(self):
        try:
            tags = super(XGBClassifier, self).__sklearn_tags__()
        except Exception:
            from sklearn.utils._tags import Tags
            tags = Tags()
        tags.estimator_type = "classifier"
        tags.classifier_tags = ClassifierTags()
        return tags
    XGBClassifier.__sklearn_tags__ = _xgb_sklearn_tags
except (ImportError, AttributeError):
    pass


from train_model import (
    build_dataset_from_local_parquet, FULL_FEATURES, EMBARGO_BARS,
    TEST_SPLIT, CALIB_SPLIT, N_FEATURES, temporal_symbol_split
)
from gate10 import evaluate_gate_10
from execution_policy import get_file_hash

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

CANDIDATE_MODEL_FILE = "candidate_model.pkl"
META_MODEL_FILE = "candidate_meta_model.pkl"
N_META_FEATURES = 25
META_SYSTEM_FEATURES = [
    "meta_primary_conf",
    "meta_primary_side_code",
    "meta_rsi_directional",
    "meta_macd_directional",
    "meta_trend_aligned",
]


def augment_meta_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    side_code = df["primary_side"].map({"BUY": 1.0, "SELL": -1.0}).fillna(0.0)
    df["meta_primary_side_code"] = side_code
    df["meta_primary_conf"] = pd.to_numeric(df["primary_conf"], errors="coerce").fillna(0.5)
    rsi = pd.to_numeric(df["rsi"], errors="coerce").fillna(50.0) if "rsi" in df.columns else pd.Series(50.0, index=df.index)
    macd = pd.to_numeric(df["macd_hist"], errors="coerce").fillna(0.0) if "macd_hist" in df.columns else pd.Series(0.0, index=df.index)
    trend = pd.to_numeric(df["trend"], errors="coerce").fillna(0.0) if "trend" in df.columns else pd.Series(0.0, index=df.index)
    df["meta_rsi_directional"] = (rsi - 50.0) * side_code
    df["meta_macd_directional"] = macd * side_code
    df["meta_trend_aligned"] = trend * side_code
    return df


def _fold_selected_features(tr: pd.DataFrame, af: list, n_features: int, target_to_int: dict, no_trade_idx: int) -> list:
    x = tr[af].replace([np.inf, -np.inf], np.nan).fillna(0)
    y = tr["target"].map(target_to_int).fillna(no_trade_idx).astype(int).values
    scanner = XGBClassifier(n_estimators=100, random_state=42, n_jobs=-1, eval_metric="mlogloss")
    scanner.fit(x, y)
    ranked = [af[i] for i in np.argsort(scanner.feature_importances_)[::-1]]
    essential = [f for f in ["volume_ratio", "volume_spike", "obv_slope", "bb_width", "atr_pct", "volatility", "vwap_dev"] if f in af]
    selected = essential[:]
    for f in ranked:
        if f not in selected:
            selected.append(f)
        if len(selected) >= n_features:
            break
    return selected


def get_primary_predictions(ds: pd.DataFrame, primary_pipeline: dict) -> pd.DataFrame:
    af = primary_pipeline["all_features"]
    for f in af:
        if f not in ds.columns:
            ds[f] = 0.0

    X  = ds[af].replace([np.inf, -np.inf], np.nan).fillna(0)
    Xs = primary_pipeline["selector"].transform(X)

    preds     = primary_pipeline["ensemble"].predict(Xs)
    probas    = primary_pipeline["ensemble"].predict_proba(Xs)
    label_map = primary_pipeline["label_map"]

    ds = ds.copy()
    ds["primary_side"] = [label_map[int(p)] for p in preds]
    ds["primary_conf"] = probas.max(axis=1)
    if "pred_id" not in ds.columns:
        ds["pred_id"] = [f"{row['symbol']}_{int(row['open_time'])}" for _, row in ds.iterrows()]
    return ds


def build_meta_labels(ds: pd.DataFrame) -> pd.DataFrame:
    directional = ds[ds["primary_side"] != "NO_TRADE"].copy()
    directional["meta_label"] = (directional["primary_side"] == directional["target"]).astype(int)
    return directional


def build_oof_meta_training(primary_train: pd.DataFrame, primary_pipeline: dict) -> pd.DataFrame:
    if primary_train.empty:
        return primary_train.copy()

    af = primary_pipeline["all_features"]
    label_map = {int(k): v for k, v in primary_pipeline["label_map"].items()}
    target_to_int = {v: k for k, v in label_map.items()}
    no_trade_idx = target_to_int.get("NO_TRADE")
    if no_trade_idx is None:
        raise ValueError("Primary pipeline label map is missing NO_TRADE")

    production_ensemble = primary_pipeline["ensemble"]
    base_estimator = getattr(production_ensemble, "estimator", None)
    if hasattr(base_estimator, "estimator"):
        base_estimator = base_estimator.estimator
    if base_estimator is None:
        base_estimator = getattr(production_ensemble, "base_estimator", None)
    if base_estimator is None:
        raise ValueError("Primary ensemble does not expose an unfrozen cloneable estimator for OOF")

    parts = []
    for sym, grp in primary_train.groupby("symbol", sort=False):
        grp = grp.sort_values("open_time").reset_index(drop=True)
        n = len(grp)
        fold_edges = np.linspace(0, n, 4, dtype=int)

        for j in range(1, 4):
            val_start = fold_edges[j - 1]
            val_end = fold_edges[j]
            train_end = val_start - EMBARGO_BARS
            if train_end < 50 or val_end <= val_start:
                continue

            tr = grp.iloc[:train_end].copy()
            va = grp.iloc[val_start:val_end].copy()
            if len(tr) < 50:
                continue

            for f in af:
                if f not in tr.columns: tr[f] = 0.0
                if f not in va.columns: va[f] = 0.0

            fold_features = _fold_selected_features(tr, af, min(N_FEATURES, len(af)), target_to_int, no_trade_idx)
            Xtr = tr[fold_features].replace([np.inf, -np.inf], np.nan).fillna(0).values
            Xva = va[fold_features].replace([np.inf, -np.inf], np.nan).fillna(0).values
            ytr = tr["target"].map(target_to_int).fillna(no_trade_idx).astype(int).values

            keep_signal = np.where(ytr != no_trade_idx)[0]
            keep_nt = np.where(ytr == no_trade_idx)[0]
            target_nt = min(len(keep_nt), len(keep_signal))
            if target_nt == 0 or len(keep_signal) == 0:
                continue

            rng = np.random.default_rng(42 + j)
            keep_nt = rng.choice(keep_nt, size=target_nt, replace=False)
            keep = np.sort(np.concatenate([keep_signal, keep_nt]))

            est = clone(base_estimator)
            est.fit(Xtr[keep], ytr[keep])
            pred = est.predict(Xva)
            proba = est.predict_proba(Xva)

            va["primary_side"] = [label_map[int(x)] for x in pred]
            va["primary_conf"] = proba.max(axis=1)
            va["pred_id"] = [f"{row['symbol']}_{int(row['open_time'])}" for _, row in va.iterrows()]
            parts.append(va)

    if not parts:
        return primary_train.iloc[:0].copy()

    out = pd.concat(parts, ignore_index=True)
    out = out[out["primary_side"] != "NO_TRADE"].copy()
    out["meta_label"] = (out["primary_side"] == out["target"]).astype(int)
    return augment_meta_features(out)


def train_meta_model():
    log.info("Loading primary candidate artifact...")
    primary_pipeline = joblib.load(CANDIDATE_MODEL_FILE)
    ds = build_dataset_from_local_parquet()

    primary_train, primary_calib, primary_test = temporal_symbol_split(
        ds, TEST_SPLIT, CALIB_SPLIT, EMBARGO_BARS
    )
    if primary_train.empty or primary_calib.empty or primary_test.empty:
        raise ValueError("Primary eras are incomplete; refusing meta-model training")

    log.info(f"PRIMARY ERAS: train={len(primary_train):,} calib={len(primary_calib):,} locked_test={len(primary_test):,}")

    meta_train = build_oof_meta_training(primary_train, primary_pipeline)
    if len(meta_train) < 50 or meta_train["meta_label"].nunique() < 2:
        raise ValueError("Insufficient two-class OOF meta-training data")

    primary_calib_pred = augment_meta_features(build_meta_labels(get_primary_predictions(primary_calib.copy(), primary_pipeline)))
    if len(primary_calib_pred) < 20 or primary_calib_pred["meta_label"].nunique() < 2:
        raise ValueError("Insufficient two-class primary calibration data")

    primary_test_pred = augment_meta_features(build_meta_labels(get_primary_predictions(primary_test.copy(), primary_pipeline)))
    if len(primary_test_pred) < 20 or primary_test_pred["meta_label"].nunique() < 2:
        raise ValueError("Insufficient two-class locked-test data")

    meta_feature_universe = list(dict.fromkeys(FULL_FEATURES + META_SYSTEM_FEATURES))
    for f in meta_feature_universe:
        for part in (meta_train, primary_calib_pred, primary_test_pred):
            if f not in part.columns:
                part[f] = 0.0

    X_train_full = meta_train[meta_feature_universe].replace([np.inf, -np.inf], np.nan).fillna(0).values
    y_train = meta_train["meta_label"].values

    scanner = XGBClassifier(n_estimators=100, random_state=42, n_jobs=-1, eval_metric="logloss")
    scanner.fit(X_train_full, y_train)
    top_idx = np.argsort(scanner.feature_importances_)[::-1][:N_META_FEATURES]
    meta_features = [meta_feature_universe[i] for i in top_idx]
    for f in META_SYSTEM_FEATURES:
        if f not in meta_features:
            meta_features.append(f)
    meta_features = meta_features[:min(N_META_FEATURES + len(META_SYSTEM_FEATURES), len(meta_feature_universe))]

    X_train = meta_train[meta_features].replace([np.inf, -np.inf], np.nan).fillna(0).values
    X_calib = primary_calib_pred[meta_features].replace([np.inf, -np.inf], np.nan).fillna(0).values
    y_calib = primary_calib_pred["meta_label"].values

    X_test = primary_test_pred[meta_features].replace([np.inf, -np.inf], np.nan).fillna(0).values
    y_test = primary_test_pred["meta_label"].values

    meta_xgb = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.03, subsample=0.85, colsample_bytree=0.85, min_child_weight=3, eval_metric="logloss", random_state=42, n_jobs=-1)
    meta_rf = RandomForestClassifier(n_estimators=300, max_depth=10, min_samples_leaf=5, random_state=42, n_jobs=-1)
    meta_ensemble = VotingClassifier(estimators=[("xgb", meta_xgb), ("rf", meta_rf)], voting="soft", weights=[2, 1])
    meta_ensemble.fit(X_train, y_train)

    calibrated_meta = CalibratedClassifierCV(estimator=FrozenEstimator(meta_ensemble), method="isotonic")
    calibrated_meta.fit(X_calib, y_calib)

    classes = list(calibrated_meta.classes_)
    if 1 not in classes:
        raise ValueError(f"Positive class '1' missing from meta classes: {classes}")
    pos_idx = classes.index(1)

    calib_proba = calibrated_meta.predict_proba(X_calib)[:, pos_idx]
    test_proba  = calibrated_meta.predict_proba(X_test)[:, pos_idx]

    best_thresh, best_score = 0.50, 0.0
    for thresh in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
        mask = calib_proba >= thresh
        if mask.sum() < 10:
            continue
        score = float(y_calib[mask].mean()) * np.sqrt(mask.sum())
        if score > best_score:
            best_score, best_thresh = score, thresh

    primary_baseline_threshold = float(primary_pipeline.get("recommended_threshold", 0.45))
    friction_r = 0.12

    # ──────────────────────────────────────────────────────────────
    # 1. Evaluate strictly in Synthetic Barrier Mode using gate10.py
    # ──────────────────────────────────────────────────────────────
    synth_df = pd.DataFrame({
        "pred_id": primary_test_pred["pred_id"],
        "evidence_type": "SYNTHETIC_BARRIER",
        "primary_selected": (primary_test_pred["primary_conf"] >= primary_baseline_threshold),
        "meta_selected": (primary_test_pred["primary_conf"] >= primary_baseline_threshold) & (test_proba >= best_thresh),
        "thesis_label": primary_test_pred["meta_label"],
        "net_r": np.where(primary_test_pred["meta_label"] == 1, 3.5 - friction_r, -2.5 - friction_r)
    })

    gate10_result = evaluate_gate_10(synth_df)
    log.info(f"Synthetic Research Gate 10 Result: {json.dumps(gate10_result, indent=2)}")

    meta_pipeline = {
        "meta_ensemble": calibrated_meta,
        "meta_features": meta_features,
        "meta_positive_class": 1,
        "meta_positive_idx": pos_idx,
        "recommended_meta_threshold": best_thresh,
        "base_rate": float(y_test.mean() * 100),
        "gate10_summary": gate10_result,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    
    # 2. Dump exclusively to candidate artifact
    joblib.dump(meta_pipeline, META_MODEL_FILE)
    log.info(f"✅ Saved meta-model pipeline: {META_MODEL_FILE}")

    with open("meta_model_performance.json", "w") as f:
        json.dump(gate10_result, f, indent=2)

    # 3. Append cryptographic identity into Candidate Manifest
    meta_sha256 = get_file_hash(META_MODEL_FILE)
    try:
        with open("candidate_manifest.json", "r") as f:
            manifest = json.load(f)
            
        manifest["decision_policy_hash"] = meta_sha256
        
        with open("candidate_manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
        log.info(f"✅ Appended Meta-Model SHA256 to Candidate Manifest.")
    except Exception as e:
        log.error(f"Failed to append hash to candidate manifest: {e}")


if __name__ == "__main__":
    t0 = time.time()
    train_meta_model()
    log.info(f"Meta-model training complete in {(time.time()-t0)/60:.1f} min")
