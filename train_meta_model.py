#!/usr/bin/env python3
# train_meta_model.py — Research Pipeline: Unwrapped OOF Meta-Training, Anti-Leakage Audits & Candidate Appending

import os
import sys
import json
import time
import logging
import hashlib
from pathlib import Path
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

try:
    from config import ATR_STOP_MULT, ATR_TARGET1_MULT
except ImportError:
    ATR_STOP_MULT = 2.5
    ATR_TARGET1_MULT = 3.5

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

CANDIDATE_MODEL_FILE = "candidate_model.pkl"
META_MODEL_FILE = "candidate_meta_model.pkl"
CANDIDATE_MANIFEST = Path("candidate_manifest.json")
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
    tr = tr.loc[:, ~tr.columns.duplicated()].copy()
    af_clean = list(dict.fromkeys([f for f in af if f in tr.columns]))

    x = tr[af_clean].replace([np.inf, -np.inf], np.nan).fillna(0)
    y = tr["target"].map(target_to_int).fillna(no_trade_idx).astype(int).values

    scanner = XGBClassifier(n_estimators=100, max_depth=4, random_state=42, n_jobs=-1, eval_metric="mlogloss")
    scanner.fit(x, y)

    ranked = [af_clean[i] for i in np.argsort(scanner.feature_importances_)[::-1]]
    essential = [f for f in ["volume_ratio", "volume_spike", "obv_slope", "bb_width", "atr_pct", "volatility", "vwap_dev"] if f in af_clean]
    selected = essential[:]
    for f in ranked:
        if f not in selected:
            selected.append(f)
        if len(selected) >= n_features:
            break
    return selected


def get_primary_predictions(ds: pd.DataFrame, primary_pipeline: dict) -> pd.DataFrame:
    ds = ds.loc[:, ~ds.columns.duplicated()].copy()
    af = list(dict.fromkeys(primary_pipeline["all_features"]))

    for f in af:
        if f not in ds.columns:
            ds[f] = 0.0

    X = ds[af].replace([np.inf, -np.inf], np.nan).fillna(0)
    Xs = primary_pipeline["selector"].transform(X)

    probas = primary_pipeline["ensemble"].predict_proba(Xs)
    
    # Manual pairwise logic bypassing default argmax
    inv_map = {v: k for k, v in primary_pipeline["label_map"].items()}
    buy_idx = inv_map.get("BUY", 0)
    sell_idx = inv_map.get("SELL", 2)
    nt_idx = inv_map.get("NO_TRADE", 1)
    
    thresh_buy = primary_pipeline.get("recommended_threshold_buy", 0.36)
    thresh_sell = primary_pipeline.get("recommended_threshold_sell", 0.36)

    primary_sides = []
    primary_confs = []
    
    for p in probas:
        pb, ps = p[buy_idx], p[sell_idx]
        if pb >= thresh_buy and pb > ps:
            primary_sides.append("BUY")
            primary_confs.append(pb)
        elif ps >= thresh_sell and ps > pb:
            primary_sides.append("SELL")
            primary_confs.append(ps)
        else:
            primary_sides.append("NO_TRADE")
            primary_confs.append(p[nt_idx])

    ds = ds.copy()
    ds["primary_side"] = primary_sides
    ds["primary_conf"] = primary_confs
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

    primary_train = primary_train.loc[:, ~primary_train.columns.duplicated()].copy()
    af = list(dict.fromkeys(primary_pipeline["all_features"]))
    
    inv_map = {v: k for k, v in primary_pipeline["label_map"].items()}
    target_to_int = inv_map
    buy_idx = inv_map.get("BUY", 0)
    sell_idx = inv_map.get("SELL", 2)
    nt_idx = inv_map.get("NO_TRADE", 1)
    
    thresh_buy = primary_pipeline.get("recommended_threshold_buy", 0.36)
    thresh_sell = primary_pipeline.get("recommended_threshold_sell", 0.36)

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

            tr = tr.loc[:, ~tr.columns.duplicated()].copy()
            va = va.loc[:, ~va.columns.duplicated()].copy()

            for f in af:
                if f not in tr.columns: tr[f] = 0.0
                if f not in va.columns: va[f] = 0.0

            fold_features = _fold_selected_features(tr, af, min(N_FEATURES, len(af)), target_to_int, nt_idx)
            Xtr = tr[fold_features].replace([np.inf, -np.inf], np.nan).fillna(0).values
            Xva = va[fold_features].replace([np.inf, -np.inf], np.nan).fillna(0).values
            ytr = tr["target"].map(target_to_int).fillna(nt_idx).astype(int).values

            keep_signal = np.where(ytr != nt_idx)[0]
            keep_nt = np.where(ytr == nt_idx)[0]
            target_nt = min(len(keep_nt), len(keep_signal))
            if target_nt == 0 or len(keep_signal) == 0:
                continue

            rng = np.random.default_rng(42 + j)
            keep_nt = rng.choice(keep_nt, size=target_nt, replace=False)
            keep = np.sort(np.concatenate([keep_signal, keep_nt]))

            est = clone(base_estimator)
            est.fit(Xtr[keep], ytr[keep])
            
            # Manual pairwise logic bypassing default argmax for OOF
            proba = est.predict_proba(Xva)
            primary_sides = []
            primary_confs = []
            for p in proba:
                pb, ps = p[buy_idx], p[sell_idx]
                if pb >= thresh_buy and pb > ps:
                    primary_sides.append("BUY")
                    primary_confs.append(pb)
                elif ps >= thresh_sell and ps > pb:
                    primary_sides.append("SELL")
                    primary_confs.append(ps)
                else:
                    primary_sides.append("NO_TRADE")
                    primary_confs.append(p[nt_idx])

            va["primary_side"] = primary_sides
            va["primary_conf"] = primary_confs
            va["pred_id"] = [f"{row['symbol']}_{int(row['open_time'])}" for _, row in va.iterrows()]
            parts.append(va)

    if not parts:
        return primary_train.iloc[:0].copy()

    out = pd.concat(parts, ignore_index=True)
    out = out[out["primary_side"] != "NO_TRADE"].copy()
    out["meta_label"] = (out["primary_side"] == out["target"]).astype(int)
    return augment_meta_features(out)


def audit_meta_anti_leakage(meta_train: pd.DataFrame, primary_calib_pred: pd.DataFrame,
                            primary_test_pred: pd.DataFrame, meta_features: list[str]) -> None:
    log.info(f"\n{'='*85}")
    log.info("META-MODEL ANTI-LEAKAGE & OOF PURITY AUDIT")
    log.info(f"{'='*85}")

    t_train_max = meta_train["open_time"].max()
    t_calib_min = primary_calib_pred["open_time"].min()
    t_calib_max = primary_calib_pred["open_time"].max()
    t_test_min  = primary_test_pred["open_time"].min()

    gap_tc = (t_calib_min - t_train_max) / (3600 * 1000)
    gap_ct = (t_test_min - t_calib_max) / (3600 * 1000)

    log.info(f"[1] Meta Partition Chronology:")
    log.info(f"    OOF Meta-Train End: {datetime.fromtimestamp(t_train_max/1000, tz=timezone.utc).isoformat()}")
    log.info(f"    Calib Pred Start:   {datetime.fromtimestamp(t_calib_min/1000, tz=timezone.utc).isoformat()} (Embargo Gap: {gap_tc:.1f}h)")
    log.info(f"    Calib Pred End:     {datetime.fromtimestamp(t_calib_max/1000, tz=timezone.utc).isoformat()}")
    log.info(f"    Test Pred Start:    {datetime.fromtimestamp(t_test_min/1000, tz=timezone.utc).isoformat()} (Embargo Gap: {gap_ct:.1f}h)")

    if t_train_max >= t_calib_min or t_calib_max >= t_test_min:
        raise ValueError("CRITICAL LEAKAGE: Meta-training chronological split inversion or zero embargo!")
    log.info("    ✓ PASS: Meta training, calibration, and test splits are strictly ordered.")

    forbidden_features = {"target", "meta_label", "future_close", "barrier_hit"}
    intersection = forbidden_features.intersection(set(meta_features))
    log.info(f"[2] Forbidden Feature Contamination Scan:")
    if intersection:
        raise ValueError(f"CRITICAL LEAKAGE: Forbidden target labels in meta-feature set: {intersection}")
    log.info("    ✓ PASS: Zero target label contamination detected in meta-feature space.")

    log.info(f"[3] Meta-Feature Correlation Scan (Leak Ceiling > 0.50):")
    y_train = meta_train["meta_label"].values
    high_corr = False
    corrs = []
    for f in meta_features:
        s = pd.to_numeric(meta_train[f], errors="coerce").fillna(0.0).values
        c = float(np.abs(np.corrcoef(s, y_train)[0, 1]))
        if np.isnan(c): c = 0.0
        corrs.append((f, c))
        if c > 0.50 and f != "meta_primary_conf":
            log.warning(f"    🚨 SUSPICIOUS META-LEAK: Feature '{f}' correlation with meta-label = {c:.4f}")
            high_corr = True

    for name, c in sorted(corrs, key=lambda x: x[1], reverse=True)[:5]:
        log.info(f"      - {name:<26}: {c:.4f}")

    if high_corr:
        raise ValueError("CRITICAL LEAKAGE: Meta-feature displays unnatural predictive correlation.")
    log.info("    ✓ PASS: All meta-feature correlations fall within expected statistical bounds.")
    log.info(f"{'='*85}\n")


def train_meta_model():
    log.info("Loading primary candidate artifact...")
    if not Path(CANDIDATE_MODEL_FILE).exists():
        raise FileNotFoundError(f"Primary artifact {CANDIDATE_MODEL_FILE} not found. Run train_model.py first.")

    primary_pipeline = joblib.load(CANDIDATE_MODEL_FILE)
    ds = build_dataset_from_local_parquet()
    ds = ds.loc[:, ~ds.columns.duplicated()].copy()

    primary_train, primary_calib, primary_test = temporal_symbol_split(
        ds, TEST_SPLIT, CALIB_SPLIT, EMBARGO_BARS
    )
    if primary_train.empty or primary_calib.empty or primary_test.empty:
        raise ValueError("Primary eras are incomplete; refusing meta-model training")

    log.info(f"PRIMARY ERAS: train={len(primary_train):,} calib={len(primary_calib):,} locked_test={len(primary_test):,}")

    log.info("Generating Out-of-Fold (OOF) meta-training labels across primary train...")
    meta_train = build_oof_meta_training(primary_train, primary_pipeline)
    if len(meta_train) < 50 or meta_train["meta_label"].nunique() < 2:
        raise ValueError("Insufficient two-class OOF meta-training data")

    log.info("Generating primary predictions for calibration and locked test sets...")
    primary_calib_pred = augment_meta_features(build_meta_labels(get_primary_predictions(primary_calib.copy(), primary_pipeline)))
    primary_test_pred = augment_meta_features(build_meta_labels(get_primary_predictions(primary_test.copy(), primary_pipeline)))

    meta_feature_universe = list(dict.fromkeys(FULL_FEATURES + META_SYSTEM_FEATURES))
    for f in meta_feature_universe:
        for part in (meta_train, primary_calib_pred, primary_test_pred):
            if f not in part.columns:
                part[f] = 0.0

    meta_train = meta_train.loc[:, ~meta_train.columns.duplicated()].copy()
    primary_calib_pred = primary_calib_pred.loc[:, ~primary_calib_pred.columns.duplicated()].copy()
    primary_test_pred = primary_test_pred.loc[:, ~primary_test_pred.columns.duplicated()].copy()

    X_train_full = meta_train[meta_feature_universe].replace([np.inf, -np.inf], np.nan).fillna(0).values
    y_train = meta_train["meta_label"].values

    log.info("Selecting optimal meta-features...")
    scanner = XGBClassifier(n_estimators=100, max_depth=4, random_state=42, n_jobs=-1, eval_metric="logloss")
    scanner.fit(X_train_full, y_train)
    top_idx = np.argsort(scanner.feature_importances_)[::-1][:N_META_FEATURES]
    meta_features = [meta_feature_universe[i] for i in top_idx]
    for f in META_SYSTEM_FEATURES:
        if f not in meta_features:
            meta_features.append(f)
    meta_features = meta_features[:min(N_META_FEATURES + len(META_SYSTEM_FEATURES), len(meta_feature_universe))]

    audit_meta_anti_leakage(meta_train, primary_calib_pred, primary_test_pred, meta_features)

    X_train = meta_train[meta_features].replace([np.inf, -np.inf], np.nan).fillna(0).values
    X_calib = primary_calib_pred[meta_features].replace([np.inf, -np.inf], np.nan).fillna(0).values
    y_calib = primary_calib_pred["meta_label"].values

    X_test = primary_test_pred[meta_features].replace([np.inf, -np.inf], np.nan).fillna(0).values
    y_test = primary_test_pred["meta_label"].values

    log.info("Training Meta-Model Ensemble (XGBoost + Random Forest)...")
    
    # Increased regularization to collapse the Meta-Model overfitting (-0.51R out-of-sample)
    meta_xgb = XGBClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.02, subsample=0.70,
        colsample_bytree=0.70, min_child_weight=8, reg_alpha=2.0, reg_lambda=5.0,
        random_state=42, n_jobs=-1
    )
    meta_rf = RandomForestClassifier(
        n_estimators=200, max_depth=5, min_samples_leaf=15, max_features="sqrt", random_state=42, n_jobs=-1
    )
    meta_ensemble = VotingClassifier(
        estimators=[("xgb", meta_xgb), ("rf", meta_rf)], voting="soft", weights=[2, 1]
    )
    meta_ensemble.fit(X_train, y_train)

    calibrated_meta = CalibratedClassifierCV(estimator=FrozenEstimator(meta_ensemble), method="isotonic")
    calibrated_meta.fit(X_calib, y_calib)

    classes = list(calibrated_meta.classes_)
    if 1 not in classes:
        raise ValueError(f"Positive class '1' missing from meta classes: {classes}")
    pos_idx = classes.index(1)

    calib_proba = calibrated_meta.predict_proba(X_calib)[:, pos_idx]
    test_proba  = calibrated_meta.predict_proba(X_test)[:, pos_idx]

    thresh_buy = float(primary_pipeline.get("recommended_threshold_buy", primary_pipeline.get("recommended_threshold", 0.36)))
    thresh_sell = float(primary_pipeline.get("recommended_threshold_sell", primary_pipeline.get("recommended_threshold", 0.36)))
    friction_r = 0.12

    calib_primary_selected = (
        ((primary_calib_pred["primary_side"] == "BUY") & (primary_calib_pred["primary_conf"] >= thresh_buy)) |
        ((primary_calib_pred["primary_side"] == "SELL") & (primary_calib_pred["primary_conf"] >= thresh_sell))
    )
    test_primary_selected = (
        ((primary_test_pred["primary_side"] == "BUY") & (primary_test_pred["primary_conf"] >= thresh_buy)) |
        ((primary_test_pred["primary_side"] == "SELL") & (primary_test_pred["primary_conf"] >= thresh_sell))
    )

    base_prec = float(y_calib[calib_primary_selected].mean() * 100) if calib_primary_selected.sum() > 0 else 0.0

    best_thresh, best_score = 0.50, 0.0

    log.info(f"\n{'='*95}")
    log.info(f"META-MODEL THRESHOLD CALIBRATION SWEEP (Primary Baseline Precision: {base_prec:.1f}%)")
    log.info(f"{'='*95}")
    log.info(
        f"{'Thresh':<8} | {'Trades':<8} | {'Precision':<10} | {'Prec Lift':<10} | "
        f"{'Retention':<10} | {'Theo EV_R':<10} | {'Score':<8}"
    )
    log.info(f"{'-'*95}")

    for thresh in [0.45, 0.48, 0.50, 0.52, 0.55, 0.58, 0.60, 0.65]:
        mask = (calib_proba >= thresh) & calib_primary_selected
        n_trades = int(mask.sum())
        if n_trades < 10:
            log.info(f"{thresh:<8.2f} | {n_trades:<8} | {'SKIPPED (<10 qualified trades)':<58}")
            continue

        prec = float(y_calib[mask].mean())
        prec_lift = (prec * 100.0) - base_prec
        retention = (n_trades / max(calib_primary_selected.sum(), 1)) * 100.0
        ev_r = prec * ATR_TARGET1_MULT - (1.0 - prec) * ATR_STOP_MULT - friction_r
        score = prec * np.sqrt(n_trades)

        log.info(
            f"{thresh:<8.2f} | {n_trades:<8} | {prec*100:>8.1f}%  | {prec_lift:>+8.1f}%  | "
            f"{retention:>8.1f}%  | {ev_r:>+8.2f}R | {score:>8.2f}"
        )

        if score > best_score:
            best_score, best_thresh = score, thresh

    log.info(f"{'-'*95}")
    log.info(f"✅ Selected Optimal Meta Threshold: {best_thresh:.2f} (Score: {best_score:.2f})")
    log.info(f"{'='*95}\n")

    synth_df = pd.DataFrame({
        "pred_id": primary_test_pred["pred_id"],
        "open_time": primary_test_pred["open_time"],
        "evidence_type": "SYNTHETIC_BARRIER",
        "primary_selected": test_primary_selected,
        "meta_selected": test_primary_selected & (test_proba >= best_thresh),
        "thesis_label": primary_test_pred["meta_label"],
        "net_r": np.where(primary_test_pred["meta_label"] == 1, ATR_TARGET1_MULT - friction_r, -ATR_STOP_MULT - friction_r)
    })

    gate10_result = evaluate_gate_10(synth_df)
    log.info(f"Synthetic Research Gate 10 Result:\n{json.dumps(gate10_result, indent=2)}")

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

    joblib.dump(meta_pipeline, META_MODEL_FILE, compress=3)
    log.info(f"✅ Saved compressed meta-model pipeline: {META_MODEL_FILE}")

    with open("meta_model_performance.json", "w") as f:
        json.dump(gate10_result, f, indent=2)

    meta_sha256 = get_file_hash(META_MODEL_FILE)
    try:
        if CANDIDATE_MANIFEST.exists():
            with open(CANDIDATE_MANIFEST, "r") as f:
                manifest = json.load(f)

            manifest["decision_policy_hash"] = meta_sha256

            with open(CANDIDATE_MANIFEST, "w") as f:
                json.dump(manifest, f, indent=2)
            log.info(f"✅ Appended Meta-Model SHA256 ({meta_sha256[:8]}...) to Candidate Manifest.")
    except Exception as e:
        log.error(f"Failed to append hash to candidate manifest: {e}")


if __name__ == "__main__":
    t0 = time.time()
    train_meta_model()
    log.info(f"Meta-model research training complete in {(time.time()-t0)/60:.1f} min")
