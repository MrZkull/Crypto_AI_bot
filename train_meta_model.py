# train_meta_model.py — P1: Meta-labeling (López de Prado, "Advances in Financial
# Machine Learning"). Separates the PRIMARY decision (which direction — handled by
# your existing pro_crypto_ai_model.pkl ensemble) from the META decision (given the
# primary called a direction, should you actually act on it, and how confident
# should you be). This is a different architecture from "retrain the same 3-class
# model with more data" — it's specifically aimed at the walk-forward-accuracy-near-
# baseline problem, since "was this specific directional call right, yes/no" is a
# much easier binary problem than "BUY vs SELL vs NO_TRADE" from scratch.
#
# WHAT THIS DOES NOT DO: replace pro_crypto_ai_model.pkl. It trains a SECOND,
# separate model (meta_pipeline.pkl) that sits on top of it. generate_signal() in
# trade_executor.py would need a small change to use both — see the worked example
# at the bottom of this file.
#
# Usage: python train_meta_model.py
# (Run AFTER train_model.py — this needs pro_crypto_ai_model.pkl to already exist.)

import json, logging, time
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, accuracy_score
from sklearn.frozen import FrozenEstimator
from xgboost import XGBClassifier

from train_model import build_dataset, FULL_FEATURES, MODEL_FILE, EMBARGO_BARS, TEST_SPLIT, CALIB_SPLIT

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

META_MODEL_FILE = "meta_pipeline.pkl"
N_META_FEATURES = 25  # meta-model can use fewer features than the primary — it's
                       # answering a narrower question ("was THIS call right")


def get_primary_predictions(ds: pd.DataFrame, primary_pipeline: dict) -> pd.DataFrame:
    """Run the EXISTING trained model over the full dataset to get its primary
    side call for every row. This is the 'primary model' in meta-labeling terms —
    we're not retraining it, just using its existing calls as the base signal."""
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
    """Keep only rows where the primary model called a direction (BUY/SELL) — a
    NO_TRADE primary call has nothing for the meta-model to evaluate. Meta-label
    is binary: did the primary's called direction match the true triple-barrier
    outcome (from make_targets(), already in ds['target'])."""
    directional = ds[ds["primary_side"] != "NO_TRADE"].copy()
    directional["meta_label"] = (directional["primary_side"] == directional["target"]).astype(int)
    return directional


def per_regime_split(ds: pd.DataFrame, test_split: float, calib_split: float, embargo: int):
    """Same embargoed per-regime split pattern as train_model.py's train() — kept
    as a local copy rather than importing, since train_model.py doesn't currently
    expose this as a standalone reusable function. If you refactor train_model.py
    to extract this into a shared helper, this should call that instead."""
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

    return (pd.concat(train_parts, ignore_index=True),
            pd.concat(calib_parts, ignore_index=True) if calib_parts else train_parts[0].iloc[:0],
            pd.concat(test_parts, ignore_index=True) if test_parts else train_parts[0].iloc[:0])


def train_meta_model():
    log.info("Loading primary model (pro_crypto_ai_model.pkl)...")
    primary_pipeline = joblib.load(MODEL_FILE)

    log.info("Building dataset (reusing train_model.py's build_dataset — same data, same regimes)...")
    ds = build_dataset()
    ds = ds.sort_values("open_time").reset_index(drop=True)

    log.info("Getting primary model's directional calls across the full dataset...")
    ds = get_primary_predictions(ds, primary_pipeline)

    directional = build_meta_labels(ds)
    n_dir = len(directional)
    n_correct = directional["meta_label"].sum()
    log.info(f"Primary model called a direction on {n_dir:,} rows "
             f"({n_dir/len(ds)*100:.1f}% of dataset) — {n_correct:,} were correct "
             f"({n_correct/n_dir*100:.1f}% base rate)")

    if n_dir < 5000:
        log.error("Too few directional calls to train a meta-model reliably — "
                   "check that primary_pipeline is actually calling BUY/SELL, not "
                   "always NO_TRADE (if so, fix the primary model first).")
        return

    train_df, calib_df, test_df = per_regime_split(directional, TEST_SPLIT, CALIB_SPLIT, EMBARGO_BARS)
    log.info(f"Meta split: train={len(train_df):,}  calib={len(calib_df):,}  test={len(test_df):,}")

    for f in FULL_FEATURES:
        for part in (train_df, calib_df, test_df):
            if f not in part.columns:
                part[f] = 0.0

    log.info("Importance scan for meta-model features...")
    X_train_full = train_df[FULL_FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0)
    y_train = train_df["meta_label"].values

    scanner = XGBClassifier(n_estimators=100, random_state=42, n_jobs=-1, eval_metric="logloss")
    scanner.fit(X_train_full, y_train)
    top_idx = np.argsort(scanner.feature_importances_)[::-1][:N_META_FEATURES]
    meta_features = [FULL_FEATURES[i] for i in top_idx]
    log.info(f"Top {len(meta_features)} meta-features selected: {meta_features[:10]}...")

    X_train = train_df[meta_features].replace([np.inf, -np.inf], np.nan).fillna(0).values
    X_calib = calib_df[meta_features].replace([np.inf, -np.inf], np.nan).fillna(0).values
    X_test  = test_df[meta_features].replace([np.inf, -np.inf], np.nan).fillna(0).values
    y_calib = calib_df["meta_label"].values
    y_test  = test_df["meta_label"].values

    log.info("Training meta-model (binary: was the primary call correct?)...")
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

    from sklearn.ensemble import VotingClassifier
    meta_ensemble = VotingClassifier(
        estimators=[("xgb", meta_xgb), ("rf", meta_rf)], voting="soft", weights=[2, 1],
    )
    meta_ensemble.fit(X_train, y_train)

    log.info("Calibrating meta-model on held-out calibration split...")
    calibrated_meta = CalibratedClassifierCV(estimator=FrozenEstimator(meta_ensemble), method="isotonic")
    calibrated_meta.fit(X_calib, y_calib)

    y_pred = calibrated_meta.predict(X_test)
    y_proba = calibrated_meta.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    log.info(f"\n{'='*60}")
    log.info(f"META-MODEL TEST ACCURACY: {acc*100:.1f}%  (base rate was {n_correct/n_dir*100:.1f}%)")
    log.info(f"  If this ISN'T meaningfully above the base rate, the meta-model")
    log.info(f"  isn't adding real signal — the primary's own confidence may")
    log.info(f"  already be capturing what's learnable here.")
    log.info(f"{'='*60}")
    log.info(f"  precision (call is correct): {report.get('1', {}).get('precision', 0):.1%}")
    log.info(f"  recall    (call is correct): {report.get('1', {}).get('recall', 0):.1%}")

    # Threshold sweep — at each meta-confidence cutoff, what fraction of ORIGINAL
    # primary calls would you act on, and what's the resulting hit rate?
    log.info("\n── Meta-confidence threshold sweep ──────────────────────")
    best_thresh, best_score = 0.5, 0.0
    for thresh in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
        mask = y_proba >= thresh
        n_sel = mask.sum()
        if n_sel < 20:
            continue
        hit_rate = y_test[mask].mean()
        # Simple EV-style score, same sqrt(n) volume weighting as train_model.py's
        # threshold selector, so both pipelines optimize consistently.
        score = hit_rate * np.sqrt(n_sel)
        log.info(f"  {thresh:.2f}   n={n_sel:>6}   hit_rate={hit_rate*100:.1f}%   score={score:.1f}")
        if score > best_score:
            best_score, best_thresh = score, thresh

    log.info(f"\n  -> Best meta-threshold: {best_thresh:.2f}")

    meta_pipeline = {
        "meta_ensemble": calibrated_meta,
        "meta_features": meta_features,
        "recommended_meta_threshold": best_thresh,
        "base_rate": float(n_correct / n_dir),
        "test_accuracy": float(acc),
        "trained_at": pd.Timestamp.utcnow().isoformat(),
        "primary_model_trained_at": primary_pipeline.get("trained_at", "unknown"),
    }
    joblib.dump(meta_pipeline, META_MODEL_FILE)
    log.info(f"\n✅ Saved: {META_MODEL_FILE}")

    with open("meta_model_performance.json", "w") as f:
        json.dump({
            "test_accuracy": round(acc * 100, 1),
            "base_rate": round(n_correct / n_dir * 100, 1),
            "n_directional_calls": int(n_dir),
            "recommended_meta_threshold": best_thresh,
            "meta_features": meta_features,
        }, f, indent=2)


# ─────────────────────────────────────────────────────────────────────────
# WORKED EXAMPLE — how generate_signal() in trade_executor.py would use this.
# NOT wired in automatically — this is reference code for when you're ready
# to integrate it, after confirming via backtest.py that it actually helps.
# ─────────────────────────────────────────────────────────────────────────
"""
# In generate_signal(), after the existing primary prediction:
#   sig  = pipeline["label_map"][int(pred)]
#   conf = round(float(max(prob))*100, 1)
# add:

    if sig != "NO_TRADE":
        meta_pipeline = joblib.load(META_MODEL_FILE)  # cache this at module load, not per-call
        mf = meta_pipeline["meta_features"]
        X_meta = pd.DataFrame([row[mf].values], columns=mf).replace([np.inf,-np.inf],0).fillna(0)
        meta_conf = meta_pipeline["meta_ensemble"].predict_proba(X_meta)[0][1]

        if meta_conf < meta_pipeline["recommended_meta_threshold"]:
            log.info(f"    [META] primary said {sig} but meta-confidence {meta_conf:.1%} "
                     f"below threshold — skip")
            return None

        # Use meta_conf (not the primary's raw softmax conf) for position sizing —
        # it's a more honest measure of "how likely is this specific call to work."
        conf = meta_conf * 100
"""

if __name__ == "__main__":
    t0 = time.time()
    train_meta_model()
    log.info(f"\nDone in {(time.time()-t0)/60:.1f} min")
