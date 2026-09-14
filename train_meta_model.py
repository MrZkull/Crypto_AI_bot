# train_meta_model.py — Research Pipeline: Unwrapped OOF Meta-Training & 6-Tier Gate 10

import json, logging, time
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score
from sklearn.base import clone
from sklearn.frozen import FrozenEstimator
from xgboost import XGBClassifier

from train_model import (
    build_dataset_from_local_parquet, FULL_FEATURES, MODEL_FILE, EMBARGO_BARS,
    TEST_SPLIT, CALIB_SPLIT, N_FEATURES, temporal_symbol_split
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

META_MODEL_FILE = "meta_pipeline.pkl"
N_META_FEATURES = 25
META_SYSTEM_FEATURES = [
    "meta_primary_conf",
    "meta_primary_side_code",
    "meta_rsi_directional",
    "meta_macd_directional",
    "meta_trend_aligned",
]


class Gate10Policy:
    MIN_TEST_TRADES: int = 50
    MIN_NET_EV_R: float = 0.05
    MIN_PRECISION_LIFT_PCT: float = 2.5
    MIN_RETENTION_RATE: float = 0.35
    CONFIDENCE_LEVEL: float = 0.95
    BOOTSTRAP_ROUNDS: int = 10_000
    BLOCK_SIZE_MS: int = 24 * 60 * 60 * 1000  # 24-hour calendar clusters


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


def evaluate_gate_10(
    primary_test_pred: pd.DataFrame,
    test_proba: np.ndarray | pd.Series | dict,
    meta_threshold: float,
    primary_threshold: float,
    realized_returns_by_id: pd.Series | dict = None,
    friction_r: float = 0.12
) -> dict:
    required_cols = {"pred_id", "open_time", "meta_label", "primary_conf", "primary_side"}
    missing = required_cols - set(primary_test_pred.columns)
    if missing:
        raise ValueError(f"CRITICAL: primary_test_pred missing columns: {missing}")

    if primary_test_pred["pred_id"].duplicated().any():
        raise ValueError("CRITICAL: Duplicate pred_id entries found in primary_test_pred.")

    y_test = primary_test_pred["meta_label"].values
    unique_labels = set(np.unique(y_test))
    if not unique_labels.issubset({0, 1}):
        raise ValueError(f"CRITICAL: Non-binary meta_label detected: {unique_labels}")

    df = primary_test_pred.copy()

    if isinstance(test_proba, (pd.Series, dict)):
        proba_map = pd.Series(test_proba)
        if proba_map.index.duplicated().any():
            raise ValueError("CRITICAL: Duplicate pred_id detected in test_proba map.")
        df["meta_prob"] = df["pred_id"].map(proba_map)
        if df["meta_prob"].isna().any():
            missing_prob_ids = df.loc[df["meta_prob"].isna(), "pred_id"].tolist()
            raise ValueError(f"CRITICAL: {len(missing_prob_ids)} predictions unaligned in test_proba map.")
    else:
        test_proba_arr = np.asarray(test_proba, dtype=float)
        if len(test_proba_arr) != len(df):
            raise ValueError(
                f"CRITICAL: test_proba length ({len(test_proba_arr)}) != "
                f"primary_test_pred rows ({len(df)})."
            )
        df["meta_prob"] = test_proba_arr

    if "is_primary_eligible" in df.columns:
        df["primary_selected"] = df["is_primary_eligible"].astype(bool)
    else:
        df["primary_selected"] = df["primary_conf"] >= primary_threshold

    df["meta_selected"] = df["primary_selected"] & (df["meta_prob"] >= meta_threshold)

    is_production_mode = realized_returns_by_id is not None
    eval_mode = "PRODUCTION_REALIZED_RETURNS" if is_production_mode else "RESEARCH_SYNTHETIC_PAYOFF"

    if is_production_mode:
        if not isinstance(realized_returns_by_id, (pd.Series, dict)):
            raise TypeError("realized_returns_by_id must be a pd.Series or dict keyed by pred_id")

        returns_map = pd.Series(realized_returns_by_id)
        if returns_map.index.duplicated().any():
            dup_ids = returns_map.index[returns_map.index.duplicated()].unique().tolist()
            raise ValueError(f"CRITICAL: Duplicate pred_id in realized return map: {dup_ids}")

        df["net_r"] = df["pred_id"].map(returns_map)
        if df["net_r"].isna().any():
            missing_ids = df.loc[df["net_r"].isna(), "pred_id"].tolist()
            raise ValueError(f"CRITICAL: {len(missing_ids)} predictions unaligned in realized_returns_by_id.")
        if not np.all(np.isfinite(df["net_r"])):
            raise ValueError("CRITICAL: Non-finite realized net_r values detected.")
    else:
        df["net_r"] = np.where(df["meta_label"] == 1, 3.5 - friction_r, -2.5 - friction_r)

    primary_sample = df[df["primary_selected"]]
    meta_sample = df[df["meta_selected"]]

    n_primary_active = len(primary_sample)
    n_meta = len(meta_sample)

    if n_primary_active == 0:
        return {
            "passed_production_gate": False,
            "passed_research_gate": False,
            "tier_failed": "SAMPLE_VALIDITY",
            "evaluation_mode": eval_mode,
            "reason": f"Primary policy generated 0 eligible trades at threshold {primary_threshold}"
        }

    if n_meta < Gate10Policy.MIN_TEST_TRADES:
        return {
            "passed_production_gate": False,
            "passed_research_gate": False,
            "tier_failed": "SAMPLE_VALIDITY",
            "evaluation_mode": eval_mode,
            "reason": f"Filtered trade count {n_meta} < minimum threshold {Gate10Policy.MIN_TEST_TRADES}"
        }

    mean_net_r = float(meta_sample["net_r"].mean())
    if mean_net_r < Gate10Policy.MIN_NET_EV_R:
        return {
            "passed_production_gate": False,
            "passed_research_gate": False,
            "tier_failed": "ECONOMIC_VIABILITY",
            "evaluation_mode": eval_mode,
            "reason": f"Mean Net-R {mean_net_r:.4f} < hurdle {Gate10Policy.MIN_NET_EV_R}R"
        }

    rng = np.random.default_rng(42)
    boot_means = np.empty(Gate10Policy.BOOTSTRAP_ROUNDS)

    if is_production_mode:
        meta_sample = meta_sample.copy()
        meta_sample["block_id"] = meta_sample["open_time"] // Gate10Policy.BLOCK_SIZE_MS
        unique_blocks = meta_sample["block_id"].unique()
        n_blocks = len(unique_blocks)
        block_returns = [meta_sample.loc[meta_sample["block_id"] == b, "net_r"].values for b in unique_blocks]

        for b in range(Gate10Policy.BOOTSTRAP_ROUNDS):
            sampled_block_indices = rng.choice(n_blocks, size=n_blocks, replace=True)
            resampled_returns = np.concatenate([block_returns[i] for i in sampled_block_indices])
            boot_means[b] = np.mean(resampled_returns)
    else:
        returns_arr = meta_sample["net_r"].values
        for b in range(Gate10Policy.BOOTSTRAP_ROUNDS):
            boot_means[b] = np.mean(rng.choice(returns_arr, size=n_meta, replace=True))

    tail = (1.0 - Gate10Policy.CONFIDENCE_LEVEL) / 2.0
    ci_lower = float(np.percentile(boot_means, tail * 100))
    ci_upper = float(np.percentile(boot_means, (1.0 - tail) * 100))

    if ci_lower <= 0.0:
        return {
            "passed_production_gate": False,
            "passed_research_gate": False,
            "tier_failed": "STATISTICAL_SIGNIFICANCE",
            "evaluation_mode": eval_mode,
            "reason": f"Bootstrap 95% CI lower bound {ci_lower:.4f} <= 0.0R"
        }

    meta_precision = float(meta_sample["meta_label"].mean())
    primary_precision = float(primary_sample["meta_label"].mean())
    primary_mean_r = float(primary_sample["net_r"].mean())

    precision_lift_pct = (meta_precision - primary_precision) * 100.0
    r_lift = mean_net_r - primary_mean_r

    if precision_lift_pct < Gate10Policy.MIN_PRECISION_LIFT_PCT or r_lift <= 0.0:
        return {
            "passed_production_gate": False,
            "passed_research_gate": False,
            "tier_failed": "INCREMENTAL_PERFORMANCE",
            "evaluation_mode": eval_mode,
            "reason": (
                f"Precision lift {precision_lift_pct:.2f}% < {Gate10Policy.MIN_PRECISION_LIFT_PCT}% "
                f"or R-lift {r_lift:.4f} <= 0.0R"
            )
        }

    retention_rate = n_meta / n_primary_active
    if retention_rate < Gate10Policy.MIN_RETENTION_RATE:
        return {
            "passed_production_gate": False,
            "passed_research_gate": False,
            "tier_failed": "CAPACITY",
            "evaluation_mode": eval_mode,
            "reason": (
                f"Trade retention {retention_rate*100:.1f}% < minimum "
                f"{Gate10Policy.MIN_RETENTION_RATE*100:.1f}% of primary setups"
            )
        }

    production_candidate_eligible = is_production_mode and (ci_lower > 0.0)
    return {
        "production_candidate_eligible": production_candidate_eligible,
        "passed_research_gate": True,
        "evaluation_mode": eval_mode,
        "bootstrap_method": "24H_CALENDAR_CLUSTER_CONSERVATIVE" if is_production_mode else "IID_RESEARCH",
        "n_primary_active": n_primary_active,
        "n_meta_retained": n_meta,
        "retention_rate_pct": round(retention_rate * 100, 2),
        "primary_precision_pct": round(primary_precision * 100, 2),
        "meta_precision_pct": round(meta_precision * 100, 2),
        "precision_lift_pct": round(precision_lift_pct, 2),
        "primary_mean_net_r": round(primary_mean_r, 4),
        "meta_mean_net_r": round(mean_net_r, 4),
        "r_lift": round(r_lift, 4),
        "bootstrap_ci_95": [round(ci_lower, 4), round(ci_upper, 4)],
        "production_status": "ELIGIBLE_FOR_TESTNET_MATRIX" if production_candidate_eligible else "RESEARCH_ONLY"
    }


def train_meta_model():
    log.info("Loading primary model artifact...")
    primary_pipeline = joblib.load(MODEL_FILE)
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
    gate10_result = evaluate_gate_10(
        primary_test_pred=primary_test_pred,
        test_proba=test_proba,
        meta_threshold=best_thresh,
        primary_threshold=primary_baseline_threshold,
        friction_r=0.12
    )

    log.info(f"Gate 10 Result: {json.dumps(gate10_result, indent=2)}")

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
    joblib.dump(meta_pipeline, META_MODEL_FILE)
    log.info(f"✅ Saved meta-model pipeline: {META_MODEL_FILE}")

    with open("meta_model_performance.json", "w") as f:
        json.dump(gate10_result, f, indent=2)


if __name__ == "__main__":
    t0 = time.time()
    train_meta_model()
    log.info(f"Meta-model training complete in {(time.time()-t0)/60:.1f} min")
