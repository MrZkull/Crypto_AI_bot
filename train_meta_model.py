#!/usr/bin/env python3
# train_meta_model.py — Research Pipeline: Direction-Aware Meta-Training & Gate 10

import argparse
import hashlib
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import clone
from sklearn.frozen import FrozenEstimator
from xgboost import XGBClassifier

# Scikit-Learn 1.6+ compatibility patch for XGBoost
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

try:
    from train_model import (
        build_dataset_from_local_parquet, MODEL_FILE, EMBARGO_BARS,
        TEST_SPLIT, CALIB_SPLIT, N_FEATURES, temporal_symbol_split, FULL_FEATURES
    )
except ImportError:
    MODEL_FILE = "pro_crypto_ai_model.pkl"
    EMBARGO_BARS = 24
    TEST_SPLIT = 0.20
    CALIB_SPLIT = 0.15
    N_FEATURES = 35
    FULL_FEATURES = []
    build_dataset_from_local_parquet = None
    temporal_symbol_split = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

META_MODEL_FILE = Path("meta_pipeline.pkl")
CANDIDATE_META_FILE = Path("candidate_meta_model.pkl")
META_MANIFEST_FILE = Path("meta_model_manifest.json")
CANDIDATE_META_MANIFEST_FILE = Path("candidate_meta_model_manifest.json")
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
    BLOCK_SIZE_MS: int = 24 * 60 * 60 * 1000


COMPRESSION_LEVEL = 3
MAX_MODEL_BYTES = 95 * 1024 * 1024

def _sha256_file(path) -> str:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Required file not found: {path}")
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()



def _optional_source_hashes() -> dict[str, str | None]:
    hashes = {}
    for name in ("train_model.py", "feature_engineering.py", "config.py"):
        path = Path(name)
        hashes[name] = _sha256_file(path) if path.is_file() else None
    return hashes


def _policy_hash() -> str:
    payload = {
        "min_test_trades": Gate10Policy.MIN_TEST_TRADES,
        "min_net_ev_r": Gate10Policy.MIN_NET_EV_R,
        "min_precision_lift_pct": Gate10Policy.MIN_PRECISION_LIFT_PCT,
        "min_retention_rate": Gate10Policy.MIN_RETENTION_RATE,
        "confidence_level": Gate10Policy.CONFIDENCE_LEVEL,
        "bootstrap_rounds": Gate10Policy.BOOTSTRAP_ROUNDS,
        "block_size_ms": Gate10Policy.BLOCK_SIZE_MS,
        "n_meta_features": N_META_FEATURES,
        "meta_system_features": META_SYSTEM_FEATURES,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def _dump_compressed_model(payload: dict, output_path: Path) -> int:
    tmp = output_path.with_suffix(output_path.suffix + ".tmp")
    try:
        joblib.dump(payload, tmp, compress=COMPRESSION_LEVEL)
        size = tmp.stat().st_size
        if size > MAX_MODEL_BYTES:
            raise RuntimeError(
                f"FATAL: compressed meta artifact is {size/(1024*1024):.2f} MiB; "
                f"safety ceiling is {MAX_MODEL_BYTES/(1024*1024):.0f} MiB"
            )
        os.replace(tmp, output_path)
        log.info(
            "Meta artifact: %s | %.2f MiB | compression=%d",
            output_path, size/(1024*1024), COMPRESSION_LEVEL
        )
        return size
    except Exception:
        tmp.unlink(missing_ok=True)
        raise

def _write_json(path, payload):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(tmp, path)


def _get_oof_base_estimator(primary_pipeline: dict):
    """Extract a cloneable unfitted classifier from the production ensemble wrapper."""
    candidate = primary_pipeline["ensemble"]
    seen = set()

    for _ in range(8):
        if id(candidate) in seen:
            break
        seen.add(id(candidate))

        # VotingClassifier retains its unfitted estimator definitions in `estimators`.
        # Cloning the wrapper gives an unfitted OOF ensemble with the same topology.
        if isinstance(candidate, VotingClassifier):
            return clone(candidate)

        if isinstance(candidate, FrozenEstimator):
            candidate = candidate.estimator
            continue

        nested = getattr(candidate, "estimator", None)
        if nested is not None and nested is not candidate:
            candidate = nested
            continue

        nested = getattr(candidate, "base_estimator", None)
        if nested is not None and nested is not candidate:
            candidate = nested
            continue

        break

    if hasattr(candidate, "fit") and hasattr(candidate, "predict_proba"):
        try:
            return clone(candidate)
        except Exception as exc:
            raise ValueError(
                "Primary ensemble classifier is not cloneable for OOF"
            ) from exc

    raise ValueError(
        "Primary ensemble does not expose a cloneable unfitted classifier for OOF"
    )


def _audit_meta_partition_isolation(
    meta_train: pd.DataFrame,
    primary_calib_pred: pd.DataFrame,
    primary_test_pred: pd.DataFrame,
) -> None:
    """Fail closed if OOF/calibration/test prediction IDs overlap."""
    parts = {
        "meta_train": set(meta_train["pred_id"].astype(str)),
        "primary_calib": set(primary_calib_pred["pred_id"].astype(str)),
        "primary_test": set(primary_test_pred["pred_id"].astype(str)),
    }
    names = list(parts)

    for i, left in enumerate(names):
        for right in names[i + 1:]:
            overlap = parts[left] & parts[right]
            if overlap:
                raise ValueError(
                    "CRITICAL META PARTITION LEAKAGE: "
                    f"{left} and {right} share pred_id values; "
                    f"sample={sorted(overlap)[:10]}"
                )

    log.info(
        "Meta partition isolation audit: PASS "
        f"(OOF={len(parts['meta_train']):,}, "
        f"calib={len(parts['primary_calib']):,}, "
        f"test={len(parts['primary_test']):,})"
    )


def _validate_empirical_returns(
    df: pd.DataFrame,
    realized_returns_by_id: pd.Series,
) -> pd.Series:
    returns_map = pd.Series(realized_returns_by_id)
    aligned = pd.to_numeric(
        df["pred_id"].map(returns_map),
        errors="coerce",
    )

    if aligned.isna().any():
        missing = df.loc[aligned.isna(), "pred_id"].astype(str).tolist()
        raise ValueError(
            "CRITICAL: realized-return map is missing "
            f"{len(missing)} selected prediction IDs; sample={missing[:10]}"
        )

    values = aligned.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        bad = df.loc[~np.isfinite(values), "pred_id"].astype(str).tolist()
        raise ValueError(
            "CRITICAL: realized-return map contains non-finite values "
            f"for {len(bad)} IDs; sample={bad[:10]}"
        )

    return aligned


def _gate10_failure(tier: str, eval_mode: str, reason: str) -> dict:
    return {
        "production_candidate_eligible": False,
        "passed_production_gate": False,
        "passed_research_gate": False,
        "tier_failed": tier,
        "evaluation_mode": eval_mode,
        "research_only": eval_mode != "PRODUCTION_REALIZED_RETURNS",
        "reason": reason,
    }


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
    af = primary_pipeline.get("all_features", FULL_FEATURES)
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

    af = primary_pipeline.get("all_features", FULL_FEATURES)
    label_map = {int(k): v for k, v in primary_pipeline["label_map"].items()}
    target_to_int = {v: k for k, v in label_map.items()}
    no_trade_idx = target_to_int.get("NO_TRADE")
    if no_trade_idx is None:
        raise ValueError("Primary pipeline label map is missing NO_TRADE")

    base_estimator = _get_oof_base_estimator(primary_pipeline)

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
    test_proba: np.ndarray,
    meta_threshold: float,
    primary_pipeline: dict,
    realized_returns_by_id: pd.Series = None,
    friction_r: float = 0.12
) -> dict:
    required_cols = {
        "pred_id", "open_time", "meta_label",
        "primary_conf", "primary_side",
    }
    missing = required_cols - set(primary_test_pred.columns)
    if missing:
        raise ValueError(
            f"CRITICAL: primary_test_pred missing columns: {missing}"
        )

    df = primary_test_pred.copy()
    test_proba = np.asarray(test_proba, dtype=float).reshape(-1)

    if len(test_proba) != len(df):
        raise ValueError(
            "CRITICAL: test_proba length does not match primary_test_pred"
        )
    if not np.isfinite(test_proba).all():
        raise ValueError("CRITICAL: test_proba contains non-finite values")

    df["meta_prob"] = test_proba

    # Direction-aware primary selection, preserving the original sentinels.
    thresh_buy = float(primary_pipeline.get("recommended_threshold_buy", 0.40))
    thresh_sell = float(primary_pipeline.get("recommended_threshold_sell", 0.40))
    buy_active = thresh_buy <= 1.0
    sell_active = thresh_sell <= 1.0

    df["primary_selected"] = (
        (buy_active & (df["primary_side"] == "BUY") & (df["primary_conf"] >= thresh_buy)) |
        (sell_active & (df["primary_side"] == "SELL") & (df["primary_conf"] >= thresh_sell))
    )
    df["meta_selected"] = (
        df["primary_selected"] & (df["meta_prob"] >= meta_threshold)
    )

    is_production_mode = realized_returns_by_id is not None
    eval_mode = (
        "PRODUCTION_REALIZED_RETURNS"
        if is_production_mode
        else "RESEARCH_SYNTHETIC_PAYOFF"
    )

    buy_tp_mult = float(primary_pipeline.get("buy_tp_mult", 3.5))
    buy_sl_mult = float(primary_pipeline.get("buy_sl_mult", 2.5))
    sell_tp_mult = float(primary_pipeline.get("sell_tp_mult", 3.5))
    sell_sl_mult = float(primary_pipeline.get("sell_sl_mult", 2.5))

    if is_production_mode:
        # The gate judges exactly the rows retained by the meta policy.
        meta_rows = df[df["meta_selected"]].copy()
        df["net_r"] = np.nan

        if not meta_rows.empty:
            aligned = _validate_empirical_returns(
                meta_rows,
                realized_returns_by_id,
            )
            df.loc[meta_rows.index, "net_r"] = aligned.to_numpy(dtype=float)
    else:
        # Research-only synthetic payoff. It cannot authorize production.
        df["net_r"] = np.where(
            df["meta_label"] == 1,
            np.where(
                df["primary_side"] == "BUY",
                buy_tp_mult - friction_r,
                sell_tp_mult - friction_r,
            ),
            np.where(
                df["primary_side"] == "BUY",
                -buy_sl_mult - friction_r,
                -sell_sl_mult - friction_r,
            )
        )

    primary_sample = df[df["primary_selected"]].copy()
    meta_sample = df[df["meta_selected"]].copy()

    n_primary_active = len(primary_sample)
    n_meta = len(meta_sample)

    if n_primary_active == 0 or n_meta < Gate10Policy.MIN_TEST_TRADES:
        return _gate10_failure(
            "SAMPLE_VALIDITY",
            eval_mode,
            f"Active trades {n_meta} < minimum threshold "
            f"{Gate10Policy.MIN_TEST_TRADES} "
            f"(Primary active: {n_primary_active})",
        )

    meta_returns = meta_sample["net_r"].to_numpy(dtype=float)
    if not np.isfinite(meta_returns).all():
        return _gate10_failure(
            "EMPIRICAL_DATA_VALIDITY" if is_production_mode else "RESEARCH_DATA_VALIDITY",
            eval_mode,
            "Selected net-R contains missing/non-finite values",
        )

    mean_net_r = float(meta_sample["net_r"].mean())
    if not np.isfinite(mean_net_r):
        return _gate10_failure(
            "STATISTICAL_VALIDITY",
            eval_mode,
            "Mean Net-R is non-finite",
        )

    if mean_net_r < Gate10Policy.MIN_NET_EV_R:
        return _gate10_failure(
            "ECONOMIC_VIABILITY",
            eval_mode,
            f"Mean Net-R {mean_net_r:.4f} < hurdle "
            f"{Gate10Policy.MIN_NET_EV_R}R",
        )

    rng = np.random.default_rng(42)
    boot_means = np.empty(
        Gate10Policy.BOOTSTRAP_ROUNDS,
        dtype=float,
    )

    if is_production_mode:
        meta_sample["block_id"] = (
            pd.to_numeric(meta_sample["open_time"], errors="coerce")
            // Gate10Policy.BLOCK_SIZE_MS
        )

        if meta_sample["block_id"].isna().any():
            return _gate10_failure(
                "EMPIRICAL_DATA_VALIDITY",
                eval_mode,
                "Selected empirical observations contain invalid open_time values",
            )

        unique_blocks = (
            meta_sample["block_id"].drop_duplicates().to_numpy()
        )
        n_blocks = len(unique_blocks)

        if n_blocks == 0:
            return _gate10_failure(
                "SAMPLE_VALIDITY",
                eval_mode,
                "No 24h blocks available for bootstrap",
            )

        block_returns = [
            meta_sample.loc[
                meta_sample["block_id"] == block,
                "net_r",
            ].to_numpy(dtype=float)
            for block in unique_blocks
        ]

        for b in range(Gate10Policy.BOOTSTRAP_ROUNDS):
            sampled_blocks = rng.choice(
                n_blocks,
                size=n_blocks,
                replace=True,
            )
            resampled = np.concatenate(
                [block_returns[i] for i in sampled_blocks]
            )
            boot_means[b] = np.mean(resampled)

    else:
        for b in range(Gate10Policy.BOOTSTRAP_ROUNDS):
            boot_means[b] = np.mean(
                rng.choice(
                    meta_returns,
                    size=n_meta,
                    replace=True,
                )
            )

    if not np.isfinite(boot_means).all():
        return _gate10_failure(
            "STATISTICAL_VALIDITY",
            eval_mode,
            "Bootstrap distribution contains non-finite values",
        )

    tail = (1.0 - Gate10Policy.CONFIDENCE_LEVEL) / 2.0
    ci_lower = float(
        np.percentile(
            boot_means,
            tail * 100,
        )
    )
    ci_upper = float(
        np.percentile(
            boot_means,
            (1.0 - tail) * 100,
        )
    )

    if ci_lower <= 0.0:
        return _gate10_failure(
            "STATISTICAL_SIGNIFICANCE",
            eval_mode,
            f"Bootstrap 95% CI lower bound {ci_lower:.4f} <= 0.0R",
        )

    meta_precision = float(meta_sample["meta_label"].mean())
    primary_precision = float(primary_sample["meta_label"].mean())
    primary_mean_r = float(primary_sample["net_r"].mean())

    precision_lift_pct = (
        meta_precision - primary_precision
    ) * 100.0
    r_lift = mean_net_r - primary_mean_r

    # This policy field existed in the baseline but was previously not enforced.
    if precision_lift_pct < Gate10Policy.MIN_PRECISION_LIFT_PCT:
        return _gate10_failure(
            "PRECISION_LIFT",
            eval_mode,
            f"Precision lift {precision_lift_pct:.2f}pp < "
            f"minimum {Gate10Policy.MIN_PRECISION_LIFT_PCT:.2f}pp",
        )

    retention_rate = n_meta / n_primary_active
    if retention_rate < Gate10Policy.MIN_RETENTION_RATE:
        return _gate10_failure(
            "CAPACITY",
            eval_mode,
            f"Trade retention {retention_rate*100:.1f}% < minimum "
            f"{Gate10Policy.MIN_RETENTION_RATE*100:.1f}%",
        )

    # Only empirical realized-return mode can ever become production-eligible.
    production_candidate_eligible = (
        is_production_mode and ci_lower > 0.0
    )

    return {
        "production_candidate_eligible": production_candidate_eligible,
        "passed_production_gate": production_candidate_eligible,
        "passed_research_gate": True,
        "evaluation_mode": eval_mode,
        "research_only": not is_production_mode,
        "n_primary_active": n_primary_active,
        "n_meta_retained": n_meta,
        "retention_rate_pct": round(retention_rate * 100, 2),
        "primary_precision_pct": round(primary_precision * 100, 2),
        "meta_precision_pct": round(meta_precision * 100, 2),
        "precision_lift_pct": round(precision_lift_pct, 2),
        "primary_mean_net_r": round(primary_mean_r, 4),
        "meta_mean_net_r": round(mean_net_r, 4),
        "r_lift": round(r_lift, 4),
        "bootstrap_ci_95": [
            round(ci_lower, 4),
            round(ci_upper, 4),
        ],
        "production_status": (
            "ELIGIBLE_FOR_TESTNET_MATRIX"
            if production_candidate_eligible
            else "RESEARCH_ONLY"
        ),
        "gate10_policy": {
            "min_test_trades": Gate10Policy.MIN_TEST_TRADES,
            "min_net_ev_r": Gate10Policy.MIN_NET_EV_R,
            "min_precision_lift_pct": Gate10Policy.MIN_PRECISION_LIFT_PCT,
            "min_retention_rate": Gate10Policy.MIN_RETENTION_RATE,
            "confidence_level": Gate10Policy.CONFIDENCE_LEVEL,
            "bootstrap_rounds": Gate10Policy.BOOTSTRAP_ROUNDS,
            "block_size_ms": Gate10Policy.BLOCK_SIZE_MS,
        },
    }


def train_meta_model(primary_model_path=None, candidate_mode=False) -> dict:
    if build_dataset_from_local_parquet is None or temporal_symbol_split is None:
        raise ImportError("Training mode requires train_model.py; use --test for self-testing.")

    if primary_model_path is None:
        primary_model_path = "candidate_model.pkl" if candidate_mode else MODEL_FILE

    primary_model_path = Path(primary_model_path)

    if not primary_model_path.is_file():
        raise FileNotFoundError(
            f"Primary model input does not exist: {primary_model_path}"
        )

    output_model_path = (
        Path(CANDIDATE_META_FILE)
        if candidate_mode
        else Path(META_MODEL_FILE)
    )
    manifest_path = (
        Path(CANDIDATE_META_MANIFEST_FILE)
        if candidate_mode
        else Path(META_MANIFEST_FILE)
    )
    artifact_role = "candidate_meta" if candidate_mode else "research_meta"

    log.info(
        "Loading primary model artifact: %s",
        primary_model_path,
    )
    primary_pipeline = joblib.load(primary_model_path)

    if not isinstance(primary_pipeline, dict):
        raise ValueError("Primary model artifact must be a dict")

    required_primary_keys = {
        "ensemble",
        "selector",
        "all_features",
        "label_map",
    }
    missing_primary = required_primary_keys - set(primary_pipeline)
    if missing_primary:
        raise ValueError(
            "Primary model artifact missing required keys: "
            f"{sorted(missing_primary)}"
        )

    sell_tp = float(primary_pipeline.get("sell_tp_mult", 3.5))
    sell_sl = float(primary_pipeline.get("sell_sl_mult", 2.5))
    ds = build_dataset_from_local_parquet(sell_tp_mult=sell_tp, sell_sl_mult=sell_sl)

    primary_train, primary_calib, primary_test = temporal_symbol_split(
        ds, TEST_SPLIT, CALIB_SPLIT, EMBARGO_BARS
    )
    if primary_train.empty or primary_calib.empty or primary_test.empty:
        raise ValueError("Primary eras are incomplete; refusing meta-model training")

    meta_train = build_oof_meta_training(primary_train, primary_pipeline)
    primary_calib_pred = augment_meta_features(build_meta_labels(get_primary_predictions(primary_calib.copy(), primary_pipeline)))
    primary_test_pred = augment_meta_features(build_meta_labels(get_primary_predictions(primary_test.copy(), primary_pipeline)))

    _audit_meta_partition_isolation(
        meta_train,
        primary_calib_pred,
        primary_test_pred,
    )

    active_features = primary_pipeline.get("all_features", FULL_FEATURES)
    meta_feature_universe = list(dict.fromkeys(active_features + META_SYSTEM_FEATURES))
    for f in meta_feature_universe:
        for part in (meta_train, primary_calib_pred, primary_test_pred):
            if f not in part.columns: part[f] = 0.0

    X_train_full = meta_train[meta_feature_universe].replace([np.inf, -np.inf], np.nan).fillna(0).values
    y_train = meta_train["meta_label"].values

    scanner = XGBClassifier(n_estimators=100, random_state=42, n_jobs=-1, eval_metric="logloss")
    scanner.fit(X_train_full, y_train)
    top_idx = np.argsort(scanner.feature_importances_)[::-1][:N_META_FEATURES]
    meta_features = [meta_feature_universe[i] for i in top_idx]
    for f in META_SYSTEM_FEATURES:
        if f not in meta_features: meta_features.append(f)

    X_train = meta_train[meta_features].replace([np.inf, -np.inf], np.nan).fillna(0).values
    X_calib = primary_calib_pred[meta_features].replace([np.inf, -np.inf], np.nan).fillna(0).values
    y_calib = primary_calib_pred["meta_label"].values
    X_test = primary_test_pred[meta_features].replace([np.inf, -np.inf], np.nan).fillna(0).values

    meta_xgb = XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.03, subsample=0.85, colsample_bytree=0.85, min_child_weight=8, reg_lambda=5.0, eval_metric="logloss", random_state=42, n_jobs=-1)
    meta_rf = RandomForestClassifier(n_estimators=300, max_depth=5, min_samples_leaf=15, random_state=42, n_jobs=-1)
    meta_ensemble = VotingClassifier(estimators=[("xgb", meta_xgb), ("rf", meta_rf)], voting="soft", weights=[2, 1])
    meta_ensemble.fit(X_train, y_train)

    calibrated_meta = CalibratedClassifierCV(estimator=FrozenEstimator(meta_ensemble), method="isotonic")
    calibrated_meta.fit(X_calib, y_calib)

    classes = list(calibrated_meta.classes_)
    pos_idx = classes.index(1)
    calib_proba = calibrated_meta.predict_proba(X_calib)[:, pos_idx]
    test_proba  = calibrated_meta.predict_proba(X_test)[:, pos_idx]

    best_thresh, best_score = 0.50, 0.0
    log.info(f"\n{'='*70}")
    log.info("META-MODEL THRESHOLD CALIBRATION SWEEP")
    log.info(f"{'='*70}")
    log.info(f"{'Thresh':<8} | {'Trades':<8} | {'Precision':<10} | {'Score':<8}")
    log.info(f"{'-'*70}")

    for thresh in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
        mask = calib_proba >= thresh
        if mask.sum() < 10:
            log.info(f"{thresh:<8.2f} | {mask.sum():<8} | {'SKIPPED (<10 trades)':<20}")
            continue
        prec = float(y_calib[mask].mean())
        score = prec * np.sqrt(mask.sum())
        log.info(f"{thresh:<8.2f} | {mask.sum():<8} | {prec*100:>8.1f}%  | {score:>8.2f}")
        if score > best_score:
            best_score, best_thresh = score, thresh
    log.info(f"{'-'*70}\n")

    gate10_result = evaluate_gate_10(
        primary_test_pred=primary_test_pred,
        test_proba=test_proba,
        meta_threshold=best_thresh,
        primary_pipeline=primary_pipeline,
        friction_r=0.12
    )

    log.info(f"Gate 10 Result: {json.dumps(gate10_result, indent=2)}")

    meta_pipeline = {
        "meta_ensemble": calibrated_meta,
        "meta_features": meta_features,
        "meta_positive_class": 1,
        "meta_positive_idx": pos_idx,
        "recommended_meta_threshold": best_thresh,
        "gate10_summary": gate10_result,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }

    # Only emit the candidate-named artifact when the run explicitly trains from a candidate primary.
    output_model_path = CANDIDATE_META_FILE if candidate_mode else META_MODEL_FILE
    manifest_path = CANDIDATE_META_MANIFEST_FILE if candidate_mode else META_MANIFEST_FILE

    primary_path = Path(primary_model_path)
    primary_sha256 = _sha256_file(primary_path)
    source_hashes = _optional_source_hashes()
    script_sha256 = _sha256_file(Path(__file__))
    source_hashes[Path(__file__).name] = script_sha256

    meta_size_bytes = _dump_compressed_model(meta_pipeline, output_model_path)
    meta_sha256 = _sha256_file(output_model_path)
    decision_policy_hash = _policy_hash()
    meta_feature_hash = hashlib.sha256(
        json.dumps(sorted(meta_features), separators=(",", ":")).encode()
    ).hexdigest()

    identity_payload = {
        "mode": "CANDIDATE" if candidate_mode else "PRODUCTION_RESEARCH",
        "primary_model_sha256": primary_sha256,
        "meta_model_sha256": meta_sha256,
        "script_sha256": script_sha256,
        "feature_engineering_sha256": source_hashes.get("feature_engineering.py"),
        "decision_policy_hash": decision_policy_hash,
    }
    candidate_id = hashlib.sha256(
        json.dumps(identity_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    manifest = {
        "candidate_id": candidate_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "CANDIDATE" if candidate_mode else "PRODUCTION_RESEARCH",
        "status": "AWAITING_PROSPECTIVE_EVIDENCE" if candidate_mode else "RESEARCH_ONLY",
        "primary_model_file": str(primary_path),
        "primary_model_sha256": primary_sha256,
        "meta_model_file": str(output_model_path),
        "meta_model_sha256": meta_sha256,
        "meta_model_size_bytes": meta_size_bytes,
        "compression_level": COMPRESSION_LEVEL,
        "script_sha256": script_sha256,
        "source_hashes": source_hashes,
        "decision_policy_hash": decision_policy_hash,
        "meta_feature_hash": meta_feature_hash,
        "meta_features": meta_features,
        "gate10_result": gate10_result,
        "production_promotion_authorized": False,
    }

    _write_json(manifest_path, manifest)
    _write_json("meta_model_performance.json", gate10_result)

    log.info(f"✅ Saved meta-model artifact: {output_model_path}")
    log.info(f"✅ Saved provenance manifest: {manifest_path}")
    log.info("Production promotion authorization: FALSE")
    return manifest




def run_self_tests() -> None:
    """Embedded deterministic Gate10/anti-leakage tests; no market data or network access."""
    log.info("Running embedded Gate10 and anti-leakage self-tests...")
    original_rounds = Gate10Policy.BOOTSTRAP_ROUNDS
    Gate10Policy.BOOTSTRAP_ROUNDS = 200
    try:
        n = 100
        df = pd.DataFrame({
            "pred_id": [f"TEST_{i}" for i in range(n)],
            "open_time": np.arange(n, dtype=np.int64) * 900000,
            "meta_label": np.array([1] * 55 + [0] * 45),
            "primary_conf": np.full(n, 0.9),
            "primary_side": np.array(["BUY"] * n),
        })
        meta_proba = np.array([0.9] * 60 + [0.1] * 40)
        pipeline = {
            "recommended_threshold_buy": 0.4,
            "recommended_threshold_sell": 0.4,
            "buy_tp_mult": 3.5,
            "buy_sl_mult": 2.5,
            "sell_tp_mult": 3.5,
            "sell_sl_mult": 2.5,
        }

        research = evaluate_gate_10(df, meta_proba, 0.5, pipeline)
        assert research["evaluation_mode"] == "RESEARCH_SYNTHETIC_PAYOFF"
        assert research["research_only"] is True
        assert research["passed_production_gate"] is False
        assert research["passed_research_gate"] is True
        assert research["precision_lift_pct"] >= Gate10Policy.MIN_PRECISION_LIFT_PCT
        log.info("PASS 1/5 — synthetic research isolation")

        realized = pd.Series(3.38, index=df.loc[:59, "pred_id"])
        empirical = evaluate_gate_10(
            df, meta_proba, 0.5, pipeline, realized_returns_by_id=realized
        )
        assert empirical["evaluation_mode"] == "PRODUCTION_REALIZED_RETURNS"
        assert empirical["production_candidate_eligible"] is True
        assert empirical["passed_production_gate"] is True
        log.info("PASS 2/5 — empirical realized-return path")

        incomplete = realized.drop(realized.index[-1])
        try:
            evaluate_gate_10(
                df, meta_proba, 0.5, pipeline, realized_returns_by_id=incomplete
            )
        except ValueError as exc:
            assert "realized-return map is missing" in str(exc)
        else:
            raise AssertionError("Missing empirical return did not fail closed")
        log.info("PASS 3/5 — incomplete empirical ledger blocked")

        overlap = df.iloc[:10].copy()
        try:
            _audit_meta_partition_isolation(
                overlap, overlap.copy(), df.iloc[20:30].copy()
            )
        except ValueError as exc:
            assert "META PARTITION LEAKAGE" in str(exc)
        else:
            raise AssertionError("Partition overlap did not fail closed")
        log.info("PASS 4/5 — OOF/calib/test overlap blocked")

        assert _policy_hash() == _policy_hash()
        assert len(_policy_hash()) == 64
        log.info("PASS 5/5 — deterministic policy fingerprint")

        print("\n" + "=" * 64)
        print("ALL EMBEDDED GATE 10 / ANTI-LEAKAGE SELF-TESTS PASSED")
        print("No external market data, model files, or network access used.")
        print("=" * 64 + "\n")
    finally:
        Gate10Policy.BOOTSTRAP_ROUNDS = original_rounds


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Train the CryptoBot AI leakage-controlled research meta-model."
    )
    parser.add_argument(
        "--candidate",
        action="store_true",
        help="Use candidate_model.pkl and emit candidate_meta_model.pkl.",
    )
    parser.add_argument(
        "--primary-model",
        default=None,
        help="Explicit primary model input path.",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run the embedded Gate10 and anti-leakage self-test suite.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.test:
        run_self_tests()
        raise SystemExit(0)

    t0 = time.time()
    result = train_meta_model(
        primary_model_path=args.primary_model,
        candidate_mode=args.candidate,
    )
    log.info(
        "Meta-model result: %s",
        json.dumps(result, indent=2, sort_keys=True),
    )
    log.info(
        "Meta-model training complete in %.1f min",
        (time.time() - t0) / 60,
    )
