#!/usr/bin/env python3
# phase2a_feature_ablation.py
# Phase 2A: Raw Funding & BTC Feature Ablation Study
#
# PURPOSE
# -------
# Controlled research experiment to determine whether:
#   1) raw fundingRate
#   2) BTC contextual metrics
#   3) both together
#
# add measurable value over the canonical BASELINE feature set.
#
# RESEARCH PRINCIPLES
# -------------------
# - Canonical local Parquet data only
# - Fail-closed on missing/invalid required features
# - Per-symbol funding validation
# - No silent zero-filling of missing experiment features
# - Chronological train/calibration/test split inherited from train_model.py
# - Frozen TP/SL geometry
# - Locked test thresholds selected only from calibration data
# - SELL disabled with sentinel 1.01 when no positive calibration evidence exists
# - No production model export
# - Structured CSV + JSON artifacts for permanent auditability

import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from feature_engineering import ALL_FEATURES
from train_model import (
    ATR_STOP_MULT,
    ATR_TARGET1_MULT,
    CALIB_SPLIT,
    EMBARGO_BARS,
    MIN_BUY_THRESHOLD_FLOOR,
    MIN_SELL_THRESHOLD_FLOOR,
    N_FEATURES,
    TEST_SPLIT,
    build_dataset_from_local_parquet,
    temporal_symbol_split,
    undersample_no_trade,
)

# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Artifact directory
# ---------------------------------------------------------------------

ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# Research constants
# ---------------------------------------------------------------------

FRICTION_R = 0.12

# These are the Phase 2A features we are explicitly testing.
EXPERIMENTAL_FEATURES = {
    "fundingRate",
    "btc_corr_20",
    "btc_beta_20",
    "btc_rel_strength",
}

# Build baseline strictly from canonical ALL_FEATURES while excluding
# every Phase 2A experimental feature.
BASE_FEATURES = [
    f for f in ALL_FEATURES if f not in EXPERIMENTAL_FEATURES
]

# Essential features preserved by the feature-selection stage.
ESSENTIAL_FEATURES = [
    "volume_ratio",
    "volume_spike",
    "obv_slope",
    "bb_width",
    "atr_pct",
    "volatility",
    "vwap_dev",
]

# ---------------------------------------------------------------------
# Scikit-learn / XGBoost compatibility patch
# ---------------------------------------------------------------------

# Some sklearn versions rely on estimator tags during calibration /
# meta-estimator validation. This keeps XGBClassifier compatible with
# the environment used by the workflow.

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


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _safe_std(series: pd.Series) -> float:
    """Return finite standard deviation or 0.0."""
    s = pd.to_numeric(series, errors="coerce").dropna()

    if len(s) == 0:
        return 0.0

    value = float(s.std())

    if not np.isfinite(value):
        return 0.0

    return value


def _validate_required_columns(ds: pd.DataFrame) -> None:
    """Fail closed if canonical dataset is missing mandatory columns."""

    required = {
        "symbol",
        "close_time",
        "target",
    }

    missing = sorted(required - set(ds.columns))

    if missing:
        raise ValueError(
            f"FATAL: Canonical dataset is missing mandatory columns: {missing}"
        )


def validate_funding_integrity(ds: pd.DataFrame) -> None:
    """
    Strict per-symbol funding validation.

    We intentionally do not accept:
      - missing funding column
      - all-NaN funding
      - insufficient funding observations
      - zero variance funding
      - missing observation timestamps
      - non-finite funding values

    This prevents a broken funding archive from masquerading as
    a legitimate negative experiment result.
    """

    log.info("Validating funding integrity per symbol...")

    if "fundingRate" not in ds.columns:
        raise ValueError(
            "FATAL: 'fundingRate' column is completely missing "
            "from the canonical dataset."
        )

    _validate_required_columns(ds)

    symbols = sorted(ds["symbol"].dropna().astype(str).unique())

    if not symbols:
        raise ValueError(
            "FATAL: No symbols found in canonical dataset."
        )

    for sym, grp in ds.groupby("symbol", sort=True):

        if grp.empty:
            raise ValueError(
                f"FATAL: Symbol {sym} has zero rows."
            )

        # Observation timestamps must exist.
        if grp["close_time"].isna().any():
            raise ValueError(
                f"FATAL: Symbol {sym} has missing close_time values."
            )

        # Observation timestamps must be finite.
        close_numeric = pd.to_numeric(
            grp["close_time"], errors="coerce"
        )

        if not np.isfinite(close_numeric.to_numpy()).all():
            raise ValueError(
                f"FATAL: Symbol {sym} has non-finite close_time values."
            )

        # Funding values.
        fr = pd.to_numeric(
            grp["fundingRate"], errors="coerce"
        ).dropna()

        if len(fr) == 0:
            raise ValueError(
                f"FATAL: Symbol {sym} has no non-null funding observations."
            )

        if len(fr) < 100:
            raise ValueError(
                f"FATAL: Symbol {sym} has insufficient funding observations "
                f"(N={len(fr)})."
            )

        if not np.isfinite(fr.to_numpy()).all():
            raise ValueError(
                f"FATAL: Symbol {sym} contains non-finite funding values."
            )

        std = _safe_std(fr)

        if std == 0.0:
            raise ValueError(
                f"FATAL: Symbol {sym} has zero-variance funding. "
                "Phase 2A would be compromised."
            )

        # Funding column must contain more than one distinct value.
        unique_values = fr.nunique(dropna=True)

        if unique_values < 2:
            raise ValueError(
                f"FATAL: Symbol {sym} has fewer than 2 unique funding values."
            )

        # We also verify the full observation range is sensible.
        min_close = float(close_numeric.min())
        max_close = float(close_numeric.max())

        if max_close <= min_close:
            raise ValueError(
                f"FATAL: Symbol {sym} has invalid close_time range."
            )

        # Funding must exist at least somewhere across the observation
        # interval. Exact PIT semantics are inherited from the canonical
        # data-building pipeline.
        funding_mask = grp["fundingRate"].notna()

        if not funding_mask.any():
            raise ValueError(
                f"FATAL: Symbol {sym} has no usable funding coverage."
            )

        log.info(
            "  ✓ %-12s | rows=%6d | funding_N=%6d | "
            "unique=%4d | std=%.6g",
            sym,
            len(grp),
            len(fr),
            unique_values,
            std,
        )

    log.info(
        "✓ Funding integrity validated across %d active symbols.",
        len(symbols),
    )


def validate_experiment_features(
    ds: pd.DataFrame,
    experiment_grid: list[dict],
) -> None:
    """
    Ensure every feature used by any experiment actually exists.

    IMPORTANT:
    We deliberately do NOT create a missing feature with zeros.
    A missing feature is a fatal integrity error.
    """

    all_required = set()

    for cfg in experiment_grid:
        all_required.update(cfg["features"])

    missing = sorted(
        f for f in all_required
        if f not in ds.columns
    )

    if missing:
        raise ValueError(
            "FATAL: Required experiment features are missing from "
            f"the canonical dataset: {missing}"
        )

    # Explicit non-finite checks for experimental features.
    for feature in sorted(EXPERIMENTAL_FEATURES):
        if feature not in ds.columns:
            continue

        vals = pd.to_numeric(
            ds[feature], errors="coerce"
        )

        finite_count = np.isfinite(vals.fillna(0.0).to_numpy()).sum()

        if finite_count == 0:
            raise ValueError(
                f"FATAL: Experimental feature '{feature}' contains no "
                "finite numeric information."
            )

    log.info(
        "✓ All experiment features are present in canonical dataset."
    )


def _validate_feature_matrix(
    df: pd.DataFrame,
    features: list[str],
    label: str,
) -> pd.DataFrame:
    """
    Build a clean feature matrix.

    NaNs/infinities inside an existing feature are handled consistently
    with the existing training convention, but missing columns are fatal.
    """

    missing = [
        f for f in features
        if f not in df.columns
    ]

    if missing:
        raise ValueError(
            f"FATAL: {label} is missing required features: {missing}"
        )

    X = df[features].copy()

    X = X.replace(
        [np.inf, -np.inf],
        np.nan,
    ).fillna(0.0)

    values = X.to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError(
            f"FATAL: {label} still contains non-finite values "
            "after cleaning."
        )

    return X


def _class_indices(
    le: LabelEncoder,
) -> tuple[int, int, int]:
    """
    Resolve class indices robustly.
    """

    classes = list(le.classes_)

    if "NO_TRADE" not in classes:
        raise ValueError(
            "FATAL: target does not contain NO_TRADE class."
        )

    if "BUY" not in classes:
        raise ValueError(
            "FATAL: target does not contain BUY class."
        )

    if "SELL" not in classes:
        raise ValueError(
            "FATAL: target does not contain SELL class."
        )

    nt_idx = classes.index("NO_TRADE")
    buy_idx = classes.index("BUY")
    sell_idx = classes.index("SELL")

    return nt_idx, buy_idx, sell_idx


# ---------------------------------------------------------------------
# Single ablation run
# ---------------------------------------------------------------------

def run_single_ablation(
    ds: pd.DataFrame,
    feature_set: list[str],
    reserved_features: list[str],
) -> dict:
    """
    Train one isolated Phase 2A variant.

    Returns detailed side-specific calibration metrics and locked
    chronological test accuracy.
    """

    ds_variant = ds.loc[
        :,
        ~ds.columns.duplicated()
    ].copy()

    active_feats = list(
        dict.fromkeys(feature_set)
    )

    if not active_feats:
        raise ValueError(
            "FATAL: Empty feature set supplied to ablation."
        )

    # ---------------------------------------------------------------
    # FAIL-CLOSED feature existence check
    # ---------------------------------------------------------------

    missing = [
        f for f in active_feats
        if f not in ds_variant.columns
    ]

    if missing:
        raise ValueError(
            "FATAL: Variant requested features that do not exist: "
            f"{missing}"
        )

    # ---------------------------------------------------------------
    # Target encoder
    # ---------------------------------------------------------------

    le = LabelEncoder()
    le.fit(ds_variant["target"])

    nt_idx, buy_idx, sell_idx = _class_indices(le)

    # ---------------------------------------------------------------
    # Chronological train/calibration/test split
    # ---------------------------------------------------------------

    train_df, calib_df, test_df = temporal_symbol_split(
        ds_variant,
        TEST_SPLIT,
        CALIB_SPLIT,
        EMBARGO_BARS,
    )

    if len(train_df) == 0:
        raise ValueError(
            "FATAL: Empty training split."
        )

    if len(calib_df) == 0:
        raise ValueError(
            "FATAL: Empty calibration split."
        )

    if len(test_df) == 0:
        raise ValueError(
            "FATAL: Empty test split."
        )

    # ---------------------------------------------------------------
    # Feature matrices
    # ---------------------------------------------------------------

    X_train_raw = _validate_feature_matrix(
        train_df,
        active_feats,
        "Training matrix",
    )

    X_calib = _validate_feature_matrix(
        calib_df,
        active_feats,
        "Calibration matrix",
    )

    X_test = _validate_feature_matrix(
        test_df,
        active_feats,
        "Test matrix",
    )

    y_train_raw = le.transform(
        train_df["target"]
    )

    y_calib = le.transform(
        calib_df["target"]
    )

    y_test = le.transform(
        test_df["target"]
    )

    # ---------------------------------------------------------------
    # Feature selection
    # ---------------------------------------------------------------

    scanner = XGBClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        eval_metric="mlogloss",
    )

    scanner.fit(
        X_train_raw,
        y_train_raw,
    )

    top_idx = np.argsort(
        scanner.feature_importances_
    )[::-1]

    selected = [
        f for f in ESSENTIAL_FEATURES
        if f in active_feats
    ]

    # Reserve Phase 2 experimental features from being accidentally
    # removed by the feature cutoff.
    selected.extend(
        f for f in reserved_features
        if f in active_feats
    )

    selected = list(
        dict.fromkeys(selected)
    )

    for idx in top_idx:
        feat = active_feats[idx]

        if feat not in selected:
            selected.append(feat)

        if len(selected) >= N_FEATURES:
            break

    # If the feature set has fewer than N_FEATURES, that's okay.
    selected = selected[:max(
        len(selected),
        min(N_FEATURES, len(active_feats))
    )]

    # ---------------------------------------------------------------
    # Reserved-feature invariant
    # ---------------------------------------------------------------

    missing_reserved = [
        f for f in reserved_features
        if f not in selected
    ]

    if missing_reserved:
        raise ValueError(
            "FATAL: Variant collapsed. Reserved experiment features "
            f"were removed by selector: {missing_reserved}"
        )

    if not selected:
        raise ValueError(
            "FATAL: Feature selector produced zero selected features."
        )

    log.info(
        "Selected %d / %d features.",
        len(selected),
        len(active_feats),
    )

    if reserved_features:
        log.info(
            "Reserved features retained: %s",
            reserved_features,
        )

    # ---------------------------------------------------------------
    # Apply same no-trade undersampling logic
    # ---------------------------------------------------------------

    X_train_selected = X_train_raw[selected]

    X_train_sel, y_train = undersample_no_trade(
        X_train_selected,
        y_train_raw,
        nt_idx,
    )

    Xtr = X_train_sel.values
    Xcal = X_calib[selected].values
    Xte = X_test[selected].values

    if len(Xtr) == 0:
        raise ValueError(
            "FATAL: Training matrix became empty after undersampling."
        )

    # ---------------------------------------------------------------
    # Asymmetric sample weights
    # ---------------------------------------------------------------

    sw_asym = np.ones(
        len(y_train),
        dtype=float,
    )

    sw_asym[
        y_train == buy_idx
    ] = 2.0

    sw_asym[
        y_train == sell_idx
    ] = 2.0

    # ---------------------------------------------------------------
    # Base models
    # ---------------------------------------------------------------

    xgb = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.03,
        subsample=0.85,
        random_state=42,
        n_jobs=-1,
        eval_metric="mlogloss",
    )

    xgb.fit(
        Xtr,
        y_train,
        sample_weight=sw_asym,
    )

    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1,
        class_weight={
            nt_idx: 1.0,
            buy_idx: 2.0,
            sell_idx: 2.0,
        },
    )

    rf.fit(
        Xtr,
        y_train,
    )

    gb = HistGradientBoostingClassifier(
        max_iter=150,
        max_depth=5,
        learning_rate=0.04,
        random_state=42,
        class_weight={
            nt_idx: 1.0,
            buy_idx: 2.0,
            sell_idx: 2.0,
        },
    )

    gb.fit(
        Xtr,
        y_train,
    )

    # ---------------------------------------------------------------
    # Soft voting ensemble
    # ---------------------------------------------------------------

    ensemble = VotingClassifier(
        estimators=[
            ("xgb", xgb),
            ("rf", rf),
            ("gb", gb),
        ],
        voting="soft",
        weights=[3, 2, 1],
    )

    ensemble.fit(
        Xtr,
        y_train,
    )

    # ---------------------------------------------------------------
    # Isotonic calibration on calibration split
    # ---------------------------------------------------------------

    calibrated_ensemble = CalibratedClassifierCV(
        estimator=FrozenEstimator(ensemble),
        method="isotonic",
    )

    calibrated_ensemble.fit(
        Xcal,
        y_calib,
    )

    calib_probas = calibrated_ensemble.predict_proba(
        Xcal
    )

    # ---------------------------------------------------------------
    # Threshold search
    # ---------------------------------------------------------------

    best_thresh_buy = float(
        MIN_BUY_THRESHOLD_FLOOR
    )

    best_thresh_sell = float(
        MIN_SELL_THRESHOLD_FLOOR
    )

    best_score_buy = 0.0
    best_score_sell = 0.0

    best_buy_metrics = {
        "n": 0,
        "prec": 0.0,
        "ev": 0.0,
        "recall": 0.0,
    }

    best_sell_metrics = {
        "n": 0,
        "prec": 0.0,
        "ev": 0.0,
        "recall": 0.0,
    }

    thresholds_tested = [
        0.35,
        0.40,
        0.45,
        0.50,
        0.55,
        0.60,
        0.65,
        0.70,
    ]

    for thresh in thresholds_tested:

        y_pred_calib = []

        for p in calib_probas:

            pred_class = int(
                np.argmax(p)
            )

            if (
                pred_class != nt_idx
                and p[pred_class] >= thresh
            ):
                y_pred_calib.append(
                    pred_class
                )
            else:
                y_pred_calib.append(
                    nt_idx
                )

        y_pred_calib = np.asarray(
            y_pred_calib,
            dtype=int,
        )

        buy_mask = (
            y_pred_calib == buy_idx
        )

        sell_mask = (
            y_pred_calib == sell_idx
        )

        # -----------------------------------------------------------
        # Precision
        # -----------------------------------------------------------

        buy_n = int(
            buy_mask.sum()
        )

        sell_n = int(
            sell_mask.sum()
        )

        buy_prec = (
            float(
                (
                    y_calib[buy_mask]
                    == buy_idx
                ).mean()
            )
            if buy_n > 0
            else 0.0
        )

        sell_prec = (
            float(
                (
                    y_calib[sell_mask]
                    == sell_idx
                ).mean()
            )
            if sell_n > 0
            else 0.0
        )

        # -----------------------------------------------------------
        # Recall
        # -----------------------------------------------------------

        true_buy_n = int(
            (y_calib == buy_idx).sum()
        )

        true_sell_n = int(
            (y_calib == sell_idx).sum()
        )

        buy_recall = (
            float(
                (
                    y_pred_calib[y_calib == buy_idx]
                    == buy_idx
                ).mean()
            )
            if true_buy_n > 0
            else 0.0
        )

        sell_recall = (
            float(
                (
                    y_pred_calib[y_calib == sell_idx]
                    == sell_idx
                ).mean()
            )
            if true_sell_n > 0
            else 0.0
        )

        # -----------------------------------------------------------
        # Fixed economic geometry
        # -----------------------------------------------------------

        buy_ev = (
            buy_prec * ATR_TARGET1_MULT
            - (1.0 - buy_prec) * ATR_STOP_MULT
            - FRICTION_R
        )

        sell_ev = (
            sell_prec * ATR_TARGET1_MULT
            - (1.0 - sell_prec) * ATR_STOP_MULT
            - FRICTION_R
        )

        # -----------------------------------------------------------
        # Direction-specific scores
        #
        # Score rewards:
        #   positive EV
        #   precision
        #   recall
        #   sample volume
        #
        # We keep the same structural style used in the earlier
        # research pipeline.
        # -----------------------------------------------------------

        buy_score = (
            buy_ev
            * buy_recall
            * np.sqrt(max(buy_n, 1))
            if buy_ev > 0.0
            else 0.0
        )

        sell_score = (
            sell_ev
            * sell_recall
            * np.sqrt(max(sell_n, 1))
            if sell_ev > 0.0
            else 0.0
        )

        # -----------------------------------------------------------
        # BUY threshold selection
        # -----------------------------------------------------------

        if (
            thresh >= MIN_BUY_THRESHOLD_FLOOR
            and buy_n > 15
            and buy_score > best_score_buy
        ):
            best_score_buy = float(
                buy_score
            )

            best_thresh_buy = float(
                thresh
            )

            best_buy_metrics = {
                "n": buy_n,
                "prec": float(buy_prec),
                "ev": float(buy_ev),
                "recall": float(buy_recall),
            }

        # -----------------------------------------------------------
        # SELL threshold selection
        # -----------------------------------------------------------

        if (
            thresh >= MIN_SELL_THRESHOLD_FLOOR
            and sell_n > 15
            and sell_score > best_score_sell
        ):
            best_score_sell = float(
                sell_score
            )

            best_thresh_sell = float(
                thresh
            )

            best_sell_metrics = {
                "n": sell_n,
                "prec": float(sell_prec),
                "ev": float(sell_ev),
                "recall": float(sell_recall),
            }

    # ---------------------------------------------------------------
    # FAIL-CLOSED threshold lock
    #
    # No positive calibration evidence means:
    #   threshold = 1.01
    #
    # Since model probabilities are <= 1.0, this effectively disables
    # that direction.
    # ---------------------------------------------------------------

    final_thresh_buy = (
        max(
            MIN_BUY_THRESHOLD_FLOOR,
            best_thresh_buy,
        )
        if best_score_buy > 0.0
        else 1.01
    )

    final_thresh_sell = (
        max(
            MIN_SELL_THRESHOLD_FLOOR,
            best_thresh_sell,
        )
        if best_score_sell > 0.0
        else 1.01
    )

    # ---------------------------------------------------------------
    # Locked test evaluation
    # ---------------------------------------------------------------

    test_probas = calibrated_ensemble.predict_proba(
        Xte
    )

    y_pred_test = []

    for p in test_probas:

        pred_class = int(
            np.argmax(p)
        )

        if (
            pred_class == buy_idx
            and p[pred_class] >= final_thresh_buy
        ):
            y_pred_test.append(
                buy_idx
            )

        elif (
            pred_class == sell_idx
            and p[pred_class] >= final_thresh_sell
        ):
            y_pred_test.append(
                sell_idx
            )

        else:
            y_pred_test.append(
                nt_idx
            )

    y_pred_test = np.asarray(
        y_pred_test,
        dtype=int,
    )

    acc = float(
        accuracy_score(
            y_test,
            y_pred_test,
        )
    )

    # ---------------------------------------------------------------
    # Additional locked-test side counts
    # ---------------------------------------------------------------

    test_buy_n = int(
        (y_pred_test == buy_idx).sum()
    )

    test_sell_n = int(
        (y_pred_test == sell_idx).sum()
    )

    test_no_trade_n = int(
        (y_pred_test == nt_idx).sum()
    )

    # ---------------------------------------------------------------
    # Research diagnostics
    # ---------------------------------------------------------------

    log.info(
        "Variant complete | total_features=%d | selected=%d | "
        "BUY(th=%.2f,N=%d,Prec=%.1f%%,EV=%.2fR,Score=%.3f) | "
        "SELL(th=%.2f,N=%d,Prec=%.1f%%,EV=%.2fR,Score=%.3f) | "
        "TestAcc=%.2f%%",
        len(active_feats),
        len(selected),
        final_thresh_buy,
        best_buy_metrics["n"],
        best_buy_metrics["prec"] * 100.0,
        best_buy_metrics["ev"],
        best_score_buy,
        final_thresh_sell,
        best_sell_metrics["n"],
        best_sell_metrics["prec"] * 100.0,
        best_sell_metrics["ev"],
        best_score_sell,
        acc * 100.0,
    )

    return {
        "Accuracy": float(acc),

        "Total_Features": int(
            len(active_feats)
        ),

        "Selected_Features": int(
            len(selected)
        ),

        "Selected_Feature_Names": list(
            selected
        ),

        "BUY_Thresh": float(
            final_thresh_buy
        ),

        "BUY_N": int(
            best_buy_metrics["n"]
        ),

        "BUY_Prec": float(
            best_buy_metrics["prec"]
        ),

        "BUY_EV": float(
            best_buy_metrics["ev"]
        ),

        "BUY_Recall": float(
            best_buy_metrics["recall"]
        ),

        "BUY_Score": float(
            best_score_buy
        ),

        "SELL_Thresh": float(
            final_thresh_sell
        ),

        "SELL_N": int(
            best_sell_metrics["n"]
        ),

        "SELL_Prec": float(
            best_sell_metrics["prec"]
        ),

        "SELL_EV": float(
            best_sell_metrics["ev"]
        ),

        "SELL_Recall": float(
            best_sell_metrics["recall"]
        ),

        "SELL_Score": float(
            best_score_sell
        ),

        "Test_BUY_N": int(
            test_buy_n
        ),

        "Test_SELL_N": int(
            test_sell_n
        ),

        "Test_NO_TRADE_N": int(
            test_no_trade_n
        ),

        "Train_Rows": int(
            len(train_df)
        ),

        "Calibration_Rows": int(
            len(calib_df)
        ),

        "Test_Rows": int(
            len(test_df)
        ),

        "Friction_R": float(
            FRICTION_R
        ),

        "TP_R": float(
            ATR_TARGET1_MULT
        ),

        "SL_R": float(
            ATR_STOP_MULT
        ),
    }


# ---------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------

def run_feature_ablation():
    t0 = time.time()

    log.info("=" * 100)
    log.info("PHASE 2A — RAW FUNDING & BTC FEATURE ABLATION")
    log.info("=" * 100)

    # ---------------------------------------------------------------
    # Load canonical datasets
    # ---------------------------------------------------------------

    log.info(
        "Loading canonical Parquet dataset across all active symbols..."
    )

    ds = build_dataset_from_local_parquet()

    if ds is None or ds.empty:
        raise ValueError(
            "FATAL: Canonical dataset is empty."
        )

    log.info(
        "Canonical dataset loaded: rows=%d | columns=%d | symbols=%d",
        len(ds),
        len(ds.columns),
        ds["symbol"].nunique()
        if "symbol" in ds.columns
        else 0,
    )

    # ---------------------------------------------------------------
    # Define experiment grid
    # ---------------------------------------------------------------

    experiment_grid = [
        {
            "name": "Baseline",
            "features": BASE_FEATURES,
        },
        {
            "name": "Baseline + Funding",
            "features": BASE_FEATURES + [
                "fundingRate",
            ],
        },
        {
            "name": "Baseline + BTC Metrics",
            "features": BASE_FEATURES + [
                "btc_corr_20",
                "btc_beta_20",
                "btc_rel_strength",
            ],
        },
        {
            "name": "Kitchen Sink",
            "features": BASE_FEATURES + [
                "fundingRate",
                "btc_corr_20",
                "btc_beta_20",
                "btc_rel_strength",
            ],
        },
    ]

    # ---------------------------------------------------------------
    # Integrity checks
    # ---------------------------------------------------------------

    validate_funding_integrity(ds)

    validate_experiment_features(
        ds,
        experiment_grid,
    )

    # ---------------------------------------------------------------
    # Manifest
    # ---------------------------------------------------------------

    log.info("")
    log.info("=" * 100)
    log.info("PHASE 2A EXPERIMENT MANIFEST")
    log.info("=" * 100)

    log.info(
        "Target geometry : TP=%.2fR / SL=%.2fR",
        ATR_TARGET1_MULT,
        ATR_STOP_MULT,
    )

    log.info(
        "Friction         : %.2fR (FROZEN)",
        FRICTION_R,
    )

    log.info(
        "Chronological lock: inherited from train_model.py"
    )

    log.info(
        "Embargo bars     : %d",
        EMBARGO_BARS,
    )

    log.info(
        "N feature cap    : %d",
        N_FEATURES,
    )

    log.info(
        "Baseline features: %d",
        len(BASE_FEATURES),
    )

    for cfg in experiment_grid:

        extras = [
            f
            for f in cfg["features"]
            if f not in BASE_FEATURES
        ]

        log.info(
            "Config: %-28s | Pool=%3d | Extras=%s",
            cfg["name"],
            len(cfg["features"]),
            extras,
        )

    log.info("=" * 100)
    log.info("")

    # ---------------------------------------------------------------
    # Run variants
    # ---------------------------------------------------------------

    results = []

    for index, config in enumerate(
        experiment_grid,
        start=1,
    ):

        log.info("")
        log.info("=" * 100)

        log.info(
            "🚀 TRAINING VARIANT %d/%d: %s",
            index,
            len(experiment_grid),
            config["name"],
        )

        log.info("=" * 100)

        reserved = [
            f
            for f in config["features"]
            if f not in BASE_FEATURES
        ]

        metrics = run_single_ablation(
            ds=ds,
            feature_set=config["features"],
            reserved_features=reserved,
        )

        results.append(
            {
                "Config": config["name"],
                **metrics,
            }
        )

    # ---------------------------------------------------------------
    # Final ASCII summary
    # ---------------------------------------------------------------

    log.info("")
    log.info("=" * 150)
    log.info("🎯 PHASE 2A EXPERIMENT RESULTS")
    log.info("=" * 150)

    header = (
        f"{'Variant':<28} | "
        f"{'Feats':>5} | "
        f"{'Acc':>7} | "
        f"{'B-Th':>5} | "
        f"{'B-N':>5} | "
        f"{'B-Prec':>7} | "
        f"{'B-EV':>7} | "
        f"{'B-Score':>8} | "
        f"{'S-Th':>5} | "
        f"{'S-N':>5} | "
        f"{'S-Prec':>7} | "
        f"{'S-EV':>7} | "
        f"{'S-Score':>8}"
    )

    log.info(header)
    log.info("-" * 150)

    for r in results:

        log.info(
            f"{r['Config']:<28} | "
            f"{r['Selected_Features']:>5} | "
            f"{r['Accuracy'] * 100:>6.1f}% | "
            f"{r['BUY_Thresh']:>5.2f} | "
            f"{r['BUY_N']:>5} | "
            f"{r['BUY_Prec'] * 100:>6.1f}% | "
            f"{r['BUY_EV']:>6.2f}R | "
            f"{r['BUY_Score']:>8.3f} | "
            f"{r['SELL_Thresh']:>5.2f} | "
            f"{r['SELL_N']:>5} | "
            f"{r['SELL_Prec'] * 100:>6.1f}% | "
            f"{r['SELL_EV']:>6.2f}R | "
            f"{r['SELL_Score']:>8.3f}"
        )

    log.info("=" * 150)

    # ---------------------------------------------------------------
    # Explicit SELL comparison
    # ---------------------------------------------------------------

    log.info("")
    log.info("=" * 100)
    log.info("SELL-SIDE EVIDENCE")
    log.info("=" * 100)

    for r in results:

        log.info(
            "%-28s | Threshold=%5.2f | N=%5d | "
            "Precision=%6.1f%% | EV=%6.2fR | Score=%7.3f",
            r["Config"],
            r["SELL_Thresh"],
            r["SELL_N"],
            r["SELL_Prec"] * 100.0,
            r["SELL_EV"],
            r["SELL_Score"],
        )

    log.info("=" * 100)

    # ---------------------------------------------------------------
    # Save structured artifacts
    # ---------------------------------------------------------------

    csv_path = (
        ARTIFACTS_DIR
        / "phase2a_ablation.csv"
    )

    json_path = (
        ARTIFACTS_DIR
        / "phase2a_ablation.json"
    )

    manifest_path = (
        ARTIFACTS_DIR
        / "phase2a_ablation_manifest.json"
    )

    # CSV cannot cleanly represent selected-feature arrays, so create
    # a flat copy for the table.
    csv_rows = []

    for r in results:

        row = dict(r)

        row["Selected_Feature_Names"] = (
            ",".join(
                row["Selected_Feature_Names"]
            )
        )

        csv_rows.append(row)

    pd.DataFrame(
        csv_rows
    ).to_csv(
        csv_path,
        index=False,
    )

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            results,
            f,
            indent=2,
        )

    manifest = {
        "experiment": "Phase 2A Raw Funding & BTC Feature Ablation",
        "target_geometry": {
            "tp_r": float(ATR_TARGET1_MULT),
            "sl_r": float(ATR_STOP_MULT),
        },
        "friction_r": float(FRICTION_R),
        "chronological_lock": True,
        "embargo_bars": int(EMBARGO_BARS),
        "n_feature_cap": int(N_FEATURES),
        "baseline_feature_count": int(
            len(BASE_FEATURES)
        ),
        "experimental_features": sorted(
            EXPERIMENTAL_FEATURES
        ),
        "variants": [
            {
                "name": cfg["name"],
                "feature_count": len(
                    cfg["features"]
                ),
                "features": cfg["features"],
            }
            for cfg in experiment_grid
        ],
        "dataset_rows": int(len(ds)),
        "dataset_symbols": int(
            ds["symbol"].nunique()
        ),
    }

    with open(
        manifest_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            manifest,
            f,
            indent=2,
        )

    elapsed_min = (
        time.time() - t0
    ) / 60.0

    log.info("")
    log.info("=" * 100)
    log.info(
        "✅ PHASE 2A COMPLETE in %.1f minutes",
        elapsed_min,
    )

    log.info(
        "CSV artifact     : %s",
        csv_path,
    )

    log.info(
        "JSON artifact    : %s",
        json_path,
    )

    log.info(
        "Manifest artifact: %s",
        manifest_path,
    )

    log.info("=" * 100)


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------

if __name__ == "__main__":
    run_feature_ablation()
