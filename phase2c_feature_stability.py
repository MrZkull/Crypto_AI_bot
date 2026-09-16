#!/usr/bin/env python3
"""
phase2c_feature_stability.py

Phase 2C — Multi-Window Feature Stability Test

Purpose:
    Test whether the Phase 2B feature-group effects survive across
    multiple chronological out-of-sample windows.

Frozen protocol:
    - Canonical local Parquet dataset
    - TP=3.5R / SL=2.5R
    - Friction=0.12R
    - 24-bar embargo
    - 30-feature baseline
    - Same Phase 2A model family and calibration
    - Thresholds selected ONLY on each window's calibration split
    - No production/candidate model export

This is research-only.
"""

import json
import logging
import math
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
    BASE_FEATURES,
    EMBARGO_BARS,
    N_FEATURES,
    build_dataset_from_local_parquet,
    undersample_no_trade,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

FRICTION_R = 0.12
TP_R = 3.5
SL_R = 2.5
WINDOW_FRACTION = 0.10

BTC_FEATURES = {
    "btc_corr_20",
    "btc_beta_20",
    "btc_rel_strength",
}

GROUPS = {
    "Momentum & Trend": [
        "rsi", "macd", "macd_signal", "macd_hist", "adx",
        "plus_di", "minus_di", "trend", "ema20_vs_ema50",
        "price_vs_ema200", "regime_uptrend",
    ],
    "Volatility & Bands": [
        "atr", "atr_pct", "bb_width", "bb_pos", "volatility",
    ],
    "Volume Dynamics": [
        "volume_ratio", "volume_spike", "obv_slope",
        "vwap_dev", "taker_buy_ratio",
    ],
    "Time Cycles": [
        "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    ],
    "Higher Timeframe Context": [
        "rsi_1h", "adx_1h", "trend_1h", "rsi_4h", "trend_4h",
    ],
}

VARIANTS = {
    "Baseline": list(BASE_FEATURES),
    **{
        f"Minus {name}": [
            f for f in BASE_FEATURES if f not in features
        ]
        for name, features in GROUPS.items()
    },
    "Minus Momentum & Trend + Volume Dynamics": [
        f for f in BASE_FEATURES
        if f not in set(GROUPS["Momentum & Trend"])
        and f not in set(GROUPS["Volume Dynamics"])
    ],
}

ESSENTIAL = [
    "volume_ratio",
    "volume_spike",
    "obv_slope",
    "bb_width",
    "atr_pct",
    "volatility",
    "vwap_dev",
]

THRESHOLDS = [
    0.35, 0.40, 0.45, 0.50,
    0.55, 0.60, 0.65, 0.70,
]


def validate_schema(ds: pd.DataFrame) -> None:
    canonical = set(ALL_FEATURES)
    if set(BASE_FEATURES) - canonical:
        raise ValueError("FATAL: BASE_FEATURES contains non-canonical features.")

    expected = canonical - BTC_FEATURES
    if set(BASE_FEATURES) != expected:
        raise ValueError(
            "FATAL: Phase 2C baseline does not match Phase 2A's "
            "30-feature baseline."
        )

    for name, features in GROUPS.items():
        if not set(features) <= set(BASE_FEATURES):
            raise ValueError(
                f"FATAL: Phase 2C group '{name}' is outside baseline."
            )

    if ds.empty:
        raise ValueError("FATAL: Dataset is empty.")

    required = {"symbol", "open_time", "target"}
    missing = sorted(required - set(ds.columns))
    if missing:
        raise ValueError(f"FATAL: Dataset missing columns: {missing}")


def rolling_split(
    ds: pd.DataFrame,
    test_block: int,
) -> list[tuple[str, pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """
    Four expanding/rolling chronological windows.

    For each symbol:
      train -> embargo -> calibration -> embargo -> test

    Test blocks are placed at approximately 60%, 70%, 80%, and 90%
    of each symbol's chronological history.
    """
    origins = [0.60, 0.70, 0.80, 0.90]
    windows = []

    for wno, origin in enumerate(origins, 1):
        train_parts, calib_parts, test_parts = [], [], []

        for _, grp in ds.groupby("symbol", sort=False):
            grp = grp.sort_values("open_time").reset_index(drop=True)
            n = len(grp)
            test_start = int(n * origin)
            calib_end = test_start - EMBARGO_BARS
            calib_start = calib_end - test_block
            train_end = calib_start - EMBARGO_BARS

            if train_end <= 100 or calib_start < 0 or test_start >= n:
                continue

            train_parts.append(grp.iloc[:train_end])
            calib_parts.append(grp.iloc[calib_start:calib_end])
            test_parts.append(grp.iloc[test_start:test_start + test_block])

        train = pd.concat(train_parts, ignore_index=True)
        calib = pd.concat(calib_parts, ignore_index=True)
        test = pd.concat(test_parts, ignore_index=True)

        windows.append(
            (f"W{wno}", train, calib, test)
        )

    return windows


def feature_matrix(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise ValueError(f"FATAL: Missing features: {missing}")

    return (
        df[features]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )


def score_side(precision, recall, n, tp=TP_R, sl=SL_R):
    ev = precision * tp - (1.0 - precision) * sl - FRICTION_R
    score = (
        ev * recall * math.sqrt(max(n, 1))
        if ev > 0 else 0.0
    )
    return ev, score


def run_window(
    train: pd.DataFrame,
    calib: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
) -> dict:
    le = LabelEncoder()
    le.fit(train["target"])

    classes = list(le.classes_)
    required = {"NO_TRADE", "BUY", "SELL"}
    if not required <= set(classes):
        raise ValueError(
            f"FATAL: Window missing target class: {required - set(classes)}"
        )

    nt = classes.index("NO_TRADE")
    buy = classes.index("BUY")
    sell = classes.index("SELL")

    Xtr_raw = feature_matrix(train, features)
    Xcal = feature_matrix(calib, features)
    Xte = feature_matrix(test, features)

    ytr_raw = le.transform(train["target"])
    ycal = le.transform(calib["target"])
    yte = le.transform(test["target"])

    scanner = XGBClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        eval_metric="mlogloss",
    )
    scanner.fit(Xtr_raw, ytr_raw)

    selected = [
        f for f in ESSENTIAL
        if f in features
    ]

    importance_idx = np.argsort(
        scanner.feature_importances_
    )[::-1]

    for idx in importance_idx:
        feat = features[idx]
        if feat not in selected:
            selected.append(feat)
        if len(selected) >= min(N_FEATURES, len(features)):
            break

    selected = selected[:min(N_FEATURES, len(features))]

    Xtr_sel = Xtr_raw[selected]
    Xtr, ytr = undersample_no_trade(
        Xtr_sel,
        ytr_raw,
        nt,
    )

    xgb = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.03,
        subsample=0.85,
        random_state=42,
        n_jobs=-1,
        eval_metric="mlogloss",
    )

    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1,
        class_weight={nt: 1.0, buy: 2.0, sell: 2.0},
    )

    gb = HistGradientBoostingClassifier(
        max_iter=150,
        max_depth=5,
        learning_rate=0.04,
        random_state=42,
        class_weight={nt: 1.0, buy: 2.0, sell: 2.0},
    )

    xgb.fit(Xtr.values, ytr)
    rf.fit(Xtr.values, ytr)
    gb.fit(Xtr.values, ytr)

    ensemble = VotingClassifier(
        estimators=[
            ("xgb", xgb),
            ("rf", rf),
            ("gb", gb),
        ],
        voting="soft",
        weights=[3, 2, 1],
    )
    ensemble.fit(Xtr.values, ytr)

    calibrated = CalibratedClassifierCV(
        estimator=FrozenEstimator(ensemble),
        method="isotonic",
    )
    calibrated.fit(
        Xcal[selected].values,
        ycal,
    )

    pcal = calibrated.predict_proba(Xcal[selected].values)

    best = {
        "buy_score": 0.0,
        "sell_score": 0.0,
        "buy_thresh": 1.01,
        "sell_thresh": 1.01,
        "buy_n": 0,
        "sell_n": 0,
        "buy_prec": 0.0,
        "sell_prec": 0.0,
        "buy_ev": 0.0,
        "sell_ev": 0.0,
        "buy_recall": 0.0,
        "sell_recall": 0.0,
    }

    for threshold in THRESHOLDS:
        yp = []
        for p in pcal:
            if (
                p[buy] >= threshold
                and p[buy] > p[sell]
            ):
                yp.append(buy)
            elif (
                p[sell] >= threshold
                and p[sell] > p[buy]
            ):
                yp.append(sell)
            else:
                yp.append(nt)

        yp = np.asarray(yp)
        bm = yp == buy
        sm = yp == sell

        bn = int(bm.sum())
        sn = int(sm.sum())

        bp = (
            float((ycal[bm] == buy).mean())
            if bn else 0.0
        )
        sp = (
            float((ycal[sm] == sell).mean())
            if sn else 0.0
        )

        br = (
            float((yp[ycal == buy] == buy).mean())
            if (ycal == buy).sum() else 0.0
        )
        sr = (
            float((yp[ycal == sell] == sell).mean())
            if (ycal == sell).sum() else 0.0
        )

        bev, bs = score_side(bp, br, bn)
        sev, ss = score_side(sp, sr, sn)

        if bn > 15 and bs > best["buy_score"]:
            best.update(
                buy_score=float(bs),
                buy_thresh=float(threshold),
                buy_n=bn,
                buy_prec=bp,
                buy_ev=bev,
                buy_recall=br,
            )

        if sn > 15 and ss > best["sell_score"]:
            best.update(
                sell_score=float(ss),
                sell_thresh=float(threshold),
                sell_n=sn,
                sell_prec=sp,
                sell_ev=sev,
                sell_recall=sr,
            )

    ptest = calibrated.predict_proba(Xte[selected].values)
    ypred = []

    for p in ptest:
        if (
            p[buy] >= best["buy_thresh"]
            and p[buy] > p[sell]
        ):
            ypred.append(buy)
        elif (
            p[sell] >= best["sell_thresh"]
            and p[sell] > p[buy]
        ):
            ypred.append(sell)
        else:
            ypred.append(nt)

    accuracy = float(
        accuracy_score(yte, np.asarray(ypred))
    )

    return {
        "Accuracy": accuracy,
        "Selected": len(selected),
        "Selected_Features": selected,
        "BUY_Thresh": best["buy_thresh"],
        "BUY_N": best["buy_n"],
        "BUY_Prec": best["buy_prec"],
        "BUY_EV": best["buy_ev"],
        "BUY_Recall": best["buy_recall"],
        "BUY_Score": best["buy_score"],
        "SELL_Thresh": best["sell_thresh"],
        "SELL_N": best["sell_n"],
        "SELL_Prec": best["sell_prec"],
        "SELL_EV": best["sell_ev"],
        "SELL_Recall": best["sell_recall"],
        "SELL_Score": best["sell_score"],
        "Train_Rows": len(train),
        "Calibration_Rows": len(calib),
        "Test_Rows": len(test),
    }


def main():
    t0 = time.time()

    log.info("=" * 120)
    log.info("PHASE 2C — MULTI-WINDOW FEATURE STABILITY TEST")
    log.info("=" * 120)

    ds = build_dataset_from_local_parquet(
        sell_tp_mult=3.5,
        sell_sl_mult=2.5,
    )
    validate_schema(ds)

    log.info(
        "Dataset: rows=%d | symbols=%d",
        len(ds),
        ds["symbol"].nunique(),
    )

    test_block = min(
        int(len(g) * WINDOW_FRACTION)
        for _, g in ds.groupby("symbol")
    )

    if test_block < 100:
        raise ValueError(
            f"FATAL: Test block too small: {test_block}"
        )

    windows = rolling_split(ds, test_block)

    if len(windows) != 4:
        raise ValueError(
            f"FATAL: Expected 4 rolling windows, got {len(windows)}"
        )

    log.info(
        "Using %d chronological windows; block=%d rows/symbol",
        len(windows),
        test_block,
    )

    rows = []

    for variant, features in VARIANTS.items():
        log.info("")
        log.info("=" * 120)
        log.info("VARIANT: %s | Features=%d", variant, len(features))
        log.info("=" * 120)

        for window_name, train, calib, test in windows:
            result = run_window(
                train,
                calib,
                test,
                features,
            )

            row = {
                "Variant": variant,
                "Window": window_name,
                **result,
            }
            rows.append(row)

            log.info(
                "%s | Acc=%.2f%% | BUY th=%.2f N=%d "
                "Prec=%.1f%% EV=%.2fR Score=%.3f | "
                "SELL th=%.2f N=%d Prec=%.1f%% EV=%.2fR Score=%.3f",
                window_name,
                result["Accuracy"] * 100,
                result["BUY_Thresh"],
                result["BUY_N"],
                result["BUY_Prec"] * 100,
                result["BUY_EV"],
                result["BUY_Score"],
                result["SELL_Thresh"],
                result["SELL_N"],
                result["SELL_Prec"] * 100,
                result["SELL_EV"],
                result["SELL_Score"],
            )

    df = pd.DataFrame(rows)

    summary_rows = []
    for variant, grp in df.groupby("Variant", sort=False):
        summary_rows.append(
            {
                "Variant": variant,
                "Windows": len(grp),
                "Mean_Test_Accuracy": grp["Accuracy"].mean(),
                "Std_Test_Accuracy": grp["Accuracy"].std(ddof=0),
                "Mean_BUY_Score": grp["BUY_Score"].mean(),
                "Std_BUY_Score": grp["BUY_Score"].std(ddof=0),
                "Positive_BUY_Windows": int(
                    (grp["BUY_Score"] > 0).sum()
                ),
                "Mean_SELL_Score": grp["SELL_Score"].mean(),
                "Std_SELL_Score": grp["SELL_Score"].std(ddof=0),
                "Positive_SELL_Windows": int(
                    (grp["SELL_Score"] > 0).sum()
                ),
                "Mean_BUY_N": grp["BUY_N"].mean(),
                "Mean_SELL_N": grp["SELL_N"].mean(),
            }
        )

    summary = pd.DataFrame(summary_rows)

    log.info("")
    log.info("=" * 150)
    log.info("PHASE 2C STABILITY SUMMARY")
    log.info("=" * 150)
    log.info(
        "%-52s | Acc Mean | Acc SD | BUY Score Mean | BUY +Win | SELL Score Mean | SELL +Win",
        "Variant",
    )
    log.info("-" * 150)

    for _, r in summary.iterrows():
        log.info(
            "%-52s | %8.2f%% | %6.2f%% | %14.3f | %8d | %15.3f | %8d",
            r["Variant"],
            r["Mean_Test_Accuracy"] * 100,
            r["Std_Test_Accuracy"] * 100,
            r["Mean_BUY_Score"],
            r["Positive_BUY_Windows"],
            r["Mean_SELL_Score"],
            r["Positive_SELL_Windows"],
        )

    log.info("=" * 150)

    detail_path = ARTIFACTS_DIR / "phase2c_feature_stability_detail.csv"
    summary_path = ARTIFACTS_DIR / "phase2c_feature_stability_summary.csv"
    json_path = ARTIFACTS_DIR / "phase2c_feature_stability.json"

    detail = df.copy()
    detail["Selected_Features"] = detail[
        "Selected_Features"
    ].apply(lambda x: ",".join(x))

    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)

    payload = {
        "experiment": "Phase 2C Multi-Window Feature Stability Test",
        "frozen_geometry": {
            "tp_r": TP_R,
            "sl_r": SL_R,
            "friction_r": FRICTION_R,
            "embargo_bars": EMBARGO_BARS,
        },
        "windows": len(windows),
        "test_block_per_symbol": test_block,
        "results": json.loads(
            detail.to_json(orient="records")
        ),
        "summary": json.loads(
            summary.to_json(orient="records")
        ),
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    log.info("")
    log.info("PHASE 2C COMPLETE in %.1f minutes", (time.time() - t0) / 60)
    log.info("Detail CSV  : %s", detail_path)
    log.info("Summary CSV : %s", summary_path)
    log.info("JSON        : %s", json_path)


if __name__ == "__main__":
    main()
