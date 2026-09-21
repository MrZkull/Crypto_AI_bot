#!/usr/bin/env python3
# tests/test_calibration_integrity.py — Regression gate for temperature calibration vs ranking collapse

import numpy as np
import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import roc_auc_score


def test_temperature_scaling_preserves_directional_auc():
    """
    Regression Test: Ensures temperature scaling preserves directional
    separation (ROC-AUC) and does not collapse minority class rankings.
    """
    rng = np.random.default_rng(42)
    n_samples = 1200
    n_features = 10

    X = rng.normal(size=(n_samples, n_features))
    
    # Synthetic 3-class target: 0: NO_TRADE, 1: BUY, 2: SELL
    logits_buy = 0.8 * X[:, 0] - 0.5 * X[:, 1]
    logits_sell = -0.7 * X[:, 0] + 0.6 * X[:, 1]
    
    y = np.zeros(n_samples, dtype=int)
    y[logits_buy > 0.8] = 1
    y[logits_sell > 0.8] = 2

    # Split Train / Calib / Test
    X_tr, y_tr = X[:600], y[:600]
    X_cal, y_cal = X[600:900], y[600:900]
    X_te, y_te = X[900:], y[900:]

    base_rf = RandomForestClassifier(n_estimators=50, max_depth=4, random_state=42)
    base_rf.fit(X_tr, y_tr)

    raw_probs_te = base_rf.predict_proba(X_te)
    raw_sell_auc = roc_auc_score((y_te == 2).astype(int), raw_probs_te[:, 2])

    # Fit temperature scaling
    temp_calibrator = CalibratedClassifierCV(
        estimator=FrozenEstimator(base_rf),
        method="temperature"
    )
    temp_calibrator.fit(X_cal, y_cal)

    cal_probs_te = temp_calibrator.predict_proba(X_te)
    cal_sell_auc = roc_auc_score((y_te == 2).astype(int), cal_probs_te[:, 2])

    # Assertion 1: AUC degradation must not exceed 2% relative
    assert cal_sell_auc >= raw_sell_auc * 0.98, (
        f"Calibration severely degraded SELL AUC: raw={raw_sell_auc:.4f}, calibrated={cal_sell_auc:.4f}"
    )

    # Assertion 2: Calibrated SELL candidates must retain volume (anti-collapse guard)
    p_sell = cal_probs_te[:, 2]
    p_buy = cal_probs_te[:, 1]
    sell_candidates = ((p_sell >= 0.34) & (p_sell > p_buy)).sum()

    assert sell_candidates > 10, (
        f"Catastrophic ranking collapse detected: only {sell_candidates} SELL candidates survived calibration"
    )
