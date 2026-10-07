import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import trade_executor


class _StrictSelector:
    feature_names = [
        "close",
        "atr",
        "deliberately_missing_feature",
    ]

    def transform(self, X):
        return X[self.feature_names].to_numpy()


class _HighConfidenceEnsemble:
    def predict(self, X):
        return np.array([0])

    def predict_proba(self, X):
        return np.array([[0.95, 0.03, 0.02]])


def _make_15m(rows=35):
    base = 1700000000000
    data = []

    for i in range(rows):
        open_time = base + i * 15 * 60 * 1000
        data.append(
            {
                "open_time": open_time,
                "close_time": open_time + 15 * 60 * 1000 - 1,
                "open": 200.0,
                "high": 201.0,
                "low": 199.0,
                "close": 200.0,
                "volume": 1000.0,
            }
        )

    return pd.DataFrame(data)


def _make_1h(rows=8):
    base = 1700000000000
    data = []

    for i in range(rows):
        open_time = base + i * 60 * 60 * 1000
        data.append(
            {
                "open_time": open_time,
                "close_time": open_time + 60 * 60 * 1000 - 1,
                "open": 200.0,
                "high": 201.0,
                "low": 199.0,
                "close": 200.0,
                "volume": 1000.0,
            }
        )

    return pd.DataFrame(data)


class TestProductionFeatureContract(unittest.TestCase):

    def setUp(self):
        self._original_liveness = dict(trade_executor._SCAN_LIVENESS)

        trade_executor._SCAN_LIVENESS.update(
            {
                "scan_ran": True,
                "symbols_attempted": 1,
                "symbols_scored": 0,
                "predictions_saved": 0,
                "features_zero_filled": 0,
                "feature_validation_rejects": 0,
                "skip_reason": None,
            }
        )

    def tearDown(self):
        trade_executor._SCAN_LIVENESS.clear()
        trade_executor._SCAN_LIVENESS.update(self._original_liveness)

    def test_missing_selected_feature_fails_closed_and_is_counted(self):
        df15 = _make_15m()
        df1h = _make_1h()

        pipeline = {
            "best_features": [
                "close",
                "atr",
                "deliberately_missing_feature",
            ],
            "selector": _StrictSelector(),
            "ensemble": _HighConfidenceEnsemble(),
            "label_map": {
                0: "BUY",
                1: "SELL",
                2: "NO_TRADE",
            },
            "recommended_threshold_buy": 0.40,
            "recommended_threshold_sell": 0.45,
            "trained_at": "test-strict-contract",
        }

        saved = []

        def fake_get_data(symbol, interval, canonical_observation=False):
            if interval == trade_executor.TIMEFRAME_ENTRY:
                return df15.copy()

            if interval == trade_executor.TIMEFRAME_CONFIRM:
                return df1h.copy()

            if interval == trade_executor.TIMEFRAME_TREND:
                raise AssertionError(
                    "Production generate_signal must derive 4H locally"
                )

            raise AssertionError(
                f"Unexpected interval: {interval}"
            )

        observation_now = (
            int(df15["close_time"].iloc[-1])
            + 60 * 1000
        )

        with patch.object(
            trade_executor,
            "get_data",
            side_effect=fake_get_data,
        ), patch.object(
            trade_executor,
            "_merge_extra_features_live",
            side_effect=lambda df, btc: df.copy(),
        ), patch.object(
            trade_executor,
            "compute_ensemble_disagreement",
            return_value={
                "ensemble_disagreement": 0.0,
                "high_disagreement": False,
            },
        ), patch.object(
            trade_executor,
            "get_required_confidence",
            return_value=35.0,
        ), patch.object(
            trade_executor,
            "save_prediction",
            side_effect=saved.append,
        ), patch.object(
            trade_executor.time,
            "time",
            return_value=observation_now / 1000.0,
        ):
            result = trade_executor.generate_signal(
                "ETH-PERPETUAL",
                pipeline,
                {
                    "min_adx": 0.0,
                    "min_score": 0.0,
                },
            )

        self.assertIsNone(result)

        # Contract-invalid observations are rejected before inference.
        # Therefore no prediction record is persisted with a fabricated
        # signal/confidence.
        self.assertEqual(len(saved), 0)

        self.assertEqual(
            trade_executor._SCAN_LIVENESS[
                "features_zero_filled"
            ],
            1,
        )
        self.assertEqual(
            trade_executor._SCAN_LIVENESS[
                "feature_validation_rejects"
            ],
            1,
        )

        # Missing selected features must never become zero-valued model input
        # and the symbol must never be counted as successfully scored.
        self.assertEqual(
            trade_executor._SCAN_LIVENESS[
                "symbols_scored"
            ],
            0,
        )


if __name__ == "__main__":
    unittest.main()
