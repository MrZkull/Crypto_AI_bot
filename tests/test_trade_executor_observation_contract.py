import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import trade_executor


class _FakeSelector:
    def transform(self, X):
        return np.asarray(X, dtype=float)


class _FakeEnsemble:
    def predict(self, X):
        return np.array([0])

    def predict_proba(self, X):
        return np.array([[0.95, 0.05]])


class TestProductionObservationContract(unittest.TestCase):

    @staticmethod
    def _make_15m_data():
        base = 1_700_000_000_000
        rows = []

        for i in range(35):
            rows.append({
                "open_time": base + i * 15 * 60 * 1000,
                "close_time": base + i * 15 * 60 * 1000 + 15 * 60 * 1000 - 1,
                "open": 2000.0,
                "high": 2010.0,
                "low": 1990.0,
                "close": 2005.0 + i,
                "volume": 1000.0,
                "atr": 10.0,
                "rsi": 55.0,
                "adx": 30.0,
                "trend": 1.0,
                "ema20": 2050.0,
                "ema50": 2000.0,
                "macd_hist": 1.0,
            })

        return pd.DataFrame(rows)

    @staticmethod
    def _make_htf_data(rows=8):
        data = []

        for i in range(rows):
            data.append({
                "open_time": 1_700_000_000_000 + i * 60 * 60 * 1000,
                "close_time": 1_700_000_000_000 + i * 60 * 60 * 1000 + 60 * 60 * 1000 - 1,
                "open": 2000.0,
                "high": 2010.0,
                "low": 1990.0,
                "close": 2005.0 + i,
                "volume": 1000.0,
                "rsi": 55.0,
                "adx": 30.0,
                "trend": 1.0,
                "ema20": 2050.0,
                "ema50": 2000.0,
                "macd_hist": 1.0,
            })

        return pd.DataFrame(data)

    def test_generate_signal_exposes_real_15m_open_time(self):
        df15 = self._make_15m_data()
        # Make synthetic 15m candles satisfy production integrity checks.
        df15["high"] = df15[["open", "close"]].max(axis=1) + 10.0
        df15["low"] = df15[["open", "close"]].min(axis=1) - 10.0
        df15["close_time"] = df15["open_time"].astype("int64") + 15 * 60 * 1000 - 1
        df1h = self._make_htf_data()
        # Build a valid completed 1h fixture for production sanitization.
        df1h["open_time"] = df1h["open_time"].iloc[0] + df1h.index.to_series() * (60 * 60 * 1000)
        df1h["high"] = df1h[["open", "close"]].max(axis=1) + 10.0
        df1h["low"] = df1h[["open", "close"]].min(axis=1) - 10.0
        df1h["close_time"] = df1h["open_time"].astype("int64") + 60 * 60 * 1000 - 1
        df4h = self._make_htf_data()
        # Build a valid completed 4h fixture for production sanitization.
        df4h["open_time"] = df4h["open_time"].iloc[0] + df4h.index.to_series() * (4 * 60 * 60 * 1000)
        df4h["high"] = df4h[["open", "close"]].max(axis=1) + 10.0
        df4h["low"] = df4h[["open", "close"]].min(axis=1) - 10.0
        df4h["close_time"] = df4h["open_time"].astype("int64") + 4 * 60 * 60 * 1000 - 1

        expected_open_time = int(df15.iloc[-1]["open_time"])

        pipeline = {
            "all_features": [
                "close",
                "atr",
                "rsi",
                "adx",
                "trend",
                "ema20",
                "ema50",
                "rsi_1h",
                "adx_1h",
                "trend_1h",
                "rsi_4h",
                "trend_4h",
            ],
            "selector": _FakeSelector(),
            "ensemble": _FakeEnsemble(),
            "label_map": {
                0: "BUY",
                1: "SELL",
            },
            "recommended_threshold_buy": 0.40,
            "recommended_threshold_sell": 0.45,
            "trained_at": "test",
        }

        def fake_get_data(symbol, interval, canonical_observation=False):
            if interval == trade_executor.TIMEFRAME_ENTRY:
                return df15.copy()
            if interval == trade_executor.TIMEFRAME_CONFIRM:
                return df1h.copy()
            if interval == trade_executor.TIMEFRAME_TREND:
                raise AssertionError(
                    "generate_signal must derive 4h context locally from completed 1h candles"
                )
            raise AssertionError(f"Unexpected interval: {interval}")

        with patch.object(trade_executor, "get_data", side_effect=fake_get_data), \
             patch.object(trade_executor, "add_indicators", side_effect=lambda df: df.copy()), \
             patch.object(
                 trade_executor,
                 "_merge_extra_features_live",
                 side_effect=lambda df, btc: df.copy(),
             ), \
             patch.object(
                 trade_executor,
                 "compute_ensemble_disagreement",
                 return_value={
                     "ensemble_disagreement": 0.0,
                     "high_disagreement": False,
                 },
             ), \
             patch.object(
                 trade_executor,
                 "get_required_confidence",
                 return_value=35.0,
             ), \
             patch.object(
                 trade_executor,
                 "save_prediction",
                 return_value=None,
             ):

            result = trade_executor.generate_signal(
                "ETH-PERPETUAL",
                pipeline,
                {
                    "min_adx": 0.0,
                    "min_score": 0.0,
                },
                btc_momentum=None,
                whale_flow=None,
                fng_data=None,
                btc_df15_live=None,
            )

        self.assertIsNotNone(result)
        self.assertEqual(result["signal"], "BUY")
        self.assertIn("open_time", result)
        self.assertIsInstance(result["open_time"], int)
        self.assertEqual(result["open_time"], expected_open_time)
        self.assertIn("observation_contract", result)
        self.assertEqual(
            result["observation_contract"]["logical_symbol"],
            "ETH-PERPETUAL",
        )
        self.assertEqual(
            result["observation_contract"]["interval"],
            "15m",
        )
        self.assertEqual(
            result["observation_contract"]["observation_source"],
            {"exchange": "binance", "market_type": "spot"},
        )
        self.assertEqual(
            result["observation_identity"],
            [
                "ETH-PERPETUAL",
                expected_open_time,
                "15m",
                "binance",
                "spot",
            ],
        )

    def test_forming_15m_candle_is_excluded(self):
        df15 = self._make_15m_data()
        # Make synthetic 15m candles satisfy production integrity checks.
        df15["high"] = df15[["open", "close"]].max(axis=1) + 10.0
        df15["low"] = df15[["open", "close"]].min(axis=1) - 10.0
        df15["close_time"] = df15["open_time"].astype("int64") + 15 * 60 * 1000 - 1

        base_open = int(df15.iloc[-1]["open_time"])
        forming_open = base_open + 15 * 60 * 1000

        forming = df15.iloc[-1].copy()
        forming["open_time"] = forming_open
        forming["close_time"] = forming_open + 15 * 60 * 1000 - 1

        df15 = pd.concat(
            [
                df15,
                pd.DataFrame([forming]),
            ],
            ignore_index=True,
        )

        observation_now = forming_open + 15 * 60 * 1000 - 1000
        expected_open_time = base_open

        df1h = self._make_htf_data()
        # Build a valid completed 1h fixture for production sanitization.
        df1h["open_time"] = df1h["open_time"].iloc[0] + df1h.index.to_series() * (60 * 60 * 1000)
        df1h["high"] = df1h[["open", "close"]].max(axis=1) + 10.0
        df1h["low"] = df1h[["open", "close"]].min(axis=1) - 10.0
        df1h["close_time"] = df1h["open_time"].astype("int64") + 60 * 60 * 1000 - 1
        df4h = self._make_htf_data()
        # Build a valid completed 4h fixture for production sanitization.
        df4h["open_time"] = df4h["open_time"].iloc[0] + df4h.index.to_series() * (4 * 60 * 60 * 1000)
        df4h["high"] = df4h[["open", "close"]].max(axis=1) + 10.0
        df4h["low"] = df4h[["open", "close"]].min(axis=1) - 10.0
        df4h["close_time"] = df4h["open_time"].astype("int64") + 4 * 60 * 60 * 1000 - 1

        pipeline = {
            "all_features": [
                "close",
                "atr",
                "rsi",
                "adx",
                "trend",
                "ema20",
                "ema50",
                "rsi_1h",
                "adx_1h",
                "trend_1h",
                "rsi_4h",
                "trend_4h",
            ],
            "selector": _FakeSelector(),
            "ensemble": _FakeEnsemble(),
            "label_map": {
                0: "BUY",
                1: "SELL",
            },
            "recommended_threshold_buy": 0.40,
            "recommended_threshold_sell": 0.45,
            "trained_at": "test",
        }

        def fake_get_data(symbol, interval, canonical_observation=False):
            if interval == trade_executor.TIMEFRAME_ENTRY:
                return df15.copy()
            if interval == trade_executor.TIMEFRAME_CONFIRM:
                return df1h.copy()
            if interval == trade_executor.TIMEFRAME_TREND:
                raise AssertionError(
                    "generate_signal must derive 4h context locally from completed 1h candles"
                )
            raise AssertionError(f"Unexpected interval: {interval}")

        with patch.object(trade_executor, "get_data", side_effect=fake_get_data), \
             patch.object(trade_executor, "add_indicators", side_effect=lambda df: df.copy()), \
             patch.object(
                 trade_executor,
                 "_merge_extra_features_live",
                 side_effect=lambda df, btc: df.copy(),
             ), \
             patch.object(
                 trade_executor,
                 "compute_ensemble_disagreement",
                 return_value={
                     "ensemble_disagreement": 0.0,
                     "high_disagreement": False,
                 },
             ), \
             patch.object(
                 trade_executor,
                 "get_required_confidence",
                 return_value=35.0,
             ), \
             patch.object(
                 trade_executor,
                 "save_prediction",
                 return_value=None,
             ), \
             patch.object(
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

        self.assertIsNotNone(result)
        self.assertEqual(result["signal"], "BUY")
        self.assertEqual(result["open_time"], expected_open_time)

    def test_generate_signal_open_time_is_not_pred_id_timestamp(self):
        df15 = self._make_15m_data()
        # Make synthetic 15m candles satisfy production integrity checks.
        df15["high"] = df15[["open", "close"]].max(axis=1) + 10.0
        df15["low"] = df15[["open", "close"]].min(axis=1) - 10.0
        df15["close_time"] = df15["open_time"].astype("int64") + 15 * 60 * 1000 - 1
        df1h = self._make_htf_data()
        # Build a valid completed 1h fixture for production sanitization.
        df1h["open_time"] = df1h["open_time"].iloc[0] + df1h.index.to_series() * (60 * 60 * 1000)
        df1h["high"] = df1h[["open", "close"]].max(axis=1) + 10.0
        df1h["low"] = df1h[["open", "close"]].min(axis=1) - 10.0
        df1h["close_time"] = df1h["open_time"].astype("int64") + 60 * 60 * 1000 - 1
        df4h = self._make_htf_data()
        # Build a valid completed 4h fixture for production sanitization.
        df4h["open_time"] = df4h["open_time"].iloc[0] + df4h.index.to_series() * (4 * 60 * 60 * 1000)
        df4h["high"] = df4h[["open", "close"]].max(axis=1) + 10.0
        df4h["low"] = df4h[["open", "close"]].min(axis=1) - 10.0
        df4h["close_time"] = df4h["open_time"].astype("int64") + 4 * 60 * 60 * 1000 - 1

        expected_open_time = int(df15.iloc[-1]["open_time"])

        pipeline = {
            "all_features": [
                "close",
                "atr",
                "rsi",
                "adx",
                "trend",
                "ema20",
                "ema50",
                "rsi_1h",
                "adx_1h",
                "trend_1h",
                "rsi_4h",
                "trend_4h",
            ],
            "selector": _FakeSelector(),
            "ensemble": _FakeEnsemble(),
            "label_map": {
                0: "BUY",
                1: "SELL",
            },
            "recommended_threshold_buy": 0.40,
            "recommended_threshold_sell": 0.45,
            "trained_at": "test",
        }

        def fake_get_data(symbol, interval, canonical_observation=False):
            if interval == trade_executor.TIMEFRAME_ENTRY:
                return df15.copy()
            if interval == trade_executor.TIMEFRAME_CONFIRM:
                return df1h.copy()
            if interval == trade_executor.TIMEFRAME_TREND:
                raise AssertionError(
                    "generate_signal must derive 4h context locally from completed 1h candles"
                )
            raise AssertionError(f"Unexpected interval: {interval}")

        with patch.object(trade_executor, "get_data", side_effect=fake_get_data), \
             patch.object(trade_executor, "add_indicators", side_effect=lambda df: df.copy()), \
             patch.object(
                 trade_executor,
                 "_merge_extra_features_live",
                 side_effect=lambda df, btc: df.copy(),
             ), \
             patch.object(
                 trade_executor,
                 "compute_ensemble_disagreement",
                 return_value={
                     "ensemble_disagreement": 0.0,
                     "high_disagreement": False,
                 },
             ), \
             patch.object(
                 trade_executor,
                 "get_required_confidence",
                 return_value=35.0,
             ), \
             patch.object(
                 trade_executor,
                 "save_prediction",
                 return_value=None,
             ):

            result = trade_executor.generate_signal(
                "ETH-PERPETUAL",
                pipeline,
                {
                    "min_adx": 0.0,
                    "min_score": 0.0,
                },
            )

        self.assertIsNotNone(result)
        self.assertNotEqual(result["open_time"], 0)
        self.assertEqual(result["open_time"], expected_open_time)
        self.assertNotEqual(result["open_time"], int(result["pred_id"].split("_")[-1]))


if __name__ == "__main__":
    unittest.main()
