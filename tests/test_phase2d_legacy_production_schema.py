from pathlib import Path
from unittest.mock import patch
import pandas as pd
import phase2d_prospective_validator as validator

EXPECTED = [
    "volume_ratio",
    "volume_spike",
    "obv_slope",
    "bb_width",
    "atr_pct",
    "volatility",
    "vwap_dev",
    "trend_1h",
    "rsi_4h",
    "dow_sin",
    "dow_cos",
    "trend_4h",
    "price_vs_ema200",
    "hour_cos",
    "hour_sin",
    "rsi_1h",
    "regime_transitional",
    "adx",
    "ema20_vs_ema50",
    "adx_1h",
    "ema200",
    "bb_high",
    "ema50_slope",
    "bb_low",
    "vol_regime",
    "ema50",
    "btc_beta_20",
    "ema20",
    "macd_signal",
    "price_vs_ema50",
    "btc_corr_20",
    "ema9",
    "atr",
    "ema20_slope",
    "macd_hist",
]


def test_locked_production_schema_contract():
    assert validator.LOCKED_PRODUCTION_SELECTED_FEATURES == EXPECTED
    assert len(validator.LOCKED_PRODUCTION_SELECTED_FEATURES) == 35


def test_legacy_feature_builder_reproduces_locked_selected_fields():
    n = 260
    base = 100.0
    rows = []

    for i in range(n):
        price = base + i * 0.05
        rows.append(
            {
                "open_time": 1760000000000 + i * 15 * 60 * 1000,
                "close_time": 1760000000000 + (i + 1) * 15 * 60 * 1000,
                "open": price - 0.02,
                "high": price + 0.10,
                "low": price - 0.10,
                "close": price + 0.03,
                "volume": 1000.0 + (i % 10) * 20,
                "taker_buy_base_vol": 500.0 + (i % 7) * 5,
            }
        )

    raw15 = pd.DataFrame(rows)
    btc15 = raw15[["open_time", "close"]].copy().rename(columns={"close": "btc_close"})

    raw1h = raw15.iloc[::4].copy().reset_index(drop=True)

    with patch("phase2d_prospective_validator.fetch_deribit_1h", return_value=raw1h):
        frame = validator.build_production_features(raw15, "ETHUSDT", btc15)

    missing = [f for f in EXPECTED if f not in frame.columns]
    assert not missing, f"Missing legacy production features: {missing}"
    assert "_had_missing_inputs" in frame.columns
    assert frame["_had_missing_inputs"].dtype == bool
