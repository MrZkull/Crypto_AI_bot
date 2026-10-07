import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pandas.testing as pdt

from legacy_production_features import (
    LOCKED_PRODUCTION_SELECTED_FEATURES,
    SOURCE_FUNCTION_SHA256,
    _legacy_production_indicators,
    add_legacy_production_indicators,
)


def _load_reference():
    path = (
        Path(__file__).resolve().parent
        / "fixtures"
        / "frozen_v2_0_legacy_reference.py"
    )

    spec = importlib.util.spec_from_file_location(
        "frozen_v2_0_legacy_reference",
        path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load frozen legacy reference")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fixture(rows=260):
    base = 1700000000000
    i = np.arange(rows, dtype=float)

    close = 100.0 + 0.08 * i + 2.0 * np.sin(i / 9.0)
    open_ = close + 0.35 * np.sin(i / 5.0)

    high = (
        np.maximum(open_, close)
        + 1.0
        + 0.15 * np.cos(i / 7.0)
    )

    low = (
        np.minimum(open_, close)
        - 1.0
        - 0.15 * np.cos(i / 7.0)
    )

    volume = (
        1000.0
        + 150.0 * np.sin(i / 11.0)
        + 50.0 * np.cos(i / 17.0)
    )

    return pd.DataFrame(
        {
            "open_time": (
                base
                + np.arange(rows, dtype=np.int64) * 15 * 60 * 1000
            ),
            "close_time": (
                base
                + (
                    np.arange(rows, dtype=np.int64) + 1
                ) * 15 * 60 * 1000
                - 1
            ),
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


def test_locked_source_hash():
    reference = _load_reference()

    assert SOURCE_FUNCTION_SHA256 == (
        reference.REFERENCE_SOURCE_FUNCTION_SHA256
    )


def test_legacy_feature_formulas_match_frozen_reference():
    reference = _load_reference()

    df = _fixture()

    actual = _legacy_production_indicators(df)
    expected = reference._legacy_production_indicators(df)

    actual.attrs = {}
    expected.attrs = {}

    pdt.assert_frame_equal(
        actual,
        expected,
        check_dtype=False,
        check_exact=True,
    )


def test_locked_selected_feature_schema_is_35_features():
    assert len(LOCKED_PRODUCTION_SELECTED_FEATURES) == 35
    assert len(set(LOCKED_PRODUCTION_SELECTED_FEATURES)) == 35


def test_regime_transitional_matches_locked_builder_rule():
    df = _fixture()

    out = add_legacy_production_indicators(df)

    expected = (out["trend"] == 0).astype(float)

    pdt.assert_series_equal(
        out["regime_transitional"],
        expected,
        check_names=False,
    )
