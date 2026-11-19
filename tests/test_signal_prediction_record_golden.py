from __future__ import annotations

import hashlib
import json
from datetime import datetime as RealDateTime, timezone
from unittest.mock import patch

import pandas as pd

import trade_executor


class FrozenDateTime:
    @staticmethod
    def now(tz=None):
        value = RealDateTime(2026, 11, 16, 10, 0, 0, tzinfo=timezone.utc)
        return value if tz is not None else value.replace(tzinfo=None)


def test_prediction_record_runtime_golden_vector():
    """Pin deterministic runtime prediction serialization without a live model.

    The vector covers _pred_record's production record builder, observation
    identity, stop/TP reference calculations and canonical serialized bytes.
    It intentionally does not claim to validate the trained model's scores.
    """
    contract = trade_executor.build_observation_contract(
        logical_symbol="ETHUSDT",
        observation_market="BINANCE_SPOT",
        observation_source={"exchange": "binance", "market_type": "spot"},
        execution_instrument=None,
        interval="15m",
        open_time=1_700_000_000_000,
        close_time=1_700_000_899_999,
        retrieved_at_ms=1_700_001_000_000,
        htf_sources={"1h": None, "4h": None},
        btc_source=None,
    )
    row = pd.Series({
        "open_time": 1_700_000_000_000,
        "close_time": 1_700_000_899_999,
        "close": 2000.0,
        "atr": 20.0,
        "rsi": 55.0,
        "adx": 30.0,
        "_observation_contract": contract,
    })
    pipeline = {"trained_at": "frozen-v2.0-runtime-vector"}
    disagreement = {"ensemble_disagreement": 0.125, "high_disagreement": False}

    with patch.object(trade_executor, "_candidate_identity_fields", return_value={}), \
         patch.object(trade_executor, "datetime", FrozenDateTime), \
         patch.object(trade_executor, "ATR_STOP_MULT", 2.5), \
         patch.object(trade_executor, "ATR_TARGET1_MULT", 3.5):
        record = trade_executor._pred_record(
            "ETHUSDT", "BUY", 81.0, pipeline, row, disagreement,
            pred_id="golden-eth-1700000000000",
        )

    assert record["observation_contract"] == contract
    assert record["observation_identity"] == [
        "ETHUSDT", 1_700_000_000_000, "15m", "binance", "spot"
    ]
    assert record["predicted_stop"] == 1950.0
    assert record["predicted_tp1"] == 2070.0
    assert record["generated_at"] == "2026-11-16T10:00:00+00:00"

    serialized = json.dumps(
        record, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False, default=str,
    ).encode("utf-8")
    assert hashlib.sha256(serialized).hexdigest().upper() == "CA6B394972C246868549873E3EBDBF16C154325BC7A02F34140B66B26832793A"
