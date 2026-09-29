#!/usr/bin/env python3
# market_data_integrity.py — Zero-Repair Market Data Cadence & Lookahead Integrity Engine

import math
import numpy as np
import pandas as pd

TIMEFRAME_MS = {
    "1m":  1 * 60 * 1000,
    "3m":  3 * 60 * 1000,
    "5m":  5 * 60 * 1000,
    "15m": 15 * 60 * 1000,
    "30m": 30 * 60 * 1000,
    "1h":  60 * 60 * 1000,
    "2h":  2 * 60 * 60 * 1000,
    "4h":  4 * 60 * 60 * 1000,
    "6h":  6 * 60 * 60 * 1000,
    "8h":  8 * 60 * 60 * 1000,
    "12h": 12 * 60 * 60 * 1000,
    "1d":  24 * 60 * 60 * 1000,
}


def interval_ms(interval_str: str) -> int:
    """Return interval duration in milliseconds."""
    if interval_str not in TIMEFRAME_MS:
        raise ValueError(
            f"Unsupported timeframe: '{interval_str}'. "
            f"Supported: {list(TIMEFRAME_MS.keys())}"
        )
    return TIMEFRAME_MS[interval_str]


OBSERVATION_CONTRACT_VERSION = "1.0"

OBSERVATION_IDENTITY_FIELDS = (
    "logical_symbol",
    "open_time",
    "interval",
    "observation_source",
)


def _contract_source(source, field_name="observation_source"):
    if not isinstance(source, dict):
        raise ValueError(f"{field_name} must be a dict")

    exchange = str(source.get("exchange", "")).strip().lower()
    market_type = str(source.get("market_type", "")).strip().lower()

    if not exchange:
        raise ValueError(f"{field_name}.exchange is required")
    if not market_type:
        raise ValueError(f"{field_name}.market_type is required")

    return {
        "exchange": exchange,
        "market_type": market_type,
    }


def _contract_ts(name, value):
    try:
        value = int(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{name} must be an integer timestamp")

    if value <= 0:
        raise ValueError(f"{name} must be > 0")

    return value


def _contract_source_record(
    record,
    *,
    name,
    expected_interval,
    default_symbol=None,
    default_market=None,
    default_source=None,
    allow_defaults=False,
):
    if record is None:
        return None

    if not isinstance(record, dict):
        raise ValueError(f"{name} must be a dict or None")

    if allow_defaults:
        logical_symbol_value = record.get("logical_symbol", default_symbol)
        observation_market_value = record.get("observation_market", default_market)
        observation_source_value = record.get("observation_source", default_source)
    else:
        logical_symbol_value = record.get("logical_symbol")
        observation_market_value = record.get("observation_market")
        observation_source_value = record.get("observation_source")

    if logical_symbol_value in (None, ""):
        raise ValueError(f"{name}.logical_symbol is required")
    if observation_market_value in (None, ""):
        raise ValueError(f"{name}.observation_market is required")
    if observation_source_value is None:
        raise ValueError(f"{name}.observation_source is required")

    logical_symbol = str(logical_symbol_value).strip().upper()
    observation_market = str(observation_market_value).strip().upper()

    if not logical_symbol:
        raise ValueError(f"{name}.logical_symbol is required")
    if not observation_market:
        raise ValueError(f"{name}.observation_market is required")

    source = _contract_source(
        observation_source_value,
        f"{name}.observation_source",
    )

    interval = record.get("interval", expected_interval)
    if interval != expected_interval:
        raise ValueError(
            f"{name}.interval must be {expected_interval!r}, got {interval!r}"
        )

    open_time = _contract_ts(
        f"{name}.open_time",
        record.get("open_time"),
    )
    close_time = _contract_ts(
        f"{name}.close_time",
        record.get("close_time"),
    )

    expected_close_time = open_time + interval_ms(interval) - 1
    if close_time != expected_close_time:
        raise ValueError(
            f"{name}.close_time does not match {interval}: "
            f"expected {expected_close_time}, got {close_time}"
        )

    return {
        "logical_symbol": logical_symbol,
        "observation_market": observation_market,
        "observation_source": source,
        "interval": interval,
        "open_time": open_time,
        "close_time": close_time,
    }


def build_observation_contract(
    *,
    logical_symbol,
    observation_market,
    observation_source,
    execution_instrument,
    interval,
    open_time,
    close_time,
    retrieved_at_ms=None,
    htf_sources=None,
    btc_source=None,
):
    """Build and strictly validate the schema-versioned Observation Contract.

    Stable observation identity is:
        logical_symbol + open_time + interval + observation_source

    Retrieval timestamp and execution instrument are metadata and are
    intentionally excluded from observation identity/comparability.
    """
    logical_symbol = str(logical_symbol).strip().upper()
    if not logical_symbol:
        raise ValueError("logical_symbol is required")

    observation_market = str(observation_market).strip().upper()
    if not observation_market:
        raise ValueError("observation_market is required")

    observation_source = _contract_source(observation_source)

    if interval not in ("15m", "1h", "4h"):
        raise ValueError(
            f"Observation Contract interval must be 15m, 1h, or 4h; got {interval!r}"
        )

    open_time = _contract_ts("open_time", open_time)
    close_time = _contract_ts("close_time", close_time)

    expected_close_time = open_time + interval_ms(interval) - 1
    if close_time != expected_close_time:
        raise ValueError(
            "Primary close_time does not match interval: "
            f"expected {expected_close_time}, got {close_time}"
        )

    if execution_instrument is not None:
        execution_instrument = str(execution_instrument).strip() or None

    if retrieved_at_ms is not None:
        retrieved_at_ms = _contract_ts(
            "retrieved_at_ms",
            retrieved_at_ms,
        )
        if retrieved_at_ms < close_time:
            raise ValueError(
                "retrieved_at_ms cannot be earlier than primary close_time"
            )

    normalized_htf = {
        "1h": None,
        "4h": None,
    }

    for timeframe in ("1h", "4h"):
        record = (
            htf_sources.get(timeframe)
            if isinstance(htf_sources, dict)
            else None
        )

        if record is None:
            continue

        normalized = _contract_source_record(
            record,
            name=f"htf_sources.{timeframe}",
            expected_interval=timeframe,
            default_symbol=logical_symbol,
            default_market=observation_market,
            default_source=observation_source,
            allow_defaults=True,
        )

        if normalized["close_time"] > close_time:
            raise ValueError(
                f"htf_sources.{timeframe}.close_time is after primary close_time"
            )

        normalized_htf[timeframe] = normalized

    normalized_btc = None

    if btc_source is not None:
        normalized_btc = _contract_source_record(
            btc_source,
            name="btc_source",
            expected_interval="15m",
            default_symbol="BTCUSDT",
            default_market=observation_market,
            default_source=observation_source,
        )

        if normalized_btc["logical_symbol"] != "BTCUSDT":
            raise ValueError(
                "btc_source.logical_symbol must be BTCUSDT"
            )

        if normalized_btc["close_time"] > close_time:
            raise ValueError(
                "btc_source.close_time is after primary close_time"
            )

    contract = {
        "contract_version": OBSERVATION_CONTRACT_VERSION,
        "logical_symbol": logical_symbol,
        "observation_market": observation_market,
        "observation_source": observation_source,
        "execution_instrument": execution_instrument,
        "interval": interval,
        "open_time": open_time,
        "close_time": close_time,
        "retrieved_at_ms": retrieved_at_ms,
        "htf_sources": normalized_htf,
        "btc_source": normalized_btc,
    }

    validate_observation_contract(contract)
    return contract


def validate_observation_contract(contract):
    """Strictly validate an Observation Contract without modifying it."""
    if not isinstance(contract, dict):
        raise ValueError("Observation Contract must be a dict")

    if contract.get("contract_version") != OBSERVATION_CONTRACT_VERSION:
        raise ValueError(
            "Unsupported Observation Contract version: "
            f"{contract.get('contract_version')!r}"
        )

    logical_symbol = str(
        contract.get("logical_symbol", "")
    ).strip().upper()

    if not logical_symbol:
        raise ValueError("logical_symbol is required")

    observation_market = str(
        contract.get("observation_market", "")
    ).strip().upper()

    if not observation_market:
        raise ValueError("observation_market is required")

    _contract_source(contract.get("observation_source"))

    interval = contract.get("interval")

    if interval not in ("15m", "1h", "4h"):
        raise ValueError(
            f"Unsupported Observation Contract interval: {interval!r}"
        )

    open_time = _contract_ts(
        "open_time",
        contract.get("open_time"),
    )

    close_time = _contract_ts(
        "close_time",
        contract.get("close_time"),
    )

    expected_close_time = open_time + interval_ms(interval) - 1

    if close_time != expected_close_time:
        raise ValueError(
            f"Invalid primary close_time: expected {expected_close_time}, "
            f"got {close_time}"
        )

    retrieved_at_ms = contract.get("retrieved_at_ms")

    if retrieved_at_ms is not None:
        retrieved_at_ms = _contract_ts(
            "retrieved_at_ms",
            retrieved_at_ms,
        )

        if retrieved_at_ms < close_time:
            raise ValueError(
                "retrieved_at_ms cannot be earlier than primary close_time"
            )

    htf_sources = contract.get("htf_sources")

    if not isinstance(htf_sources, dict):
        raise ValueError("htf_sources must be a dict")

    unknown_htf = set(htf_sources) - {"1h", "4h"}
    if unknown_htf:
        raise ValueError(
            f"Unsupported htf_sources keys: {sorted(unknown_htf)!r}"
        )

    for timeframe in ("1h", "4h"):
        record = htf_sources.get(timeframe)

        if record is None:
            continue

        normalized = _contract_source_record(
            record,
            name=f"htf_sources.{timeframe}",
            expected_interval=timeframe,
        )

        if normalized["close_time"] > close_time:
            raise ValueError(
                f"htf_sources.{timeframe}.close_time is after primary close_time"
            )

    btc_source = contract.get("btc_source")

    if btc_source is not None:
        normalized_btc = _contract_source_record(
            btc_source,
            name="btc_source",
            expected_interval="15m",
        )

        if normalized_btc["logical_symbol"] != "BTCUSDT":
            raise ValueError(
                "btc_source.logical_symbol must be BTCUSDT"
            )

        if normalized_btc["close_time"] > close_time:
            raise ValueError(
                "btc_source.close_time is after primary close_time"
            )

    execution_instrument = contract.get("execution_instrument")

    if (
        execution_instrument is not None
        and not str(execution_instrument).strip()
    ):
        raise ValueError(
            "execution_instrument cannot be blank"
        )

    return True


def observation_identity(contract):
    """Return stable observation identity.

    Retrieval time and execution instrument are deliberately excluded.
    """
    validate_observation_contract(contract)

    source = contract["observation_source"]

    return (
        contract["logical_symbol"],
        int(contract["open_time"]),
        contract["interval"],
        source["exchange"],
        source["market_type"],
    )



def sanitize_closed_candles(
    raw,
    interval_ms_value: int = None,
    observation_time_ms: int = None,
    candle_duration_ms: int = None,
    **kwargs,
) -> list[dict]:
    """
    Strict zero-repair candle sanitizer.

    Accepted input forms:
      - pandas DataFrame
      - list[dict]
      - exchange-style array rows: [open_time, open, high, low, close, volume, close_time, ...]
      - None

    No values are repaired or interpolated. Invalid rows are rejected.
    """
    if raw is None:
        return []

    duration_ms = (
        interval_ms_value
        if interval_ms_value is not None
        else candle_duration_ms
    )
    if duration_ms is None:
        duration_ms = kwargs.get("duration_ms") or kwargs.get("interval_ms")

    rows = raw.to_dict("records") if isinstance(raw, pd.DataFrame) else raw
    if isinstance(rows, dict):
        rows = rows.get("data", rows.get("result", []))

    clean = []
    seen_open_times = set()

    for r in rows or []:
        try:
            if isinstance(r, dict):
                o = int(r["open_time"])
                c = int(r["close_time"])
                vals = {
                    k: float(r[k])
                    for k in ("open", "high", "low", "close", "volume")
                }
                extra = {
                    k: r[k]
                    for k in r
                    if k not in vals and k not in ("open_time", "close_time")
                }
            else:
                o = int(r[0])
                c = int(r[6])
                vals = {
                    "open": float(r[1]),
                    "high": float(r[2]),
                    "low": float(r[3]),
                    "close": float(r[4]),
                    "volume": float(r[5]),
                }
                extra = {}
                if len(r) > 9:
                    extra["taker_buy_base_vol"] = float(r[9])

            if c <= o:
                continue

            if duration_ms is not None:
                expected_c_time = o + int(duration_ms) - 1
                if c != expected_c_time:
                    continue

            if observation_time_ms is not None and c > int(observation_time_ms):
                continue

            if not all(math.isfinite(v) for v in vals.values()):
                continue

            if (
                vals["open"] <= 0
                or vals["high"] <= 0
                or vals["low"] <= 0
                or vals["close"] <= 0
                or vals["volume"] < 0
            ):
                continue

            if (
                vals["high"] < max(vals["open"], vals["close"])
                or vals["low"] > min(vals["open"], vals["close"])
            ):
                continue

            if o in seen_open_times:
                continue

            seen_open_times.add(o)
            clean.append({"open_time": o, "close_time": c, **vals, **extra})

        except (TypeError, ValueError, KeyError, OverflowError):
            continue

    clean.sort(key=lambda x: x["open_time"])
    return clean


def sanitize_ohlcv_frame(
    df: pd.DataFrame,
    interval_ms_value: int = None,
    observation_time_ms: int = None,
    candle_duration_ms: int = None,
    **kwargs,
) -> pd.DataFrame:
    """Return a sanitized pandas DataFrame while preserving extra columns."""
    duration_ms = (
        interval_ms_value
        if interval_ms_value is not None
        else candle_duration_ms
    )
    rows = sanitize_closed_candles(
        df,
        interval_ms_value=duration_ms,
        observation_time_ms=observation_time_ms,
        **kwargs,
    )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    if observation_time_ms is not None:
        out = out[out["close_time"] <= int(observation_time_ms)]
    return out.reset_index(drop=True)


def merge_completed_htf(
    base_df: pd.DataFrame,
    htf_df: pd.DataFrame,
    feature_cols: list = None,
    prefix: str = "htf",
    value_cols: list = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Point-in-time HTF alignment.

    The observation timestamp is the completed LTF candle close_time.
    An HTF feature row is usable only when:

        HTF source close_time <= LTF observation close_time

    Equal timestamps are explicitly allowed because the HTF candle is complete
    at that instant.
    """
    if base_df.empty or htf_df.empty:
        return base_df.copy()

    cols_to_merge = feature_cols if feature_cols is not None else value_cols
    if cols_to_merge is None:
        cols_to_merge = kwargs.get("cols", [])

    ltf = base_df.copy()
    htf = htf_df.copy()

    if "close_time" not in ltf.columns or "close_time" not in htf.columns:
        raise ValueError(
            "Both LTF and HTF frames must contain 'close_time' "
            "for point-in-time alignment"
        )

    valid_cols = [c for c in cols_to_merge if c in htf.columns]
    if not valid_cols:
        return ltf

    ltf["_obs_time"] = pd.to_numeric(ltf["close_time"], errors="coerce")
    h = htf[["close_time"] + valid_cols].copy()
    h["close_time"] = pd.to_numeric(h["close_time"], errors="coerce")
    h = h.dropna(subset=["close_time"])
    h = h.sort_values("close_time")

    rename_map = {c: f"{prefix}_{c}" for c in valid_cols}
    rename_map["close_time"] = f"{prefix}_source_close_time"
    h = h.rename(columns=rename_map)

    merged = pd.merge_asof(
        ltf.sort_values("_obs_time"),
        h,
        left_on="_obs_time",
        right_on=f"{prefix}_source_close_time",
        direction="backward",
        allow_exact_matches=True,
    )

    return (
        merged.drop(columns=["_obs_time"], errors="ignore")
        .sort_values("open_time")
        .reset_index(drop=True)
    )
