#!/usr/bin/env python3
"""Deterministic verification of F08 historical-resolution provenance.

The canonical HISTORICAL_BACKFILL observation rows intentionally omit the
resolver's volatile wall-clock ``resolved_at``.  This module stores that value
in a separate, SHA-256-bound JSONL artifact and independently derives the
selected 15-minute candle's close boundary from the exact raw OHLC window.

``outcome_bar_close_time`` uses the same interval convention as the pinned
Deribit fetcher: candle ``open_time`` + 900,000 ms (the interval close boundary,
exclusive).  It is not the resolver process timestamp.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import Any

EXPERIMENT_ID = "PHASE2D-HARDENED-20260924-F08"
LEGACY_CUTOFF_UTC = "2026-10-16T14:08:36.946361+00:00"
LEGACY_CUTOFF = datetime.fromisoformat(LEGACY_CUTOFF_UTC)
PROVENANCE_SCHEMA_VERSION = 1
PROVENANCE_RECORD_TYPE = "historical_backfill_resolution_provenance"
PROVENANCE_FILENAME = "resolution_provenance.jsonl"
CANDLE_RESOLUTION_MINUTES = 15
CANDLE_INTERVAL_MS = CANDLE_RESOLUTION_MINUTES * 60 * 1000
LOOKAHEAD_BARS = 24
EXPECTED_MODEL_COUNTS = {"candidate": 74, "production": 544}
RESOLVED_AT_SEMANTICS = (
    "Exact UTC timestamp returned by the pinned production resolver during "
    "this reconciliation execution; not the historical market-outcome time."
)
OUTCOME_BAR_CLOSE_TIME_SEMANTICS = (
    "15-minute candle interval close boundary, calculated as the selected raw "
    "candle timestamp plus 900000 milliseconds, matching the pinned validator close_time convention."
)


class ResolutionProvenanceError(ValueError):
    """Raised when per-prediction historical resolution evidence is invalid."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def canonical_jsonl_bytes(rows: list[dict]) -> bytes:
    if not rows:
        raise ResolutionProvenanceError("Resolution provenance cannot be empty")
    return b"\n".join(canonical_json_bytes(row) for row in rows) + b"\n"


def _strict_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ResolutionProvenanceError(f"{label} must be an integer")
    return value


def _utc_timestamp(value: Any, *, label: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ResolutionProvenanceError(f"{label} is missing")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ResolutionProvenanceError(f"{label} is invalid: {value!r}") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise ResolutionProvenanceError(f"{label} must contain an explicit UTC offset")
    if not (value.endswith("+00:00") or value.endswith("Z")):
        raise ResolutionProvenanceError(f"{label} must use the canonical UTC offset (+00:00 or Z)")
    parsed = parsed.astimezone(timezone.utc)
    if parsed <= LEGACY_CUTOFF:
        raise ResolutionProvenanceError(f"{label} must be after the legacy cutoff")
    return parsed


def _iso_utc_from_ms(value: int) -> str:
    try:
        return datetime.fromtimestamp(value / 1000, tz=timezone.utc).isoformat(
            timespec="milliseconds"
        ).replace("+00:00", "Z")
    except (OverflowError, OSError, ValueError) as exc:
        raise ResolutionProvenanceError(
            f"Candle timestamp is outside the supported UTC range: {value}"
        ) from exc


def index_raw_windows(raw_window_rows: list[dict]) -> dict[tuple[str, int], dict]:
    """Validate raw 15m windows and index them by exact symbol/open_time."""
    if not isinstance(raw_window_rows, list):
        raise ResolutionProvenanceError("Raw OHLC windows must be a list")
    indexed: dict[tuple[str, int], dict] = {}
    for row_number, window in enumerate(raw_window_rows, start=1):
        if not isinstance(window, dict):
            raise ResolutionProvenanceError(f"Raw OHLC window {row_number} is not an object")
        symbol = window.get("symbol")
        if not isinstance(symbol, str) or not symbol:
            raise ResolutionProvenanceError(f"Raw OHLC window {row_number} has invalid symbol")
        try:
            open_time = _strict_int(window.get("open_time"), label=f"raw window {row_number} open_time")
        except ResolutionProvenanceError:
            raise
        key = (symbol, open_time)
        if key in indexed:
            raise ResolutionProvenanceError(f"Duplicate raw OHLC window: {key}")
        request = window.get("request")
        if not isinstance(request, dict) or str(request.get("resolution", "15")) != "15":
            raise ResolutionProvenanceError(f"Raw OHLC window {key} is not 15-minute data")
        candles = window.get("candles")
        if not isinstance(candles, list) or len(candles) < LOOKAHEAD_BARS + 2:
            raise ResolutionProvenanceError(
                f"Raw OHLC window {key} has fewer than {LOOKAHEAD_BARS} subsequent candles"
            )
        timestamps: list[int] = []
        for candle_number, candle in enumerate(candles, start=1):
            if not isinstance(candle, dict):
                raise ResolutionProvenanceError(
                    f"Raw OHLC window {key} candle {candle_number} is not an object"
                )
            timestamp = _strict_int(
                candle.get("timestamp"),
                label=f"raw OHLC window {key} candle {candle_number} timestamp",
            )
            try:
                high = float(candle["high"])
                low = float(candle["low"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ResolutionProvenanceError(
                    f"Raw OHLC window {key} candle {candle_number} has invalid high/low"
                ) from exc
            if not math.isfinite(high) or not math.isfinite(low) or high < low:
                raise ResolutionProvenanceError(
                    f"Raw OHLC window {key} candle {candle_number} has invalid OHLC range"
                )
            timestamps.append(timestamp)
        if timestamps != sorted(timestamps) or len(timestamps) != len(set(timestamps)):
            raise ResolutionProvenanceError(f"Raw OHLC timestamps are not strictly ordered for {key}")
        if timestamps.count(open_time) != 1:
            raise ResolutionProvenanceError(f"Raw OHLC window does not contain one exact entry candle: {key}")
        entry_index = timestamps.index(open_time)
        if entry_index + LOOKAHEAD_BARS >= len(timestamps):
            raise ResolutionProvenanceError(f"Raw OHLC window lacks 24 candles after entry: {key}")
        indexed[key] = window
    return indexed


def _build_one(
    result: dict,
    raw_windows: dict[tuple[str, int], dict],
    reconciliation_id: str,
) -> dict:
    model = result.get("model")
    prediction_id = result.get("id")
    symbol = result.get("symbol")
    if model not in EXPECTED_MODEL_COUNTS:
        raise ResolutionProvenanceError(f"Invalid historical model: {model!r}")
    if not isinstance(prediction_id, str) or not prediction_id:
        raise ResolutionProvenanceError("Historical prediction ID is missing")
    if not isinstance(symbol, str) or not symbol:
        raise ResolutionProvenanceError(f"Historical prediction {prediction_id} has invalid symbol")
    prediction_open_time = _strict_int(
        result.get("open_time"), label=f"{model}/{prediction_id} open_time"
    )
    status = result.get("status")
    if status not in {"TP", "SL", "EXPIRED"}:
        raise ResolutionProvenanceError(f"Unsupported historical outcome for {model}/{prediction_id}: {status!r}")
    outcome_bar = _strict_int(
        result.get("outcome_bar"), label=f"{model}/{prediction_id} outcome_bar"
    )
    if not 1 <= outcome_bar <= LOOKAHEAD_BARS:
        raise ResolutionProvenanceError(f"Invalid outcome_bar for {model}/{prediction_id}: {outcome_bar}")
    if status == "EXPIRED" and outcome_bar != LOOKAHEAD_BARS:
        raise ResolutionProvenanceError(
            f"EXPIRED prediction {model}/{prediction_id} must resolve at bar {LOOKAHEAD_BARS}"
        )
    resolved_at = result.get("resolved_at")
    _utc_timestamp(resolved_at, label=f"{model}/{prediction_id} resolved_at")

    key = (symbol, prediction_open_time)
    window = raw_windows.get(key)
    if window is None:
        raise ResolutionProvenanceError(
            f"No exact raw OHLC window for {model}/{prediction_id}: {key}"
        )
    candles = window["candles"]
    timestamps = [int(candle["timestamp"]) for candle in candles]
    entry_index = timestamps.index(prediction_open_time)
    outcome_index = entry_index + outcome_bar
    if outcome_index >= len(candles):
        raise ResolutionProvenanceError(
            f"Raw OHLC window does not contain outcome bar {outcome_bar} for {model}/{prediction_id}"
        )
    outcome_candle = candles[outcome_index]
    candle_open_time_ms = int(outcome_candle["timestamp"])
    if candle_open_time_ms <= prediction_open_time:
        raise ResolutionProvenanceError(
            f"Outcome candle is not after entry for {model}/{prediction_id}"
        )
    candle_close_time_ms = candle_open_time_ms + CANDLE_INTERVAL_MS
    return {
        "record_type": PROVENANCE_RECORD_TYPE,
        "provenance_schema_version": PROVENANCE_SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "reconciliation_id": reconciliation_id,
        "model": model,
        "prediction_id": prediction_id,
        "symbol": symbol,
        "prediction_open_time_ms": prediction_open_time,
        "status": status,
        "outcome_bar": outcome_bar,
        # Preserve the exact string returned by the pinned production resolver.
        "resolved_at": resolved_at,
        "outcome_bar_open_time_ms": candle_open_time_ms,
        "outcome_bar_close_time_ms": candle_close_time_ms,
        "outcome_bar_close_time": _iso_utc_from_ms(candle_close_time_ms),
        "raw_window_sha256": hashlib.sha256(canonical_json_bytes(window)).hexdigest(),
    }


def build_resolution_provenance_records(
    result_records: list[dict],
    raw_windows: dict[tuple[str, int], dict],
    reconciliation_id: str,
    *,
    expected_counts: dict[str, int] | None = None,
) -> list[dict]:
    """Create the non-deterministic per-ID provenance sidecar from resolver output."""
    if not isinstance(reconciliation_id, str) or not reconciliation_id:
        raise ResolutionProvenanceError("A reconciliation_id is required for provenance")
    if not isinstance(result_records, list):
        raise ResolutionProvenanceError("Resolver result records must be a list")
    if not isinstance(raw_windows, dict):
        raise ResolutionProvenanceError("Raw OHLC windows must be indexed by (symbol, open_time)")
    raw_windows = index_raw_windows(list(raw_windows.values()))
    expected_counts = dict(expected_counts or EXPECTED_MODEL_COUNTS)
    records: list[dict] = []
    seen: set[tuple[str, str]] = set()
    counts: Counter = Counter()
    for result in result_records:
        if not isinstance(result, dict):
            raise ResolutionProvenanceError("Resolver output contains a non-object")
        identity = (result.get("model"), result.get("id"))
        if identity in seen:
            raise ResolutionProvenanceError(f"Duplicate resolver result: {identity}")
        seen.add(identity)
        record = _build_one(result, raw_windows, reconciliation_id)
        records.append(record)
        counts[record["model"]] += 1
    records.sort(key=lambda row: (row["model"], row["prediction_id"]))
    if dict(counts) != expected_counts:
        raise ResolutionProvenanceError(
            f"Resolution provenance population mismatch: {dict(counts)} != {expected_counts}"
        )
    if len(records) != sum(expected_counts.values()):
        raise ResolutionProvenanceError(
            f"Expected {sum(expected_counts.values())} provenance records, got {len(records)}"
        )
    return records


def verify_resolution_provenance_records(
    provenance_records: list[dict],
    observation_rows: list[dict],
    raw_windows: dict[tuple[str, int], dict],
    reconciliation_id: str,
    *,
    expected_counts: dict[str, int] | None = None,
) -> dict:
    """Verify provenance maps one-to-one to outcomes and exact raw OHLC bars."""
    if not isinstance(raw_windows, dict):
        raise ResolutionProvenanceError("Raw OHLC windows must be indexed by (symbol, open_time)")
    raw_windows = index_raw_windows(list(raw_windows.values()))
    expected_counts = dict(expected_counts or EXPECTED_MODEL_COUNTS)
    expected_total = sum(expected_counts.values())
    if len(provenance_records) != expected_total:
        raise ResolutionProvenanceError(
            f"Expected {expected_total} provenance records, got {len(provenance_records)}"
        )
    outcomes: dict[tuple[str, str], dict] = {}
    for row in observation_rows:
        if not isinstance(row, dict):
            raise ResolutionProvenanceError("Historical observation row is not an object")
        identity = (row.get("model"), row.get("id"))
        if identity in outcomes:
            raise ResolutionProvenanceError(f"Duplicate historical observation row: {identity}")
        outcomes[identity] = row
    if len(outcomes) != expected_total:
        raise ResolutionProvenanceError(
            f"Expected {expected_total} historical outcome rows, got {len(outcomes)}"
        )

    provenance_ids: set[tuple[str, str]] = set()
    counts: Counter = Counter()
    previous_key: tuple[str, str] | None = None
    for record in provenance_records:
        if not isinstance(record, dict):
            raise ResolutionProvenanceError("Resolution provenance contains a non-object")
        identity = (record.get("model"), record.get("prediction_id"))
        if identity in provenance_ids:
            raise ResolutionProvenanceError(f"Duplicate provenance identity: {identity}")
        provenance_ids.add(identity)
        if previous_key is not None and identity < previous_key:
            raise ResolutionProvenanceError("Resolution provenance is not sorted by model/prediction_id")
        previous_key = identity
        counts[record.get("model")] += 1
        outcome = outcomes.get(identity)
        if outcome is None:
            raise ResolutionProvenanceError(f"Provenance has no matching outcome row: {identity}")
        expected = _build_one(
            {
                **outcome,
                "id": outcome.get("id"),
                "resolved_at": record.get("resolved_at"),
            },
            raw_windows,
            reconciliation_id,
        )
        if record != expected:
            mismatches = sorted(
                key for key in set(record) | set(expected)
                if record.get(key) != expected.get(key)
            )
            raise ResolutionProvenanceError(
                f"Per-prediction provenance mismatch for {identity}: {mismatches}"
            )
    if provenance_ids != set(outcomes):
        missing = sorted(set(outcomes) - provenance_ids)[:5]
        extra = sorted(provenance_ids - set(outcomes))[:5]
        raise ResolutionProvenanceError(
            f"Provenance/outcome population mismatch; missing={missing}, extra={extra}"
        )
    if dict(counts) != expected_counts:
        raise ResolutionProvenanceError(
            f"Provenance model counts mismatch: {dict(counts)} != {expected_counts}"
        )
    return {
        "status": "PASS",
        "records_verified": expected_total,
        "by_model": dict(counts),
        "raw_windows_referenced": len({
            (row.get("symbol"), row.get("prediction_open_time_ms"))
            for row in provenance_records
        }),
        "close_time_convention": "15m candle open timestamp + 900000ms interval boundary",
    }
