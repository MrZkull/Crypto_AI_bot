from __future__ import annotations

import copy

import pytest

import phase2d_f08_resolution_provenance as provenance


def _fixture():
    raw_windows = {}
    results = []
    status_groups = {
        "candidate": ["TP"] * 12 + ["SL"] * 62,
        "production": ["TP"] * 280 + ["SL"] * 245 + ["EXPIRED"] * 19,
    }
    for model, statuses in status_groups.items():
        for index, status in enumerate(statuses):
            if model == "candidate":
                unique_no = index
            elif index < 45:
                unique_no = index
            else:
                unique_no = 74 + index - 45
            symbol = f"SYM{unique_no:03d}"
            open_time = 1_700_000_000_000 + unique_no * 900_000
            key = (symbol, open_time)
            raw_windows.setdefault(key, {
                "symbol": symbol,
                "open_time": open_time,
                "request": {"resolution": "15"},
                "candles": [
                    {"timestamp": open_time + bar * 900_000, "high": 101.0, "low": 99.0}
                    for bar in range(26)
                ],
            })
            outcome_bar = 24 if status == "EXPIRED" else (index % 5) + 1
            results.append({
                "id": f"{model}-{index:04d}",
                "model": model,
                "symbol": symbol,
                "open_time": open_time,
                "status": status,
                "outcome_bar": outcome_bar,
                "resolved_at": "2026-11-05T12:00:00+00:00",
            })
    assert len(raw_windows) == 573
    return results, raw_windows


def _bundle_rows(results):
    return [{key: value for key, value in row.items() if key != "resolved_at"} for row in results]


def test_builds_618_one_to_one_provenance_records_with_two_distinct_times():
    results, raw_windows = _fixture()
    rows = provenance.build_resolution_provenance_records(
        results, raw_windows, "F08-HIST-BF-test"
    )

    assert len(rows) == 618
    assert sum(row["model"] == "candidate" for row in rows) == 74
    assert sum(row["model"] == "production" for row in rows) == 544
    assert all(row["record_type"] == provenance.PROVENANCE_RECORD_TYPE for row in rows)
    assert all(row["resolved_at"] == "2026-11-05T12:00:00+00:00" for row in rows)
    assert all(row["outcome_bar_close_time"].endswith("Z") for row in rows)
    assert all(
        row["outcome_bar_close_time_ms"] - row["outcome_bar_open_time_ms"] == 900_000
        for row in rows
    )
    # Resolver wall-clock time is preserved, while outcome close comes from OHLC.
    assert rows[0]["resolved_at"] != rows[0]["outcome_bar_close_time"]


def test_expired_outcome_uses_the_24th_subsequent_candle_close():
    results, raw_windows = _fixture()
    rows = provenance.build_resolution_provenance_records(
        results, raw_windows, "F08-HIST-BF-test"
    )
    expired = next(row for row in rows if row["status"] == "EXPIRED")
    assert expired["outcome_bar"] == 24
    assert expired["outcome_bar_open_time_ms"] == (
        expired["prediction_open_time_ms"] + 24 * 900_000
    )
    assert expired["outcome_bar_close_time_ms"] == (
        expired["prediction_open_time_ms"] + 25 * 900_000
    )


def test_verifier_reconciles_every_record_to_outcome_and_raw_window():
    results, raw_windows = _fixture()
    provenance_rows = provenance.build_resolution_provenance_records(
        results, raw_windows, "F08-HIST-BF-test"
    )
    verified = provenance.verify_resolution_provenance_records(
        provenance_rows,
        _bundle_rows(results),
        raw_windows,
        "F08-HIST-BF-test",
    )
    assert verified["status"] == "PASS"
    assert verified["records_verified"] == 618
    assert verified["raw_windows_referenced"] == 573


@pytest.mark.parametrize("bad_timestamp", [
    "2026-11-05T12:00:00+05:30",
    "2026-11-05T12:00:00-00:00",
    "2026-10-16T14:08:36.946361+00:00",
    "not-a-time",
    None,
])
def test_builder_rejects_invalid_or_non_utc_resolved_at(bad_timestamp):
    results, raw_windows = _fixture()
    results[0]["resolved_at"] = bad_timestamp
    with pytest.raises(provenance.ResolutionProvenanceError, match="resolved_at"):
        provenance.build_resolution_provenance_records(
            results, raw_windows, "F08-HIST-BF-test"
        )


def test_verifier_rejects_modified_outcome_bar_close_time():
    results, raw_windows = _fixture()
    provenance_rows = provenance.build_resolution_provenance_records(
        results, raw_windows, "F08-HIST-BF-test"
    )
    tampered = copy.deepcopy(provenance_rows)
    tampered[0]["outcome_bar_close_time_ms"] += 1
    with pytest.raises(provenance.ResolutionProvenanceError, match="provenance mismatch"):
        provenance.verify_resolution_provenance_records(
            tampered, _bundle_rows(results), raw_windows, "F08-HIST-BF-test"
        )


def test_outcome_close_time_uses_selected_raw_bar_even_when_exchange_window_has_a_gap():
    results, raw_windows = _fixture()
    result = results[0]
    window = raw_windows[(result["symbol"], result["open_time"])]
    # Preserve the resolver's row-index semantics: outcome_bar=1 refers to
    # the next returned candle, whose timestamp may be later than one interval
    # if the exchange source omitted an empty interval.
    for index in range(1, len(window["candles"])):
        window["candles"][index]["timestamp"] += 900_000
    rows = provenance.build_resolution_provenance_records(
        [result], raw_windows, "F08-HIST-BF-test",
        expected_counts={"candidate": 1},
    )
    assert rows[0]["outcome_bar"] == 1
    assert rows[0]["outcome_bar_open_time_ms"] == window["candles"][1]["timestamp"]
    assert rows[0]["outcome_bar_close_time_ms"] == window["candles"][1]["timestamp"] + 900_000


def test_verifier_rejects_raw_window_substitution():
    results, raw_windows = _fixture()
    provenance_rows = provenance.build_resolution_provenance_records(
        results, raw_windows, "F08-HIST-BF-test"
    )
    changed_windows = copy.deepcopy(raw_windows)
    first_key = next(iter(changed_windows))
    changed_windows[first_key]["candles"][1]["high"] += 0.5
    with pytest.raises(provenance.ResolutionProvenanceError, match="provenance mismatch"):
        provenance.verify_resolution_provenance_records(
            provenance_rows, _bundle_rows(results), changed_windows, "F08-HIST-BF-test"
        )


def test_verifier_rejects_missing_duplicate_or_unsorted_records():
    results, raw_windows = _fixture()
    provenance_rows = provenance.build_resolution_provenance_records(
        results, raw_windows, "F08-HIST-BF-test"
    )
    with pytest.raises(provenance.ResolutionProvenanceError, match="Expected 618"):
        provenance.verify_resolution_provenance_records(
            provenance_rows[:-1], _bundle_rows(results), raw_windows, "F08-HIST-BF-test"
        )
    duplicate = provenance_rows.copy()
    duplicate[-1] = copy.deepcopy(duplicate[0])
    with pytest.raises(provenance.ResolutionProvenanceError, match="Duplicate provenance"):
        provenance.verify_resolution_provenance_records(
            duplicate, _bundle_rows(results), raw_windows, "F08-HIST-BF-test"
        )
    unsorted = list(reversed(provenance_rows))
    with pytest.raises(provenance.ResolutionProvenanceError, match="not sorted"):
        provenance.verify_resolution_provenance_records(
            unsorted, _bundle_rows(results), raw_windows, "F08-HIST-BF-test"
        )


def test_raw_window_rejects_non_15_minute_resolution_metadata():
    results, raw_windows = _fixture()
    first_key = (results[0]["symbol"], results[0]["open_time"])
    raw_windows[first_key]["request"]["resolution"] = "60"
    with pytest.raises(provenance.ResolutionProvenanceError, match="not 15-minute data"):
        provenance.build_resolution_provenance_records(
            [results[0]], raw_windows, "F08-HIST-BF-test",
            expected_counts={"candidate": 1},
        )


def test_builder_rejects_invalid_outcome_bar_or_missing_exact_window():
    results, raw_windows = _fixture()
    results[0]["outcome_bar"] = 25
    with pytest.raises(provenance.ResolutionProvenanceError, match="outcome_bar"):
        provenance.build_resolution_provenance_records(
            results, raw_windows, "F08-HIST-BF-test"
        )

    results, raw_windows = _fixture()
    raw_windows.pop((results[0]["symbol"], results[0]["open_time"]))
    with pytest.raises(provenance.ResolutionProvenanceError, match="No exact raw OHLC window"):
        provenance.build_resolution_provenance_records(
            results, raw_windows, "F08-HIST-BF-test"
        )
