from __future__ import annotations

import pandas as pd
import pytest
import requests

import phase2d_prospective_validator as validator


class FakeResponse:
    def __init__(self, payload=None, status_code=200):
        self._payload = payload if payload is not None else {"result": {}}
        self.status_code = status_code

    def json(self):
        return self._payload


def _payload(times):
    return {
        "result": {
            "ticks": list(times),
            "open": [100.0 + i for i, _ in enumerate(times)],
            "high": [101.0 + i for i, _ in enumerate(times)],
            "low": [99.0 + i for i, _ in enumerate(times)],
            "close": [100.5 + i for i, _ in enumerate(times)],
            "volume": [10.0 + i for i, _ in enumerate(times)],
        }
    }


def test_range_fetch_paginates_sorts_deduplicates_and_bounds_each_request(monkeypatch):
    start = ((1_700_000_000_000 // validator.ENTRY_MS) * validator.ENTRY_MS)
    interval = validator.ENTRY_MS
    requested = []

    def fake_get(_url, *, params, timeout):
        requested.append((dict(params), timeout))
        page_start = params["start_timestamp"]
        page_end = params["end_timestamp"] + 1
        times = list(range(page_start, page_end, interval))
        # A duplicate inside the first returned page must not duplicate output.
        payload = _payload(times)
        if len(requested) == 1 and times:
            # Identical duplicate at a candle boundary is safely de-duplicated.
            times = [times[1], times[0], times[0], *times[2:]]
            payload = _payload(times)
            for column in ("open", "high", "low", "close", "volume"):
                payload["result"][column][2] = payload["result"][column][1]
            payload["result"]["ticks"] = times
        return FakeResponse(payload)

    monkeypatch.setattr(validator.requests, "get", fake_get)
    result = validator.fetch_deribit_range(
        "ETHUSDT", "15", start, start + 7 * interval,
        page_size=3, max_pages=3,
    )

    assert len(requested) == 3
    assert [p[0]["start_timestamp"] for p in requested] == [
        start, start + 3 * interval, start + 6 * interval
    ]
    assert all(p[0]["end_timestamp"] < p[0]["start_timestamp"] + 3 * interval for p in requested)
    assert result["open_time"].tolist() == [start + i * interval for i in range(7)]
    assert result["open_time"].is_unique
    assert result["open_time"].is_monotonic_increasing
    assert result.attrs["pages_fetched"] == 3
    assert (result["close_time"] <= int(__import__("time").time() * 1000)).all()


def test_range_fetch_rejects_requests_exceeding_page_budget_without_network(monkeypatch):
    called = False

    def fake_get(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("network must not be called")

    monkeypatch.setattr(validator.requests, "get", fake_get)
    with pytest.raises(validator.HistoricalRangeLimitError):
        validator.fetch_deribit_range(
            "ETHUSDT", "15", ((1_700_000_000_000 // validator.ENTRY_MS) * validator.ENTRY_MS),
            ((1_700_000_000_000 // validator.ENTRY_MS) * validator.ENTRY_MS) + 10 * validator.ENTRY_MS,
            page_size=3, max_pages=2,
        )
    assert called is False


@pytest.mark.parametrize("status", [429, 500, 503])
def test_range_fetch_classifies_rate_limit_and_server_errors_as_transient(monkeypatch, status):
    monkeypatch.setattr(
        validator.requests, "get",
        lambda *args, **kwargs: FakeResponse(status_code=status),
    )
    with pytest.raises(validator.TransientMarketDataError):
        validator.fetch_deribit_range(
            "ETHUSDT", "15", ((1_700_000_000_000 // validator.ENTRY_MS) * validator.ENTRY_MS),
            ((1_700_000_000_000 // validator.ENTRY_MS) * validator.ENTRY_MS) + validator.ENTRY_MS,
        )


def test_range_fetch_classifies_network_timeout_as_transient(monkeypatch):
    def fail(*args, **kwargs):
        raise requests.Timeout("test timeout")

    monkeypatch.setattr(validator.requests, "get", fail)
    with pytest.raises(validator.TransientMarketDataError):
        validator.fetch_deribit_range(
            "ETHUSDT", "15", ((1_700_000_000_000 // validator.ENTRY_MS) * validator.ENTRY_MS),
            ((1_700_000_000_000 // validator.ENTRY_MS) * validator.ENTRY_MS) + validator.ENTRY_MS,
        )


def test_range_fetch_classifies_http_404_as_unavailable(monkeypatch):
    monkeypatch.setattr(
        validator.requests, "get",
        lambda *args, **kwargs: FakeResponse(status_code=404),
    )
    with pytest.raises(validator.HistoricalDataUnavailableError):
        validator.fetch_deribit_range(
            "ETHUSDT", "15", ((1_700_000_000_000 // validator.ENTRY_MS) * validator.ENTRY_MS),
            ((1_700_000_000_000 // validator.ENTRY_MS) * validator.ENTRY_MS) + validator.ENTRY_MS,
        )


def test_range_fetch_classifies_successful_empty_range_as_unavailable(monkeypatch):
    monkeypatch.setattr(
        validator.requests, "get",
        lambda *args, **kwargs: FakeResponse({"result": {"ticks": [], "open": [], "high": [], "low": [], "close": [], "volume": []}}),
    )
    with pytest.raises(validator.HistoricalDataUnavailableError):
        validator.fetch_deribit_range(
            "ETHUSDT", "15", ((1_700_000_000_000 // validator.ENTRY_MS) * validator.ENTRY_MS),
            ((1_700_000_000_000 // validator.ENTRY_MS) * validator.ENTRY_MS) + validator.ENTRY_MS,
        )


def test_range_fetch_rejects_malformed_ohlcv_arrays_as_transient(monkeypatch):
    malformed = _payload([((1_700_000_000_000 // validator.ENTRY_MS) * validator.ENTRY_MS), ((1_700_000_000_000 // validator.ENTRY_MS) * validator.ENTRY_MS) + validator.ENTRY_MS])
    malformed["result"]["close"] = [100.0]
    monkeypatch.setattr(
        validator.requests, "get", lambda *args, **kwargs: FakeResponse(malformed)
    )
    with pytest.raises(validator.TransientMarketDataError):
        validator.fetch_deribit_range(
            "ETHUSDT", "15", ((1_700_000_000_000 // validator.ENTRY_MS) * validator.ENTRY_MS),
            ((1_700_000_000_000 // validator.ENTRY_MS) * validator.ENTRY_MS) + 2 * validator.ENTRY_MS,
        )



def test_range_fetch_rejects_conflicting_duplicate_candles(monkeypatch):
    start = ((1_700_000_000_000 // validator.ENTRY_MS) * validator.ENTRY_MS)
    times = [start, start]
    payload = _payload(times)
    # Same timestamp but conflicting close is not a safe de-duplication case.
    payload["result"]["close"][1] += 5.0
    monkeypatch.setattr(
        validator.requests, "get", lambda *args, **kwargs: FakeResponse(payload)
    )
    with pytest.raises(validator.TransientMarketDataError, match="conflicting OHLCV"):
        validator.fetch_deribit_range(
            "ETHUSDT", "15", start, start + validator.ENTRY_MS
        )


def test_range_fetch_requires_interval_aligned_bounds(monkeypatch):
    monkeypatch.setattr(
        validator.requests, "get",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("network called")),
    )
    with pytest.raises(ValueError, match="align"):
        validator.fetch_deribit_range(
            "ETHUSDT", "15", 1_700_000_000_001,
            1_700_000_000_001 + validator.ENTRY_MS,
        )

def _complete_window(entry_open):
    opens = [entry_open + i * validator.ENTRY_MS for i in range(25)]
    return pd.DataFrame({
        "open_time": opens,
        "close_time": [value + validator.ENTRY_MS for value in opens],
        "open": [100.0] * 25,
        "high": [101.0] * 25,
        "low": [99.0] * 25,
        "close": [100.5] * 25,
        "volume": [10.0] * 25,
    })


def test_prediction_window_requires_exact_entry_plus_24_completed_followups():
    entry = ((1_700_000_000_000 // validator.ENTRY_MS) * validator.ENTRY_MS)
    pred = {"open_time": entry}
    window = _complete_window(entry)
    now = entry + 25 * validator.ENTRY_MS
    assert validator.prediction_window_status(pred, window, now) == "COMPLETE"

    missing = window.drop(index=12)
    assert validator.prediction_window_status(pred, missing, now) == "CANDLE_GAP"

    duplicate = pd.concat([window, window.iloc[[12]]], ignore_index=True)
    assert validator.prediction_window_status(pred, duplicate, now) == "DUPLICATE_CANDLES"

    without_entry = window.iloc[1:].copy()
    assert validator.prediction_window_status(pred, without_entry, now) == "HISTORY_UNAVAILABLE"

    intruder = pd.concat([window.iloc[:8], window.iloc[[8]].assign(open_time=entry + 8 * validator.ENTRY_MS + 60_000, close_time=entry + 9 * validator.ENTRY_MS + 60_000), window.iloc[8:]], ignore_index=True)
    assert validator.prediction_window_status(pred, intruder, now) == "CANDLE_GAP"

    out_of_order = window.iloc[[0, 2, 1, *range(3, 25)]].reset_index(drop=True)
    assert validator.prediction_window_status(pred, out_of_order, now) == "CANDLE_GAP"


def test_prediction_window_does_not_resolve_before_final_followup_is_closed():
    entry = ((1_700_000_000_000 // validator.ENTRY_MS) * validator.ENTRY_MS)
    pred = {"open_time": entry}
    window = _complete_window(entry)
    final_close = entry + 25 * validator.ENTRY_MS
    assert validator.prediction_window_status(pred, window, final_close - 1) == "NOT_MATURE"

    late_close = window.copy()
    late_close.loc[24, "close_time"] = final_close + 11
    assert validator.prediction_window_status(pred, late_close, final_close + 10) == "NOT_MATURE"


def test_prediction_window_rejects_invalid_prediction_timestamp():
    assert validator.prediction_window_status({}, pd.DataFrame(), 2_000_000_000_000) == "INVALID_PREDICTION_TIME"

def _pending_resolution_metrics():
    return {
        "pending_not_mature": 0,
        "history_window_fetches": 0,
        "history_window_cache_hits": 0,
        "transient_fetch_failures": 0,
        "history_unavailable": 0,
        "history_gaps": 0,
        "range_fetch_budget_exhausted": 0,
    }


def test_pending_resolution_uses_historical_fallback(monkeypatch):
    entry = (1_700_000_000_000 // validator.ENTRY_MS) * validator.ENTRY_MS
    prediction = {"id": "p1", "symbol": "ETHUSDT", "open_time": entry}
    now_ms = entry + 25 * validator.ENTRY_MS
    historical = _complete_window(entry)
    calls = []

    def fake_fetch(symbol, open_time):
        calls.append((symbol, open_time))
        return historical

    monkeypatch.setattr(validator, "fetch_deribit_15m_window", fake_fetch)

    metrics = _pending_resolution_metrics()
    cache = {}
    fetch_state = {"fetches": 0}

    frame, status = validator.fetch_pending_resolution_window(
        prediction,
        {"ETHUSDT": pd.DataFrame()},
        now_ms,
        cache,
        fetch_state,
        metrics,
    )

    assert status == "COMPLETE"
    assert frame is historical
    assert calls == [("ETHUSDT", entry)]
    assert fetch_state["fetches"] == 1
    assert metrics["history_window_fetches"] == 1


def test_pending_resolution_historical_fallback_is_cached_across_models(monkeypatch):
    entry = (1_700_000_000_000 // validator.ENTRY_MS) * validator.ENTRY_MS
    prediction = {"id": "p1", "symbol": "ETHUSDT", "open_time": entry}
    now_ms = entry + 25 * validator.ENTRY_MS
    historical = _complete_window(entry)
    calls = []

    def fake_fetch(symbol, open_time):
        calls.append((symbol, open_time))
        return historical

    monkeypatch.setattr(validator, "fetch_deribit_15m_window", fake_fetch)

    cache = {}
    fetch_state = {"fetches": 0}
    candidate_metrics = _pending_resolution_metrics()
    production_metrics = _pending_resolution_metrics()
    recent = {"ETHUSDT": pd.DataFrame()}

    first = validator.fetch_pending_resolution_window(
        prediction, recent, now_ms, cache, fetch_state, candidate_metrics
    )
    second = validator.fetch_pending_resolution_window(
        prediction, recent, now_ms, cache, fetch_state, production_metrics
    )

    assert first[1] == second[1] == "COMPLETE"
    assert calls == [("ETHUSDT", entry)]
    assert fetch_state["fetches"] == 1
    assert candidate_metrics["history_window_fetches"] == 1
    assert production_metrics["history_window_fetches"] == 0
    assert production_metrics["history_window_cache_hits"] == 1


def test_pending_resolution_does_not_fetch_before_window_matures(monkeypatch):
    entry = (1_700_000_000_000 // validator.ENTRY_MS) * validator.ENTRY_MS
    prediction = {"id": "p1", "symbol": "ETHUSDT", "open_time": entry}
    now_ms = entry + 25 * validator.ENTRY_MS - 1

    def forbidden_fetch(*args):
        raise AssertionError("Historical fetch must not run before maturity")

    monkeypatch.setattr(
        validator, "fetch_deribit_15m_window", forbidden_fetch
    )

    metrics = _pending_resolution_metrics()
    frame, status = validator.fetch_pending_resolution_window(
        prediction,
        {"ETHUSDT": pd.DataFrame()},
        now_ms,
        {},
        {"fetches": 0},
        metrics,
    )

    assert status == "NOT_MATURE"
    assert metrics["pending_not_mature"] == 1
    assert metrics["history_window_fetches"] == 0


def test_pending_resolution_respects_per_run_fetch_budget(monkeypatch):
    entry = (1_700_000_000_000 // validator.ENTRY_MS) * validator.ENTRY_MS
    prediction = {"id": "p1", "symbol": "ETHUSDT", "open_time": entry}
    now_ms = entry + 25 * validator.ENTRY_MS

    def forbidden_fetch(*args):
        raise AssertionError("Fetch budget must prevent network access")

    monkeypatch.setattr(
        validator, "fetch_deribit_15m_window", forbidden_fetch
    )

    metrics = _pending_resolution_metrics()
    frame, status = validator.fetch_pending_resolution_window(
        prediction,
        {"ETHUSDT": pd.DataFrame()},
        now_ms,
        {},
        {"fetches": validator.MAX_PENDING_RANGE_FETCHES_PER_RUN},
        metrics,
    )

    assert status == "FETCH_BUDGET_EXHAUSTED"
    assert frame is None
    assert metrics["range_fetch_budget_exhausted"] == 1
    assert metrics["history_window_fetches"] == 0


def test_pending_resolution_retains_prediction_after_transient_fetch_failure(monkeypatch):
    entry = (1_700_000_000_000 // validator.ENTRY_MS) * validator.ENTRY_MS
    prediction = {"id": "p1", "symbol": "ETHUSDT", "open_time": entry}
    now_ms = entry + 25 * validator.ENTRY_MS

    def failed_fetch(*args):
        raise validator.TransientMarketDataError("simulated timeout")

    monkeypatch.setattr(
        validator, "fetch_deribit_15m_window", failed_fetch
    )

    metrics = _pending_resolution_metrics()
    frame, status = validator.fetch_pending_resolution_window(
        prediction,
        {"ETHUSDT": pd.DataFrame()},
        now_ms,
        {},
        {"fetches": 0},
        metrics,
    )

    assert status == "TRANSIENT_FETCH_FAILURE"
    assert frame is None
    assert metrics["transient_fetch_failures"] == 1


def test_main_pending_resolution_loop_uses_tested_fallback_helper():
    import inspect

    source = inspect.getsource(validator.main)
    assert "fetch_pending_resolution_window(" in source
