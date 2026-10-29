import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest
import paper_execution_v2 as paper


INTERVAL = paper.CANDLE_INTERVAL_MS


def make_eligible_state(prediction):
    """State fixture whose prospective cutoff admits this test prediction."""
    state = paper._new_state()
    normalized = paper._normalize_prediction(prediction)
    if normalized is None:
        raise AssertionError("Test prediction must normalize successfully")
    state["prospective_start_time_ms"] = normalized["generated_at_ms"]
    return state



def make_contract(open_time):
    return {
        "contract_version": "1.0",
        "logical_symbol": "ETHUSDT",
        "observation_market": "BINANCE_SPOT",
        "observation_source": {"exchange": "binance", "market_type": "spot"},
        "execution_instrument": None,
        "interval": "15m",
        "open_time": open_time,
        "close_time": open_time + INTERVAL - 1,
        "retrieved_at_ms": open_time + INTERVAL + 1000,
        "htf_sources": {},
        "btc_source": None,
    }


def make_prediction():
    signal_open = (1_700_000_000_000 // INTERVAL) * INTERVAL
    signal_close = signal_open + INTERVAL - 1
    generated_ms = signal_close + 60_000
    generated = datetime.fromtimestamp(
        generated_ms / 1000, tz=timezone.utc
    ).isoformat()
    return {
        "pred_id": f"ETHUSDT_{signal_open}",
        "logical_symbol": "ETHUSDT",
        "symbol": "ETHUSDT",
        "predicted_signal": "BUY",
        "reject_reason": None,
        "open_time": signal_open,
        "interval": "15m",
        "entry_ref": 100.0,
        "atr_ref": 1.0,
        "was_executed": False,
        "observation_contract": make_contract(signal_open),
        "model_version": "test-model",
        "generated_at": generated,
    }


def make_candle(open_time, *, opn=100.0, high=101.0, low=99.0, close=100.0):
    return {
        "open_time": open_time,
        "close_time": open_time + INTERVAL - 1,
        "open": opn,
        "high": high,
        "low": low,
        "close": close,
    }


def setup_trade():
    prediction = paper._normalize_prediction(make_prediction())
    entry_open = prediction["entry_bar_open_time_ms"]
    entry = make_candle(entry_open, opn=123.0, high=124.0, low=122.0, close=123.5)
    return prediction, entry, paper._build_position(prediction, entry)


def test_v2_uses_separate_policy_state_and_ledger():
    assert paper.PAPER_POLICY_VERSION == "stage2e-v2"
    assert paper.PAPER_EXECUTION_MODE == "PAPER_ONLY"
    assert paper.PAPER_STATE_FILE.name == "paper_state_v2.json"
    assert paper.PAPER_EVENTS_FILE.name == "paper_events_v2.jsonl"
    assert paper.PAPER_SCHEMA_VERSION == 2


def test_next_entry_bar_is_strictly_after_generated_at():
    normalized = paper._normalize_prediction(make_prediction())
    generated_ms = normalized["generated_at_ms"]
    entry_open = normalized["entry_bar_open_time_ms"]
    assert entry_open > generated_ms
    assert entry_open % INTERVAL == 0
    assert entry_open - INTERVAL <= generated_ms < entry_open


def test_prediction_with_naive_generated_at_is_rejected():
    prediction = make_prediction()
    prediction["generated_at"] = "2026-10-22T10:00:00"
    with pytest.raises(ValueError, match="timezone"):
        paper._normalize_prediction(prediction)


def test_entry_uses_next_bar_open_not_signal_close(tmp_path):
    prediction = make_prediction()
    normalized = paper._normalize_prediction(prediction)
    entry_open = normalized["entry_bar_open_time_ms"]
    entry = make_candle(entry_open, opn=123.0, high=124.0, low=122.0, close=123.5)
    state = make_eligible_state(prediction)
    events_path = tmp_path / "paper_events_v2.jsonl"

    with patch.object(paper, "PAPER_EVENTS_FILE", events_path), patch.object(
        paper, "fetch_closed_candles", return_value=[entry]
    ):
        dirty, actions = paper._open_eligible_predictions(state, [prediction])

    position = state["positions"][normalized["pred_id"]]
    assert dirty
    assert actions[0]["action"] == "OPENED"
    assert position["entry_price"] == 123.0
    assert position["simulated_entry"] == 123.0
    assert position["entry_price_source"] == "NEXT_BAR_OPEN"
    assert position["entry_bar_open_time_ms"] == entry_open
    assert position["next_candle_open_time_ms"] == entry_open
    assert position["expires_at_ms"] == entry_open + 24 * INTERVAL
    assert position["execution_mode"] == "PAPER_ONLY"
    event = json.loads(events_path.read_text(encoding="utf-8").splitlines()[0])
    assert event["entry_price_source"] == "NEXT_BAR_OPEN"
    assert event["entry_bar_open_time_ms"] == entry_open


def test_entry_waits_when_next_bar_is_not_closed_yet(tmp_path):
    prediction = make_prediction()
    state = make_eligible_state(prediction)
    events_path = tmp_path / "paper_events_v2.jsonl"

    with patch.object(paper, "PAPER_EVENTS_FILE", events_path), patch.object(
        paper, "fetch_closed_candles", return_value=[]
    ):
        dirty, actions = paper._open_eligible_predictions(state, [prediction])

    assert not dirty
    assert actions == []
    assert state["positions"] == {}
    assert state["decisions"] == {}


def test_missing_expected_entry_candle_is_recorded_as_invalidated(tmp_path):
    prediction = make_prediction()
    normalized = paper._normalize_prediction(prediction)
    entry_open = normalized["entry_bar_open_time_ms"]
    state = make_eligible_state(prediction)
    events_path = tmp_path / "paper_events_v2.jsonl"

    with patch.object(paper, "PAPER_EVENTS_FILE", events_path), patch.object(
        paper,
        "fetch_closed_candles",
        return_value=[make_candle(entry_open + INTERVAL)],
    ):
        dirty, actions = paper._open_eligible_predictions(state, [prediction])

    assert dirty
    assert state["positions"] == {}
    assert state["decisions"][normalized["pred_id"]]["status"] == "ENTRY_CANDLE_GAP_INVALIDATED"


def test_monitor_processes_missed_candles_chronologically_and_idempotently(tmp_path):
    _prediction, entry, trade = setup_trade()
    state = paper._new_state()
    state["positions"][trade["pred_id"]] = trade
    events_path = tmp_path / "paper_events_v2.jsonl"
    candles = [make_candle(entry["open_time"] + i * INTERVAL) for i in range(3)]
    processed = []

    def process(current_trade, candle):
        processed.append(candle["open_time"])
        return {
            "modified": False,
            "closed": False,
            "events": [{"event": "CANDLE_OBSERVED"}],
        }

    with patch.object(paper, "PAPER_EVENTS_FILE", events_path), patch.object(
        paper, "fetch_closed_candles", side_effect=[candles, []]
    ), patch.object(paper.monitor, "process_trade_candle", side_effect=process):
        dirty1, actions1 = paper._monitor_open_positions(state)
        dirty2, actions2 = paper._monitor_open_positions(state)

    assert dirty1 and not dirty2
    assert actions1 == [] and actions2 == []
    assert processed == [c["open_time"] for c in candles]
    assert trade["bars_processed"] == 3
    assert trade["last_processed_candle_close_time"] == candles[-1]["close_time"]
    assert trade["next_candle_open_time_ms"] == candles[-1]["open_time"] + INTERVAL
    events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()]
    assert len(events) == 3
    assert len({event["event_id"] for event in events}) == 3


def test_gap_invalidates_position_without_fabricated_pnl(tmp_path):
    _prediction, entry, trade = setup_trade()
    state = paper._new_state()
    state["positions"][trade["pred_id"]] = trade
    events_path = tmp_path / "paper_events_v2.jsonl"
    candles = [
        make_candle(entry["open_time"]),
        make_candle(entry["open_time"] + 2 * INTERVAL),
    ]

    with patch.object(paper, "PAPER_EVENTS_FILE", events_path), patch.object(
        paper, "fetch_closed_candles", return_value=candles
    ), patch.object(
        paper.monitor,
        "process_trade_candle",
        return_value={"modified": False, "closed": False, "events": []},
    ):
        dirty, actions = paper._monitor_open_positions(state)

    assert dirty
    assert trade["status"] == "INVALIDATED"
    assert trade["exit_reason"] == "DATA_GAP_INVALIDATED"
    assert "net_pnl_usd" not in trade
    assert len(actions) == 1
    event = json.loads(events_path.read_text(encoding="utf-8").splitlines()[0])
    assert event["event"] == "PAPER_DATA_GAP_INVALIDATED"


def test_expiry_uses_entry_bar_wall_clock_horizon(tmp_path):
    _prediction, entry, trade = setup_trade()
    state = paper._new_state()
    state["positions"][trade["pred_id"]] = trade
    events_path = tmp_path / "paper_events_v2.jsonl"
    candles = [
        make_candle(entry["open_time"] + i * INTERVAL, opn=100.0, high=101.0, low=99.0, close=100.0)
        for i in range(24)
    ]
    with patch.object(paper, "PAPER_EVENTS_FILE", events_path), patch.object(
        paper, "fetch_closed_candles", return_value=candles
    ), patch.object(
        paper.monitor,
        "process_trade_candle",
        return_value={"modified": False, "closed": False, "events": []},
    ):
        dirty, actions = paper._monitor_open_positions(state)

    assert dirty
    assert trade["status"] == "CLOSED"
    assert trade["exit_reason"] == "EXPIRED"
    assert trade["resolved_candle_close_time"] == candles[-1]["close_time"]
    assert trade["expires_at_ms"] == candles[-1]["close_time"] + 1
    assert actions[-1]["reason"] == "EXPIRED"


def test_history_fetch_failure_preserves_cursor_and_position(tmp_path):
    _prediction, _entry, trade = setup_trade()
    state = paper._new_state()
    state["positions"][trade["pred_id"]] = trade
    original_cursor = trade["next_candle_open_time_ms"]

    with patch.object(paper, "fetch_closed_candles", side_effect=RuntimeError("offline")):
        report = paper._new_lifecycle_report()
        dirty, actions = paper._monitor_open_positions(state, report=report)

    assert not dirty
    assert actions == []
    assert trade["status"] == "ACTIVE"
    assert trade["next_candle_open_time_ms"] == original_cursor
    assert report["history_fetch_failed"] == 1


def test_run_lock_wrapper_calls_v2_runner_under_file_lock():
    with patch.object(paper, "_run_once_v2_impl", return_value={"ok": True}) as runner, patch.object(
        paper, "PAPER_STATE_FILE", Path("paper_state_v2_test.json")
    ):
        assert paper.run_once() == {"ok": True}
        runner.assert_called_once()


class FakeBinanceResponse:
    status = 200

    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


def raw_kline(open_time):
    return [
        open_time,
        "100.0",
        "103.0",
        "99.0",
        "101.0",
        "12.0",
        open_time + INTERVAL - 1,
        "1200.0",
        25,
        "6.0",
        "600.0",
        "0",
    ]


def test_fetch_closed_candles_orders_rows_and_excludes_unclosed_bars():
    start = (1_700_000_000_000 // INTERVAL) * INTERVAL
    rows = [
        raw_kline(start + INTERVAL),
        raw_kline(start),
        raw_kline(start + 2 * INTERVAL),
    ]
    end_time = start + 2 * INTERVAL - 1

    with patch.object(
        paper,
        "urlopen",
        return_value=FakeBinanceResponse(rows),
    ) as request:
        candles = paper.fetch_closed_candles(
            "ETHUSDT",
            start,
            end_time,
        )

    assert [c["open_time"] for c in candles] == [start, start + INTERVAL]
    assert [c["close_time"] for c in candles] == [
        start + INTERVAL - 1,
        start + 2 * INTERVAL - 1,
    ]
    request.assert_called_once()


def test_fetch_closed_candles_paginates_without_skipping_or_duplicating():
    start = (1_700_000_000_000 // INTERVAL) * INTERVAL
    all_rows = [
        raw_kline(start + index * INTERVAL)
        for index in range(paper.PAPER_CANDLE_PAGE_SIZE + 5)
    ]
    first_page = all_rows[:paper.PAPER_CANDLE_PAGE_SIZE]
    second_page = all_rows[paper.PAPER_CANDLE_PAGE_SIZE:]
    end_time = (
        start
        + (len(all_rows) - 1) * INTERVAL
        + INTERVAL
        - 1
    )

    with patch.object(
        paper,
        "urlopen",
        side_effect=[
            FakeBinanceResponse(first_page),
            FakeBinanceResponse(second_page),
        ],
    ) as request:
        candles = paper.fetch_closed_candles(
            "ETHUSDT",
            start,
            end_time,
        )

    expected_opens = [start + i * INTERVAL for i in range(len(all_rows))]
    assert [c["open_time"] for c in candles] == expected_opens
    assert len({c["open_time"] for c in candles}) == len(all_rows)
    assert request.call_count == 2


def test_capacity_full_decision_is_sticky_after_entry_candle_closes(tmp_path):
    prediction = make_prediction()
    normalized = paper._normalize_prediction(prediction)
    entry_open = normalized["entry_bar_open_time_ms"]
    state = make_eligible_state(prediction)
    for index in range(paper.PAPER_MAX_OPEN_POSITIONS):
        state["positions"][f"dummy-{index}"] = {
            "status": "ACTIVE",
            "symbol": f"DUMMY{index}USDT",
            "side": "SELL",
        }

    with patch.object(paper, "PAPER_EVENTS_FILE", tmp_path / "events.jsonl"), patch.object(
        paper, "fetch_closed_candles", return_value=[make_candle(entry_open)]
    ):
        dirty, _ = paper._open_eligible_predictions(state, [prediction])

    assert dirty
    assert state["decisions"][normalized["pred_id"]]["status"] == "CAPACITY_FULL"

    state["positions"].pop("dummy-0")
    with patch.object(paper, "PAPER_EVENTS_FILE", tmp_path / "events.jsonl"), patch.object(
        paper, "fetch_closed_candles", return_value=[make_candle(entry_open)]
    ):
        _, actions = paper._open_eligible_predictions(state, [prediction])

    assert actions == []
    assert normalized["pred_id"] not in state["positions"]


def test_direction_limit_decision_is_sticky_after_entry_candle_closes(tmp_path):
    prediction = make_prediction()
    normalized = paper._normalize_prediction(prediction)
    entry_open = normalized["entry_bar_open_time_ms"]
    state = make_eligible_state(prediction)
    for index in range(paper.PAPER_MAX_SAME_DIRECTION):
        state["positions"][f"same-side-{index}"] = {
            "status": "ACTIVE",
            "symbol": f"OTHER{index}USDT",
            "side": "BUY",
        }

    with patch.object(paper, "PAPER_EVENTS_FILE", tmp_path / "events.jsonl"), patch.object(
        paper, "fetch_closed_candles", return_value=[make_candle(entry_open)]
    ):
        dirty, actions = paper._open_eligible_predictions(state, [prediction])

    assert dirty
    assert actions == []
    assert state["decisions"][normalized["pred_id"]]["status"] == "DIRECTION_LIMIT"


def test_full_capacity_does_not_reject_unclosed_future_entry(tmp_path):
    prediction = make_prediction()
    normalized = paper._normalize_prediction(prediction)
    state = make_eligible_state(prediction)
    for index in range(paper.PAPER_MAX_OPEN_POSITIONS):
        state["positions"][f"dummy-{index}"] = {
            "status": "ACTIVE",
            "symbol": f"DUMMY{index}USDT",
            "side": "SELL",
        }

    with patch.object(paper, "PAPER_EVENTS_FILE", tmp_path / "events.jsonl"), patch.object(
        paper, "fetch_closed_candles", return_value=[]
    ):
        dirty, actions = paper._open_eligible_predictions(state, [prediction])

    assert not dirty
    assert actions == []
    assert normalized["pred_id"] not in state["decisions"]


def test_module_does_not_import_live_execution_path():
    source = Path(paper.__file__).read_text(encoding="utf-8").lower()
    for forbidden in (
        "deribitclient",
        "execute_trade",
        "place_market_order",
        "place_limit_order",
    ):
        assert forbidden not in source


def test_v2_first_run_persists_fixed_prospective_baseline(monkeypatch, tmp_path):
    import json
    import paper_execution_v2 as paper

    state_path = tmp_path / "paper_state_v2.json"
    events_path = tmp_path / "paper_events_v2.jsonl"
    predictions_path = tmp_path / "predictions.json"

    archived_timestamp = "2030-01-01T00:00:00+00:00"
    predictions_path.write_text(
        json.dumps([{"generated_at": archived_timestamp}]),
        encoding="utf-8",
    )

    monkeypatch.setattr(paper, "PAPER_STATE_FILE", state_path)
    monkeypatch.setattr(paper, "PAPER_EVENTS_FILE", events_path)
    monkeypatch.setattr(paper, "PREDICTIONS_FILE", predictions_path)

    paper.run_once()

    first_state = json.loads(state_path.read_text(encoding="utf-8"))
    first_cutoff = first_state["prospective_start_time_ms"]
    archived_ms = paper._parse_generated_at_ms(archived_timestamp)

    assert first_state["prospective_initialized_at_ms"] > 0
    assert first_cutoff > archived_ms
    assert (
        first_state["prospective_baseline_max_prediction_generated_at_ms"]
        == archived_ms
    )

    # A changed archive must not advance a persisted baseline.
    predictions_path.write_text(
        json.dumps([{"generated_at": "2030-01-02T00:00:00+00:00"}]),
        encoding="utf-8",
    )
    paper.run_once()

    second_state = json.loads(state_path.read_text(encoding="utf-8"))
    assert second_state["prospective_start_time_ms"] == first_cutoff
    assert (
        second_state["prospective_baseline_max_prediction_generated_at_ms"]
        == archived_ms
    )


def test_v2_excludes_pre_baseline_predictions_and_attempts_new_ones(monkeypatch):
    import paper_execution_v2 as paper

    state = paper._new_state()
    state["prospective_initialized_at_ms"] = 1_999
    state["prospective_start_time_ms"] = 2_000
    calls = []

    def fake_normalize(raw):
        generated_ms = 1_999 if raw["pred_id"] == "old" else 2_001
        return {
            "pred_id": raw["pred_id"],
            "symbol": "BTCUSDT" if raw["pred_id"] == "old" else "ETHUSDT",
            "side": "BUY",
            "open_time": generated_ms,
            "generated_at": str(generated_ms),
            "generated_at_ms": generated_ms,
            "entry_bar_open_time_ms": 9_000,
        }

    def fake_fetch(symbol, start_open_time_ms, *args, **kwargs):
        calls.append((symbol, start_open_time_ms))
        return []

    monkeypatch.setattr(paper, "_normalize_prediction", fake_normalize)
    monkeypatch.setattr(paper, "fetch_closed_candles", fake_fetch)

    report = paper._new_decision_report()
    dirty, actions = paper._open_eligible_predictions(
        state,
        [{"pred_id": "old"}, {"pred_id": "new"}],
        report=report,
    )

    assert report["pre_baseline_predictions"] == 1
    assert calls == [("ETHUSDT", 9_000)]
    assert report["waiting_for_signal_candle"] == 1
    assert dirty is False
    assert actions == []
    assert state["positions"] == {}
    assert state["decisions"] == {}


def test_run_once_is_idempotent_across_persisted_restarts(
    monkeypatch, tmp_path
):
    prediction = make_prediction()
    normalized = paper._normalize_prediction(prediction)
    pred_id = normalized["pred_id"]
    entry_open = normalized["entry_bar_open_time_ms"]

    candles = [
        make_candle(
            entry_open,
            opn=123.0,
            high=124.0,
            low=122.0,
            close=123.5,
        ),
        make_candle(
            entry_open + INTERVAL,
            opn=123.5,
            high=124.0,
            low=123.0,
            close=123.4,
        ),
        make_candle(
            entry_open + 2 * INTERVAL,
            opn=123.4,
            high=124.0,
            low=123.0,
            close=123.6,
        ),
    ]

    state_path = tmp_path / "paper_state_v2.json"
    events_path = tmp_path / "paper_events_v2.jsonl"
    predictions_path = tmp_path / "predictions.json"

    # Create a valid persisted prospective state that admits this fixture.
    state = paper._new_state()
    generated_ms = normalized["generated_at_ms"]
    initialized_ms = generated_ms - 2
    baseline_ms = generated_ms - 1

    state["prospective_initialized_at_ms"] = initialized_ms
    state["prospective_initialized_at_utc"] = datetime.fromtimestamp(
        initialized_ms / 1000, timezone.utc
    ).isoformat()
    state["prospective_start_time_ms"] = baseline_ms
    state["prospective_start_at_utc"] = datetime.fromtimestamp(
        baseline_ms / 1000, timezone.utc
    ).isoformat()

    state_path.write_text(json.dumps(state), encoding="utf-8")
    predictions_path.write_text(
        json.dumps([prediction]), encoding="utf-8"
    )

    monkeypatch.setattr(paper, "PAPER_STATE_FILE", state_path)
    monkeypatch.setattr(paper, "PAPER_EVENTS_FILE", events_path)
    monkeypatch.setattr(paper, "PREDICTIONS_FILE", predictions_path)

    def fake_fetch_closed_candles(
        symbol, start_open_time_ms, end_time_ms=None
    ):
        start_ms = int(start_open_time_ms)
        return [
            candle
            for candle in candles
            if int(candle["open_time"]) >= start_ms
        ]

    monkeypatch.setattr(
        paper, "fetch_closed_candles", fake_fetch_closed_candles
    )

    # First run opens the paper position and consumes the supplied candles.
    first_result = paper.run_once()
    first_state = json.loads(state_path.read_text(encoding="utf-8"))
    first_events = [
        json.loads(line)
        for line in events_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert first_result["reconciliation"]["ok"] is True
    assert first_result["decisions"]["opened"] == 1
    assert first_result["lifecycle"]["candles_processed"] == 3
    assert first_result["state_changed"] is True
    assert first_state["positions"][pred_id]["bars_processed"] == 3
    assert first_state["positions"][pred_id]["next_candle_open_time_ms"] == (
        candles[-1]["open_time"] + INTERVAL
    )

    first_event_ids = [event["event_id"] for event in first_events]
    assert len(first_event_ids) == len(set(first_event_ids))

    # A second run reloads persisted state and must not repeat completed work.
    second_result = paper.run_once()
    second_state = json.loads(state_path.read_text(encoding="utf-8"))
    second_events = [
        json.loads(line)
        for line in events_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert second_result["reconciliation"]["ok"] is True
    assert second_result["decisions"]["opened"] == 0
    assert second_result["decisions"]["already_processed"] == 1
    assert second_result["lifecycle"]["candles_processed"] == 0
    assert second_result["state_changed"] is False
    assert second_state == first_state
    assert second_events == first_events


def test_run_once_recovers_after_event_append_before_state_save(
    monkeypatch, tmp_path
):
    prediction = make_prediction()
    normalized = paper._normalize_prediction(prediction)
    pred_id = normalized["pred_id"]
    entry_open = normalized["entry_bar_open_time_ms"]

    candles = [
        make_candle(
            entry_open, opn=123.0, high=124.0, low=122.0, close=123.5
        ),
        make_candle(
            entry_open + INTERVAL,
            opn=123.5, high=127.0, low=123.0, close=126.0,
        ),
        make_candle(
            entry_open + 2 * INTERVAL,
            opn=126.0, high=126.2, low=123.0, close=125.5,
        ),
    ]

    state_path = tmp_path / "paper_state_v2.json"
    events_path = tmp_path / "paper_events_v2.jsonl"
    predictions_path = tmp_path / "predictions.json"

    # An existing valid baseline prevents first-run initialization from
    # calling _save_state before the simulated failure point.
    state = paper._new_state()
    generated_ms = normalized["generated_at_ms"]
    initialized_ms = generated_ms - 2
    baseline_ms = generated_ms - 1

    state["prospective_initialized_at_ms"] = initialized_ms
    state["prospective_initialized_at_utc"] = datetime.fromtimestamp(
        initialized_ms / 1000, timezone.utc
    ).isoformat()
    state["prospective_start_time_ms"] = baseline_ms
    state["prospective_start_at_utc"] = datetime.fromtimestamp(
        baseline_ms / 1000, timezone.utc
    ).isoformat()

    state_path.write_text(json.dumps(state), encoding="utf-8")
    predictions_path.write_text(
        json.dumps([prediction]), encoding="utf-8"
    )

    monkeypatch.setattr(paper, "PAPER_STATE_FILE", state_path)
    monkeypatch.setattr(paper, "PAPER_EVENTS_FILE", events_path)
    monkeypatch.setattr(paper, "PREDICTIONS_FILE", predictions_path)

    def fake_fetch_closed_candles(
        symbol, start_open_time_ms, end_time_ms=None
    ):
        start_ms = int(start_open_time_ms)
        return [
            candle for candle in candles
            if int(candle["open_time"]) >= start_ms
        ]

    monkeypatch.setattr(
        paper, "fetch_closed_candles", fake_fetch_closed_candles
    )

    original_save_state = paper._save_state

    def simulated_save_failure(_state):
        raise OSError("simulated failure before state persistence")

    # The run should write its events, then fail at the final state save.
    monkeypatch.setattr(paper, "_save_state", simulated_save_failure)
    with pytest.raises(OSError, match="simulated failure"):
        paper.run_once()

    disk_state_after_failure = json.loads(
        state_path.read_text(encoding="utf-8")
    )
    events_after_failure = [
        json.loads(line)
        for line in events_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    # The original state remains on disk, but the OPEN and TP1 events exist.
    assert disk_state_after_failure["positions"] == {}
    assert sum(
        event["event_id"] == f"OPEN:{pred_id}"
        for event in events_after_failure
    ) == 1
    assert sum(
        event.get("event") == "PARTIAL_TP1_FILLED"
        for event in events_after_failure
    ) == 1

    # Restore persistence and replay the same deterministic candle history.
    monkeypatch.setattr(paper, "_save_state", original_save_state)
    retry_result = paper.run_once()

    persisted_state = json.loads(state_path.read_text(encoding="utf-8"))
    persisted_events = [
        json.loads(line)
        for line in events_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert retry_result["reconciliation"]["ok"] is True
    position = persisted_state["positions"][pred_id]
    assert position["status"] == "ACTIVE"
    assert position["state"] == "PARTIAL_TP1"
    assert position["bars_processed"] == 3
    assert position["remaining_qty"] == pytest.approx(position["qty"] * 0.5)

    event_ids = [event["event_id"] for event in persisted_events]
    assert len(event_ids) == len(set(event_ids))
    assert sum(event_id == f"OPEN:{pred_id}" for event_id in event_ids) == 1

    tp1_events = [
        event for event in persisted_events
        if event.get("event") == "PARTIAL_TP1_FILLED"
    ]
    assert len(tp1_events) == 1
    assert position["tp1_realized_gross_pnl"] == pytest.approx(
        tp1_events[0]["tp1_realized_gross_pnl"]
    )
