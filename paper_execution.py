#!/usr/bin/env python3
"""Stage 2E paper execution and reconciliation.

PAPER ONLY:
- Consumes accepted production predictions from predictions.json.
- Requires a valid Stage 2C Observation Contract.
- Uses the existing closed-candle shadow lifecycle for TP1/TP2/SL.
- Adds the fixed 24-bar prospective horizon as an EXPIRED outcome.
- Never imports or calls the live exchange execution path.
- Never mutates predictions.json or production trade state.

The paper layer is intentionally separate from F08 evidence and from the
candidate shadow state/ledger.
"""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

from filelock import FileLock

import candidate_market_monitor as monitor
from execution_policy import fee, gross_pnl
from market_data_integrity import (
    observation_identity,
    validate_observation_contract,
)

PAPER_STATE_FILE = Path("paper_state.json")
PAPER_EVENTS_FILE = Path("paper_events.jsonl")
PREDICTIONS_FILE = Path("predictions.json")

PAPER_SCHEMA_VERSION = 1
PAPER_POLICY_VERSION = "stage2e-v1"
PAPER_EXECUTION_MODE = "PAPER_ONLY"

PAPER_INITIAL_BALANCE_USD = 10_000.0
PAPER_RISK_PER_TRADE = 0.03
PAPER_MAX_OPEN_POSITIONS = 8
PAPER_MAX_SAME_DIRECTION = 4

PAPER_STOP_MULT = 2.5
PAPER_TP1_MULT = 3.5
PAPER_TP2_MULT = 7.5
PAPER_FEE_RATE = 0.0006

CANDLE_INTERVAL_MS = 15 * 60 * 1000
PAPER_HORIZON_BARS = 24


def _load_json(path: Path, default):
    if not path.exists():
        return default
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return default


def _save_json_atomic(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp, path)


def _read_events() -> list[dict]:
    if not PAPER_EVENTS_FILE.exists():
        return []

    events = []
    with PAPER_EVENTS_FILE.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Corrupt paper event ledger at line {line_no}: {exc}"
                ) from exc
            if not isinstance(record, dict):
                raise RuntimeError(
                    f"Paper event at line {line_no} is not an object"
                )
            events.append(record)
    return events


def _event_exists(event_id: str) -> bool:
    return any(
        item.get("event_id") == event_id
        for item in _read_events()
    )


def _append_event(event: dict) -> bool:
    event_id = str(event.get("event_id") or "").strip()
    if not event_id:
        raise ValueError("Paper event requires event_id")

    lock = FileLock(str(PAPER_EVENTS_FILE) + ".lock")
    with lock:
        if _event_exists(event_id):
            return False

        PAPER_EVENTS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with PAPER_EVENTS_FILE.open(
            "a",
            encoding="utf-8",
            newline="\n",
        ) as handle:
            handle.write(
                json.dumps(
                    event,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )
    return True


def _new_state() -> dict:
    return {
        "schema_version": PAPER_SCHEMA_VERSION,
        "policy_version": PAPER_POLICY_VERSION,
        "execution_mode": PAPER_EXECUTION_MODE,
        "initial_balance_usd": PAPER_INITIAL_BALANCE_USD,
        "risk_per_trade": PAPER_RISK_PER_TRADE,
        "positions": {},
        "decisions": {},
    }


def _load_state() -> dict:
    state = _load_json(PAPER_STATE_FILE, None)
    if not isinstance(state, dict) or not state:
        return _new_state()

    if state.get("schema_version") != PAPER_SCHEMA_VERSION:
        raise RuntimeError(
            f"Unsupported paper state schema: {state.get('schema_version')!r}"
        )
    if state.get("policy_version") != PAPER_POLICY_VERSION:
        raise RuntimeError(
            f"Paper policy mismatch: {state.get('policy_version')!r}"
        )
    if state.get("execution_mode") != PAPER_EXECUTION_MODE:
        raise RuntimeError(
            f"Paper execution mode mismatch: {state.get('execution_mode')!r}"
        )

    state.setdefault("positions", {})
    state.setdefault("decisions", {})
    return state


def _save_state(state: dict) -> None:
    lock = FileLock(str(PAPER_STATE_FILE) + ".lock")
    with lock:
        _save_json_atomic(PAPER_STATE_FILE, state)


def _normalize_prediction(prediction: dict) -> dict | None:
    if not isinstance(prediction, dict):
        return None

    pred_id = str(prediction.get("pred_id") or "").strip()
    signal = str(prediction.get("predicted_signal") or "").upper().strip()

    if not pred_id or signal not in {"BUY", "SELL"}:
        return None

    if prediction.get("reject_reason") not in (None, ""):
        return None

    contract = prediction.get("observation_contract")
    if not isinstance(contract, dict):
        return None

    validate_observation_contract(contract)
    identity = tuple(observation_identity(contract))

    logical_symbol = str(
        prediction.get("logical_symbol") or ""
    ).upper()
    interval = str(prediction.get("interval") or "")

    if identity[0] != logical_symbol:
        raise ValueError(f"Prediction logical symbol mismatch: {pred_id}")
    if identity[1] != int(prediction.get("open_time", 0)):
        raise ValueError(f"Prediction observation timestamp mismatch: {pred_id}")
    if identity[2] != interval:
        raise ValueError(f"Prediction interval mismatch: {pred_id}")
    if identity[3] != "binance" or identity[4] != "spot":
        raise ValueError(
            f"Non-canonical observation source in prediction: {pred_id}"
        )

    entry = float(prediction.get("entry_ref", 0.0))
    atr = float(prediction.get("atr_ref", 0.0))

    if not math.isfinite(entry) or entry <= 0:
        raise ValueError(f"Invalid prediction entry_ref: {pred_id}")
    if not math.isfinite(atr) or atr <= 0:
        raise ValueError(f"Invalid prediction atr_ref: {pred_id}")

    return {
        "pred_id": pred_id,
        "symbol": identity[0],
        "side": signal,
        "open_time": identity[1],
        "interval": identity[2],
        "entry_ref": entry,
        "atr_ref": atr,
        "observation_contract": contract,
        "observation_identity": list(identity),
        "was_executed_live": bool(prediction.get("was_executed", False)),
        "model_version": prediction.get("model_version"),
        "generated_at": prediction.get("generated_at"),
    }


def _latest_candle_open_time(candle: dict) -> int:
    return int(candle["close_time"]) - CANDLE_INTERVAL_MS + 1


def _candle_observation_identity(trade: dict, candle: dict) -> list:
    """Build the canonical Binance Spot identity for a closed lifecycle candle."""
    close_time = int(candle["close_time"])
    derived_open_time = _latest_candle_open_time(candle)

    supplied_open_time = candle.get("open_time")
    if supplied_open_time is not None:
        supplied_open_time = int(supplied_open_time)
        if supplied_open_time != derived_open_time:
            raise ValueError(
                f"Lifecycle candle open/close mismatch for {trade['pred_id']}"
            )

    return [
        str(trade["logical_symbol"]).upper(),
        derived_open_time,
        str(trade["interval"]),
        "binance",
        "spot",
    ]


def _active_positions(state: dict) -> list[dict]:
    return [
        value
        for value in state.get("positions", {}).values()
        if value.get("status") == "ACTIVE"
    ]


def _symbol_is_active(state: dict, symbol: str) -> bool:
    symbol = symbol.upper()
    return any(
        str(position.get("symbol", "")).upper() == symbol
        for position in _active_positions(state)
    )


def _direction_count(state: dict, side: str) -> int:
    return sum(
        1
        for position in _active_positions(state)
        if str(position.get("side", "")).upper() == side
    )


def _build_position(prediction: dict) -> dict:
    entry = float(prediction["entry_ref"])
    atr = float(prediction["atr_ref"])

    if prediction["side"] == "BUY":
        stop = entry - atr * PAPER_STOP_MULT
        tp1 = entry + atr * PAPER_TP1_MULT
        tp2 = entry + atr * PAPER_TP2_MULT
    else:
        stop = entry + atr * PAPER_STOP_MULT
        tp1 = entry - atr * PAPER_TP1_MULT
        tp2 = entry - atr * PAPER_TP2_MULT

    stop_distance = abs(entry - stop)
    initial_risk_usd = PAPER_INITIAL_BALANCE_USD * PAPER_RISK_PER_TRADE
    qty = initial_risk_usd / stop_distance

    values = (stop, tp1, tp2, qty, initial_risk_usd)
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError(
            f"Non-finite paper brackets for {prediction['pred_id']}"
        )
    if min(stop, tp1, tp2, qty, initial_risk_usd) <= 0:
        raise ValueError(
            f"Invalid paper brackets for {prediction['pred_id']}"
        )

    return {
        "pred_id": prediction["pred_id"],
        "symbol": prediction["symbol"],
        "logical_symbol": prediction["symbol"],
        "side": prediction["side"],
        "open_time": prediction["open_time"],
        "interval": prediction["interval"],
        "observation_contract": prediction["observation_contract"],
        "observation_identity": prediction["observation_identity"],
        "entry_observation_open_time": prediction["open_time"],
        "entry_observation_close_time": int(
            prediction["observation_contract"]["close_time"]
        ),
        "entry_observation_identity": list(
            prediction["observation_identity"]
        ),
        "model_version": prediction.get("model_version"),
        "generated_at": prediction.get("generated_at"),

        "opened_at_ms": int(time.time() * 1000),
        "execution_mode": PAPER_EXECUTION_MODE,
        "paper_policy_version": PAPER_POLICY_VERSION,

        "entry_price": prediction["entry_ref"],
        "entry_price_source": "SIGNAL_CLOSE_PROXY",
        "simulated_entry": prediction["entry_ref"],
        "entry_ref": prediction["entry_ref"],
        "atr_ref": prediction["atr_ref"],
        "stop": stop,
        "tp1": tp1,
        "tp2": tp2,

        "qty": qty,
        "remaining_qty": qty,
        "initial_risk_usd": initial_risk_usd,
        "entry_fee_usd": fee(
            prediction["entry_ref"] * qty,
            PAPER_FEE_RATE,
        ),

        "state": "OPEN",
        "status": "ACTIVE",
        "bars_processed": 0,
        "last_processed_candle_close_time": int(prediction["observation_contract"]["close_time"]),

        "live_was_executed_at_signal": prediction["was_executed_live"],
    }


def _decision_event_id(pred_id: str, reason: str) -> str:
    return f"DECISION:{pred_id}:{reason}"


def _lifecycle_event_id(
    pred_id: str,
    event_name: str,
    candle_close_time: int,
) -> str:
    return f"{event_name}:{pred_id}:{int(candle_close_time)}"


def _record_decision(
    state: dict,
    pred: dict,
    reason: str,
    **extra,
) -> None:
    state["decisions"][pred["pred_id"]] = {
        "status": reason,
        "timestamp_ms": int(time.time() * 1000),
        **extra,
    }

    _append_event({
        "event": "PAPER_DECISION",
        "event_id": _decision_event_id(pred["pred_id"], reason),
        "reason": reason,
        "pred_id": pred["pred_id"],
        "symbol": pred["symbol"],
        "side": pred["side"],
        "open_time": pred["open_time"],
        "observation_identity": pred["observation_identity"],
        "execution_mode": PAPER_EXECUTION_MODE,
        "paper_policy_version": PAPER_POLICY_VERSION,
        **extra,
    })


def _open_eligible_predictions(
    state: dict,
    predictions: list[dict],
) -> tuple[bool, list[dict]]:
    dirty = False
    actions: list[dict] = []

    normalized = []
    for raw in predictions:
        try:
            pred = _normalize_prediction(raw)
        except (TypeError, ValueError, KeyError):
            continue
        if pred is not None:
            normalized.append(pred)

    normalized.sort(
        key=lambda item: (item["open_time"], item["pred_id"])
    )

    for pred in normalized:
        pred_id = pred["pred_id"]

        if pred_id in state["positions"] or pred_id in state["decisions"]:
            continue

        active = _active_positions(state)

        if len(active) >= PAPER_MAX_OPEN_POSITIONS:
            continue

        if _symbol_is_active(state, pred["symbol"]):
            _record_decision(
                state,
                pred,
                "DUPLICATE_SYMBOL_ACTIVE",
            )
            dirty = True
            continue

        if _direction_count(state, pred["side"]) >= PAPER_MAX_SAME_DIRECTION:
            continue

        candle = monitor.fetch_latest_candle(pred["symbol"])
        if not candle:
            continue

        latest_open_time = _latest_candle_open_time(candle)

        if latest_open_time < pred["open_time"]:
            continue

        # A production prediction is executable only on its exact signal candle.
        # A later candle means the simulated entry window has been missed; never
        # fabricate an entry at a later market price.
        if latest_open_time > pred["open_time"]:
            _record_decision(
                state,
                pred,
                "MISSED_ENTRY_WINDOW",
                latest_closed_open_time=latest_open_time,
            )
            dirty = True
            continue

        position = _build_position(pred)
        validate_observation_contract(
            position["observation_contract"]
        )

        if tuple(position["observation_identity"]) != tuple(
            observation_identity(position["observation_contract"])
        ):
            raise ValueError(
                f"Paper observation identity mismatch: {pred_id}"
            )

        state["positions"][pred_id] = position
        state["decisions"][pred_id] = {
            "status": "OPENED",
            "timestamp_ms": int(time.time() * 1000),
        }

        _append_event({
            "event": "PAPER_POSITION_OPENED",
            "event_id": f"OPEN:{pred_id}",
            "pred_id": pred_id,
            "symbol": pred["symbol"],
            "side": pred["side"],
            "open_time": pred["open_time"],
            "entry_price": pred["entry_ref"],
            "entry_price_source": "SIGNAL_CLOSE_PROXY",
            "entry_observation_open_time": pred["open_time"],
            "entry_observation_close_time": int(
                pred["observation_contract"]["close_time"]
            ),
            "entry_observation_identity": list(
                pred["observation_identity"]
            ),
            "atr_ref": pred["atr_ref"],
            "stop": position["stop"],
            "tp1": position["tp1"],
            "tp2": position["tp2"],
            "qty": position["qty"],
            "initial_risk_usd": position["initial_risk_usd"],
            "entry_fee_usd": position["entry_fee_usd"],
            "observation_identity": pred["observation_identity"],
            "execution_mode": PAPER_EXECUTION_MODE,
            "paper_policy_version": PAPER_POLICY_VERSION,
            "live_was_executed_at_signal": pred["was_executed_live"],
        })

        actions.append({
            "action": "OPENED",
            "pred_id": pred_id,
            "symbol": pred["symbol"],
        })
        dirty = True

    return dirty, actions


def _expire_position(
    trade: dict,
    candle: dict,
) -> dict:
    """Close any remaining paper quantity at the horizon candle close."""
    exit_price = float(candle["close"])
    if not math.isfinite(exit_price) or exit_price <= 0:
        raise ValueError(
            f"Invalid expiry price for {trade['pred_id']}"
        )

    remaining_qty = float(
        trade.get(
            "remaining_qty",
            trade.get("qty", 0.0),
        )
    )
    if remaining_qty <= 0:
        raise ValueError(
            f"Invalid remaining quantity at expiry: {trade['pred_id']}"
        )

    final_gross = gross_pnl(
        trade["side"],
        float(trade["entry_price"]),
        exit_price,
        remaining_qty,
    )
    exit_fee = fee(
        exit_price * remaining_qty,
        PAPER_FEE_RATE,
    )

    total_gross = (
        float(trade.get("tp1_realized_gross_pnl", 0.0))
        + final_gross
    )
    total_fees = (
        float(trade.get("entry_fee_usd", 0.0))
        + float(trade.get("tp1_fee_usd", 0.0))
        + exit_fee
    )
    net_pnl_usd = total_gross - total_fees
    initial_risk_usd = float(trade["initial_risk_usd"])
    net_r = net_pnl_usd / initial_risk_usd

    trade.update({
        "status": "CLOSED",
        "state": "CLOSED",
        "exit_price": exit_price,
        "realized_gross_pnl": total_gross,
        "final_exit_fee_usd": exit_fee,
        "total_fees_usd": total_fees,
        "net_pnl_usd": net_pnl_usd,
        "net_r": net_r,
        "exit_reason": "EXPIRED",
        "remaining_qty": 0.0,
        "final_exit_qty": remaining_qty,
        "resolved_at_ms": int(time.time() * 1000),
        "resolved_candle_close_time": int(candle["close_time"]),
    })

    return {
        "event": "PAPER_EXPIRED",
        "event_id": _lifecycle_event_id(
            trade["pred_id"],
            "PAPER_EXPIRED",
            int(candle["close_time"]),
        ),
        "pred_id": trade["pred_id"],
        "symbol": trade["symbol"],
        "side": trade["side"],
        "open_time": trade["open_time"],
        "candle_close_time": int(candle["close_time"]),
        "candle_open_time": _latest_candle_open_time(candle),
        "candle_observation_identity": _candle_observation_identity(trade, candle),
        "exit_price": exit_price,
        "exit_reason": "EXPIRED",
        "remaining_qty": remaining_qty,
        "realized_gross_pnl": total_gross,
        "total_fees_usd": total_fees,
        "net_pnl_usd": net_pnl_usd,
        "net_r": net_r,
        "observation_identity": trade["observation_identity"],
        "execution_mode": PAPER_EXECUTION_MODE,
        "paper_policy_version": PAPER_POLICY_VERSION,
    }


def _monitor_open_positions(state: dict) -> tuple[bool, list[dict]]:
    dirty = False
    actions: list[dict] = []

    for pred_id, trade in list(state["positions"].items()):
        if trade.get("status") != "ACTIVE":
            continue

        candle = monitor.fetch_latest_candle(str(trade["symbol"]))
        if not candle:
            continue

        candle_close_time = int(candle["close_time"])
        if candle_close_time <= int(
            trade.get("last_processed_candle_close_time", 0)
        ):
            continue

        outcome = monitor.process_trade_candle(trade, candle)
        candle_open_time = _latest_candle_open_time(candle)
        candle_identity = _candle_observation_identity(trade, candle)
        trade["last_observation_open_time"] = candle_open_time
        trade["last_observation_close_time"] = candle_close_time
        trade["last_observation_identity"] = candle_identity

        trade["last_processed_candle_close_time"] = candle_close_time
        trade["bars_processed"] = int(trade.get("bars_processed", 0)) + 1
        trade["last_observation_open_time"] = int(
            candle["open_time"]
        )
        trade["last_observation_close_time"] = candle_close_time
        trade["last_observation_identity"] = [
            str(trade["logical_symbol"]).upper(),
            int(candle["open_time"]),
            trade["interval"],
            "binance",
            "spot",
        ]

        # Persist candle-consumption progress even when the lifecycle itself
        # produces no TP/SL/state event. Otherwise the same closed candle can
        # be reprocessed on the next workflow run after an otherwise no-op bar.
        dirty = True

        for lifecycle_event in outcome.get("events", []):
            event = dict(lifecycle_event)
            event.update({
                "event_id": _lifecycle_event_id(
                    pred_id,
                    str(event.get("event", "LIFECYCLE")),
                    candle_close_time,
                ),
                "pred_id": pred_id,
                "symbol": trade["symbol"],
                "side": trade["side"],
                "open_time": trade["open_time"],
                "candle_open_time": int(candle["open_time"]),
                "candle_close_time": candle_close_time,
                "candle_open_time": candle_open_time,
                "candle_observation_identity": candle_identity,
                "candle_observation_identity": [
                    str(trade["logical_symbol"]).upper(),
                    int(candle["open_time"]),
                    trade["interval"],
                    "binance",
                    "spot",
                ],
                "observation_identity": trade["observation_identity"],
                "execution_mode": PAPER_EXECUTION_MODE,
                "paper_policy_version": PAPER_POLICY_VERSION,
            })
            _append_event(event)

        if outcome.get("modified"):
            dirty = True

        if outcome.get("closed"):
            trade["resolved_at_ms"] = int(time.time() * 1000)
            trade["resolved_candle_close_time"] = candle_close_time
            actions.append({
                "action": "CLOSED",
                "pred_id": pred_id,
                "symbol": trade["symbol"],
                "reason": trade.get("exit_reason"),
                "net_r": trade.get("net_r"),
            })
            dirty = True
            continue

        if (
            trade["bars_processed"] >= PAPER_HORIZON_BARS
            and trade.get("status") == "ACTIVE"
        ):
            expiry_event = _expire_position(trade, candle)
            _append_event(expiry_event)
            actions.append({
                "action": "CLOSED",
                "pred_id": pred_id,
                "symbol": trade["symbol"],
                "reason": "EXPIRED",
                "net_r": trade.get("net_r"),
            })
            dirty = True

        elif outcome.get("modified"):
            actions.append({
                "action": "UPDATED",
                "pred_id": pred_id,
                "symbol": trade["symbol"],
                "state": trade.get("state"),
            })

    return dirty, actions


def reconcile_state(state: dict) -> dict:
    """Fail-closed structural reconciliation of paper state and event ledger."""
    checked = 0
    failures: list[str] = []

    if state.get("execution_mode") != PAPER_EXECUTION_MODE:
        failures.append("EXECUTION_MODE_NOT_PAPER_ONLY")

    if state.get("policy_version") != PAPER_POLICY_VERSION:
        failures.append("PAPER_POLICY_VERSION_MISMATCH")

    try:
        events = _read_events()
    except RuntimeError as exc:
        failures.append(str(exc))
        events = []

    event_ids = [str(event.get("event_id") or "") for event in events]
    duplicates = {
        event_id
        for event_id in event_ids
        if event_id and event_ids.count(event_id) > 1
    }
    if duplicates:
        failures.append(
            "DUPLICATE_EVENT_IDS:" + ",".join(sorted(duplicates))
        )

    for pred_id, trade in state.get("positions", {}).items():
        checked += 1

        try:
            contract = trade["observation_contract"]
            validate_observation_contract(contract)
            identity = list(observation_identity(contract))

            if identity != trade.get("observation_identity"):
                failures.append(
                    f"{pred_id}:OBSERVATION_IDENTITY_MISMATCH"
                )
            if identity[0] != str(
                trade.get("logical_symbol") or ""
            ).upper():
                failures.append(
                    f"{pred_id}:LOGICAL_SYMBOL_MISMATCH"
                )
            if int(trade.get("open_time", 0)) != int(identity[1]):
                failures.append(
                    f"{pred_id}:OPEN_TIME_MISMATCH"
                )
            if trade.get("execution_mode") != PAPER_EXECUTION_MODE:
                failures.append(
                    f"{pred_id}:EXECUTION_MODE_MISMATCH"
                )

            entry_identity = trade.get("entry_observation_identity")
            if entry_identity is not None:
                if list(entry_identity) != list(identity):
                    failures.append(
                        f"{pred_id}:ENTRY_OBSERVATION_IDENTITY_MISMATCH"
                    )

                if int(
                    trade.get("entry_observation_open_time", 0)
                ) != int(identity[1]):
                    failures.append(
                        f"{pred_id}:ENTRY_OBSERVATION_OPEN_TIME_MISMATCH"
                    )

                expected_entry_close = (
                    int(identity[1]) + CANDLE_INTERVAL_MS - 1
                )
                if int(
                    trade.get("entry_observation_close_time", 0)
                ) != expected_entry_close:
                    failures.append(
                        f"{pred_id}:ENTRY_OBSERVATION_CLOSE_TIME_MISMATCH"
                    )

            if trade.get("status") == "ACTIVE":
                last_processed = int(
                    trade.get(
                        "last_processed_candle_close_time",
                        identity[1] + CANDLE_INTERVAL_MS - 1,
                    )
                )
                entry_close = int(
                    trade["observation_contract"]["close_time"]
                )

                if last_processed > entry_close:
                    last_open = trade.get("last_observation_open_time")
                    last_close = trade.get("last_observation_close_time")
                    last_identity = trade.get("last_observation_identity")

                    if (
                        last_open is None
                        or last_close is None
                        or last_identity is None
                    ):
                        failures.append(
                            f"{pred_id}:MISSING_LAST_OBSERVATION_PROVENANCE"
                        )
                    else:
                        expected_last_identity = [
                            identity[0],
                            int(last_open),
                            identity[2],
                            identity[3],
                            identity[4],
                        ]

                        if list(last_identity) != expected_last_identity:
                            failures.append(
                                f"{pred_id}:LAST_OBSERVATION_IDENTITY_MISMATCH"
                            )

                        if int(last_close) != last_processed:
                            failures.append(
                                f"{pred_id}:LAST_OBSERVATION_CLOSE_TIME_MISMATCH"
                            )

                        if int(last_close) < entry_close:
                            failures.append(
                                f"{pred_id}:LAST_OBSERVATION_BEFORE_ENTRY"
                            )

            if trade.get("status") == "ACTIVE":
                last_open = trade.get("last_observation_open_time")
                last_close = trade.get("last_observation_close_time")
                last_identity = trade.get("last_observation_identity")
                if last_open is not None and last_close is not None:
                    expected_identity = [
                        identity[0],
                        int(last_open),
                        identity[2],
                        identity[3],
                        identity[4],
                    ]
                    if last_identity != expected_identity:
                        failures.append(
                            f"{pred_id}:LAST_OBSERVATION_IDENTITY_MISMATCH"
                        )
                    if int(last_close) < int(trade["observation_contract"]["close_time"]):
                        failures.append(
                            f"{pred_id}:LAST_OBSERVATION_BEFORE_ENTRY"
                        )

            for field in (
                "entry_price",
                "atr_ref",
                "stop",
                "tp1",
                "tp2",
                "qty",
                "remaining_qty",
                "initial_risk_usd",
            ):
                value = float(trade.get(field, 0))
                if not math.isfinite(value) or value <= 0:
                    failures.append(
                        f"{pred_id}:INVALID_{field.upper()}"
                    )

            status = trade.get("status")
            if status == "ACTIVE" and float(
                trade.get("remaining_qty", 0)
            ) <= 0:
                failures.append(
                    f"{pred_id}:ACTIVE_WITHOUT_REMAINING_QTY"
                )

            if status == "CLOSED":
                reason = trade.get("exit_reason")
                if reason == "AMBIGUOUS_BARRIER":
                    pass
                elif trade.get("net_r") is None:
                    failures.append(
                        f"{pred_id}:CLOSED_WITHOUT_NET_R"
                    )

        except (KeyError, TypeError, ValueError):
            failures.append(
                f"{pred_id}:INVALID_OBSERVATION_CONTRACT"
            )

    return {
        "ok": not failures,
        "checked_positions": checked,
        "failures": failures,
        "paper_policy_version": PAPER_POLICY_VERSION,
        "execution_mode": PAPER_EXECUTION_MODE,
        "event_count": len(events),
    }


def run_once() -> dict:
    state = _load_state()
    predictions = _load_json(PREDICTIONS_FILE, [])

    dirty_open, open_actions = _open_eligible_predictions(
        state,
        predictions if isinstance(predictions, list) else [],
    )
    dirty_monitor, monitor_actions = _monitor_open_positions(state)

    reconciliation = reconcile_state(state)
    if not reconciliation["ok"]:
        raise RuntimeError(
            "Paper reconciliation failed: "
            + ", ".join(reconciliation["failures"])
        )

    dirty = dirty_open or dirty_monitor

    if dirty:
        _save_state(state)

    active = len(_active_positions(state))
    closed = sum(
        1
        for position in state.get("positions", {}).values()
        if position.get("status") == "CLOSED"
    )

    return {
        "paper_policy_version": PAPER_POLICY_VERSION,
        "execution_mode": PAPER_EXECUTION_MODE,
        "active_positions": active,
        "closed_positions": closed,
        "open_actions": open_actions,
        "monitor_actions": monitor_actions,
        "reconciliation": reconciliation,
        "state_changed": dirty,
    }


if __name__ == "__main__":
    print(json.dumps(run_once(), sort_keys=True))




