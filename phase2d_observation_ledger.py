from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

LEDGER_SCHEMA_VERSION = 1
RESOLVED_STATUSES = frozenset({"TP", "SL", "EXPIRED", "AMBIGUOUS", "INVALID"})
OBSERVATION_RECORD_TYPE = "observation"
EVENT_RECORD_TYPE = "event"


def observation_key(experiment_id: str, result: dict) -> str:
    return (
        f"{experiment_id}:"
        f"{result.get('model')}:"
        f"{result.get('symbol')}:"
        f"{int(result['open_time'])}"
    )


def _read_records(path: Path) -> list[dict]:
    if not path.exists():
        return []

    records: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Corrupt observation ledger JSON at {path}:{line_no}: {exc}"
                ) from exc
            if not isinstance(record, dict):
                raise RuntimeError(
                    f"Observation ledger record at {path}:{line_no} is not an object"
                )
            records.append(record)
    return records


def load_observations(path: Path, experiment_id: str | None = None) -> list[dict]:
    rows: list[dict] = []
    for record in _read_records(path):
        if record.get("record_type") != OBSERVATION_RECORD_TYPE:
            continue
        if record.get("ledger_schema_version") != LEDGER_SCHEMA_VERSION:
            raise RuntimeError(
                f"Unsupported observation ledger schema: {record.get('ledger_schema_version')}"
            )
        if experiment_id is not None and record.get("experiment_id") != experiment_id:
            continue
        if record.get("status") not in RESOLVED_STATUSES:
            raise RuntimeError(
                f"Invalid observation ledger status: {record.get('status')!r}"
            )
        rows.append(record)
    return rows


def load_events(path: Path, experiment_id: str | None = None) -> list[dict]:
    events: list[dict] = []
    for record in _read_records(path):
        if record.get("record_type") != EVENT_RECORD_TYPE:
            continue
        if record.get("ledger_schema_version") != LEDGER_SCHEMA_VERSION:
            raise RuntimeError(
                f"Unsupported event ledger schema: {record.get('ledger_schema_version')}"
            )
        if experiment_id is not None and record.get("experiment_id") != experiment_id:
            continue
        events.append(record)
    return events


def _event_exists(path: Path, event: str, experiment_id: str) -> bool:
    return any(
        item.get("event") == event
        and item.get("experiment_id") == experiment_id
        for item in load_events(path, experiment_id=experiment_id)
    )


def _append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")


def append_event(path: Path, event: str, experiment_id: str, payload: dict | None = None) -> bool:
    if _event_exists(path, event, experiment_id):
        return False

    record = {
        "record_type": EVENT_RECORD_TYPE,
        "ledger_schema_version": LEDGER_SCHEMA_VERSION,
        "event": event,
        "experiment_id": experiment_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if payload:
        record.update(payload)

    _append_jsonl(path, record)
    return True


def append_observation(path: Path, experiment_id: str, result: dict) -> bool:
    status = result.get("status")
    if status not in RESOLVED_STATUSES:
        raise ValueError(
            f"Cannot append unresolved observation with status={status!r}"
        )

    for field in ("model", "symbol", "open_time", "id"):
        if result.get(field) is None:
            raise ValueError(f"Resolved observation missing required field: {field}")

    key = observation_key(experiment_id, result)
    existing = None
    for record in load_observations(path, experiment_id=experiment_id):
        if record.get("observation_key") == key:
            existing = record
            break

    if existing is not None:
        metadata = {
            "record_type",
            "ledger_schema_version",
            "experiment_id",
            "observation_key",
        }
        existing_payload = {
            k: v for k, v in existing.items() if k not in metadata
        }
        if existing_payload != result:
            raise ValueError(
                f"Observation ledger conflict for immutable key: {key}"
            )
        return False

    record = dict(result)
    record.update(
        {
            "record_type": OBSERVATION_RECORD_TYPE,
            "ledger_schema_version": LEDGER_SCHEMA_VERSION,
            "experiment_id": experiment_id,
            "observation_key": key,
        }
    )
    _append_jsonl(path, record)
    return True


def initialize_observation_ledger_from_state(
    state: dict,
    path: Path,
    *,
    experiment_id: str,
    event_name: str,
    legacy_state_schema: int | None = None,
) -> dict:
    if _event_exists(path, event_name, experiment_id):
        return {
            "initialized": False,
            "event": event_name,
            "experiment_id": experiment_id,
            "migrated_resolved": 0,
            "seen_total": 0,
            "pending_at_migration": 0,
            "unrecoverable_seen": 0,
        }

    migrated_by_model: dict[str, int] = {}
    resolved_total = 0

    for model_name in ("candidate", "production"):
        model_rows = state.get("resolved", {}).get(model_name, [])
        migrated_by_model[model_name] = 0
        for result in model_rows:
            added = append_observation(
                path,
                experiment_id,
                result,
            )
            if added:
                migrated_by_model[model_name] += 1
                resolved_total += 1

    seen_total = sum(
        len(set(state.get("seen", {}).get(model_name, [])))
        for model_name in ("candidate", "production")
    )

    pending_keys_total = sum(
        len({
            f"{item.get('symbol')}:{int(item['open_time'])}"
            for item in state.get("pending", {}).get(model_name, [])
            if item.get("symbol") is not None and item.get("open_time") is not None
        })
        for model_name in ("candidate", "production")
    )

    migrated_keys = sum(migrated_by_model.values())
    unrecoverable = max(
        0,
        seen_total - migrated_keys - pending_keys_total,
    )

    payload = {
        "initialized": True,
        "event": event_name,
        "experiment_id": experiment_id,
        "legacy_state_schema": (
            state.get("schema_version")
            if legacy_state_schema is None
            else legacy_state_schema
        ),
        "migrated_resolved": resolved_total,
        "migrated_by_model": migrated_by_model,
        "seen_total": seen_total,
        "pending_at_migration": pending_keys_total,
        "unrecoverable_seen": unrecoverable,
    }

    append_event(
        path,
        event_name,
        experiment_id,
        payload,
    )
    return payload

def reconcile_resolved_observations(
    path: Path,
    experiment_id: str,
    historical_records: list[dict],
) -> dict:
    """Reconcile immutable resolved observations from historical F08 states.

    Existing records are never modified. Missing records are appended.
    Any conflicting payload for the same immutable observation key fails closed.
    """

    existing_records = load_observations(
        path,
        experiment_id=experiment_id,
    )

    existing_by_key = {
        record["observation_key"]: record
        for record in existing_records
    }

    historical_by_key: dict[str, dict] = {}

    for result in historical_records:
        if result.get("status") not in RESOLVED_STATUSES:
            continue

        key = observation_key(
            experiment_id,
            result,
        )

        existing_historical = historical_by_key.get(key)

        if (
            existing_historical is not None
            and existing_historical != result
        ):
            raise ValueError(
                "Historical observation conflict for immutable key: "
                f"{key}"
            )

        historical_by_key[key] = dict(result)

    missing: list[dict] = []
    conflicts: list[str] = []

    for key, result in historical_by_key.items():
        existing = existing_by_key.get(key)

        if existing is None:
            missing.append(result)
            continue

        metadata = {
            "record_type",
            "ledger_schema_version",
            "experiment_id",
            "observation_key",
        }

        existing_payload = {
            k: v
            for k, v in existing.items()
            if k not in metadata
        }

        if existing_payload != result:
            conflicts.append(key)

    if conflicts:
        raise ValueError(
            "Observation ledger conflicts detected: "
            + ", ".join(sorted(conflicts))
        )

    missing.sort(
        key=lambda r: (
            int(r["open_time"]),
            str(r.get("model")),
            str(r.get("symbol")),
        )
    )

    for result in missing:
        record = dict(result)
        record.update(
            {
                "record_type": OBSERVATION_RECORD_TYPE,
                "ledger_schema_version": LEDGER_SCHEMA_VERSION,
                "experiment_id": experiment_id,
                "observation_key": observation_key(
                    experiment_id,
                    result,
                ),
            }
        )
        _append_jsonl(path, record)

    added_by_model: dict[str, int] = {
        "candidate": 0,
        "production": 0,
    }

    for result in missing:
        model = str(result.get("model"))
        if model in added_by_model:
            added_by_model[model] += 1

    return {
        "experiment_id": experiment_id,
        "historical_unique": len(historical_by_key),
        "existing_before": len(existing_records),
        "added": len(missing),
        "added_by_model": added_by_model,
        "conflicts": len(conflicts),
        "remaining_missing": 0,
    }
