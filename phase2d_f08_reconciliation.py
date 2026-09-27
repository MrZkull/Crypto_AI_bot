"""One-time F08 historical observation-ledger reconciliation."""

from __future__ import annotations

import json
from pathlib import Path

import phase2d_observation_ledger as ledger


EXPERIMENT_ID = "PHASE2D-HARDENED-20260924-F08"
MIGRATION_EVENT = "F08_LEDGER_INITIALIZED_FROM_STATE"
CORRECTION_EVENT = "F08_LEDGER_MIGRATION_METADATA_CORRECTED"
RECONCILIATION_EVENT = "F08_LEDGER_HISTORICAL_RECONCILIATION"


def load_state(path: Path) -> dict:
    return json.loads(
        path.read_text(encoding="utf-8")
    )


def collect_historical_resolved(
    state_paths: list[Path],
    *,
    experiment_id: str = EXPERIMENT_ID,
) -> tuple[list[dict], dict]:
    """Collect the union of resolved observations from F08 states."""

    by_key: dict[str, dict] = {}
    source_counts: dict[str, int] = {}

    for path in sorted(
        state_paths,
        key=lambda p: str(p).lower(),
    ):
        state = load_state(path)

        actual_experiment = (
            state.get("experiment_definition", {})
            .get("experiment_id")
            or state.get("locked_models", {})
            .get("experiment_id")
        )

        if actual_experiment != experiment_id:
            continue

        source_counts[str(path)] = 0

        for model_name in ("candidate", "production"):
            for result in (
                state.get("resolved", {})
                .get(model_name, [])
            ):
                if result.get("status") not in ledger.RESOLVED_STATUSES:
                    continue

                key = ledger.observation_key(
                    experiment_id,
                    result,
                )

                previous = by_key.get(key)

                if previous is not None and previous != result:
                    raise ValueError(
                        "Historical F08 observation conflict for key: "
                        f"{key}"
                    )

                if previous is None:
                    by_key[key] = dict(result)
                    source_counts[str(path)] += 1

    records = sorted(
        by_key.values(),
        key=lambda r: (
            int(r["open_time"]),
            str(r.get("model")),
            str(r.get("symbol")),
        ),
    )

    return records, {
        "state_files_examined": len(state_paths),
        "state_files_f08": len(source_counts),
        "historical_unique_observations": len(records),
        "source_counts": source_counts,
    }


def correct_migration_metadata(
    state: dict,
    observation_ledger_path: Path,
    *,
    source_schema: int = 5,
    experiment_id: str = EXPERIMENT_ID,
) -> dict:
    migration = (
        state.get("observation_ledger", {})
        .get("migration")
    )

    if not isinstance(migration, dict):
        return {
            "corrected": False,
            "reason": "missing migration metadata",
        }

    current_schema = migration.get(
        "legacy_state_schema"
    )

    if current_schema == source_schema:
        return {
            "corrected": False,
            "reason": "already correct",
            "legacy_state_schema": source_schema,
        }

    events = ledger.load_events(
        observation_ledger_path,
        experiment_id=experiment_id,
    )

    initialization_events = [
        event
        for event in events
        if event.get("event") == MIGRATION_EVENT
    ]

    if initialization_events:
        original_event_schema = initialization_events[-1].get(
            "legacy_state_schema"
        )

        ledger.append_event(
            observation_ledger_path,
            CORRECTION_EVENT,
            experiment_id,
            {
                "original_legacy_state_schema": original_event_schema,
                "corrected_legacy_state_schema": source_schema,
                "reason": "controlled F08 schema-5 to schema-6 migration metadata correction",
            },
        )

    migration["legacy_state_schema"] = source_schema

    state.setdefault(
        "observation_ledger",
        {},
    )["migration"] = migration

    return {
        "corrected": True,
        "legacy_state_schema": source_schema,
        "original_legacy_state_schema": current_schema,
        "initialization_event_found": bool(
            initialization_events
        ),
    }


def reconcile_ledger(
    ledger_path: Path,
    historical_records: list[dict],
    *,
    experiment_id: str = EXPERIMENT_ID,
) -> dict:
    result = ledger.reconcile_resolved_observations(
        ledger_path,
        experiment_id,
        historical_records,
    )

    ledger.append_event(
        ledger_path,
        RECONCILIATION_EVENT,
        experiment_id,
        result,
    )

    return result
