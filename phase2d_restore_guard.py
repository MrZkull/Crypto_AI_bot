from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


BLOCK_MS = 86_400_000

EXPECTED_F08_EVENTS = {
    "F08_LEDGER_INITIALIZED_FROM_STATE",
    "F08_LEDGER_MIGRATION_METADATA_CORRECTED",
    "F08_LEDGER_HISTORICAL_RECONCILIATION",
}


class RestoreGuardError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


def load_manifest(path: Path) -> dict:
    try:
        return json.loads(
            path.read_text(encoding="utf-8")
        )
    except Exception as exc:
        raise RestoreGuardError(
            f"Unable to load baseline manifest: {path}: {exc}"
        ) from exc


def ledger_stats(
    ledger_path: Path,
    experiment_id: str,
) -> dict:
    if not ledger_path.exists():
        raise RestoreGuardError(
            f"Observation ledger missing: {ledger_path}"
        )

    seen_keys = set()
    utc_blocks = set()
    events = set()
    reconciliation_remaining = []

    candidate = 0
    production = 0
    total = 0

    with ledger_path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except Exception as exc:
                raise RestoreGuardError(
                    f"Invalid JSON at {ledger_path}:{line_no}: {exc}"
                ) from exc

            if record.get("experiment_id") != experiment_id:
                raise RestoreGuardError(
                    f"Experiment mismatch at {ledger_path}:{line_no}"
                )

            event = record.get("event")

            if event in EXPECTED_F08_EVENTS:
                events.add(event)

                if event == "F08_LEDGER_HISTORICAL_RECONCILIATION":
                    reconciliation_remaining.append(
                        record.get("remaining_missing")
                    )

            if record.get("record_type") != "observation":
                continue

            model = record.get("model")
            symbol = record.get("symbol")
            open_time = record.get("open_time")

            if model not in {"candidate", "production"}:
                raise RestoreGuardError(
                    f"Invalid model at {ledger_path}:{line_no}: {model!r}"
                )

            if not symbol or open_time is None:
                raise RestoreGuardError(
                    f"Missing observation identity at "
                    f"{ledger_path}:{line_no}"
                )

            key = (
                model,
                str(symbol),
                int(open_time),
            )

            if key in seen_keys:
                raise RestoreGuardError(
                    f"Duplicate observation key: {key}"
                )

            seen_keys.add(key)
            total += 1

            if model == "candidate":
                candidate += 1
            else:
                production += 1

            utc_blocks.add(
                int(open_time) // BLOCK_MS
            )

    missing_events = (
        EXPECTED_F08_EVENTS - events
    )

    if missing_events:
        raise RestoreGuardError(
            "Missing required F08 ledger events: "
            + ", ".join(sorted(missing_events))
        )

    if not reconciliation_remaining:
        raise RestoreGuardError(
            "F08 historical reconciliation event is missing."
        )

    if any(
        value != 0
        for value in reconciliation_remaining
    ):
        raise RestoreGuardError(
            "F08 historical reconciliation reports "
            "remaining_missing != 0."
        )

    return {
        "observation_count": total,
        "candidate_observation_count": candidate,
        "production_observation_count": production,
        "unique_utc_blocks": len(utc_blocks),
        "sha256": sha256_file(ledger_path),
    }


def validate_against_baseline(
    stats: dict,
    baseline: dict,
) -> tuple[bool, dict]:
    policy = baseline["restore_policy"]

    mismatches = {}

    checks = {
        "observation_count": (
            stats["observation_count"],
            policy["observation_count_must_be_gte"],
        ),
        "candidate_observation_count": (
            stats["candidate_observation_count"],
            policy["candidate_observation_count_must_be_gte"],
        ),
        "production_observation_count": (
            stats["production_observation_count"],
            policy["production_observation_count_must_be_gte"],
        ),
        "unique_utc_blocks": (
            stats["unique_utc_blocks"],
            policy["unique_utc_blocks_must_be_gte"],
        ),
    }

    for name, (actual, required) in checks.items():
        if actual < required:
            mismatches[name] = {
                "actual": actual,
                "required": required,
            }

    baseline_total = (
        baseline["ledger"]["observation_count"]
    )
    baseline_sha = (
        baseline["ledger"]["sha256"]
    )

    if (
        stats["observation_count"] == baseline_total
        and stats["sha256"] != baseline_sha
    ):
        mismatches["equal_count_ledger_sha256"] = {
            "actual": stats["sha256"],
            "required": baseline_sha,
        }

    if mismatches:
        return False, mismatches

    return True, {}


def assert_preserves_baseline(
    candidate_ledger: Path,
    baseline_ledger: Path,
) -> None:
    if not baseline_ledger.exists():
        raise RestoreGuardError(
            f"Canonical baseline ledger missing: {baseline_ledger}"
        )

    baseline_bytes = baseline_ledger.read_bytes()
    candidate_bytes = candidate_ledger.read_bytes()

    if len(candidate_bytes) < len(baseline_bytes):
        raise RestoreGuardError(
            "Candidate ledger is shorter than the canonical "
            "F08 baseline."
        )

    if not candidate_bytes.startswith(baseline_bytes):
        raise RestoreGuardError(
            "Candidate ledger is not append-only relative to the "
            "canonical F08 baseline: canonical ledger bytes were "
            "removed or changed."
        )


def inspect_artifact(
    state_path: Path,
    ledger_path: Path,
    baseline_path: Path,
    expected_experiment_id: str,
    baseline_ledger_path: Path | None = None,
) -> dict:
    try:
        state = json.loads(
            state_path.read_text(encoding="utf-8")
        )
    except Exception as exc:
        raise RestoreGuardError(
            f"Unable to load state: {state_path}: {exc}"
        ) from exc

    locked = state.get("locked_models", {})

    actual_experiment = (
        state.get("experiment_definition", {})
        .get("experiment_id")
        or locked.get("experiment_id")
    )

    if actual_experiment != expected_experiment_id:
        raise RestoreGuardError(
            "State experiment ID mismatch"
        )

    baseline = load_manifest(baseline_path)

    if baseline.get("experiment_id") != expected_experiment_id:
        raise RestoreGuardError(
            "Baseline manifest experiment ID mismatch"
        )

    stats = ledger_stats(
        ledger_path,
        expected_experiment_id,
    )

    if expected_experiment_id == "PHASE2D-HARDENED-20260924-F08":
        migration = (
            state.get("observation_ledger", {})
            .get("migration", {})
        )

        reconciliation = (
            migration.get("historical_reconciliation")
            if isinstance(migration, dict)
            else None
        )

        if not isinstance(reconciliation, dict):
            raise RestoreGuardError(
                "F08 state missing historical reconciliation metadata."
            )

        historical_count = reconciliation.get(
            "historical_unique_observations"
        )

        if (
            historical_count is None
            or stats["observation_count"] < historical_count
        ):
            raise RestoreGuardError(
                "F08 state historical reconciliation metadata "
                "exceeds ledger coverage."
            )

    compatible, reasons = (
        validate_against_baseline(
            stats,
            baseline,
        )
    )

    if not compatible:
        baseline_total = baseline["ledger"]["observation_count"]

        if stats["observation_count"] > baseline_total:
            raise RestoreGuardError(
                "FATAL_RICHER_ARTIFACT_REGRESSION: "
                + json.dumps(
                    reasons,
                    sort_keys=True,
                )
            )

        raise RestoreGuardError(
            json.dumps(
                reasons,
                sort_keys=True,
            )
        )

    if baseline_ledger_path is not None:
        baseline_stats = ledger_stats(
            baseline_ledger_path,
            expected_experiment_id,
        )

        try:
            assert_preserves_baseline(
                ledger_path,
                baseline_ledger_path,
            )
        except RestoreGuardError as exc:
            # A richer artifact that is not append-only is a fatal
            # monotonicity regression. Do not allow workflow fallback
            # to silently replace the richer history with the canonical
            # baseline.
            if (
                stats["observation_count"]
                > baseline_stats["observation_count"]
            ):
                raise RestoreGuardError(
                    "FATAL_RICHER_ARTIFACT_REGRESSION: "
                    + str(exc)
                ) from exc
            raise

    return stats


def main() -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument("--state", required=True)
    parser.add_argument("--ledger", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--baseline-ledger")
    parser.add_argument("--experiment", required=True)

    args = parser.parse_args()

    try:
        stats = inspect_artifact(
            Path(args.state),
            Path(args.ledger),
            Path(args.baseline),
            args.experiment,
            (
                Path(args.baseline_ledger)
                if args.baseline_ledger
                else None
            ),
        )
    except RestoreGuardError as exc:
        print(
            f"INCOMPATIBLE: {exc}",
            flush=True,
        )

        if str(exc).startswith(
            "FATAL_RICHER_ARTIFACT_REGRESSION:"
        ):
            return 3

        return 2

    print(
        "|".join(
            [
                str(stats["observation_count"]),
                str(stats["candidate_observation_count"]),
                str(stats["production_observation_count"]),
                str(stats["unique_utc_blocks"]),
                stats["sha256"],
            ]
        )
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
