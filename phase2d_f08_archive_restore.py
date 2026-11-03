from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

EXPERIMENT_ID = "PHASE2D-HARDENED-20260924-F08"

ARCHIVE_SHA256 = (
    "e84575e9ad95afd96509c07eaafbf59c15ddbcf20cb26e4a5f54452981a6fa59"
)

MAGIC = b"CBF08ARC"
VERSION = 1
ITERATIONS = 600_000
MIN_ITERATIONS = 500_000
MAX_ITERATIONS = 1_000_000
AAD = b"CryptoBot-AI-F08-authoritative-archive-v1"

EXPECTED_FILES = {
    "research_outputs/phase2d_state.json": (
        1_589_599,
        "4dd39ee1eefb0866dc5060719d69babf2b31f0d73ee814be5418322ac1d262e5",
    ),
    "research_outputs/phase2d_results.json": (
        18_547,
        "dae7c14a33c0296e14d9a577f0a98a725fc66f5c74034c19bd9913153b065abf",
    ),
    "research_outputs/phase2d_latest_snapshot.json": (
        168_069,
        "cbd3af7bb5425e8ac128ce383c5487d46e92f8eff33f12dc695b31990ebb2910",
    ),
    "research_outputs/experiment_ledger.jsonl": (
        44_047,
        "689459dbb7e27ac2249745924894a062cacf9c69e71eac467d17a693f74653f5",
    ),
    "research_outputs/phase2d_observations.jsonl": (
        10_732_864,
        "4ba3daaa2e4f25eae895699e01a4cd1d0a4fd9e7b64a2632bf4fcac7d56274e9",
    ),
    "phase2d_validator.log": (
        30_163,
        "9836df492ebf62726ca7cb32be543cf7dc56498a7ab483f043a3aa5181d1d388",
    ),
}

EXPECTED_MODEL_HASHES = {
    "candidate_sha256": (
        "7213987874543e5d6e18b0bac32556647d40080b3e4222532e40a225a1bf08a3"
    ),
    "production_sha256": (
        "f55b887c7f624179b3d9fee56d792c29424a589e3d8734edcd78ce1be71f2c21"
    ),
    "feature_code_hash": (
        "f60a2508e19892baeeead20d4aa41d8210f559c827affd5cd50ff541973d4683"
    ),
    "feature_schema_hash": (
        "24659f235a0385ec18789cfe7b6e7067996f77dc24f763a5e327fc7ffe3020c1"
    ),
    "validator_code_hash": (
        "2b9f16f87f8222fbb00cfd0e92a0f149dfefdd107d0bfa3079a9a97040e299c5"
    ),
}

EXPECTED_OBSERVATION_COUNTS = {
    "candidate": 2_959,
    "production": 13_249,
}

EXPECTED_PENDING_COUNTS = {
    "candidate": 74,
    "production": 544,
}

EXPECTED_SEEN_COUNTS = {
    "candidate": 3_033,
    "production": 13_793,
}

EXPECTED_EVENT_NAMES = {
    "F08_LEDGER_INITIALIZED_FROM_STATE",
    "F08_LEDGER_MIGRATION_METADATA_CORRECTED",
    "F08_LEDGER_HISTORICAL_RECONCILIATION",
}

HEADER_SIZE = 8 + 1 + 4 + 16 + 12


class ArchiveError(RuntimeError):
    """Raised when the encrypted or recovered F08 archive is invalid."""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_passphrase(env_name: str) -> str:
    value = os.environ.get(env_name, "").strip()
    if len(value.encode("utf-8")) < 32:
        raise ArchiveError(
            f"{env_name} must contain at least 32 UTF-8 bytes; "
            "use a randomly generated secret."
        )
    return value


def _derive_key(passphrase: str, salt: bytes, iterations: int) -> bytes:
    if not MIN_ITERATIONS <= iterations <= MAX_ITERATIONS:
        raise ArchiveError(
            f"Unsupported PBKDF2 iteration count: {iterations}"
        )

    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=iterations,
    )
    return kdf.derive(passphrase.encode("utf-8"))


def _encrypt_bytes(plaintext: bytes, passphrase: str) -> bytes:
    salt = os.urandom(16)
    nonce = os.urandom(12)
    iteration_bytes = ITERATIONS.to_bytes(4, "big")
    header = MAGIC + bytes([VERSION]) + iteration_bytes + salt + nonce
    key = _derive_key(passphrase, salt, ITERATIONS)
    ciphertext = AESGCM(key).encrypt(nonce, plaintext, AAD + header)
    return header + ciphertext


def _decrypt_bytes(encrypted: bytes, passphrase: str) -> bytes:
    if len(encrypted) <= HEADER_SIZE:
        raise ArchiveError("Encrypted archive is truncated.")

    if encrypted[:8] != MAGIC:
        raise ArchiveError("Encrypted archive magic does not match.")

    version = encrypted[8]
    if version != VERSION:
        raise ArchiveError(f"Unsupported encrypted archive version: {version}")

    iterations = int.from_bytes(encrypted[9:13], "big")
    salt = encrypted[13:29]
    nonce = encrypted[29:41]
    header = encrypted[:HEADER_SIZE]
    ciphertext = encrypted[HEADER_SIZE:]

    key = _derive_key(passphrase, salt, iterations)
    try:
        return AESGCM(key).decrypt(nonce, ciphertext, AAD + header)
    except Exception as exc:
        raise ArchiveError(
            "Archive authentication/decryption failed. Check the secret and "
            "the encrypted asset integrity."
        ) from exc


def encrypt_archive(
    source_zip: Path,
    destination: Path,
    passphrase: str,
) -> dict[str, Any]:
    raw = source_zip.read_bytes()
    actual = _sha256(raw)
    if actual != ARCHIVE_SHA256:
        raise ArchiveError(
            "Source ZIP SHA-256 does not match the verified authoritative archive: "
            f"{actual}"
        )

    encrypted = _encrypt_bytes(raw, passphrase)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_bytes(encrypted)
    os.replace(temporary, destination)

    return {
        "archive_sha256": actual,
        "encrypted_sha256": _sha256(encrypted),
        "encrypted_bytes": len(encrypted),
    }


def _parse_json(data: bytes, label: str) -> Any:
    try:
        return json.loads(data.decode("utf-8"))
    except Exception as exc:
        raise ArchiveError(f"{label} is not valid UTF-8 JSON: {exc}") from exc


def _validate_archive_payloads(payloads: dict[str, bytes]) -> dict[str, Any]:
    if set(payloads) != set(EXPECTED_FILES):
        missing = sorted(set(EXPECTED_FILES) - set(payloads))
        extra = sorted(set(payloads) - set(EXPECTED_FILES))
        raise ArchiveError(
            f"Archive member set mismatch. Missing={missing}; extra={extra}"
        )

    for name, (expected_size, expected_hash) in EXPECTED_FILES.items():
        raw = payloads[name]
        if len(raw) != expected_size:
            raise ArchiveError(
                f"Internal file size mismatch for {name}: "
                f"expected={expected_size}; actual={len(raw)}"
            )
        actual_hash = _sha256(raw)
        if actual_hash != expected_hash:
            raise ArchiveError(
                f"Internal SHA-256 mismatch for {name}: {actual_hash}"
            )

    state = _parse_json(
        payloads["research_outputs/phase2d_state.json"],
        "phase2d_state.json",
    )
    results = _parse_json(
        payloads["research_outputs/phase2d_results.json"],
        "phase2d_results.json",
    )
    snapshot = _parse_json(
        payloads["research_outputs/phase2d_latest_snapshot.json"],
        "phase2d_latest_snapshot.json",
    )

    if not isinstance(state, dict) or not isinstance(results, dict) or not isinstance(snapshot, dict):
        raise ArchiveError("State, results and latest snapshot must be JSON objects.")

    locked = state.get("locked_models", {})
    if state.get("schema_version") != 6:
        raise ArchiveError("Recovered state schema is not version 6.")
    if locked.get("experiment_id") != EXPERIMENT_ID:
        raise ArchiveError("Recovered state experiment identity mismatch.")

    for key, expected in EXPECTED_MODEL_HASHES.items():
        actual = locked.get(key)
        if actual != expected:
            raise ArchiveError(
                f"Recovered state locked identity mismatch for {key}: {actual}"
            )

    raw_lines = payloads["research_outputs/phase2d_observations.jsonl"].decode(
        "utf-8"
    ).splitlines()

    observations: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    observation_keys: set[tuple[str, str, int]] = set()
    counts = {"candidate": 0, "production": 0}

    for line_no, raw_line in enumerate(raw_lines, start=1):
        if not raw_line.strip():
            raise ArchiveError(
                f"Blank line in observation ledger at line {line_no}."
            )
        try:
            row = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise ArchiveError(
                f"Invalid observation JSON at line {line_no}: {exc}"
            ) from exc

        if not isinstance(row, dict):
            raise ArchiveError(f"Observation ledger line {line_no} is not an object.")
        if row.get("experiment_id") != EXPERIMENT_ID:
            raise ArchiveError(f"Experiment identity mismatch at line {line_no}.")

        if row.get("record_type") == "observation":
            model = row.get("model")
            symbol = row.get("symbol")
            open_time = row.get("open_time")
            if model not in counts or not symbol or open_time is None:
                raise ArchiveError(f"Invalid observation identity at line {line_no}.")
            if row.get("ledger_schema_version") != 1:
                raise ArchiveError(f"Unsupported ledger schema at line {line_no}.")
            if row.get("status") not in {"TP", "SL", "EXPIRED", "AMBIGUOUS", "INVALID"}:
                raise ArchiveError(f"Invalid resolved status at line {line_no}.")

            key = (model, str(symbol), int(open_time))
            if key in observation_keys:
                raise ArchiveError(f"Duplicate observation key at line {line_no}: {key}")

            expected_key = (
                f"{EXPERIMENT_ID}:{model}:{symbol}:{int(open_time)}"
            )
            if row.get("observation_key") != expected_key:
                raise ArchiveError(f"Observation key mismatch at line {line_no}.")

            observation_keys.add(key)
            counts[model] += 1
            observations.append(row)

        elif row.get("record_type") == "event":
            if row.get("ledger_schema_version") != 1:
                raise ArchiveError(f"Unsupported event schema at line {line_no}.")
            events.append(row)
        else:
            raise ArchiveError(f"Unknown ledger record type at line {line_no}.")

    if len(observations) != 16_208:
        raise ArchiveError(f"Expected 16,208 observations; got {len(observations)}.")
    if counts != EXPECTED_OBSERVATION_COUNTS:
        raise ArchiveError(f"Observation model counts mismatch: {counts}")
    if len(events) != 3 or {e.get("event") for e in events} != EXPECTED_EVENT_NAMES:
        raise ArchiveError("Expected exactly the three canonical F08 migration events.")

    for event in events:
        if event.get("record_type") != "event":
            raise ArchiveError("Invalid event record.")
    reconciliation = [
        e for e in events
        if e.get("event") == "F08_LEDGER_HISTORICAL_RECONCILIATION"
    ]
    if len(reconciliation) != 1 or reconciliation[0].get("remaining_missing") != 0:
        raise ArchiveError("Historical reconciliation is missing or incomplete.")

    pending = state.get("pending", {})
    seen = state.get("seen", {})
    if not isinstance(pending, dict) or not isinstance(seen, dict):
        raise ArchiveError("State pending/seen structures are invalid.")

    for model in ("candidate", "production"):
        pending_rows = pending.get(model, [])
        seen_rows = seen.get(model, [])
        if not isinstance(pending_rows, list) or not isinstance(seen_rows, list):
            raise ArchiveError(f"State pending/seen list invalid for {model}.")
        if len(pending_rows) != EXPECTED_PENDING_COUNTS[model]:
            raise ArchiveError(
                f"Pending count mismatch for {model}: {len(pending_rows)}"
            )
        if len(seen_rows) != EXPECTED_SEEN_COUNTS[model]:
            raise ArchiveError(f"Seen count mismatch for {model}: {len(seen_rows)}")
        if len(set(seen_rows)) != len(seen_rows):
            raise ArchiveError(f"Duplicate seen key for {model}.")

        resolved_keys = {
            f"{symbol}:{open_time}"
            for row_model, symbol, open_time in observation_keys
            if row_model == model
        }
        pending_keys: set[str] = set()
        for row in pending_rows:
            if not isinstance(row, dict) or not row.get("symbol") or row.get("open_time") is None:
                raise ArchiveError(f"Malformed pending item for {model}.")
            pending_key = f"{row['symbol']}:{int(row['open_time'])}"
            if pending_key in pending_keys:
                raise ArchiveError(f"Duplicate pending key for {model}: {pending_key}")
            pending_keys.add(pending_key)

        if resolved_keys & pending_keys:
            raise ArchiveError(f"Resolved/pending overlap detected for {model}.")
        expected_seen = resolved_keys | pending_keys
        if set(seen_rows) != expected_seen:
            missing = len(expected_seen - set(seen_rows))
            extra = len(set(seen_rows) - expected_seen)
            raise ArchiveError(
                f"State seen-key reconciliation failed for {model}: "
                f"missing={missing}; extra={extra}"
            )

    journal_raw = payloads["research_outputs/experiment_ledger.jsonl"].decode(
        "utf-8"
    )
    journal: list[dict[str, Any]] = []
    for line_no, line in enumerate(journal_raw.splitlines(), start=1):
        if not line.strip():
            raise ArchiveError(f"Blank experiment journal line {line_no}.")
        item = _parse_json(line.encode("utf-8"), f"experiment journal line {line_no}")
        if not isinstance(item, dict) or item.get("experiment_id") != EXPERIMENT_ID:
            raise ArchiveError(f"Invalid experiment journal entry at line {line_no}.")
        journal.append(item)

    if len(journal) != 97:
        raise ArchiveError(f"Expected 97 experiment journal rows; got {len(journal)}.")

    run_rows = [r for r in journal if r.get("event") == "PHASE2D_RUN"]
    checkpoint_rows = [
        r for r in journal if r.get("event") == "F08_CHECKPOINT_EVALUATED"
    ]
    if len(run_rows) != 95 or len(checkpoint_rows) != 2:
        raise ArchiveError("Experiment journal event counts mismatch.")

    checkpoint_map = {
        int(row.get("checkpoint_blocks", -1)): row
        for row in checkpoint_rows
    }
    if set(checkpoint_map) != {15, 20}:
        raise ArchiveError("Expected exactly C15 and C20 checkpoint events.")
    for checkpoint in (15, 20):
        row = checkpoint_map[checkpoint]
        if row.get("decision") != "CHECKPOINT_CONTINUE":
            raise ArchiveError(f"C{checkpoint} decision does not match canonical archive.")
        if row.get("candidate_profitable") is True:
            raise ArchiveError(f"C{checkpoint} unexpectedly marks candidate profitable.")
        if row.get("candidate_beats_production") is True:
            raise ArchiveError(f"C{checkpoint} unexpectedly marks candidate superior.")

    if any(r.get("event") == "F08_GATE_AMENDMENT" for r in journal):
        raise ArchiveError("Archive already contains an F08 gate amendment.")
    if any(
        int(r.get("checkpoint_blocks", 0)) >= 25
        for r in checkpoint_rows
    ):
        raise ArchiveError("A C25-or-later checkpoint is already present.")

    return {
        "observation_count": len(observations),
        "candidate_count": counts["candidate"],
        "production_count": counts["production"],
        "event_count": len(events),
        "journal_count": len(journal),
        "checkpoint_ids": sorted(checkpoint_map),
        "pending_candidate": EXPECTED_PENDING_COUNTS["candidate"],
        "pending_production": EXPECTED_PENDING_COUNTS["production"],
        "state_seen_candidate": EXPECTED_SEEN_COUNTS["candidate"],
        "state_seen_production": EXPECTED_SEEN_COUNTS["production"],
    }


def _read_and_validate_zip(zip_bytes: bytes) -> dict[str, bytes]:
    if _sha256(zip_bytes) != ARCHIVE_SHA256:
        raise ArchiveError(
            "Decrypted ZIP SHA-256 does not match the verified authoritative archive."
        )

    import io

    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as archive:
            names = archive.namelist()
            if len(names) != len(EXPECTED_FILES) or set(names) != set(EXPECTED_FILES):
                raise ArchiveError(
                    "ZIP member list does not exactly match the six expected files."
                )
            bad_member = archive.testzip()
            if bad_member is not None:
                raise ArchiveError(f"ZIP CRC failure in {bad_member}.")
            payloads: dict[str, bytes] = {}
            for info in archive.infolist():
                if info.is_dir() or info.filename not in EXPECTED_FILES:
                    raise ArchiveError(f"Unexpected ZIP member: {info.filename}")
                expected_size = EXPECTED_FILES[info.filename][0]
                if info.file_size != expected_size:
                    raise ArchiveError(
                        f"ZIP size mismatch for {info.filename}: {info.file_size}"
                    )
                payloads[info.filename] = archive.read(info)
    except zipfile.BadZipFile as exc:
        raise ArchiveError(f"Decrypted payload is not a valid ZIP: {exc}") from exc

    _validate_archive_payloads(payloads)
    return payloads


def _merge_validation_stats(
    archive_stats: dict[str, Any],
    guard_stats: dict[str, Any],
) -> dict[str, Any]:
    """Combine archive-level and restore-guard metrics without key ambiguity."""
    required_archive = {
        "observation_count",
        "candidate_count",
        "production_count",
        "event_count",
        "journal_count",
        "checkpoint_ids",
        "pending_candidate",
        "pending_production",
    }
    required_guard = {
        "observation_count",
        "candidate_observation_count",
        "production_observation_count",
        "unique_utc_blocks",
        "sha256",
    }
    missing_archive = required_archive - set(archive_stats)
    missing_guard = required_guard - set(guard_stats)
    if missing_archive or missing_guard:
        raise ArchiveError(
            "Validation metrics are incomplete: "
            f"missing_archive={sorted(missing_archive)}; "
            f"missing_restore_guard={sorted(missing_guard)}"
        )

    count_pairs = {
        "observation_count": (
            archive_stats["observation_count"],
            guard_stats["observation_count"],
        ),
        "candidate_count": (
            archive_stats["candidate_count"],
            guard_stats["candidate_observation_count"],
        ),
        "production_count": (
            archive_stats["production_count"],
            guard_stats["production_observation_count"],
        ),
    }
    mismatches = {
        name: {"archive": left, "restore_guard": right}
        for name, (left, right) in count_pairs.items()
        if left != right
    }
    if mismatches:
        raise ArchiveError(
            "Archive and restore-guard observation counts disagree: "
            + json.dumps(mismatches, sort_keys=True)
        )

    merged = {**archive_stats, **guard_stats}
    merged["ledger_sha256"] = guard_stats["sha256"]
    return merged


def _validate_project_state(state: dict[str, Any]) -> None:
    """Run the project's canonical state-integrity validator before restore."""
    try:
        from phase2d_prospective_validator import validate_state_integrity
    except Exception as exc:
        raise ArchiveError(
            "Could not import project state-integrity validator."
        ) from exc

    try:
        validate_state_integrity(state)
    except Exception as exc:
        raise ArchiveError(
            f"Project state-integrity validation failed: {exc}"
        ) from exc


def verify_encrypted_archive(
    encrypted_path: Path,
    passphrase: str,
    baseline_manifest: Path,
    baseline_ledger: Path,
) -> tuple[dict[str, bytes], dict[str, Any]]:
    encrypted = encrypted_path.read_bytes()
    zip_bytes = _decrypt_bytes(encrypted, passphrase)
    payloads = _read_and_validate_zip(zip_bytes)
    archive_stats = _validate_archive_payloads(payloads)

    state = _parse_json(
        payloads["research_outputs/phase2d_state.json"],
        "phase2d_state.json",
    )
    _validate_project_state(state)

    with tempfile.TemporaryDirectory(prefix="f08-archive-guard-") as temp:
        temp_root = Path(temp)
        state_path = temp_root / "phase2d_state.json"
        ledger_path = temp_root / "phase2d_observations.jsonl"

        state_path.write_bytes(
            payloads["research_outputs/phase2d_state.json"]
        )
        ledger_path.write_bytes(
            payloads["research_outputs/phase2d_observations.jsonl"]
        )

        try:
            from phase2d_restore_guard import inspect_artifact
        except Exception as exc:
            raise ArchiveError(
                "Could not import phase2d_restore_guard from the repository."
            ) from exc

        try:
            guard_stats = inspect_artifact(
                state_path,
                ledger_path,
                baseline_manifest,
                EXPERIMENT_ID,
                baseline_ledger,
            )
        except Exception as exc:
            raise ArchiveError(f"Canonical F08 restore guard failed: {exc}") from exc

    return payloads, _merge_validation_stats(archive_stats, guard_stats)


def restore_archive(
    encrypted_path: Path,
    output_dir: Path,
    validator_log_path: Path,
    passphrase: str,
    baseline_manifest: Path,
    baseline_ledger: Path,
) -> dict[str, Any]:
    payloads, validation_stats = verify_encrypted_archive(
        encrypted_path,
        passphrase,
        baseline_manifest,
        baseline_ledger,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    validator_log_path.parent.mkdir(parents=True, exist_ok=True)

    mapping = {
        "research_outputs/phase2d_state.json": output_dir / "phase2d_state.json",
        "research_outputs/phase2d_results.json": output_dir / "phase2d_results.json",
        "research_outputs/phase2d_latest_snapshot.json": output_dir / "phase2d_latest_snapshot.json",
        "research_outputs/experiment_ledger.jsonl": output_dir / "experiment_ledger.jsonl",
        "research_outputs/phase2d_observations.jsonl": output_dir / "phase2d_observations.jsonl",
        "phase2d_validator.log": validator_log_path,
    }

    staged: list[tuple[Path, Path]] = []
    with tempfile.TemporaryDirectory(
        prefix="f08-archive-stage-",
        dir=str(output_dir.parent),
    ) as temp:
        temp_root = Path(temp)
        try:
            for source_name, destination in mapping.items():
                staged_path = temp_root / destination.name
                staged_path.write_bytes(payloads[source_name])
                staged.append((staged_path, destination))

            for staged_path, destination in staged:
                destination.parent.mkdir(parents=True, exist_ok=True)
                os.replace(staged_path, destination)
        except Exception:
            raise

    return {
        **validation_stats,
        "restored_files": [str(destination) for _, destination in staged],
    }


def _print_summary(stats: dict[str, Any]) -> None:
    print("F08 AUTHORITATIVE ARCHIVE VALIDATION: PASS")
    print("Experiment:", EXPERIMENT_ID)
    print("Observations:", stats["observation_count"])
    print("Candidate:", stats["candidate_count"])
    print("Production:", stats["production_count"])
    print("Events:", stats["event_count"])
    print("Journal rows:", stats["journal_count"])
    print("Preserved checkpoints:", stats["checkpoint_ids"])
    print("Pending candidate:", stats["pending_candidate"])
    print("Pending production:", stats["pending_production"])
    print("Canonical baseline / append-only guard: PASS")
    if stats.get("ledger_sha256"):
        print("Ledger SHA256:", stats["ledger_sha256"])


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Encrypt, verify, or restore the pinned F08 authoritative archive."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    encrypt_parser = sub.add_parser("encrypt")
    encrypt_parser.add_argument("--archive", required=True)
    encrypt_parser.add_argument("--output", required=True)
    encrypt_parser.add_argument("--passphrase-env", required=True)

    verify_parser = sub.add_parser("verify")
    verify_parser.add_argument("--encrypted-archive", required=True)
    verify_parser.add_argument("--baseline-manifest", required=True)
    verify_parser.add_argument("--baseline-ledger", required=True)
    verify_parser.add_argument("--passphrase-env", required=True)

    restore_parser = sub.add_parser("restore")
    restore_parser.add_argument("--encrypted-archive", required=True)
    restore_parser.add_argument("--output-dir", required=True)
    restore_parser.add_argument("--validator-log-path", required=True)
    restore_parser.add_argument("--baseline-manifest", required=True)
    restore_parser.add_argument("--baseline-ledger", required=True)
    restore_parser.add_argument("--passphrase-env", required=True)

    args = parser.parse_args()

    try:
        passphrase = _read_passphrase(args.passphrase_env)

        if args.command == "encrypt":
            result = encrypt_archive(
                Path(args.archive),
                Path(args.output),
                passphrase,
            )
            print("F08 archive encryption: PASS")
            print("Verified plaintext ZIP SHA256:", result["archive_sha256"])
            print("Encrypted asset bytes:", result["encrypted_bytes"])
            print("Encrypted asset SHA256:", result["encrypted_sha256"])
            return 0

        if args.command == "verify":
            _, stats = verify_encrypted_archive(
                Path(args.encrypted_archive),
                passphrase,
                Path(args.baseline_manifest),
                Path(args.baseline_ledger),
            )
            _print_summary(stats)
            return 0

        stats = restore_archive(
            Path(args.encrypted_archive),
            Path(args.output_dir),
            Path(args.validator_log_path),
            passphrase,
            Path(args.baseline_manifest),
            Path(args.baseline_ledger),
        )
        _print_summary(stats)
        print("Restored files:")
        for path in stats["restored_files"]:
            print(" -", path)
        return 0

    except Exception as exc:
        print(f"FATAL: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
