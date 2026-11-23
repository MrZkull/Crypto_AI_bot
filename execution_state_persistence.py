"""Strict, durable JSON persistence for order-critical local state.

This module is intentionally small and exchange-agnostic. It does not submit,
cancel, reconcile, or otherwise mutate exchange orders. Callers must treat any
mirror mismatch, corrupt JSON, or failed write as a hard persistence failure.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable


class StatePersistenceError(RuntimeError):
    """Base error for strict order-state persistence failures."""


class StateFileCorrupt(StatePersistenceError):
    """A persisted state file is not valid, unambiguous JSON."""


class StateMirrorMismatch(StatePersistenceError):
    """Two existing state mirrors contain different values."""


@dataclass(frozen=True)
class PersistenceReceipt:
    paths: tuple[str, ...]
    sha256: str
    size_bytes: int


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _reject_nonstandard_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON numeric constant: {value}")


def _decode_json(raw: bytes, *, source: str) -> Any:
    try:
        return json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonstandard_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise StateFileCorrupt(f"Invalid or ambiguous JSON in {source}: {type(exc).__name__}") from None


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize state deterministically and reject NaN/Infinity values."""
    try:
        text = json.dumps(
            value,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
            default=str,
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise StatePersistenceError(f"State cannot be serialized safely: {type(exc).__name__}") from None
    return (text + "\n").encode("utf-8")


def _normalise_paths(paths: Iterable[str | os.PathLike[str]]) -> tuple[Path, ...]:
    normalised: list[Path] = []
    seen: set[str] = set()
    for raw in paths:
        path = Path(raw).expanduser()
        if not str(path).strip():
            raise ValueError("State mirror path must be non-empty")
        absolute = Path(os.path.abspath(os.fspath(path)))
        # Refuse a symlink at the file or any already-existing parent path;
        # resolving first would erase the evidence that the requested path was
        # redirected.
        cursor = absolute
        while True:
            if cursor.is_symlink():
                raise StatePersistenceError(f"Refusing symlinked state path: {cursor}")
            if cursor.parent == cursor:
                break
            cursor = cursor.parent
        key = os.path.normcase(str(absolute))
        if key in seen:
            raise ValueError("State mirror paths must be unique")
        seen.add(key)
        normalised.append(absolute)
    if not normalised:
        raise ValueError("At least one state mirror path is required")
    return tuple(normalised)


def read_json_mirrors(
    paths: Iterable[str | os.PathLike[str]],
    *,
    default: Any = None,
) -> Any:
    """Read one or more mirrors; refuse corruption or disagreement.

    A single existing mirror is accepted for compatibility with older local
    installs. When multiple mirrors exist, every copy must decode to the same
    JSON value. Missing files are not interpreted as an empty account/trade set.
    """
    targets = _normalise_paths(paths)
    found: list[tuple[Path, Any]] = []
    for path in targets:
        if path.is_symlink():
            raise StatePersistenceError(f"Refusing symlinked state file: {path}")
        if not path.exists():
            continue
        if not path.is_file():
            raise StatePersistenceError(f"State path is not a regular file: {path}")
        try:
            raw = path.read_bytes()
        except OSError as exc:
            raise StatePersistenceError(f"Cannot read state file {path}: {type(exc).__name__}") from None
        found.append((path, _decode_json(raw, source=str(path))))

    if not found:
        return default

    first = found[0][1]
    for path, value in found[1:]:
        if value != first:
            names = ", ".join(str(item[0]) for item in found)
            raise StateMirrorMismatch(f"State mirrors disagree: {names}")
    return first


def _fsync_directory(path: Path) -> None:
    # POSIX directory fsync closes the rename durability window. Windows does
    # not expose a portable directory fsync through Python's standard library;
    # the file itself is still flushed before os.replace.
    if os.name == "nt":
        return
    flags = getattr(os, "O_DIRECTORY", 0) | os.O_RDONLY
    try:
        fd = os.open(str(path), flags)
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_json_mirrors(
    paths: Iterable[str | os.PathLike[str]],
    value: Any,
) -> PersistenceReceipt:
    """Durably write and read-verify every state mirror.

    Every temporary file is created beside its destination so os.replace is a
    same-filesystem atomic replacement. The operation is deliberately not
    advertised as a cross-file transaction: if a later mirror fails, callers
    receive an exception and must not proceed with an order mutation.
    """
    targets = _normalise_paths(paths)
    # Never overwrite a corrupted or divergent existing state snapshot.
    read_json_mirrors(targets, default=None)
    payload = canonical_json_bytes(value)
    expected = _decode_json(payload, source="new state payload")
    pending: list[tuple[Path, Path]] = []
    replaced: list[Path] = []

    try:
        for path in targets:
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.is_symlink():
                raise StatePersistenceError(f"Refusing symlinked state file: {path}")
            if path.exists() and not path.is_file():
                raise StatePersistenceError(f"State destination is not a regular file: {path}")
            fd, temp_name = tempfile.mkstemp(
                prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
            )
            temp_path = Path(temp_name)
            try:
                with os.fdopen(fd, "wb") as stream:
                    stream.write(payload)
                    stream.flush()
                    os.fsync(stream.fileno())
            except Exception:
                try:
                    os.close(fd)
                except OSError:
                    pass
                raise
            pending.append((temp_path, path))

        for temp_path, destination in pending:
            os.replace(temp_path, destination)
            replaced.append(destination)
            _fsync_directory(destination.parent)

        observed = read_json_mirrors(targets, default=None)
        if observed != expected:
            raise StatePersistenceError("Persisted state read-back differs from the requested snapshot")

        return PersistenceReceipt(
            paths=tuple(str(path) for path in targets),
            sha256=hashlib.sha256(payload).hexdigest().upper(),
            size_bytes=len(payload),
        )
    except StatePersistenceError:
        raise
    except OSError as exc:
        changed = ", ".join(str(path) for path in replaced) or "none"
        raise StatePersistenceError(
            f"State mirror write failed ({type(exc).__name__}); replaced={changed}"
        ) from None
    finally:
        for temp_path, _ in pending:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass
