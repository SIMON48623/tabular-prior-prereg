from __future__ import annotations

import json
import os
import shutil
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CheckpointIdentity:
    registration_sha256: str
    driver_sha256: str
    host_assignment_sha256: str
    task_key: str


def checkpoint_path(root: Path, unit: Any) -> Path:
    """Return the one canonical checkpoint path for a registered plan unit."""
    safe_identifier = str(unit.identifier).replace("|", "__")
    return (
        root / str(unit.module) / safe_identifier / str(unit.variant)
        / f"fold_{int(unit.fold)}" / f"{unit.group}.json"
    )


def rows_to_columnar(rows: list[dict[str, Any]]) -> dict[str, list[Any]]:
    """Encode prediction rows as JSON-friendly column arrays."""
    if not rows:
        return {}
    columns = list(rows[0])
    expected = set(columns)
    for number, row in enumerate(rows):
        if set(row) != expected:
            raise RuntimeError(f"ragged prediction row {number}")
    return {column: [row[column] for row in rows] for column in columns}


def columnar_to_rows(columns: dict[str, list[Any]]) -> list[dict[str, Any]]:
    if not columns:
        return []
    lengths = {len(values) for values in columns.values()}
    if len(lengths) != 1:
        raise RuntimeError("columnar prediction arrays have unequal lengths")
    length = lengths.pop()
    names = list(columns)
    return [{name: columns[name][index] for name in names} for index in range(length)]


def write_checkpoint(path: Path, identity: CheckpointIdentity, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps({"identity": asdict(identity), "payload": payload}, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    try:
        directory_fd = os.open(path.parent, os.O_RDONLY)
    except OSError:
        directory_fd = None
    if directory_fd is not None:
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)


def read_checkpoint(path: Path, identity: CheckpointIdentity) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("identity") != asdict(identity):
        raise RuntimeError(f"checkpoint identity mismatch: {path}")
    return dict(document["payload"])


@contextmanager
def exclusive_lock(path: Path):
    """Crash-releasing advisory lock used for schedulers and individual units."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor: int | None = None
    used_flock = False
    try:
        try:
            import fcntl
        except ImportError:
            fcntl = None  # type: ignore[assignment]
        if fcntl is not None:
            descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                os.close(descriptor)
                descriptor = None
                raise RuntimeError(f"lock already held: {path}") from exc
            os.ftruncate(descriptor, 0)
            used_flock = True
        else:
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        os.write(descriptor, f"pid={os.getpid()}\n".encode())
        os.fsync(descriptor)
    except FileExistsError as exc:
        raise RuntimeError(f"lock already held: {path}") from exc
    try:
        yield
    finally:
        if descriptor is not None:
            if used_flock:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)
        if not used_flock:
            path.unlink(missing_ok=True)


def quarantine_checkpoint(path: Path, quarantine_root: Path, reason: str) -> Path:
    quarantine_root.mkdir(parents=True, exist_ok=True)
    destination = quarantine_root / f"{path.name}.{os.getpid()}.invalid"
    counter = 0
    while destination.exists():
        counter += 1
        destination = quarantine_root / f"{path.name}.{os.getpid()}.{counter}.invalid"
    shutil.move(str(path), str(destination))
    destination.with_suffix(destination.suffix + ".reason.txt").write_text(
        reason + "\n", encoding="utf-8"
    )
    return destination


def load_migrations(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    document = json.loads(path.resolve().read_text(encoding="utf-8"))
    if not isinstance(document, list):
        raise RuntimeError("checkpoint migration file must contain a JSON list")
    required = {"old_driver_sha256", "new_driver_sha256", "task_keys", "approved_by"}
    for number, item in enumerate(document):
        if not isinstance(item, dict) or required - set(item):
            raise RuntimeError(f"invalid checkpoint migration entry {number}")
        if not item["approved_by"] or not isinstance(item["task_keys"], list):
            raise RuntimeError(f"unapproved checkpoint migration entry {number}")
    return document


def read_checkpoint_with_migration(
    path: Path,
    identity: CheckpointIdentity,
    migrations: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    document = json.loads(path.read_text(encoding="utf-8"))
    saved = document.get("identity")
    expected = asdict(identity)
    if saved == expected:
        return dict(document["payload"]), None
    if not isinstance(saved, dict):
        raise RuntimeError(f"checkpoint identity mismatch: {path}")
    stable = ("registration_sha256", "host_assignment_sha256", "task_key")
    if any(saved.get(key) != expected[key] for key in stable):
        raise RuntimeError(f"checkpoint identity mismatch: {path}")
    for item in migrations:
        if (
            saved.get("driver_sha256") == item["old_driver_sha256"]
            and expected["driver_sha256"] == item["new_driver_sha256"]
            and identity.task_key in item["task_keys"]
        ):
            payload = dict(document["payload"])
            return payload, {
                "old_driver_sha256": item["old_driver_sha256"],
                "new_driver_sha256": item["new_driver_sha256"],
                "approved_by": item["approved_by"],
                "task_key": identity.task_key,
            }
    raise RuntimeError(
        f"checkpoint from driver {saved.get('driver_sha256')} is not approved for migration: {path}"
    )
