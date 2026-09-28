from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def _atomic_json(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(document, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def append_run_log(path: Path, record: dict[str, Any]) -> None:
    required = {
        "task_key", "elapsed_seconds", "status", "failure_reasons",
        "s_undefined", "inner_failures",
    }
    missing = required - set(record)
    if missing:
        raise RuntimeError(f"run log missing fields: {sorted(missing)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
    with path.open("a", encoding="utf-8") as stream:
        stream.write(encoded)


def write_environment_record(path: Path, record: dict[str, Any]) -> None:
    required = {"host", "lock_sha256", "weights", "determinism"}
    missing = required - set(record)
    if missing:
        raise RuntimeError(f"environment record missing fields: {sorted(missing)}")
    document: dict[str, Any] = {"hosts": {}}
    if path.exists():
        document = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(document.get("hosts"), dict):
            raise RuntimeError("invalid existing environment_run.json")
    host = str(record["host"])
    body = {key: value for key, value in record.items() if key != "host"}
    existing = document["hosts"].setdefault(host, [])
    if not isinstance(existing, list):
        raise RuntimeError(f"invalid environment record list for host {host}")
    if body not in existing:
        existing.append(body)
    _atomic_json(path, document)
