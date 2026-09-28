from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
import subprocess
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence


CLEANED_DATA_MANIFEST_SHA256 = (
    "a807db2390d5a85a9b77f1286e6ab7d1aa1a1f7faaf2396f3be8459220e0ce68"
)


@dataclass(frozen=True)
class RegisteredCheckpoint:
    model: str
    path: Path
    sha256: str
    bytes: int


@dataclass(frozen=True)
class RuntimeAttestation:
    host_label: str
    cpu_model: str
    gpu_model: str
    registered_hashes: dict[str, str]
    driver_manifest_sha256: str
    checkpoints: dict[str, RegisteredCheckpoint]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def assert_process_environment() -> None:
    expected = {
        "PYTHONHASHSEED": "13",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
    }
    mismatches = {
        name: os.environ.get(name)
        for name, value in expected.items()
        if os.environ.get(name) != value
    }
    if mismatches:
        raise RuntimeError(
            f"process environment was not fixed before interpreter start: {mismatches}"
        )


def _hash_manifest(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2 or len(parts[0]) != 64:
            raise RuntimeError(f"malformed hash manifest line {number}: {path}")
        result[parts[1].strip().replace("\\", "/")] = parts[0].lower()
    return result


def verify_registered_inputs(root: Path) -> dict[str, str]:
    root = root.resolve()
    manifest_path = root / "MANIFEST.sha256"
    if not manifest_path.is_file():
        raise RuntimeError(f"registered manifest missing: {manifest_path}")
    manifest = _hash_manifest(manifest_path)
    required_exact = {
        "code/generators.py", "code/synth_calibration.json",
        "frozen/stratifier_v2.csv", "phase0c_outputs/environment_phase1.json",
        "phase0c_outputs/environment_phase1.lock.txt",
    }
    missing = required_exact - set(manifest)
    if missing or not any(path.startswith("pipeline_code/") for path in manifest):
        raise RuntimeError(f"registered manifest lacks required entries: {sorted(missing)}")
    if not any(path.startswith("final_pool/") for path in manifest):
        raise RuntimeError("registered manifest lacks final_pool entries")
    for relative, expected in manifest.items():
        path = root / relative
        if not path.is_file():
            raise RuntimeError(f"registered file missing: {relative}")
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(
                f"registered hash mismatch for {relative}: expected {expected}, got {actual}"
            )
    return dict(sorted(manifest.items()))


def ensure_cleaned_data_manifest(root: Path, runtime_root: Path) -> dict[str, str]:
    """Create once, then verify, a complete cleaned-data file hash ledger."""
    cleaned = root.resolve() / "cleaned_data"
    if not cleaned.is_dir():
        raise RuntimeError(f"cleaned_data directory missing: {cleaned}")
    manifest_path = runtime_root.resolve() / "cleaned_data_sha256.txt"
    actual_files = sorted(path for path in cleaned.rglob("*") if path.is_file())
    actual_names = {path.relative_to(cleaned).as_posix() for path in actual_files}
    if not manifest_path.exists():
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = manifest_path.with_name(f".{manifest_path.name}.{os.getpid()}.tmp")
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            for path in actual_files:
                stream.write(f"{sha256_file(path)}  {path.relative_to(cleaned).as_posix()}\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(manifest_path)
    manifest_digest = sha256_file(manifest_path)
    if manifest_digest != CLEANED_DATA_MANIFEST_SHA256:
        raise RuntimeError(
            "cleaned_data manifest aggregate hash mismatch: "
            f"expected {CLEANED_DATA_MANIFEST_SHA256}, got {manifest_digest}"
        )
    manifest = _hash_manifest(manifest_path)
    if set(manifest) != actual_names:
        raise RuntimeError(
            "cleaned_data file set differs from its frozen ledger: "
            f"missing={sorted(set(manifest) - actual_names)[:3]} "
            f"extra={sorted(actual_names - set(manifest))[:3]}"
        )
    for relative, expected in manifest.items():
        actual = sha256_file(cleaned / relative)
        if actual != expected:
            raise RuntimeError(
                f"cleaned_data hash mismatch for {relative}: expected {expected}, got {actual}"
            )
    return dict(sorted(manifest.items()))


def machine_host_label() -> str:
    """Return the machine-derived label; operators cannot supply this value."""
    value = socket.gethostname().strip()
    if not value:
        raise RuntimeError("machine hostname is empty")
    return value


def resolve_registered_host(host_registry: Path) -> tuple[str, str]:
    """Map this machine's hostname to its frozen logical assignment role."""
    document = json.loads(host_registry.resolve().read_text(encoding="utf-8"))
    actual = machine_host_label()
    matches = [str(role) for role, hostname in document.items() if str(hostname) == actual]
    if len(matches) != 1:
        raise RuntimeError(
            f"hostname {actual!r} is not registered exactly once in {host_registry}"
        )
    return matches[0], actual


def verify_driver_manifest(driver_dir: Path) -> str:
    driver_dir = driver_dir.resolve()
    manifest_root = driver_dir.parent
    manifest_path = manifest_root / "phase1_driver_sha256.txt"
    if not manifest_path.is_file():
        raise RuntimeError(f"driver manifest missing: {manifest_path}")
    entries = _hash_manifest(manifest_path)
    for relative, expected in entries.items():
        path = manifest_root / relative
        if not path.is_file():
            raise RuntimeError(f"driver file missing: {relative}")
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(
                f"driver hash mismatch for {relative}: expected {expected}, got {actual}"
            )
    actual_files = {
        path.relative_to(manifest_root).as_posix()
        for path in driver_dir.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
    }
    if actual_files != set(entries):
        raise RuntimeError(
            f"driver manifest file set mismatch: missing={sorted(actual_files - set(entries))} "
            f"extra={sorted(set(entries) - actual_files)}"
        )
    return sha256_file(manifest_path)


def derive_runtime_identity(root: Path, driver_dir: Path, assignment_path: Path,
                            task_key: str, *, verify: bool = True) -> Any:
    from .checkpoints import CheckpointIdentity

    if verify:
        verify_registered_inputs(root)
        driver_manifest_sha256 = verify_driver_manifest(driver_dir)
    else:
        driver_manifest_sha256 = sha256_file(driver_dir.resolve().parent / "phase1_driver_sha256.txt")
    if not assignment_path.is_file():
        raise RuntimeError(f"host assignment missing: {assignment_path}")
    return CheckpointIdentity(
        registration_sha256=sha256_file(root.resolve() / "MANIFEST.sha256"),
        driver_sha256=driver_manifest_sha256,
        host_assignment_sha256=sha256_file(assignment_path.resolve()),
        task_key=task_key,
    )


def registered_checkpoint_paths(root: Path) -> dict[str, RegisteredCheckpoint]:
    environment_path = root.resolve() / "phase0c_outputs" / "environment_phase1.json"
    document = json.loads(environment_path.read_text(encoding="utf-8"))
    source = document["checkpoints"]
    mapping = {"tabpfn35": "tabpfn35", "tabpfn2": "tabpfn2_b", "tabicl2": "tabicl2"}
    result: dict[str, RegisteredCheckpoint] = {}
    for model, key in mapping.items():
        item = source[key]
        path = Path(item["path"]).expanduser().resolve()
        if not path.is_file():
            raise RuntimeError(f"registered weight missing for {model}: {path}")
        actual_hash = sha256_file(path)
        expected_hash = str(item["sha256"]).lower()
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"weight hash mismatch for {model}: expected {expected_hash}, got {actual_hash}"
            )
        actual_size = path.stat().st_size
        if actual_size != int(item["bytes"]):
            raise RuntimeError(
                f"weight size mismatch for {model}: expected {item['bytes']}, got {actual_size}"
            )
        result[model] = RegisteredCheckpoint(model, path, expected_hash, actual_size)
    return result


def set_offline_environment() -> None:
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        os.environ[name] = "1"
    os.environ["WANDB_MODE"] = "offline"
    os.environ["NO_PROXY"] = "*"
    os.environ["no_proxy"] = "*"


@contextmanager
def offline_socket_guard() -> Iterator[None]:
    original_socket = socket.socket
    original_connection = socket.create_connection

    def blocked(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("network disabled for offline Phase 1 execution")

    socket.socket = blocked  # type: ignore[assignment]
    socket.create_connection = blocked  # type: ignore[assignment]
    try:
        yield
    finally:
        socket.socket = original_socket  # type: ignore[assignment]
        socket.create_connection = original_connection  # type: ignore[assignment]


def validate_hardware(unit: Any, registered_role: str, actual_host: str,
                      cpu_model: str, gpu_model: str) -> None:
    if registered_role != str(unit.host):
        raise RuntimeError(
            f"machine-derived host {actual_host} is registered as {registered_role}, "
            f"not unit.host {unit.host}"
        )
    if unit.group == "cpu" and "8352V" not in cpu_model:
        raise RuntimeError(f"CPU unit requires Xeon Platinum 8352V, got {cpu_model}")
    if unit.group in {"gpu", "ftt"} and "RTX 4090" not in gpu_model:
        raise RuntimeError(f"GPU unit requires RTX 4090, got {gpu_model}")
    if unit.group == "none" and "8352V" not in cpu_model:
        raise RuntimeError(f"model-free unit requires Xeon Platinum 8352V, got {cpu_model}")


def assert_unit_in_plan(unit: Any, plan: Sequence[Any]) -> None:
    matches = [candidate for candidate in plan if candidate == unit]
    if len(matches) != 1:
        raise RuntimeError(
            f"unit is not in verified plan exactly once: key={unit.key}, matches={len(matches)}"
        )


def cpu_model_name() -> str:
    path = Path("/proc/cpuinfo")
    if path.exists():
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def gpu_model_name() -> str:
    import torch
    return str(torch.cuda.get_device_name(0))


def gpu_driver_version() -> str:
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            check=True, capture_output=True, text=True, timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f"unable to read NVIDIA driver version: {exc}") from exc
    versions = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if not versions:
        raise RuntimeError("nvidia-smi returned no NVIDIA driver version")
    return ",".join(versions)
