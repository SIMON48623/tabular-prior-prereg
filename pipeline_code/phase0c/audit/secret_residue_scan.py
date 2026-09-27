from __future__ import annotations

import getpass
import json
from pathlib import Path


def scan(root: Path, needles: dict[str, bytes], skip: set[Path]) -> dict[str, int]:
    counts = {"files_scanned": 0, **{name: 0 for name in needles}}
    for path in root.rglob("*"):
        if not path.is_file() or path in skip:
            continue
        try:
            data = path.read_bytes()
        except (OSError, PermissionError):
            continue
        counts["files_scanned"] += 1
        for name, needle in needles.items():
            counts[name] += data.count(needle)
    return counts


def main() -> None:
    task_root = Path("/root/autodl-tmp/tabular_prior_phase0c")
    token = getpass.getpass("Token for residue scan (input hidden): ").encode()
    password = getpass.getpass("SSH password for residue scan (input hidden): ").encode()
    secret_part = token.split(b"_", 2)[-1]
    if len(secret_part) < 6 or len(password) < 6:
        raise RuntimeError("secret input is too short")
    needles = {
        "exact_token_matches": token,
        "token_unique_prefix_matches": secret_part[:6],
        "exact_password_matches": password,
    }
    output_scan = scan(task_root / "phase0c_outputs", needles, set())
    scanner = task_root / "secret_residue_scan.py"
    result_file = task_root / "run_state" / "secret_scan.json"
    home_scan = scan(Path("/root"), needles, {scanner, result_file})
    token = b""
    password = b""
    result = {
        "prefix_redacted": True,
        "prefix_definition": "first six characters of token secret payload after product prefix",
        "phase0c_outputs": output_scan,
        "home": home_scan,
    }
    result_file.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
