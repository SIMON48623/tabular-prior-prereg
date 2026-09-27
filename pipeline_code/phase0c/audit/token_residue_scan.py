from __future__ import annotations

import getpass
import json
from pathlib import Path


def scan(root: Path, needles: list[bytes], skip: set[Path]) -> dict[str, int]:
    counts = {"files_scanned": 0, "exact_token_matches": 0, "unique_prefix_matches": 0}
    for path in root.rglob("*"):
        if not path.is_file() or path in skip:
            continue
        try:
            data = path.read_bytes()
        except (OSError, PermissionError):
            continue
        counts["files_scanned"] += 1
        counts["exact_token_matches"] += data.count(needles[0])
        counts["unique_prefix_matches"] += data.count(needles[1])
    return counts


def main() -> None:
    task_root = Path("/root/autodl-tmp/tabular_prior_phase0c")
    token = getpass.getpass("Token for residue scan (input hidden): ").encode()
    secret_part = token.split(b"_", 2)[-1]
    if len(secret_part) < 6:
        raise RuntimeError("token secret part is too short")
    needles = [token, secret_part[:6]]
    output_scan = scan(task_root / "phase0c_outputs", needles, set())
    home_skip = {
        task_root / "run_state" / "token_scan.json",
        task_root / "token_residue_scan.py",
    }
    home_scan = scan(Path("/root"), needles, home_skip)
    token = b""
    result = {
        "prefix_redacted": True,
        "prefix_definition": "first six characters of token secret payload after product prefix",
        "phase0c_outputs": output_scan,
        "home": home_scan,
    }
    target = task_root / "run_state" / "token_scan.json"
    target.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
