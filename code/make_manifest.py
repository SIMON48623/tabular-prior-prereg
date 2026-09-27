"""Write MANIFEST.sha256 at the repository root: the SHA-256 of every registered file.

Registered: README.md, .gitignore, pipeline_code_sha256.txt and everything under the directories listed in
DIRS (caches excluded). Anything else in the working folder is not registered; .gitignore keeps it out of Git.
Refuses to run while the plan still carries the pre-flight pending marker, unless --draft is given.
Usage: python3 code/make_manifest.py [--draft]
"""
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = ["README.md", ".gitignore", "pipeline_code_sha256.txt"]
DIRS = ["code", "docs", "final_pool", "frozen", "phase0b_outputs", "phase0c_outputs", "phase0c_history",
        "pool_review", "power", "preflight_outputs", "pipeline_code"]

plan = (ROOT / "docs" / "preregistration_v2.md").read_text(encoding="utf-8")
if "[PENDING PRE-FLIGHT]" in plan and "--draft" not in sys.argv:
    sys.exit("preregistration_v2.md still marked [PENDING PRE-FLIGHT]; run the pre-flight first (or pass --draft)")

paths = [ROOT / f for f in FILES if (ROOT / f).is_file()]
for d in DIRS:
    if (ROOT / d).is_dir():
        paths += [q for q in (ROOT / d).rglob("*")
                  if q.is_file() and "__pycache__" not in q.parts and q.suffix != ".pyc"]
rel = sorted(q.relative_to(ROOT).as_posix() for q in paths)
lines = [f"{hashlib.sha256((ROOT / r).read_bytes()).hexdigest()}  {r}" for r in rel]
(ROOT / "MANIFEST.sha256").write_text("\n".join(lines) + "\n")
missing = [d for d in ("preflight_outputs", "pipeline_code") if not (ROOT / d).is_dir()]
print(f"{len(lines)} files" + (f"; not yet present: {', '.join(missing)}" if missing else ""))
