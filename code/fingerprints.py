"""Fingerprints of the frozen generators, so that the executor can confirm that its environment
reproduces the registered arrays. Arrays are rounded to 10 decimals before hashing.

Usage: python3 fingerprints.py            -> prints JSON
       python3 fingerprints.py check F    -> compares with the JSON file F, exit 1 on mismatch
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import generators as G  # noqa: E402


def h(a) -> str:
    a = np.round(np.asarray(a, dtype=float), 10)
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


def compute() -> dict:
    y = (np.arange(1000) % 7 < 2).astype(int)
    out = {"numpy": np.__version__}
    for arm, lev, rep in [("noise", None, 0), ("concentrated", 0.85, 0), ("dispersed", 0.85, 1),
                          ("dispersed_noise", 0.85, 2), ("concentrated", 0.65, 2)]:
        out[f"inject|{arm}|{lev}|r{rep}"] = h(G.injection_block(y, 31, arm, lev, rep)[0])
    out["subsample|31|0|250|2"] = h(G.subsample_indices(y, 31, 0, 250, 2))
    out["removal|31"] = hashlib.sha256(json.dumps(G.removal_plan(31, {f"f{j}": 0.5 + 0.013 * ((j * 7) % 11) for j in range(12)}),
                                                  sort_keys=True).encode()).hexdigest()[:16]
    for key in [("linear", 4, 0.0, 0.7, 300, 0, "train"), ("threshold", 16, 0.3, 0.8, 1000, 5, "test")]:
        X, yy, p, inf = G.synth_draw(*key)
        out["synth|" + "|".join(map(str, key))] = h(np.column_stack([X, yy, p]))
    return out


if __name__ == "__main__":
    fp = compute()
    if len(sys.argv) > 2 and sys.argv[1] == "check":
        ref = json.loads(Path(sys.argv[2]).read_text())
        bad = [k for k in ref if k != "numpy" and ref[k] != fp.get(k)]
        print(json.dumps({"reference_numpy": ref.get("numpy"), "this_numpy": fp["numpy"], "mismatches": bad}, indent=1))
        sys.exit(1 if bad else 0)
    print(json.dumps(fp, indent=1))
