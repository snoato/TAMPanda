"""Print the conflict-set structure of grasp libraries (pure Python; no MuJoCo).

For every cell: the number of candidate motions, the subset-minimal conflict sets (an action is feasible iff at least
one of them is clear), and the cells that EVERY candidate needs empty (their intersection). Offsets are printed as
(dcol, drow) within the target's region, rows counting away from the robot. The summary counts conflicts behind the
target and in other regions over all candidates (both are 0 for the shipped libraries).

    python scripts/libraries/conflict_sets.py                       # the three shipped libraries
    python scripts/libraries/conflict_sets.py LIB.json [LIB.json ...]

Adapted from the probe that produced the blocking-structure statements in the camera-ready specs (2026-10-09).
"""
import json
import sys
from pathlib import Path

DOMAINS = Path(__file__).resolve().parents[2] / "tampanda" / "symbolic" / "domains"
SHIPPED = [DOMAINS / "access19_v2" / "data" / "library_T17C.json",
           DOMAINS / "confined_shelf" / "honest_data" / "library_v1.json",
           DOMAINS / "shelf_access_v2" / "data" / "library_M3.json"]


def parse(cell):
    region, ij = cell.split("__")
    i, j = map(int, ij.split("_"))
    return region, i, j


def conflict_sets(meta):
    """{cell: [frozenset of conflicting cells, one per candidate]} from a library's JSON metadata."""
    bits = meta["bits"]
    return {c: [frozenset(bits[k] for k in range(len(bits)) if (int(r["mask"], 16) >> k) & 1) for r in rows]
            for c, rows in meta["cells"].items()}


def minimal(sets):
    """The subset-minimal conflict sets, deduplicated, smallest first."""
    uniq = set(sets)
    return sorted((s for s in uniq if not any(o < s for o in uniq)), key=lambda s: (len(s), sorted(s)))


def offsets(cell, cells):
    r, i, j = parse(cell)
    out = []
    for o in cells:
        r2, i2, j2 = parse(o)
        out.append((i2 - i, j2 - j) if r2 == r else (r2, i2, j2))
    return sorted(out, key=str)


def report(path):
    meta = json.load(open(path))
    cs = conflict_sets(meta)
    print("== %s: %d cells, %d candidates" % (Path(path).name, len(cs), sum(len(v) for v in cs.values())))
    behind = other = 0
    for cell, sets in cs.items():
        r, i, j = parse(cell)
        for s in sets:
            for o in s:
                r2, _, j2 = parse(o)
                other += r2 != r
                behind += r2 == r and j2 > j
        mins = minimal(sets)
        need = frozenset.intersection(*sets) if sets else frozenset()
        print("  %-22s cand %d  minimal sets %d  |every candidate needs empty| %2d  smallest: %s"
              % (cell, len(sets), len(mins), len(need), offsets(cell, mins[0]) if mins else "NO CANDIDATE"))
        for extra in mins[1:]:
            print("  %-22s %38s alternative: %s" % ("", "", offsets(cell, extra)))
    print("  SUMMARY conflicts over all candidates: behind the target %d, in another region %d" % (behind, other))


if __name__ == "__main__":
    for p in sys.argv[1:] or SHIPPED:
        report(p)
