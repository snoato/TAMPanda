"""Honest feasibility for the confined shelf: a precomputed grasp library with exact conflict sets (2026-10-03).

Why. The original FAST check is a geometric occupancy rule (front-occlusion in the target's column). With the real
arm collision meshes (``base_panda_v2.xml``; tampanda's scenes only ever collided the hand / link7 capsules and the
finger pads) the arm sweeps far more than that column: bottles are 6 cm wide on a 7 cm pitch, so the open fingers
never fit beside a same-row neighbour, and reaching a deeper row sweeps the forearm across the neighbouring columns'
front rows (the robot stands at the shelf's front-left corner). Replaying the paper's plans with honest motions,
every infeasible action was of that kind, while all 34 cells are reachable on an empty shelf -- the geometry is
fine, the checker was too permissive.

How. Built like Access-19 v2's library (``access19_v2/library.py``) on THIS scene: per cell, complete pick/put motions
(home transit -> horizontal approach -> grasp -> 1.5 cm / 4 mm lift and retreat -> home transit), every IK branch
(frantik), dense exact checks at 2 mm clearance, VAMP transits re-validated on the exact meshes. Each candidate
records which cells' bottles it would touch (its conflict set, measured with every cell occupied). Bottles are
identical and contacts pairwise, so

    pick/put on cell c is feasible in occupancy S  <=>  some candidate of c has  conflicts & (S - {c}) == {}

exactly in that collision model (conservative only w.r.t. motions that weave around particular bottles). Microseconds
per query, no MuJoCo. Build / validation tooling: ``a19v2/honest_library.py`` (see README_honest.md).

    lib = HonestLibrary()                                   # DEFAULT_LIBRARY
    lib.feasible("shelf_interior__4_3", occupied_cells)     # occupied_cells: cells of the OTHER bottles (own ignored)
    plan = solve(lib, init, goal)                           # macro-move oracle, {cyl: cell} -> [(pick|put, cyl, cell)]
"""
from __future__ import annotations

import heapq
import itertools
import json
import time
from pathlib import Path

DEFAULT_LIBRARY = Path(__file__).resolve().parent / "honest_data" / "library_v1.json"


class HonestLibrary:
    def __init__(self, path=DEFAULT_LIBRARY):
        meta = json.load(open(path))
        self.path = str(path); self.meta = meta
        self.cells = list(meta["bits"]); self.bit = {c: i for i, c in enumerate(self.cells)}
        self.masks = {c: [int(r["mask"], 16) for r in rows] for c, rows in meta["cells"].items()}

    def occ(self, cells) -> int:
        m = 0
        for c in cells:
            m |= 1 << self.bit[c]
        return m

    def feasible_mask(self, cell, S: int) -> bool:
        """``S``: occupancy bitmask of the bottles around (the cell's own bit is ignored)."""
        if cell not in self.bit:
            return False
        S &= ~(1 << self.bit[cell])
        return any((m & S) == 0 for m in self.masks.get(cell, ()))

    def feasible(self, cell, occupied) -> bool:
        return cell in self.bit and self.feasible_mask(cell, self.occ(c for c in occupied if c != cell))

    def check_action(self, layout: dict, held, action) -> bool:
        """True iff FEASIBLE (tampanda convention). ``layout``: {cyl: cell} of the bottles standing in cells."""
        name, cyl, cell = action[0], action[1], action[2]
        if name not in ("pick", "put"):
            raise ValueError("confined shelf has pick/put only, got %r" % (name,))
        return self.feasible(cell, layout.values())


def solve(lib: HonestLibrary, init: dict, goal: dict, time_limit=60.0, max_nodes=2_000_000):
    """GBFS over macro moves (pick a bottle, put it on a free cell), h = misplaced bottles, ties by depth.
    Returns [(pick|put, cyl, cell), ...] or None."""
    objs = sorted(init); bit = lib.bit; cells = lib.cells
    gl = tuple(goal.get(o) for o in objs); s0 = tuple(init[o] for o in objs)
    h = lambda s: sum(1 for a, b in zip(s, gl) if b is not None and a != b)
    if h(s0) == 0:
        return []
    cnt = itertools.count(); openl = [(h(s0), 0, next(cnt), s0)]; parent = {s0: None}; t0 = time.time(); n = 0
    while openl:
        _, g, _, s = heapq.heappop(openl); n += 1
        if n > max_nodes or time.time() - t0 > time_limit:
            return None
        S = 0
        for c in s:
            S |= 1 << bit[c]
        free = [c for c in cells if not (S >> bit[c]) & 1]
        for k, c in enumerate(s):
            Sk = S & ~(1 << bit[c])
            if not lib.feasible_mask(c, Sk):
                continue
            for d in ([gl[k]] if gl[k] in free else []) + [d for d in free if d != gl[k]]:
                if not lib.feasible_mask(d, Sk):
                    continue
                t = s[:k] + (d,) + s[k + 1:]
                if t in parent:
                    continue
                parent[t] = (s, objs[k], c, d)
                if h(t) == 0:
                    plan = []
                    while parent[t] is not None:
                        p, o, a, b = parent[t]; plan[:0] = [("pick", o, a), ("put", o, b)]; t = p
                    return plan
                heapq.heappush(openl, (h(t), g + 1, next(cnt), t))
    return None


def verify(lib: HonestLibrary, init: dict, goal: dict, plan) -> bool:
    lay = dict(init); held = None
    for a, o, c in plan:
        if a == "pick":
            if held is not None or lay.get(o) != c or not lib.feasible(c, lay.values()):
                return False
            del lay[o]; held = o
        else:
            if held != o or c in lay.values() or not lib.feasible(c, lay.values()):
                return False
            lay[o] = c; held = None
    return held is None and all(lay.get(o) == c for o, c in goal.items())
