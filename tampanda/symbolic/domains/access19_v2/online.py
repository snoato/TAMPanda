"""Access-19 v2 feasibility by planning in the loop (the alternative to the precomputed library).

For every query the full motion stack runs in the ACTUAL scene, with exactly the objects that are present:

1. the cell's grasp-path IK layers (obstacle independent; computed once per cell and cached);
2. the bottleneck DP over all IK branches against the present objects, for the open path (target in its cell)
   and the held path (target in the hand); every DP-optimal path is re-checked densely (``grasp.FINE``);
3. matched (open, held) pairs in order of the geometry's reference posture, at most ``max_pairs`` mutually
   different arm configurations; for each, VAMP transits home <-> the paths' outer ends, validated against the
   exact meshes.

The action is feasible as soon as one pair has both transits. Sound in the same exact model as the library (every
accepted motion is collision-free), and on the validation set it agrees with the library on 700/700 states. It
costs seconds per query instead of microseconds, and needs MuJoCo + frantik + vamp in the calling process.

    ck = OnlineChecker()                       # T17c, reference posture from the shipped library
    ck.check_action(layout, held, ("pick", "ooi", "shelf_interior__2_6"))
    ck.last                                     # {"feasible", "s", "pairs_tried", "motion": {...} or None}
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from tampanda.symbolic.domains.access19_v2 import grasp as G
from tampanda.symbolic.domains.access19_v2 import kinematics as K
from tampanda.symbolic.domains.access19_v2 import motion as M
from tampanda.symbolic.domains.access19_v2.checker import DEFAULT_LIBRARY
from tampanda.symbolic.domains.access19_v2.layout import T17C, ShelfGeometry, parse_cell_id


def reference_posture(g: ShelfGeometry, library=DEFAULT_LIBRARY):
    """The geometry's reference posture from a library built for it (falls back to home)."""
    p = Path(library)
    if p.with_suffix(".json").exists():
        meta = json.load(open(p.with_suffix(".json")))
        if meta.get("digest") == g.digest():
            return np.asarray(meta["ref"])
    return M.HOME.copy()


class OnlineChecker:
    def __init__(self, g: ShelfGeometry = T17C, ref=None, max_pairs: int = 6, cache: bool = True):
        self.g = g; self.ref = reference_posture(g) if ref is None else np.asarray(ref)
        self.max_pairs = max_pairs; self.sc = None; self.q7s = None
        self._cells = {}; self._cache = {} if cache else None; self.last = None

    def init(self):
        if self.sc is None:
            from tampanda.symbolic.domains.access19_v2.shelf import Scene
            self.sc = Scene(self.g); self.q7s = K.q7_grid(self.g.q7_step_deg)

    def _cellpaths(self, key):
        if key not in self._cells:
            self._cells[key] = G.CellPaths(self.sc, key, self.q7s)
        return self._cells[key]

    def plan(self, key, others):
        """Search a motion for cell ``key`` with the cells ``others`` occupied. Returns (motion or None, pairs tried)."""
        self.init(); sc, g = self.sc, self.g; cp = self._cellpaths(key)
        paths = {"open": [], "held": []}
        for (kind, route), (Ls, hl0) in cp.layers.items():
            finger, hl = (g.fo, None) if kind == "open" else (g.r, hl0)
            sc.occupy(others | ({key} if kind == "open" else set()))
            for Q, _ in G.dp_paths(sc, Ls, finger, hl, g):
                if M.mj_check(sc, Q, finger, hl, G.MARGIN)[0]:
                    paths[kind].append({"Q": Q, "route": route, "held_local": hl0})
        pairs = sorted((G.posture_cost(paths["open"][a]["Q"][0], paths["held"][b]["Q"][0], self.ref), a, b)
                       for a, b in G.match_pairs(paths["open"], paths["held"]))
        direction = [0, -1, 0] if key[0] == "interior" else [0, 0, 1]
        tried = []
        for _, a, b in pairs:
            o, h = paths["open"][a], paths["held"][b]
            if any(np.abs(o["Q"][0] - t).max() < 0.3 for t in tried):
                continue
            tried.append(o["Q"][0])
            t_open = M.plan_validated(sc, o["Q"][0], M.HOME, others | {key}, None, g.fo, direction, self.q7s, G.MARGIN)
            t_held = None if t_open is None else M.plan_validated(
                sc, h["Q"][0], M.HOME, others, h["held_local"], g.r, direction, self.q7s, G.MARGIN, reverse=True)
            if t_held is not None:
                return {"open": o["Q"], "held": h["Q"], "t_open": t_open, "t_held": t_held,
                        "held_local": h["held_local"], "routes": o["route"] + h["route"]}, len(tried)
            if len(tried) >= self.max_pairs:
                break
        return None, len(tried)

    def feasible(self, cell: str, occupied_others) -> bool:
        key = parse_cell_id(cell); others = frozenset(parse_cell_id(c) for c in occupied_others if c != cell)
        ck = (others, key)
        if self._cache is not None and ck in self._cache:
            self.last = self._cache[ck]
            return self.last["feasible"]
        t0 = time.time(); motion, n = self.plan(key, set(others))
        self.last = {"feasible": motion is not None, "s": time.time() - t0, "pairs_tried": n, "motion": motion}
        if self._cache is not None:
            self._cache[ck] = self.last
        return self.last["feasible"]

    def check_action(self, layout: dict, held, action) -> bool:
        """True iff the action is physically FEASIBLE (tampanda convention), same contract as Access19V2Checker."""
        name, obj, cell = action[0], action[1], action[2]
        if name not in ("pick", "put"):
            raise ValueError("Access-19 v2 has pick/put only, got %r" % (name,))
        return self.feasible(cell, [c for o, c in layout.items() if o != obj])
