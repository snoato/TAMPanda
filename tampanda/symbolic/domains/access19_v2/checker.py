"""Access-19 v2 feasibility checker: a pure-Python lookup in the grasp library (no MuJoCo, IK or planner).

    lib = GraspLibrary.load("…/library_T17C.json")
    ck = Access19V2Checker(lib)
    ck.check_action(layout={"ooi": "shelf_interior__2_6", "blocker_0": "shelf_interior__2_0"}, held=None,
                    action=("pick", "ooi", "shelf_interior__2_6"))          # -> False (blocker_0 is in front)

Semantics (see ``library.py``): ``pick o c`` / ``put o c`` is feasible iff one of cell c's library motions --
home -> approach -> grasp -> retreat -> home with the one frontal grasp -- has no conflict with the objects in the
OTHER cells. The symbolic preconditions (o at c and the hand empty for pick; o held and c empty for put) are the
PDDL domain's job and are not re-checked here.
"""
from __future__ import annotations

import json
from pathlib import Path

DEFAULT_LIBRARY = Path(__file__).resolve().parent / "data" / "library_T17C.json"


class GraspLibrary:
    def __init__(self, meta: dict):
        if meta.get("format") != 1:
            raise ValueError("unsupported grasp-library format %r" % meta.get("format"))
        self.meta = meta
        self.bits = {cid: n for n, cid in enumerate(meta["bits"])}
        self.masks = {cid: [int(r["mask"], 16) for r in rows] for cid, rows in meta["cells"].items()}

    @classmethod
    def load(cls, path=DEFAULT_LIBRARY) -> "GraspLibrary":
        path = Path(path)
        if path.suffix != ".json":
            path = path.with_suffix(".json")
        with open(path) as fh:
            return cls(json.load(fh))

    @property
    def cells(self):
        return list(self.meta["bits"])

    def occupancy_mask(self, cells) -> int:
        m = 0
        for c in cells:
            m |= 1 << self.bits[c]
        return m

    def candidate(self, cell: str, occupied_others) -> int | None:
        """Index of the first (best-posture) library motion of ``cell`` that is clear in the given occupancy of
        the OTHER cells, or None if the action is infeasible there."""
        occ = self.occupancy_mask(c for c in occupied_others if c != cell)
        for n, m in enumerate(self.masks[cell]):
            if not m & occ:
                return n
        return None

    def feasible(self, cell: str, occupied_others) -> bool:
        return self.candidate(cell, occupied_others) is not None

    def blockers(self, cell: str, occupied_others) -> list[set]:
        """Per library motion of ``cell``: the occupied cells it conflicts with (diagnostics)."""
        occ = set(occupied_others) - {cell}
        names = self.meta["bits"]
        return [{names[b] for b in range(len(names)) if (m >> b) & 1} & occ for m in self.masks[cell]]


class Access19V2Checker:
    """``check_action`` returns True iff the action is physically FEASIBLE (tampanda convention)."""

    def __init__(self, library: GraspLibrary | None = None):
        self.lib = library or GraspLibrary.load()

    def check_action(self, layout: dict, held, action) -> bool:
        name, obj, cell = action[0], action[1], action[2]
        if name not in ("pick", "put"):
            raise ValueError("Access-19 v2 has pick/put only, got %r" % (name,))
        others = [c for o, c in layout.items() if o != obj]
        return self.lib.feasible(cell, others)
