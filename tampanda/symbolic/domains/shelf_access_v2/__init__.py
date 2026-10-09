"""Multi-level shelf (shelf-access) v2: physically honest geometry + exact feasibility library (2026-10-03).

v1 (tabletop_access ``make_access_builder``) checked only the hand/link7 capsules and finger pads; with the real arm
meshes its verified plans put the forearm 40-65 mm through the decks, its boxes sat on a 6 cm pitch (6 cm boxes:
touching, so a held box could never be lifted out with clearance) and floated 5 mm above the decks. v2 keeps the
task, the YCB-proxy boxes (6 x 6 x 9 cm) and the floor separator, and changes only the geometry (geometry "M3"):

* shelf 80 x 45 cm at (0.40, 0.58) from the robot base (off-centre: a centred Panda cannot point its hand straight
  ahead at deck height -- joint 6 would need < 0 rad), decks at 0.34 / 0.58 m, separator under the lower deck;
* 7 cm pitch both ways, boxes resting on the surfaces, FRONT grasp 2.5 cm above the box centre (keeps the wrist
  off the deck);
* the v1 grid topology: middle deck 6 x 5, top deck 6 x 5 minus 4_4 / 5_4 (as v1), floor one column per half, 5 rows
  left / 4 right (floor_right__0_4 is out of reach) -- 67 cells, every one graspable on the empty shelf.

Feasibility = precomputed grasp library (Access-19 v2 construction on this scene: complete pick/put motions over all
IK branches, dense exact checks at 2 mm, validated transits; per candidate the set of cells whose box it would
touch). Boxes are identical, contacts pairwise:  action on cell c feasible in S  <=>  some candidate of c has
conflicts & (S - {c}) == {}.  Pure Python, microseconds per query. Build / validation tooling:
``scripts/libraries/shelf_access_v2/`` (see the README there).

    lib = HonestLibrary()            # DEFAULT_LIBRARY
    lib.feasible("middle_deck__2_3", occupied_cells)
"""
from __future__ import annotations

import json
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data"
DEFAULT_LIBRARY = DATA / "library_M3.json"
GEOMETRY = DATA / "M3.kv"


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
        if cell not in self.bit:
            return False
        S &= ~(1 << self.bit[cell])
        return any((m & S) == 0 for m in self.masks.get(cell, ()))

    def feasible(self, cell, occupied) -> bool:
        return cell in self.bit and self.feasible_mask(cell, self.occ(c for c in occupied if c != cell))

    def check_action(self, layout: dict, held, action) -> bool:
        """True iff FEASIBLE (tampanda convention). ``layout``: {obj: cell} of the boxes standing in cells."""
        name, obj, cell = action[0], action[1], action[2]
        if name not in ("pick", "put"):
            raise ValueError("shelf-access has pick/put only, got %r" % (name,))
        return self.feasible(cell, [c for o, c in layout.items() if o != obj])
