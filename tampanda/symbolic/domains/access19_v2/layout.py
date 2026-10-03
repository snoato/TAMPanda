"""Access-19 v2 layout: shelf geometry parameters and cell naming (no MuJoCo; safe to import anywhere).

Layout (robot base frame, metres): a bookcase centred in front of the robot. Its open front face is at ``y = c``;
the compartment ("interior") floor is at ``z = f``, its roof at ``f + H``, the top deck surface at
``s = f + H + roof``. Both levels carry the same ``ncol x nrow`` grid of identical boxes (``2r x 2r x h``): column
``i`` runs along x (``x0 + i*px``), row ``j`` away from the robot (row 0 = front). A cell is a ``(level, i, j)``
key with ``level in {"interior", "top"}``; its PDDL name is ``shelf_<level>__<i>_<j>``.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, fields

import numpy as np

LEVELS = ("interior", "top")
# keys of the research prototype's *.geo files that are fixed here: boxes (box=1), no excluded cells (no_exclude=1),
# one shared x grid for both levels (x0t = x0, pxt = px), no mirrored interior, no designed-blocker corridors
_RESEARCH_ONLY_KEYS = {"box", "no_exclude", "x0t", "pxt", "mirror_int", "corridor", "top_corridor", "legs"}


@dataclass(frozen=True)
class ShelfGeometry:
    # placement / grid
    c: float = 0.375             # shelf front edge (y)
    x0: float = -0.08            # x of column 0 centre (interior and top)
    ncol: int = 5
    nrow: int = 7
    px: float = 0.04             # lateral pitch
    pyi: float = 0.036           # interior depth pitch
    pyt: float = 0.036           # top depth pitch
    gap0: float = 0.012          # front edge -> front face of a row-0 interior object
    gap0t: float = 0.012         # same for the top deck
    # shelf body
    f: float = 0.34              # compartment floor height
    H: float = 0.24              # cavity height
    roof: float = 0.012
    wall: float = 0.012
    side: float = 0.115          # outer column centre -> inner side-panel face
    back: float = 0.02           # back-row object back face -> back wall
    panels: int = 1              # 1: bookcase with full-height side panels, open below the compartment floor
    # objects and gripper
    r: float = 0.015             # half side of the square boxes
    h: float = 0.10              # box height
    fo: float = 0.03             # per-finger opening on approach
    grasp_depth: float = 0.007   # the ONE grasp: 0.7 cm below the box top, hand horizontal, palm facing +y
    flip: int = 0                # hand rotation about the approach axis: 0 = FRONT_QUAT, 1 = rotated by 180 deg
    # path discretisation and acceptance
    selfcol: int = 1             # arm self-collision counts (non-adjacent links)
    q7_step_deg: float = 3.0     # joint-7 grid of the analytic IK
    step: float = 0.015          # Cartesian spacing of grasp-path layers (m)
    travel: float = 0.01         # top deck: descent starts this far above the boxes in front
    margin: float = 0.003        # clearance required at grasp-path layers (DP)
    max_dq: float = 0.25         # max joint jump between consecutive layers (rad)

    @property
    def s(self) -> float:
        return self.f + self.H + self.roof

    def inner_width(self) -> float:
        return (self.ncol - 1) * self.px + 2 * self.side

    def digest(self) -> str:
        return hashlib.sha1(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:12]

    @classmethod
    def from_kv(cls, text: str, base: "ShelfGeometry | None" = None) -> "ShelfGeometry":
        """Parse ``k=v`` tokens (later tokens win), e.g. the research ``*.geo`` files; unknown keys raise."""
        known = {f.name: f.type for f in fields(cls)}
        kv = asdict(base or cls())
        for tok in text.split():
            k, v = tok.split("=", 1)
            if k == "depths":                       # research files: first grasp depth is the one grasp
                k, v = "grasp_depth", v.split(",")[0]
            if k in _RESEARCH_ONLY_KEYS:            # research-prototype switches with a fixed meaning here
                continue
            if k not in known:
                raise KeyError("unknown ShelfGeometry key %r" % k)
            kv[k] = type(kv[k])(float(v)) if isinstance(kv[k], (int, float)) else v
        return cls(**kv)


# T17c (2026-10-02): the centred bookcase. Inner width 39 cm, cavity 24 cm, 5 x 7 cells on both levels,
# 3 x 3 x 10 cm boxes; every cell pick+put-able with the one grasp, same-row neighbours block each other,
# full lower deck unloads onto the top deck (two passes, every frame collision-free).
T17C = ShelfGeometry()


# ---------------------------------------------------------------- cells
def cell_keys(g: ShelfGeometry):
    return [(lv, i, j) for lv in LEVELS for i in range(g.ncol) for j in range(g.nrow)]


def cell_id(key) -> str:
    lv, i, j = key
    return "shelf_%s__%d_%d" % (lv, i, j)


def parse_cell_id(name: str):
    region, ij = name.split("__")
    lv = region[len("shelf_"):]
    if lv not in LEVELS:
        raise ValueError("not an Access-19 v2 cell: %r" % name)
    i, j = map(int, ij.split("_"))
    return (lv, i, j)


def centre(g: ShelfGeometry, key) -> np.ndarray:
    """Bottom-centre of the object in cell ``key``."""
    lv, i, j = key
    if lv == "interior":
        return np.array([g.x0 + i * g.px, g.c + g.gap0 + g.r + j * g.pyi, g.f])
    return np.array([g.x0 + i * g.px, g.c + g.gap0t + g.r + j * g.pyt, g.s])
