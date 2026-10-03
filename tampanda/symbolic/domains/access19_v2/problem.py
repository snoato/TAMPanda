"""Access-19 v2 PDDL: domain files and a problem writer for the 5 x 7 x 2 grid (integration with rgnet).

The domain is v1's ``tabletop-access`` (``pick`` / ``put``, filter mode: the symbolic domain deliberately
over-approximates, the feasibility checker prunes) under the new name ``tabletop-access-v2`` so that rgnet dispatches
v2 problems to the v2 checker while every v1 run keeps working. Two variants ship in ``pddl/``:

* ``domain.pddl``          -- plain STRIPS + typing (for classical planners / unified-planning);
* ``domain_derived.pddl``  -- adds v1's derived ``(blocking ?obj ?goal)`` over static ``(same-column-front ?cf ?cb)``
                              facts, the structural hint v1's GNN datasets used.

Changes vs v1 problems: no excluded cells (all 35 cells per level exist); ``same-column-front`` facts are emitted
for BOTH levels (v2's top deck uses the same frontal grasp as the interior, so a box in front blocks there too;
v1's deck used a top-down approach and had none). Note that the physical blocking band is wider than one column
(see README, "What blocks what"); the hint stays v1's single-column relation unless ``blocking_band`` widens it.
"""
from __future__ import annotations

from pathlib import Path

from tampanda.symbolic.domains.access19_v2.layout import LEVELS, ShelfGeometry, T17C, cell_id

DOMAIN_NAME = "tabletop-access-v2"
PDDL_DIR = Path(__file__).resolve().parent / "pddl"
DEFAULT_OBJECTS = tuple("blocker_%d" % i for i in range(18)) + ("ooi",)


def cells(g: ShelfGeometry = T17C):
    return [cell_id((lv, i, j)) for lv in LEVELS for i in range(g.ncol) for j in range(g.nrow)]


def static_facts(g: ShelfGeometry = T17C, blocking_band: int = 0, cells_subset=None):
    """Adjacency (north = one row deeper, east = one column to +x) and ``same-column-front`` (front cell, back cell)
    for every pair within ``blocking_band`` columns of each other, on both levels; restricted to ``cells_subset``
    (v1's reduced-depth problems expose only some interior rows) when given."""
    keep = None if cells_subset is None else set(cells_subset)
    return [f for f in _static_facts(g, blocking_band) if keep is None or all(c in keep for c in f[1:-1].split()[-2:])]


def _static_facts(g, blocking_band):
    out = []
    for lv in LEVELS:
        for i in range(g.ncol):
            for j in range(g.nrow):
                c = cell_id((lv, i, j))
                if j + 1 < g.nrow:
                    out.append("(adjacent north %s %s)" % (c, cell_id((lv, i, j + 1))))
                if i + 1 < g.ncol:
                    out.append("(adjacent east %s %s)" % (c, cell_id((lv, i + 1, j))))
        for i in range(g.ncol):
            for jb in range(g.nrow):
                for jf in range(jb):
                    for i2 in range(max(0, i - blocking_band), min(g.ncol, i + blocking_band + 1)):
                        out.append("(same-column-front %s %s)" % (cell_id((lv, i2, jf)), cell_id((lv, i, jb))))
    return out


def problem_text(name: str, init: dict, goal: dict, g: ShelfGeometry = T17C, objects=None, held=None,
                 blocking_band: int = 0, comment: str = "", cells_subset=None) -> str:
    """``init`` / ``goal``: {object: cell_id}; ``held``: the object in the hand at the start (or None).
    Goal = every listed object at its cell (v1 convention: ``occupied`` atoms only)."""
    objects = list(objects or sorted(set(init) | set(goal) | ({held} if held else set())))
    allc = cells(g) if cells_subset is None else [c for c in cells(g) if c in set(cells_subset)]
    occ = set(init.values())
    if len(occ) != len(init) or not occ <= set(allc) or not set(goal.values()) <= set(allc):
        raise ValueError("init/goal must place objects on distinct v2 cells")
    lines = [";; %s" % comment] if comment else []
    lines += ["(define (problem %s)" % name, "  (:domain %s)" % DOMAIN_NAME, "  (:objects",
              "    %s - movable" % " ".join(objects), "    %s - cell" % " ".join(allc), "  )", "  (:init"]
    lines += ["    %s" % f for f in static_facts(g, blocking_band, cells_subset)]
    lines += ["    (occupied %s %s)" % (c, o) for o, c in sorted(init.items())]
    lines += ["    (empty %s)" % c for c in allc if c not in occ]
    lines += ["    (holding %s)" % held] if held else ["    (gripper-empty)"]
    lines += ["  )", "  (:goal (and"]
    lines += ["    (occupied %s %s)" % (c, o) for o, c in sorted(goal.items())]
    lines += ["  ))", ")"]
    return "\n".join(lines) + "\n"
