"""Multi-level shelf (shelf-access) v2: parametrised geometry + honest collision scene (2026-10-03).

Geometry of the current environment (tampanda tabletop_access ``make_access_builder``, reproduced exactly by the
defaults): a free-standing shelf, two decks (surfaces at ``levels``) on four corner legs, a vertical separator under
the lower deck splitting the floor compartment; YCB-proxy boxes 6 x 6 x 9 cm (FIXED -- modelled after YCB objects).
Cells are explicit grids per region: (first column centre x, n columns, first row centre y, n rows) at pitch
(px, py); rows run away from the robot (+y), the shelf opens towards -y.

The scene: Franka from ``base_panda_v2.xml`` (real link collision meshes), the static shelf as boxes, one free box per
cell (+1 spare), wrapped in ``honest_check.EnvScene`` -- so the Access-19 v2 motion search (honest_check.check_motion),
the library builder and the oracle tooling apply unchanged.

Packaged unchanged for tampanda 1.1.0 from the research tooling (msv2/msv2_scene.py) that built library_M3. The shipped
geometry is ``tampanda/symbolic/domains/shelf_access_v2/data/M3.kv``, applied over the defaults by ``from_kv``.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, replace
from types import SimpleNamespace

import numpy as np

BOX = (0.03, 0.03, 0.045)          # YCB-proxy half extents: fixed


@dataclass(frozen=True)
class MSGeometry:
    sx: float = 0.25                # deck centre (world x, y); robot base at the origin
    sy: float = 0.70
    dx: float = 0.60                # deck size
    dy: float = 0.45
    t: float = 0.012                # deck thickness
    levels: tuple = (0.30, 0.55)    # deck top surfaces (lower = "middle_deck", upper = "top_deck")
    leg: float = 0.018
    inset: float = 0.020
    sep_t: float = 0.010            # floor separator thickness (0 = none); at x = sx
    px: float = 0.06                # cell pitch
    py: float = 0.06
    # regions: name -> (x0, ncol, y0, nrow, surface z); ncol 0 drops the region
    mid: tuple = (0.098, 6, 0.583, 5)
    top: tuple = (0.098, 6, 0.583, 5)
    floor_l: tuple = (0.098, 1, 0.583, 5)
    floor_r: tuple = (0.36, 1, 0.583, 5)
    excl: tuple = ("top_deck__4_4", "top_deck__5_4")       # baseline: cells the env never offers
    gz: float = 0.0                 # FRONT grasp height above the box centre
    hp: float = 0.0                 # robot pedestal: the robot base stands hp above the floor the shelf stands on
    ped: float = 0.30               # pedestal footprint (square, centred under the robot)
    bz: float = 0.05                # box centre above its surface (v1 places boxes at +5 cm: they float 5 mm;
                                    # resting would be BOX[2] = 0.045)

    def surface(self, region):
        return {"floor_left": 0.0, "floor_right": 0.0, "middle_deck": self.levels[0], "top_deck": self.levels[1]}[region] - self.hp

    def regions(self):
        return {"floor_left": self.floor_l, "floor_right": self.floor_r, "middle_deck": self.mid, "top_deck": self.top}

    def cells(self):
        """{cell_id: box centre}; cell ids as in the PDDL (region__i_j)."""
        out = {}
        for reg, (x0, nc, y0, nr) in self.regions().items():
            for i in range(nc):
                for j in range(nr):
                    cid = "%s__%d_%d" % (reg, i, j)
                    if cid not in self.excl:
                        out[cid] = np.array([x0 + i * self.px, y0 + j * self.py, self.surface(reg) + self.bz])
        return out

    def front_y(self):
        return self.sy - self.dy / 2

    def digest(self):
        return hashlib.sha1(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:12]


V1 = MSGeometry()


def static_boxes(g: MSGeometry):
    """(name, lo, hi) of the shelf."""
    out = []; hp = g.hp
    x0, x1 = g.sx - g.dx / 2, g.sx + g.dx / 2; y0, y1 = g.sy - g.dy / 2, g.sy + g.dy / 2
    for k, z in enumerate(g.levels):
        out.append(("shelf_deck_%d" % k, [x0, y0, z - g.t - hp], [x1, y1, z - hp]))
    for k, (sxn, syn) in enumerate([(1, 1), (-1, 1), (1, -1), (-1, -1)]):
        cx = g.sx + sxn * (g.dx / 2 - g.inset - g.leg / 2); cy = g.sy + syn * (g.dy / 2 - g.inset - g.leg / 2)
        out.append(("shelf_leg_%d" % k, [cx - g.leg / 2, cy - g.leg / 2, -hp], [cx + g.leg / 2, cy + g.leg / 2, g.levels[-1] - hp]))
    if g.sep_t > 0:
        out.append(("shelf_floor_separator", [g.sx - g.sep_t / 2, y0, -hp], [g.sx + g.sep_t / 2, y1, g.levels[0] - g.t - hp]))
    if hp > 0:
        out.append(("pedestal", [-g.ped / 2, -g.ped / 2, -hp], [g.ped / 2, g.ped / 2, 0.0]))
    return out


def build(g: MSGeometry = V1, look=None):
    """(EnvScene, cells {id: centre}) for geometry g."""
    import mujoco
    import honest_check as H
    from tampanda.symbolic.domains.access19_v2.shelf import ROBOT_XML
    spec = mujoco.MjSpec.from_file(str(ROBOT_XML)); wb = spec.worldbody
    fl = wb.add_geom(); fl.name = "floor"; fl.type = mujoco.mjtGeom.mjGEOM_PLANE; fl.size = [3, 3, 0.1]
    fl.pos = [0, 0, -g.hp]
    fl.contype = 1; fl.conaffinity = 0; fl.rgba = [0.55, 0.45, 0.35, 1]
    for name, lo, hi in static_boxes(g):
        lo, hi = np.asarray(lo, float), np.asarray(hi, float)
        b = wb.add_geom(); b.name = name; b.type = mujoco.mjtGeom.mjGEOM_BOX
        b.size = (hi - lo) / 2; b.pos = (hi + lo) / 2; b.contype = 1; b.conaffinity = 0; b.rgba = [0.92, 0.92, 0.92, 1]
    if look is not None:
        from tampanda.symbolic.domains.access19_v2 import look as L
        L.apply_spec(spec, look, floor_geom=fl)
    cells = g.cells(); objs = []
    for n in range(len(cells) + 1):
        bd = wb.add_body(); bd.name = "box_%d" % n; bd.pos = [100.0 + 3 * n, 100.0, 50.0]
        bd.add_freejoint()
        gm = bd.add_geom(); gm.name = "box_%d_geom" % n; gm.type = mujoco.mjtGeom.mjGEOM_BOX; gm.size = list(BOX)
        gm.contype = 1; gm.conaffinity = 1; gm.rgba = [0.75, 0.15, 0.15, 1]
        objs.append(bd.name)
    m = spec.compile(); d = mujoco.MjData(m)
    if look is not None:
        L.apply_model(m, look)
    sc = H.EnvScene(SimpleNamespace(model=m, data=d), objs, BOX, g.front_y(), "box")
    sc.floor_z = -g.hp
    return sc, cells


def from_kv(text):
    """k=v lines (tuples as comma lists) -> MSGeometry over V1."""
    kw = {}
    for line in text.splitlines():
        line = line.split("#")[0].strip()
        if not line:
            continue
        k, v = [x.strip() for x in line.split("=", 1)]
        cur = getattr(V1, k)
        if isinstance(cur, tuple):
            parts = [p.strip() for p in v.split(",") if p.strip()]
            kw[k] = tuple(p if k == "excl" else (int(p) if p.lstrip("-").isdigit() else float(p)) for p in parts)
        else:
            kw[k] = type(cur)(v)
    return replace(V1, **kw)
