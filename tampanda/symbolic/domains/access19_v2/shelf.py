"""Access-19 v2 collision scene for a ``layout.ShelfGeometry``.

The scene compiles the Franka from ``base_panda_v2.xml`` (real link collision meshes, all compiled) and enables
arm-vs-scene contacts for every moving arm link plus the hand and fingers, optional arm self-collision, and a
geometric (vertex-in-box) check of the bolted base ``link0`` against the shelf, which MuJoCo's parent-child filter
would otherwise hide. Objects are mocap bodies, one per cell, parked far away when the cell is empty, so a scene is
compiled once and re-used for every occupancy.
"""
from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from tampanda.kinematics import panda as P
from tampanda.symbolic.domains.access19_v2 import kinematics as K
from tampanda.symbolic.domains.access19_v2.layout import (LEVELS, T17C, ShelfGeometry, cell_id, cell_keys,  # noqa: F401
                                                         centre, parse_cell_id)

ROBOT_XML = Path(__file__).resolve().parents[3] / "environments" / "assets" / "franka_emika_panda" / "base_panda_v2.xml"
ARM_BODIES = ("link0", "link1", "link2", "link3", "link4", "link5", "link6", "link7", "hand",
              "left_finger", "right_finger")
DETECT = 0.03                       # contact detection range on arm geoms: distances below this are reported
FAR = np.array([30.0, 30.0, -30.0])  # parking position of absent objects


# ---------------------------------------------------------------- scene
class Scene:
    """Compiled collision scene for one geometry. Not thread-safe (one per process)."""

    def __init__(self, g: ShelfGeometry, look: str | None = None):
        """``look``: optional render style (``look.STYLES``); visual only, default = the plain compiled scene."""
        self.g = g; self.look = look
        spec = mujoco.MjSpec.from_file(str(ROBOT_XML))
        wb = spec.worldbody
        fl = wb.add_geom(); fl.name = "floor"; fl.type = mujoco.mjtGeom.mjGEOM_PLANE; fl.size = [3, 3, 0.1]
        fl.contype = 0; fl.conaffinity = 0; fl.rgba = [0.55, 0.45, 0.35, 1]
        xl = g.x0 - g.side - g.wall; xr = g.x0 + (g.ncol - 1) * g.px + g.side + g.wall
        di = g.gap0 + 2 * g.r + (g.nrow - 1) * g.pyi + g.back
        dt = g.gap0t + 2 * g.r + (g.nrow - 1) * g.pyt + g.back
        yb = g.c + max(di + g.wall, dt)
        self.boxes = []

        def box(name, lo, hi, rgba=(0.92, 0.92, 0.92, 1)):
            lo, hi = np.asarray(lo, float), np.asarray(hi, float)
            b = wb.add_geom(); b.name = name; b.type = mujoco.mjtGeom.mjGEOM_BOX
            b.size = (hi - lo) / 2; b.pos = (hi + lo) / 2; b.contype = 1; b.conaffinity = 0; b.rgba = rgba
            self.boxes.append((name, lo, hi))

        if g.panels:
            box("base", [xl, g.c, g.f - 0.02], [xr, yb, g.f])            # compartment floor board
            box("panel_l", [xl, g.c, 0.0], [xl + g.wall, yb, g.f])       # side panels below the compartment
            box("panel_r", [xr - g.wall, g.c, 0.0], [xr, yb, g.f])
        else:
            box("base", [xl, g.c, -0.6], [xr, yb, g.f])                  # solid base
        box("roof", [xl, g.c, g.f + g.H], [xr, yb, g.s])
        box("wall_l", [xl, g.c, g.f], [xl + g.wall, yb, g.f + g.H])
        box("wall_r", [xr - g.wall, g.c, g.f], [xr, yb, g.f + g.H])
        box("back", [xl, g.c + di, g.f], [xr, yb, g.f + g.H])
        self.slots = {}
        for key in cell_keys(g):
            b = wb.add_body(); b.name = "o_%s_%d_%d" % (key[0][0], key[1], key[2]); b.mocap = True; b.pos = FAR
            gm = b.add_geom(); gm.name = b.name; gm.type = mujoco.mjtGeom.mjGEOM_BOX; gm.size = [g.r, g.r, g.h / 2]
            gm.contype = 1; gm.conaffinity = 0; gm.rgba = [0.75, 0.15, 0.15, 1]
            self.slots[key] = b.name
        hb = wb.add_body(); hb.name = "held"; hb.mocap = True; hb.pos = FAR
        hg = hb.add_geom(); hg.name = "held"; hg.type = mujoco.mjtGeom.mjGEOM_BOX; hg.size = [g.r, g.r, g.h / 2]
        hg.contype = 0; hg.conaffinity = 1; hg.rgba = [0.15, 0.3, 0.85, 1]
        if look is not None:
            from tampanda.symbolic.domains.access19_v2 import look as L
            L.apply_spec(spec, look, floor_geom=fl)
        self.m = spec.compile(); self.d = mujoco.MjData(self.m)
        m = self.m
        if look is not None:
            L.apply_model(m, look)
        _enable_arm_collisions(m)
        arm_bodies = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in ARM_BODIES if b != "link0"}
        self.arm = [k for k in range(m.ngeom) if m.geom_bodyid[k] in arm_bodies and m.geom_conaffinity[k]]
        self.held_geom = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "held")
        self.armset = set(self.arm) | {self.held_geom}
        m.geom_margin[self.arm] = DETECT; m.geom_margin[self.held_geom] = DETECT
        if g.selfcol:
            m.geom_contype[self.arm] = 2; m.geom_conaffinity[self.arm] = 3     # arm-arm (bit 2) + arm-scene (bit 1)
            for b in {int(m.geom_bodyid[k]) for k in self.arm}:              # the broad phase filters on BODY flags
                m.body_contype[b] |= 2; m.body_conaffinity[b] |= 3
        self.slot_of_geom = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n): k for k, n in self.slots.items()}
        self.mid = {k: m.body_mocapid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)] for k, n in self.slots.items()}
        self.held_mid = m.body_mocapid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "held")]
        self.jq = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "joint%d" % i)] for i in range(1, 8)]
        self.fq = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]
                   for n in ("finger_joint1", "finger_joint2")]
        l0 = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "link0")
        mujoco.mj_kinematics(m, self.d)
        self.base = np.eye(4); self.base[:3, :3] = self.d.xmat[l0].reshape(3, 3); self.base[:3, 3] = self.d.xpos[l0]
        self.base_clear = self._base_clearance(l0)
        self.self_exclude = set()
        if g.selfcol:     # arm-arm pairs already within the detection range at home are structural neighbours
            self.d.qpos[:] = 0; self.d.qpos[self.jq] = P.Q_HOME
            mujoco.mj_kinematics(m, self.d); mujoco.mj_collision(m, self.d)
            armset = set(self.arm)
            for cc in self.d.contact[:self.d.ncon]:
                if cc.geom1 in armset and cc.geom2 in armset:
                    self.self_exclude.add((min(cc.geom1, cc.geom2), max(cc.geom1, cc.geom2)))
        self.occupied = frozenset()

    def _base_clearance(self, l0) -> float:
        """Signed distance of link0's collision-mesh vertices to the shelf boxes (< 0: the base is inside)."""
        m, d = self.m, self.d
        best = 1.0
        for k in range(m.ngeom):
            if m.geom_bodyid[k] != l0 or m.geom_type[k] != mujoco.mjtGeom.mjGEOM_MESH or m.geom_group[k] != 3:
                continue
            mid = m.geom_dataid[k]
            v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
            w = v @ d.geom_xmat[k].reshape(3, 3).T + d.geom_xpos[k]
            for _, lo, hi in self.boxes:
                od = np.linalg.norm(np.maximum(np.maximum(lo - w, 0), w - hi), axis=1)
                ins = np.minimum(w - lo, hi - w).min(1)
                best = min(best, float(np.where(od > 0, od, -ins).min()))
        return best

    # ---- state
    def occupy(self, occupied) -> None:
        occupied = frozenset(occupied)
        for k, mid in self.mid.items():
            self.d.mocap_pos[mid] = FAR
        for k in occupied:
            self.d.mocap_pos[self.mid[k]] = centre(self.g, k) + np.array([0, 0, self.g.h / 2])
        self.occupied = occupied

    def site_T(self, q) -> np.ndarray:
        return K.site_T(q, self.base)

    def _pose(self, q, finger, held_local):
        m, d = self.m, self.d
        d.qpos[self.jq] = q; d.qpos[self.fq] = finger
        d.mocap_pos[self.held_mid] = FAR
        mujoco.mj_kinematics(m, d)
        if held_local is not None:
            T = self.site_T(q)
            d.mocap_pos[self.held_mid] = T[:3, 3] + T[:3, :3] @ np.asarray(held_local, float)
            mujoco.mj_kinematics(m, d)
        mujoco.mj_collision(m, d)

    def contacts(self, q, finger, held_local=None):
        """(static, per_object): smallest distance of the arm (+ held object) to the shelf / to itself, and
        to each present object, within the detection range (absent = farther than DETECT)."""
        self._pose(q, finger, held_local)
        d = self.d; static = DETECT; per = {}
        for k in range(d.ncon):
            cc = d.contact[k]; g1, g2 = cc.geom1, cc.geom2
            a1, a2 = g1 in self.armset, g2 in self.armset
            if a1 != a2:
                other = g2 if a1 else g1
                slot = self.slot_of_geom.get(other)
                if slot is None:
                    static = min(static, cc.dist)
                elif cc.dist < per.get(slot, DETECT):
                    per[slot] = cc.dist
            elif a1 and a2 and self.g.selfcol and cc.dist < static:
                if (min(g1, g2), max(g1, g2)) not in self.self_exclude and self.held_geom not in (g1, g2):
                    static = cc.dist
        return static, per

    def clearance(self, q, finger, held_local=None) -> float:
        static, per = self.contacts(q, finger, held_local)
        return min([static] + list(per.values()))

    def name(self, gid) -> str:
        b = mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_BODY, self.m.geom_bodyid[gid])
        return mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_GEOM, gid) or b


def _enable_arm_collisions(m) -> None:
    """Every arm collision geom (group 3 mesh, or any primitive pad) collides with the scene, never with the arm;
    tampanda's oversized hand / link7 capsule stand-ins are disabled (the real meshes collide instead). Raises if a
    collision mesh has no compiled convex hull (its contacts would be silently lost)."""
    arm_ids = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in ARM_BODIES if b != "link0"}
    link0 = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "link0")
    for gid in range(m.ngeom):
        if m.geom_bodyid[gid] == link0:
            m.geom_contype[gid] = 0; m.geom_conaffinity[gid] = 0
            continue
        if m.geom_bodyid[gid] not in arm_ids:
            continue
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
        if name in ("hand_capsule", "link7_capsule"):
            m.geom_contype[gid] = 0; m.geom_conaffinity[gid] = 0
            continue
        if m.geom_group[gid] == 3 or m.geom_conaffinity[gid] or m.geom_contype[gid]:
            m.geom_contype[gid] = 0; m.geom_conaffinity[gid] = 1; m.geom_margin[gid] = 0.003
    missing = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MESH, m.geom_dataid[gid]) for gid in range(m.ngeom)
               if m.geom_bodyid[gid] in arm_ids and m.geom_type[gid] == mujoco.mjtGeom.mjGEOM_MESH
               and m.geom_conaffinity[gid] and m.mesh_graphadr[m.geom_dataid[gid]] < 0]
    if missing:
        raise RuntimeError("collision meshes without compiled hulls (contacts would be silently lost): %s" % missing)
