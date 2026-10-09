"""Honest pick/put motion feasibility for the confined-shelf and shelf-access scenes, with the Access-19 v2 stack.

The environment's own scene (shelf, table, objects) is compiled with the real arm collision meshes
(``base_panda_v2.xml`` + ``access19_v2.shelf._enable_arm_collisions``, as in the clipping audit) and wrapped in a
scene adapter with the interface the v2 motion code uses (``clearance``, ``contacts``, ``site_T``, ``base``).
A grasp is one of the environment's own grasp families (tampanda ``GraspPlanner``: FRONT, TOP_DOWN_X/Y, each also
with the hand rolled 180 deg); per (cell, grasp) the motion is searched exactly like Access-19 v2:

1. Cartesian approach line into the grasp (FRONT: horizontal from 6 cm in front of the shelf; TOP_DOWN: vertical
   from 12 cm above the object), IK layers over ALL analytic branches (frantik, joint-7 grid);
2. bottleneck DP over the layers against the present objects, open path (object in its cell, hand open) and held
   path (object in the hand, lifted 4 mm); every DP path re-checked densely (``grasp.FINE``) at ``MARGIN``;
3. matched (open, held) grasp configurations, VAMP transits home <-> the paths' outer ends, re-validated densely
   against the exact meshes.

``feasible`` = some grasp has an open path, a held path and both transits. Every accepted motion is collision-free
in the exact model (arm meshes, hand, fingers, held object) with >= ``MARGIN`` clearance, except arm/scene pairs
that already penetrate at the home posture (the shoulder in its mounting table: a scene defect, reported).

Packaged unchanged for tampanda 1.1.0 from the research tooling (msv2/honest_check.py, which built library_M3). Its
a19v2/ twin, which built confined library_v1, lacks only ``GRASP_DZ`` and ``floor_z``; both default to that build's
values (0.0), so the two give the same result on the confined shelf.
"""
from __future__ import annotations

import os
import sys
import time
from types import SimpleNamespace

import mujoco
import numpy as np

from tampanda.kinematics import panda as P
from tampanda.planners.grasp_planner import _QUATS, GraspType, GRASP_CONTACT_OFFSET as GP_CONTACT
from tampanda.symbolic.domains.access19_v2 import grasp as G
from tampanda.symbolic.domains.access19_v2 import kinematics as K
from tampanda.symbolic.domains.access19_v2 import motion as M
from tampanda.symbolic.domains.access19_v2.shelf import ARM_BODIES, DETECT, _enable_arm_collisions

MARGIN = G.MARGIN                       # 2 mm dense clearance
DP = SimpleNamespace(max_dq=0.25, margin=0.003)
STEP = 0.015                            # Cartesian layer spacing
HELD_LIFT = 0.015                       # held object lifted 1.5 cm (v2's 4 mm: a 20 cm bottle tilting between IK layers
                                        # grazes the 2 mm margin to the floor; the env's own executor lifts 4 cm)
FINGER_OPEN = 0.04                      # Panda max opening per finger (objects here are 6 cm wide)
HOME = P.Q_HOME.copy()
PARK = np.array([100.0, 100.0, 50.0])
GRASP_DZ = 0.0                          # FRONT grasps: grasp point this far above the object centre


def _flip(R):                           # same grasp, hand rolled 180 deg about its approach axis
    return R @ np.diag([-1.0, -1.0, 1.0])


def grasp_families(domain):
    fr = K.quat_to_R(_QUATS[GraspType.FRONT])
    fams = {"front": fr, "front_flip": _flip(fr)}
    if domain == "shelf-access":
        for name, gt in (("top_x", GraspType.TOP_DOWN_X), ("top_y", GraspType.TOP_DOWN_Y)):
            R = K.quat_to_R(_QUATS[gt]); fams[name] = R; fams[name + "_flip"] = _flip(R)
    return fams


class EnvScene:
    """v2-style collision scene around a tampanda environment model (honest arm). One per process."""

    def __init__(self, env, objects, half_sizes, front_y, shape):
        self.m, self.d = env.model, env.data; m, d = self.m, self.d
        _enable_arm_collisions(m)
        arm_bodies = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in ARM_BODIES if b != "link0"}
        self.arm = [k for k in range(m.ngeom) if m.geom_bodyid[k] in arm_bodies and m.geom_conaffinity[k]]
        m.geom_margin[self.arm] = DETECT
        self.objects = list(objects); self.half = np.asarray(half_sizes, float); self.front_y = front_y; self.shape = shape
        self.obj_geoms, self.obj_qadr, self.geom_obj = {}, {}, {}
        for o in self.objects:
            b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, o)
            j = m.body_jntadr[b]; assert m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE, o
            self.obj_qadr[o] = m.jnt_qposadr[j]
            gs = [k for k in range(m.ngeom) if m.geom_bodyid[k] == b]
            self.obj_geoms[o] = gs
            for k in gs:
                self.geom_obj[k] = o
                m.geom_margin[k] = DETECT
        # self-collision (v2 Scene): arm-arm bit 2 + arm-scene bit 1, broad phase filters on BODY flags
        m.geom_contype[self.arm] = 2; m.geom_conaffinity[self.arm] = 3
        for b in {int(m.geom_bodyid[k]) for k in self.arm}:
            m.body_contype[b] |= 2; m.body_conaffinity[b] |= 3
        self.jq = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "joint%d" % i)] for i in range(1, 8)]
        self.fq = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in ("finger_joint1", "finger_joint2")]
        l0 = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "link0")
        self.layout, self.held = {}, None
        self.occupy({})
        mujoco.mj_kinematics(m, d)
        self.base = np.eye(4); self.base[:3, :3] = d.xmat[l0].reshape(3, 3); self.base[:3, 3] = d.xpos[l0]
        self.armset = set(self.arm)
        # static boxes (for VAMP) = every enabled non-arm, non-object box geom
        self.boxes = []
        for k in range(m.ngeom):
            if k in self.armset or k in self.geom_obj or m.geom_bodyid[k] == l0:
                continue
            if m.geom_type[k] == mujoco.mjtGeom.mjGEOM_BOX and (m.geom_contype[k] or m.geom_conaffinity[k]):
                c, h = d.geom_xpos[k].copy(), m.geom_size[k].copy()
                Rg = d.geom_xmat[k].reshape(3, 3); h = np.abs(Rg) @ h        # axis-aligned bound
                self.boxes.append((mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, k) or "box%d" % k, c - h, c + h))
        # structural pairs: arm geom / scene geom closer than MARGIN at home (link1 inside the mounting table)
        self.structural = set(); self.self_exclude = set()
        self._pose(HOME, FINGER_OPEN, None)
        for cc in d.contact[:d.ncon]:
            a1, a2 = cc.geom1 in self.armset, cc.geom2 in self.armset
            if a1 and a2:
                self.self_exclude.add((min(cc.geom1, cc.geom2), max(cc.geom1, cc.geom2)))
            elif (a1 != a2) and cc.dist < MARGIN and (cc.geom2 if a1 else cc.geom1) not in self.geom_obj:
                self.structural.add((cc.geom1, cc.geom2) if a1 else (cc.geom2, cc.geom1))
        # link1 only spins about the base axis: any static geom inside its swept cylinder (r 0.12 m, z < 0.35) is a
        # mounting defect of the scene (table corner / leg under the robot), not something a motion can avoid
        l1 = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "link1")
        l1g = [k for k in self.arm if m.geom_bodyid[k] == l1]
        for k in range(m.ngeom):
            if k in self.armset or k in self.geom_obj or m.geom_type[k] == mujoco.mjtGeom.mjGEOM_PLANE:
                continue
            if not (m.geom_contype[k] or m.geom_conaffinity[k]):
                continue
            c = d.geom_xpos[k]; h = np.abs(d.geom_xmat[k].reshape(3, 3)) @ m.geom_size[k]
            dxy = np.linalg.norm(np.maximum(np.abs(c[:2]) - h[:2], 0)) if True else 0
            near = np.linalg.norm(np.maximum(np.maximum((c - h)[:2], -(c + h)[:2]), 0))  # xy distance box->axis
            if near < 0.12 and (c - h)[2] < 0.35:
                for a in l1g:
                    self.structural.add((a, k))
        self.structural_names = sorted({(self.name(a), self.name(s)) for a, s in self.structural})

    def name(self, k):
        return mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_GEOM, k) or \
            mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_BODY, self.m.geom_bodyid[k]) or str(k)

    # ---- state
    def occupy(self, layout, held=None):
        """layout: {object: world centre (x, y, z)} of the objects resting in cells; held: object in the hand."""
        for i, o in enumerate(self.objects):
            a = self.obj_qadr[o]
            p = layout.get(o)
            self.d.qpos[a:a + 3] = PARK + [3.0 * i, 0, 0] if p is None else p
            self.d.qpos[a + 3:a + 7] = [1, 0, 0, 0]
        self.layout = dict(layout); self.held = held

    def site_T(self, q):
        return K.site_T(q, self.base)

    def _pose(self, q, finger, held_local):
        m, d = self.m, self.d
        d.qpos[self.jq] = q; d.qpos[self.fq] = finger
        if self.held is not None:
            a = self.obj_qadr[self.held]
            if held_local is None:
                d.qpos[a:a + 3] = PARK + [0, 5.0, 0]
            else:
                T = self.site_T(q); Rg = self._grasp_R
                d.qpos[a:a + 3] = T[:3, 3] + T[:3, :3] @ np.asarray(held_local, float)
                qq = np.zeros(4); mujoco.mju_mat2Quat(qq, (T[:3, :3] @ Rg.T).flatten()); d.qpos[a + 3:a + 7] = qq
        mujoco.mj_kinematics(m, d); mujoco.mj_collision(m, d)

    def contacts(self, q, finger, held_local=None):
        self._pose(q, finger, held_local)
        d = self.d; static = DETECT; per = {}
        hg = set(self.obj_geoms[self.held]) if (self.held is not None and held_local is not None) else set()
        mover = self.armset | hg
        for k in range(d.ncon):
            cc = d.contact[k]; g1, g2 = cc.geom1, cc.geom2
            a1, a2 = g1 in mover, g2 in mover
            if a1 and a2:
                if (g1 in hg) or (g2 in hg):
                    continue                                   # fingers on the held object
                if (min(g1, g2), max(g1, g2)) not in self.self_exclude and cc.dist < static:
                    static = cc.dist
                continue
            if a1 == a2:
                continue
            mv, other = (g1, g2) if a1 else (g2, g1)
            o = self.geom_obj.get(other)
            if o is None:
                if (mv, other) not in self.structural and cc.dist < static:
                    static = cc.dist
            elif o != self.held and cc.dist < per.get(o, DETECT):
                per[o] = cc.dist
        return static, per

    def clearance(self, q, finger, held_local=None):
        s, per = self.contacts(q, finger, held_local)
        return min([s] + list(per.values()))


# ------------------------------------------------------------------ VAMP with this scene
def _split_hole(lo, hi, h=0.09):
    """Axis-aligned box minus the vertical square column |x|,|y| < h around the robot axis (VAMP's base spheres)."""
    if hi[0] <= -h or lo[0] >= h or hi[1] <= -h or lo[1] >= h:
        return [(lo, hi)]
    out = []
    if hi[0] > h: out.append((np.r_[h, lo[1:]], hi.copy()))
    if lo[0] < -h: out.append((lo.copy(), np.r_[-h, hi[1:]]))
    xl, xh = max(lo[0], -h), min(hi[0], h)
    if hi[1] > h: out.append((np.r_[xl, h, lo[2]], np.r_[xh, hi[1], hi[2]]))
    if lo[1] < -h: out.append((np.r_[xl, lo[1], lo[2]], np.r_[xh, -h, hi[2]]))
    return out


def vamp_env(sc, pad=0.0):
    vamp = M._vamp(); env = vamp.Environment()
    for _, lo, hi in sc.boxes:
        for a, b in _split_hole(lo, hi):
            env.add_cuboid(vamp.Cuboid(list((a + b) / 2), [0, 0, 0], list((b - a) / 2)))
    fz = getattr(sc, "floor_z", 0.0)                                                       # floor, minus the base
    for a, b in _split_hole(np.array([-3.0, -3.0, fz - 0.105]), np.array([3.0, 3.0, fz - 0.005])):
        env.add_cuboid(vamp.Cuboid(list((a + b) / 2), [0, 0, 0], list((b - a) / 2)))
    for o, p in sc.layout.items():
        if sc.shape == "cylinder":
            env.add_capsule(vamp.Cylinder(list(p), [0, 0, 0], float(sc.half[0] + pad), float(2 * sc.half[2])))
        else:
            env.add_cuboid(vamp.Cuboid(list(p), [0, 0, 0], list(sc.half + pad)))
    return env


def attach(env, sc, held_local, R):
    vamp = M._vamp()
    T = np.eye(4, dtype=np.float32); T[:3, 3] = np.asarray(held_local) - np.array([0, 0, M.EE_FROM_SITE])
    a = vamp.Attachment(np.asfortranarray(T))
    axis = R.T @ np.array([0, 0, 1.0]); rad = float(np.linalg.norm(sc.half[:2])); hz = sc.half[2]
    n = max(2, int(np.ceil(2 * hz / rad)) + 1)
    a.add_spheres([vamp.Sphere(list(axis * s), rad) for s in np.linspace(-hz + rad, hz - rad, n)])
    env.attach(a)


def plan_validated(sc, q_end, held_local, finger, direction, R, q7s, reverse=False):
    for pad in (M.PAD, 2 * M.PAD, 3 * M.PAD):
        env = vamp_env(sc, pad)
        if held_local is not None:
            attach(env, sc, held_local, R)
        b = None
        for dvec in (direction, (np.asarray(direction, float) + np.array([0, 0, 1.0])) / np.sqrt(2), [0, 0, 1], [0, -1, 0]):
            b = M.backoff(sc, q_end, dvec, finger, held_local, env, q7s, MARGIN, R=R)
            if b:
                break
        if not b:
            continue
        for seed in range(5):
            p = M.rrtc(b[-1], HOME, env, seed) if reverse else M.rrtc(HOME, b[-1], env, seed)
            if p is None:
                continue
            full = np.vstack([np.array(b), p]) if reverse else np.vstack([p, np.array(b[::-1])])
            if M.mj_check(sc, full, finger, held_local, MARGIN)[0]:
                return full
    return None


# ------------------------------------------------------------------ grasp paths and the per-action check
def _T(R, p):
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = p
    return T


def grasp_lines(sc, centre, fam, R):
    """{kind: (site points from outside to the grasp, held_local)} for one object centre and grasp family.
    open: the open hand onto the resting object. held: ends at the grasp lifted 4 mm (v2: matches the open grasp
    configuration), FRONT rises to HELD_LIFT and slides out at that height, TOP_DOWN rises along its approach."""
    hz = R[:, 2]; c0 = np.asarray(centre, float); out = {}
    gp0 = c0 + [0, 0, GRASP_DZ if fam.startswith("front") else 0.0] - hz * GP_CONTACT
    end = {"open": gp0, "held": gp0 + [0, 0, K.HELD_LIFT], "held_low": gp0 + [0, 0, K.HELD_LIFT]}
    for kind in (("open", "held", "held_low") if fam.startswith("front") else ("open", "held")):
        e = end[kind]
        if fam.startswith("front"):      # held: slide out 1.5 cm up; held_low: slide out at 4 mm (v2) -- either retreat
            z = gp0[2] + HELD_LIFT if kind == "held" else e[2]
            ys = list(np.arange(sc.front_y - 0.06, e[1], STEP)) + [e[1]]
            pts = [np.array([e[0], y, z]) for y in ys]
            if kind == "held":
                pts += [e.copy()]
        else:
            zs = list(np.arange(e[2] + 0.12, e[2], -STEP)) + [e[2]]
            pts = [np.array([e[0], e[1], zz]) for zz in zs]
        out[kind] = (pts, R.T @ (c0 - gp0))
    return out


def check_motion(sc, target, centre, others, R, fam, q7s, max_pairs=10, stats=None):
    """Search a pick/put motion of ``target`` at ``centre`` with ``others`` = {obj: centre} present.
    Returns the motion dict or None; ``stats`` collects the stage counts."""
    st = stats if stats is not None else {}
    sc._grasp_R = R
    lines = grasp_lines(sc, centre, fam, R)
    paths = {"open": [], "held": []}
    for kind, (pts, hl) in lines.items():
        Ls = [K.ik_all(sc.base, _T(R, p), q7s) for p in pts]
        if fam.startswith("front"):                     # drop leading points no IK reaches while still outside
            k = 0
            while k < len(Ls) - 1 and len(Ls[k]) == 0 and pts[k + 1][1] <= sc.front_y - 0.03:
                k += 1
            Ls = Ls[k:]
        st["ik_grasp_" + kind] = int(len(Ls[-1])); st["ik_empty_layers_" + kind] = int(sum(len(L) == 0 for L in Ls))
        finger = FINGER_OPEN if kind == "open" else float(sc.half[0])
        if kind == "open":
            sc.occupy({**others, target: centre})
        else:
            sc.occupy(others, held=target)
        h = None if kind == "open" else hl
        dps = G.dp_paths(sc, Ls, finger, h, DP)
        st["dp_" + kind] = len(dps); n0 = len(paths["open" if kind == "open" else "held"])
        for Q, _ in dps:
            if M.mj_check(sc, Q, finger, h, MARGIN)[0]:
                paths["open" if kind == "open" else "held"].append({"Q": Q, "hl": hl})
        st["dense_" + kind] = len(paths["open" if kind == "open" else "held"]) - n0
    pairs = sorted((G.posture_cost(paths["open"][a]["Q"][0], paths["held"][b]["Q"][0], HOME), a, b)
                   for a, b in G.match_pairs(paths["open"], paths["held"]))
    st["pairs"] = len(pairs)
    direction = [0, -1, 0] if fam.startswith("front") else [0, 0, 1]
    tried = []
    for _, a, b in pairs:
        o, h = paths["open"][a], paths["held"][b]
        if any(np.abs(o["Q"][0] - t).max() < 0.3 for t in tried):
            continue
        tried.append(o["Q"][0])
        sc.occupy({**others, target: centre})
        t_open = plan_validated(sc, o["Q"][0], None, FINGER_OPEN, direction, R, q7s)
        t_held = None
        if t_open is not None:
            sc.occupy(others, held=target)
            t_held = plan_validated(sc, h["Q"][0], h["hl"], float(sc.half[0]), direction, R, q7s, reverse=True)
        if t_held is not None:
            st["transit"] = "ok"; st["tried"] = len(tried)
            return {"fam": fam, "open": o["Q"], "held": h["Q"], "t_open": t_open, "t_held": t_held}
        st["transit"] = "open_failed" if t_open is None else "held_failed"
        if len(tried) >= max_pairs:
            break
    st["tried"] = len(tried)
    return None
