"""Access-19 v2: reach-envelope maps for frontal grasps past a shared shelf front edge (2026-10-01).

Constraint formulation (Daniel's idea): the shelf is a solid body behind a front plane y = c (robot base frame).
  * TOP level: solid for z <= s (the top surface); the hand reaches over it at grasp height z_top = s + h_g.
  * INTERIOR level: solid except a horizontal slot f < z < f + H (cavity); grasp height z_int = f + h_g.
(h_g = cube height - grasp depth below the cube top.) For a lateral position x and a level, the REACH is the deepest
cube-centre depth d beyond c such that the straight horizontal frontal approach (site from c - 6 cm to the grasp
point, hand at the FRONT orientation, fingers open) has a continuous collision-free path with clearance >= margin.
All postures over the redundancy are enumerated analytically (He & Liu IK via frantik: q7 grid x 4 branches,
joint limits); the bottleneck DP is extended layer by layer and stops when no path keeps the margin.

The scene is the arm + two boxes whose pose/size is changed in place (no recompiling the arm meshes).

    python -m tampanda.symbolic.domains.access19_v2.reach out.json c=0.25 level=top|int H=0.30 \
        xs=-0.3:0.55:0.05 zs=0.25:0.85:0.03 [q7_step_deg=4] [margin=0.005]
"""
from __future__ import annotations

import json
import sys
import time

import mujoco
import numpy as np

import frantik
from tampanda.kinematics import panda as P
from tampanda.symbolic.domains.access19_v2 import verify as V
from tampanda.symbolic.domains.access19_v2.geometry import V2_ROBOT_XML, V2Params, patch_collisions

TCP_FROM_SITE = P.TCP_OFFSET - V.ATTACH_OFFSET
MAX_STEP = 0.20
DETECT = 0.03
FAR = np.array([50.0, 50.0, -50.0])


# fixed shapes, moved as mocap bodies. slab_b = the roof (12 mm) PLUS an 8 cm keep-out above it: cubes may be parked
# anywhere on the top, so an interior grasp must not put any link there (keeps interior picks independent of the top)
SLAB_HALF = {"slab_a": [2.0, 0.75, 1.0], "slab_b": [2.0, 0.75, 0.046]}


def build():
    # Static geoms cannot be resized after compiling (their bounding-volume tree is built at compile time and the
    # broad phase would miss them), so each slab is a fixed-size box on its own mocap body, positioned per query.
    spec = mujoco.MjSpec.from_file(str(V2_ROBOT_XML))
    for name, half in SLAB_HALF.items():
        b = spec.worldbody.add_body(); b.name = name; b.mocap = True; b.pos = FAR
        g = b.add_geom(); g.name = name; g.type = mujoco.mjtGeom.mjGEOM_BOX; g.size = half
        g.contype = 1; g.conaffinity = 1; g.rgba = [0.8, 0.8, 0.8, 1]
    m = spec.compile(); d = mujoco.MjData(m)
    patch_collisions(m, V2Params())
    return m, d


class Scene:
    def __init__(self):
        self.m, self.d = build()
        m = self.m
        self.slabs = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n) for n in SLAB_HALF]
        self.mocap = [m.body_mocapid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)] for n in SLAB_HALF]
        arm_bodies = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in V.ARM_BODIES if b != "link0"}
        self.arm = np.array([g for g in range(m.ngeom) if m.geom_bodyid[g] in arm_bodies
                             and (m.geom_contype[g] or m.geom_conaffinity[g])])
        self.armset = set(self.arm.tolist())
        m.geom_margin[self.arm] = DETECT
        self.jq = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "joint%d" % i)] for i in range(1, 8)]
        self.fq = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]
                   for n in ("finger_joint1", "finger_joint2")]
        mujoco.mj_kinematics(m, self.d)
        l0 = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "link0")
        self.base = np.eye(4); self.base[:3, :3] = self.d.xmat[l0].reshape(3, 3); self.base[:3, 3] = self.d.xpos[l0]

    def place_top_face(self, k, c, z_top):
        """Slab k: front face at y = c, top face at z = z_top."""
        h = SLAB_HALF[list(SLAB_HALF)[k]]
        self.d.mocap_pos[self.mocap[k]] = [0.0, c + h[1], z_top - h[2]]

    def hide(self, k):
        self.d.mocap_pos[self.mocap[k]] = FAR

    def clearance(self, q, finger):
        m, d = self.m, self.d
        d.qpos[self.jq] = q; d.qpos[self.fq] = finger
        mujoco.mj_kinematics(m, d); mujoco.mj_collision(m, d)
        best = DETECT
        for c in d.contact[:d.ncon]:
            if (c.geom1 in self.armset) != (c.geom2 in self.armset) and c.dist < best:
                best = c.dist
        return best


def solve_layer(sc, pos, R, q7_grid):
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = pos + TCP_FROM_SITE * R[:, 2]
    Tb = np.linalg.inv(sc.base) @ T
    out = []
    for q7 in q7_grid:
        for s in frantik.ik(Tb, float(q7), P.Q_HOME):
            if np.isfinite(s).all() and (s >= P.Q_MIN).all() and (s <= P.Q_MAX).all():
                out.append(s)
    return np.array(out) if out else np.zeros((0, 7))


def reach(sc, x, z, c, q7_grid, finger, margin, d_max=0.50, step=0.01):
    """Deepest cube-centre depth beyond c (m) reachable with a continuous path keeping >= margin, or -1."""
    best_d, best_quat = -1.0, None
    for qi, quat in enumerate(V.FRONT_QUATS):
        R = V.quat_to_R(quat)
        ys = np.arange(c - 0.06, c + d_max + 1e-9, step)
        val = None; prevQ = None; deepest = -1.0
        for y in ys:
            Q = solve_layer(sc, np.array([x, y, z]), R, q7_grid)
            if len(Q) == 0:
                break
            cl = np.array([sc.clearance(q, finger) for q in Q])
            if val is None:
                val = cl
            else:
                jump = np.abs(Q[:, None, :] - prevQ[None, :, :]).max(2)
                cand = np.where(jump <= MAX_STEP, val[None, :], -np.inf).max(1)
                val = np.minimum(cand, cl)
            prevQ = Q
            if val.max() < margin:
                break
            deepest = y + V.GRASP_CONTACT_OFFSET - c      # cube centre depth beyond the front edge
        if deepest > best_d:
            best_d, best_quat = deepest, qi
    return best_d, best_quat


def rng(s):
    a, b, st = map(float, s.split(":"))
    return np.round(np.arange(a, b + 1e-9, st), 4)


def main():
    out = sys.argv[1]
    kv = dict(a.split("=", 1) for a in sys.argv[2:] if "=" in a)
    c = float(kv["c"]); level = kv["level"]; H = float(kv.get("H", 0.30)); wall = float(kv.get("wall", 0.012))
    xs, zs = rng(kv.get("xs", "-0.3:0.55:0.05")), rng(kv.get("zs", "0.25:0.85:0.03"))
    hg = float(kv.get("hg", 0.07)); margin = float(kv.get("margin", 0.005)); finger = float(kv.get("finger", 0.03))
    q7_grid = np.arange(P.Q_MIN[6] + 1e-3, P.Q_MAX[6], np.radians(float(kv.get("q7_step_deg", 4))))
    sc = Scene(); t0 = time.time()
    # self-test: hand pushed 2 cm into a top surface must report negative clearance
    sc.place_top_face(0, 0.20, 0.40); sc.hide(1)
    R = V.quat_to_R(V.FRONT_QUATS[0])
    Qt = solve_layer(sc, np.array([0.2, 0.45, 0.40]), R, q7_grid)
    worst = min(sc.clearance(q, finger) for q in Qt) if len(Qt) else None
    best = max(sc.clearance(q, finger) for q in Qt) if len(Qt) else None
    print("self-test (site on the slab surface): %d postures, clearance min %.3f max %.3f (must be < 0)" % (len(Qt), worst, best))
    assert best is not None and best < 0, "collision detection broken"
    res = {"c": c, "level": level, "H": H, "wall": wall, "hg": hg, "margin": margin, "finger": finger,
           "xs": xs.tolist(), "zs": zs.tolist(), "reach": {}, "quat": {}}
    for z in zs:
        if level == "top":                  # solid behind c up to the top surface s = z - hg
            sc.place_top_face(0, c, z - hg); sc.hide(1)
        else:                               # slot between floor f = z - hg and roof f + H (roof slab 12 mm thick)
            f = z - hg
            sc.place_top_face(0, c, f); sc.place_top_face(1, c, f + H + 2 * SLAB_HALF["slab_b"][2])
        row, rq = [], []
        for x in xs:
            dd, qi = reach(sc, x, z, c, q7_grid, finger, margin)
            row.append(round(dd, 3)); rq.append(qi)
        res["reach"]["%.3f" % z] = row; res["quat"]["%.3f" % z] = rq
        print("z=%.3f  " % z + " ".join("%5.2f" % v for v in row) + "   (%.0fs)" % (time.time() - t0), flush=True)
    res["wall_s"] = round(time.time() - t0, 1)
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
