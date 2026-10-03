"""Access-19 v2 geometry verification (2026-10-01). Requirements (Daniel):
  R1  every cell of both grids (47 interior, 49 deck) is graspable when the shelf is empty;
  R2  ... and with the target cube present (the open fingers clear it);
  R3  two cubes in laterally adjacent cells of the same row block each other (grasp infeasible);
  R4  full collision model (every arm link vs shelf and cubes), joint limits respected;
  R5  reasonable RRT runtime for feasible grasps (separate stage, ``--rrt``).

A grasp is the v1 front grasp: attachment_site at FRONT orientation (palm +y), at
(cell_x, cell_y - 0.0137, cube_top - 0.010). Interior cells are approached in a straight line from 6 cm
outside the open front face; deck cells by a vertical descent from 10 cm above the deck. A cell passes R1/R2
when IK reaches every waypoint of that straight-line approach (1 cm steps, continuous joints, inside limits)
with no arm-vs-scene contact. R3 requires the same grasp to FAIL when a lateral neighbour is present.

    MUJOCO_GL=egl python -m tampanda.symbolic.domains.access19_v2.verify [--json out.json] [k=v ...]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

from tampanda.kinematics import panda as P
from tampanda.symbolic.domains.access19_v2.geometry import ARM_BODIES, V2Params, build

GRASP_CONTACT_OFFSET = 0.0137
CUBE_GRASP_OFFSET = 0.010
SAFE_Z_ABOVE_DECK = 0.10
ATTACH_OFFSET = 0.09           # attachment_site: hand frame + 0.09 m along z
FRONT_QUATS = (np.array([-0.5, 0.5, 0.5, 0.5]), np.array([-0.5, 0.5, -0.5, -0.5]))


def quat_to_R(q):
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


class Rig:
    def __init__(self, p: V2Params):
        self.p = p
        self.env, self.ws, self.cfg = build(p)
        self.m, self.d = self.env.model, self.env.data
        m = self.m
        self.arm_body = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in ARM_BODIES}
        self.link0 = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "link0")
        self.jq = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "joint%d" % i)] for i in range(1, 8)]
        self.fq = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in ("finger_joint1", "finger_joint2")]
        self.cubes = ["blocker_%d" % i for i in range(18)] + ["ooi"]
        self.cube_qadr = {c: m.jnt_qposadr[m.body_jntadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, c)]] for c in self.cubes}
        self.park_all()
        mujoco.mj_forward(m, self.d)
        T = np.eye(4); T[:3, :3] = self.d.xmat[self.link0].reshape(3, 3); T[:3, 3] = self.d.xpos[self.link0]
        self.base = T

    # ---- scene state
    def park_all(self):
        for i, c in enumerate(self.cubes):
            a = self.cube_qadr[c]
            self.d.qpos[a:a + 7] = [100 + i, 0, self.p.cube_half[2] + 0.005, 1, 0, 0, 0]

    def place(self, cube, xyz):
        a = self.cube_qadr[cube]; self.d.qpos[a:a + 7] = [*xyz, 1, 0, 0, 0]

    # ---- kinematics
    def site_T(self, q):
        F = P.fk(q, self.base)
        T = F.hand.copy(); T[:3, 3] = F.hand[:3, 3] + ATTACH_OFFSET * F.hand[:3, 2]
        return T

    def ik(self, pos, R, q0, iters=200, tol_p=0.002, tol_r=0.01):
        """Damped least squares on the attachment site, joint limits by clamping. Returns q or None."""
        q = np.array(q0, float)
        for _ in range(iters):
            T = self.site_T(q)
            ep = pos - T[:3, 3]
            Re = R @ T[:3, :3].T
            er = 0.5 * np.array([Re[2, 1] - Re[1, 2], Re[0, 2] - Re[2, 0], Re[1, 0] - Re[0, 1]])
            if np.linalg.norm(ep) < tol_p and np.linalg.norm(er) < tol_r:
                return q
            F = P.fk(q, self.base)
            J = np.zeros((6, 7)); p_site = T[:3, 3]
            for i, Ti in enumerate(F.links):
                z, o = Ti[:3, 2], Ti[:3, 3]
                J[:3, i] = np.cross(z, p_site - o); J[3:, i] = z
            e = np.concatenate([ep, er])
            lam = 0.05
            dq = J.T @ np.linalg.solve(J @ J.T + lam ** 2 * np.eye(6), e)
            # mild pull to home in the nullspace keeps solutions away from joint limits
            N = np.eye(7) - np.linalg.pinv(J) @ J
            dq += 0.05 * N @ (P.Q_HOME - q)
            q = np.clip(q + np.clip(dq, -0.3, 0.3), P.Q_MIN + 1e-3, P.Q_MAX - 1e-3)
        return None

    def contacts(self, q, finger=None, held=None):
        """Arm-vs-scene contacts at configuration q (fingers open by default). With ``held`` = a cube name, that
        cube is carried rigidly at the grasp offset, counts as part of the arm, and is closed on by the fingers
        (finger width = cube half width). Returns list of (arm_body, other_body)."""
        self.d.qpos[self.jq] = q
        if held is not None:
            finger = self.p.cube_half[0] if finger is None else finger
        self.d.qpos[self.fq] = self.p.finger_open if finger is None else finger
        if held is not None:
            T = self.site_T(q)
            c = T[:3, 3] + np.array([0.0, GRASP_CONTACT_OFFSET, -(self.p.cube_half[2] - CUBE_GRASP_OFFSET)])
            self.place(held, c)
        mujoco.mj_forward(self.m, self.d)
        held_id = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, held) if held else -1
        out = []
        for k in range(self.d.ncon):
            c = self.d.contact[k]
            b1, b2 = self.m.geom_bodyid[c.geom1], self.m.geom_bodyid[c.geom2]
            a1, a2 = b1 in self.arm_body or b1 == held_id, b2 in self.arm_body or b2 == held_id
            if a1 == a2 or c.dist > 0.001:
                continue
            arm, other = (b1, b2) if a1 else (b2, b1)
            if other == 0 and arm == self.link0:       # robot base bolted to the floor
                continue
            out.append((mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_BODY, arm),
                        mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_BODY, other)))
        return out

    # ---- grasps
    def cell_xyz(self, region, cell):
        x, y = region.pose_for(cell)[:2] if len(region.pose_for(cell)) >= 2 else region.pose_for(cell)
        return np.array([x, y, region.level_z])

    def approach_path(self, region, cell):
        c = self.cell_xyz(region, cell)
        gz = c[2] + self.p.cube_half[2] - CUBE_GRASP_OFFSET
        g = np.array([c[0], c[1] - GRASP_CONTACT_OFFSET, gz])
        if region.name == "shelf_interior":
            front_face_y = region.origin[1] - 0.02
            s = np.array([c[0], front_face_y - 0.06, gz])
        else:
            deck_top = region.level_z - self.p.cube_half[2]
            s = np.array([g[0], g[1], deck_top + SAFE_Z_ABOVE_DECK])
        n = max(2, int(np.ceil(np.linalg.norm(g - s) / 0.01)) + 1)
        return [s + (g - s) * t for t in np.linspace(0, 1, n)]

    def try_grasp(self, region, cell, seeds, held=None):
        """Returns (ok, reason, q_path). Tries both front quaternions and all seeds. ``held``: PUT mode."""
        path = self.approach_path(region, cell)
        if held is not None:                              # place: the cube hovers 3 mm above its resting pose
            path = [w + np.array([0, 0, 0.003]) for w in path]
        best = "ik_fail"
        for quat in FRONT_QUATS:
            R = quat_to_R(quat)
            for seed in seeds:
                q = self.ik(path[0], R, seed)
                if q is None:
                    continue
                qs = [q]; ok = True; why = ""
                for wp in path[1:]:
                    qn = self.ik(wp, R, qs[-1], iters=60)
                    if qn is None:
                        ok, why = False, "ik_fail_along_path"; break
                    if np.abs(qn - qs[-1]).max() > 0.25:
                        ok, why = False, "joint_jump"; break
                    qs.append(qn)
                if not ok:
                    best = why if best == "ik_fail" else best
                    continue
                for qq in qs:
                    col = self.contacts(qq, held=held)
                    if col:
                        ok, why = False, "collision:%s-%s" % col[0]; break
                if ok and held is not None:
                    # release: the cube rests on the surface, the fingers open to the approach width and must
                    # not hit anything (a lateral neighbour 2 cm away is what makes side-by-side puts infeasible)
                    rest = self.cell_xyz(region, cell)
                    self.place(held, rest)
                    col = self.contacts(qs[-1], finger=self.p.finger_open)
                    if col:
                        ok, why = False, "release:%s-%s" % col[0]
                if ok:
                    return True, "ok", qs
                best = why
        return False, best, None


def seeds_for(rig, region, cell):
    """IK seeds: home, plus a few elbow-up variants (cheap multi-start)."""
    s = [P.Q_HOME.copy()]
    for j1 in (-0.6, 0.0, 0.6, 1.2):
        q = P.Q_HOME.copy(); q[0] = j1; q[1] = 0.3; q[3] = -2.0; q[5] = 2.2; s.append(q)
    rng = np.random.default_rng(1000 * cell.ix + cell.iy)
    s += list(P.random_q(rng, rig.p.ik_seeds)) if rig.p.ik_seeds else []
    return s


def run(p: V2Params, verbose=True):
    rig = Rig(p)
    t0 = time.time()
    res = {"params": p.as_dict(), "home_contacts": rig.contacts(P.Q_HOME, finger=0.0), "cells": {}}
    pitch = p.cube_half[0] * 2 + p.inter_row_gap
    for rname in ("shelf_interior", "shelf_top"):
        region = rig.ws[rname]
        for cell in region.cells():
            ix, iy = cell.ix, cell.iy
            key = "%s__%d_%d" % (rname, ix, iy)
            rig.park_all()
            seeds = seeds_for(rig, region, cell)
            r1, why1, qs = rig.try_grasp(region, cell, seeds)                    # R1 empty shelf
            rig.place("ooi", rig.cell_xyz(region, cell))
            r2, why2, _ = rig.try_grasp(region, cell, [qs[0]] + seeds if qs else seeds)   # R2 target present
            blocks = {}
            for dx in (-1, 1):                                                   # R3 lateral neighbour
                nb = (ix + dx, iy)
                if not region.in_bounds(*nb) or region.is_excluded(*nb):
                    continue
                rig.place("blocker_0", rig.cell_xyz(region, cell) + np.array([dx * pitch, 0, 0]))
                okb, whyb, _ = rig.try_grasp(region, cell, [qs[0]] + seeds if qs else seeds)
                blocks[dx] = {"blocked": not okb, "why": whyb}
                rig.place("blocker_0", [100, 0, p.cube_half[2] + 0.005])
            rig.park_all()
            r6, why6, _ = rig.try_grasp(region, cell, [qs[0]] + seeds if qs else seeds, held="ooi")   # R6 put
            pblocks = {}
            for dx in (-1, 1):                                                   # R7 put next to neighbour
                nb = (ix + dx, iy)
                if not region.in_bounds(*nb) or region.is_excluded(*nb):
                    continue
                rig.place("blocker_0", rig.cell_xyz(region, cell) + np.array([dx * pitch, 0, 0]))
                okp, whyp, _ = rig.try_grasp(region, cell, [qs[0]] + seeds if qs else seeds, held="ooi")
                pblocks[dx] = {"blocked": not okp, "why": whyp}
                rig.place("blocker_0", [100, 0, p.cube_half[2] + 0.005])
            rig.park_all()
            res["cells"][key] = {"empty": r1, "empty_why": why1, "with_target": r2, "target_why": why2,
                                 "lateral": blocks, "put": r6, "put_why": why6, "put_lateral": pblocks}
            if verbose:
                print("%-22s empty=%-5s %-24s target=%-5s lat_blocked=%s put=%-5s %-20s put_lat_blocked=%s" % (
                    key, r1, why1 if not r1 else "", r2, {k: v["blocked"] for k, v in blocks.items()},
                    r6, why6 if not r6 else "", {k: v["blocked"] for k, v in pblocks.items()}), flush=True)
    C = res["cells"]
    summ = {}
    for rname in ("shelf_interior", "shelf_top"):
        cs = {k: v for k, v in C.items() if k.startswith(rname)}
        lat = [b["blocked"] for v in cs.values() for b in v["lateral"].values()]
        plat = [b["blocked"] for v in cs.values() if v["put"] for b in v["put_lateral"].values()]
        summ[rname] = {"n": len(cs), "R1_empty": sum(v["empty"] for v in cs.values()),
                       "R2_with_target": sum(v["with_target"] for v in cs.values()),
                       "R3_lateral_blocked": "%d/%d" % (sum(lat), len(lat)),
                       "R6_put": sum(v["put"] for v in cs.values()),
                       "R7_put_lateral_blocked": "%d/%d" % (sum(plat), len(plat))}
    res["summary"] = summ; res["wall_s"] = round(time.time() - t0, 1)
    return res


def main():
    kv = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a and not a.startswith("--"))
    out = next((sys.argv[i + 1] for i, a in enumerate(sys.argv) if a == "--json"), None)
    p = V2Params()
    for k, v in kv.items():
        cur = getattr(p, k)
        setattr(p, k, tuple(float(x) for x in v.split(",")) if isinstance(cur, tuple) else type(cur)(eval(v) if isinstance(cur, bool) else v))
    res = run(p)
    print("\nhome-pose contacts:", res["home_contacts"])
    for k, v in res["summary"].items():
        print("%-15s %s" % (k, v))
    print("wall %.0fs" % res["wall_s"])
    if out:
        json.dump(res, open(out, "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
