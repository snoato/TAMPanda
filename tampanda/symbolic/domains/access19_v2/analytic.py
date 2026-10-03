"""Access-19 v2: analytic, exhaustive grasp-clearance analysis (2026-10-01).

Complements the DLS-from-random-seeds verifier (verify.py), which can miss solutions. For every cell, grasp height
and frontal hand orientation, every waypoint of the straight-line approach is solved in CLOSED FORM over the whole
redundancy: He & Liu (2021) analytic Panda IK via ``frantik`` (q7 swept on a grid, all 4 branches, joint limits
enforced). Every solution gets a signed clearance (contact pipeline with a 5 cm detection margin on the arm geoms:
min distance arm-vs-scene, negative = penetration). A cell's score is the BOTTLENECK clearance of the best
continuous path through the waypoint layers (max over paths of the min clearance; consecutive configurations must
differ by <= MAX_STEP rad in every joint), computed by DP. The binding (arm link, scene geom) pair at the bottleneck
tells which dimension to change.

    python -m tampanda.symbolic.domains.access19_v2.analytic out.json [k=v ...] [scene=empty|target|lateral]
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
from tampanda.symbolic.domains.access19_v2.geometry import V2Params

TCP_FROM_SITE = P.TCP_OFFSET - V.ATTACH_OFFSET      # tcp = site + 0.0134 m along the hand z axis
DETECT = 0.05
MAX_STEP = 0.20                                     # rad, per joint, between consecutive waypoints


class Clearance:
    def __init__(self, rig):
        self.rig, m = rig, rig.m
        self.arm = np.array([g for g in range(m.ngeom) if m.geom_bodyid[g] in rig.arm_body
                             and m.geom_bodyid[g] != rig.link0 and (m.geom_contype[g] or m.geom_conaffinity[g])])
        self.armset = set(self.arm.tolist())
        self.base_margin = m.geom_margin.copy()

    def __call__(self, q, finger):
        rig, m, d = self.rig, self.rig.m, self.rig.d
        m.geom_margin[self.arm] = DETECT
        d.qpos[rig.jq] = q; d.qpos[rig.fq] = finger
        mujoco.mj_forward(m, d)
        best = (DETECT, -1, -1)
        for c in d.contact[:d.ncon]:
            a1, a2 = c.geom1 in self.armset, c.geom2 in self.armset
            if a1 == a2:
                continue
            if c.dist < best[0]:
                best = (float(c.dist), c.geom1 if a1 else c.geom2, c.geom2 if a1 else c.geom1)
        m.geom_margin[:] = self.base_margin
        return best


def all_ik(T_site, q7_grid):
    T = T_site.copy(); T[:3, 3] = T[:3, 3] + TCP_FROM_SITE * T[:3, 2]
    sols = []
    for q7 in q7_grid:
        S = frantik.ik(T, float(q7), P.Q_HOME)
        for s in S:
            if np.isfinite(s).all() and (s >= P.Q_MIN).all() and (s <= P.Q_MAX).all():
                sols.append(s)
    return np.array(sols) if sols else np.zeros((0, 7))


def cell_score(rig, clear, region, cell, depth, R, q7_grid, finger):
    path = rig.approach_path(region, cell, 0.0, depth)
    layers, cl, pairs = [], [], []
    for wp in path:
        T = np.eye(4); T[:3, :3] = R; T[:3, 3] = wp
        Q = all_ik(_to_base(rig, T), q7_grid)
        if len(Q) and not getattr(rig, "_ik_checked", False):        # one-time sanity: FK of an IK solution
            err = np.abs(rig.site_T(Q[0]) - T).max(); rig._ik_checked = True
            assert err < 1e-6, "analytic IK frame mismatch: %.2e" % err
        if len(Q) == 0:
            return {"score": -np.inf, "why": "no_ik", "layer": len(layers)}
        c = [clear(q, finger) for q in Q]
        layers.append(Q); cl.append(np.array([x[0] for x in c])); pairs.append([(x[1], x[2]) for x in c])
    # bottleneck DP: val[i] = best min-clearance of a continuous path ending at node i of the current layer
    val, back = cl[0].copy(), []
    for k in range(1, len(layers)):
        jump = np.abs(layers[k][:, None, :] - layers[k - 1][None, :, :]).max(2)       # (n_k, n_{k-1})
        cand = np.where(jump <= MAX_STEP, val[None, :], -np.inf)
        j = cand.argmax(1); best_prev = cand[np.arange(len(j)), j]
        val = np.minimum(best_prev, cl[k]); back.append(j)
    i = int(val.argmax()); score = float(val[i])
    if not np.isfinite(score):
        return {"score": -np.inf, "why": "no_continuous_path", "layer": len(layers)}
    # recover the path and its bottleneck
    idx = [i]
    for j in reversed(back):
        idx.append(int(j[idx[-1]]))
    idx = idx[::-1]
    k_b = int(np.argmin([cl[k][idx[k]] for k in range(len(idx))]))
    ga, gs = pairs[k_b][idx[k_b]]
    nm = lambda g: (mujoco.mj_id2name(rig.m, mujoco.mjtObj.mjOBJ_BODY, rig.m.geom_bodyid[g]) +
                    ":" + (mujoco.mj_id2name(rig.m, mujoco.mjtObj.mjOBJ_GEOM, g) or str(g))) if g >= 0 else None
    return {"score": score, "bind_arm": nm(ga), "bind_scene": nm(gs), "bind_step": k_b, "n_steps": len(idx),
            "n_sol_final": int(len(layers[-1])), "q_final": layers[-1][idx[-1]].tolist()}


def _to_base(rig, T_world):
    return np.linalg.inv(rig.base) @ T_world


def main():
    out = sys.argv[1]
    kv = dict(a.split("=", 1) for a in sys.argv[2:] if "=" in a)
    scene = kv.pop("scene", "empty"); q7_step = float(kv.pop("q7_step_deg", 3.0))
    only = kv.pop("only", None)
    p = V2Params()
    for k, v in kv.items():
        cur = getattr(p, k)
        setattr(p, k, tuple(float(x) for x in v.split(",")) if isinstance(cur, tuple) else type(cur)(v))
    rig = V.Rig(p); clear = Clearance(rig)
    q7_grid = np.arange(P.Q_MIN[6] + 1e-3, P.Q_MAX[6], np.radians(q7_step))
    pitch = p.cube_half[0] * 2 + p.inter_row_gap
    t0 = time.time(); res = {"params": p.as_dict(), "scene": scene, "q7_step_deg": q7_step, "cells": {}}
    for rname in ("shelf_interior", "shelf_top"):
        if only and only != rname:
            continue
        region = rig.ws[rname]
        for cell in region.cells():
            key = "%s__%d_%d" % (rname, cell.ix, cell.iy)
            rig.park_all()
            if scene in ("target", "lateral"):
                rig.place("ooi", rig.cell_xyz(region, cell))
            per = []
            nbs = [dx for dx in (-1, 1) if region.in_bounds(cell.ix + dx, cell.iy)
                   and not region.is_excluded(cell.ix + dx, cell.iy)] if scene == "lateral" else [None]
            for dx in nbs:
                if dx is not None:
                    rig.place("blocker_0", rig.cell_xyz(region, cell) + np.array([dx * pitch, 0, 0]))
                best = {"score": -np.inf}
                for depth in p.grasp_depths:
                    for qi, quat in enumerate(V.FRONT_QUATS):
                        r = cell_score(rig, clear, region, cell, depth, V.quat_to_R(quat), q7_grid, p.finger_open)
                        r.update(depth=depth, quat=qi)
                        if r["score"] > best["score"]:
                            best = r
                if dx is not None:
                    best["dx"] = dx
                    rig.place("blocker_0", [100, 0, p.cube_half[2] + 0.005])
                per.append(best)
            res["cells"][key] = per if scene == "lateral" else per[0]
            b = per[0]
            print("%-22s score %7.1f mm  bind %s vs %s  depth %s quat %s nsol %s" % (
                key, 1000 * b["score"] if np.isfinite(b["score"]) else -999, b.get("bind_arm"), b.get("bind_scene"),
                b.get("depth"), b.get("quat"), b.get("n_sol_final")), flush=True)
    res["wall_s"] = round(time.time() - t0, 1)
    json.dump(res, open(out, "w"), indent=1, default=float)
    print("wall %.0fs" % res["wall_s"])


if __name__ == "__main__":
    main()
