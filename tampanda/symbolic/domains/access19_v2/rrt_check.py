"""Access-19 v2, stage 2: transit motion planning with the full collision model (2026-10-01).

For every cell that passes the grasp check, take the verified staging configuration (first waypoint of the
straight-line approach) and measure, from the robot home pose:
  * lerp_free  -- the straight joint-space move home -> staging is collision-free (no planning needed);
  * rrt_ok / rrt_s / rrt_nodes -- tampanda RRTStar (seeded) finds a collision-free path, and how fast.
Also checks statically that link0 (excluded from dynamic collision queries) clears the shelf.

    MUJOCO_GL=egl python -m tampanda.symbolic.domains.access19_v2.rrt_check [--json f] [--max-iter N] [k=v ...]
"""
from __future__ import annotations

import json
import sys
import time

import mujoco
import numpy as np

from tampanda.kinematics import panda as P
from tampanda.symbolic.domains.access19_v2.geometry import V2Params
from tampanda.symbolic.domains.access19_v2.verify import (FRONT_QUATS, GRASP_CONTACT_OFFSET, SAFE_Z_ABOVE_DECK,
                                                          Rig, quat_to_R, seeds_for)


def cart_path(rig, a, b, R, q0, step=0.01, jump=0.25):
    """IK-follow the straight line a->b at orientation R from q0; returns (qs, why)."""
    n = max(2, int(np.ceil(np.linalg.norm(b - a) / step)) + 1)
    qs = [q0]
    for wp in [a + (b - a) * t for t in np.linspace(0, 1, n)][1:]:
        q = rig.ik(wp, R, qs[-1], iters=80)
        if q is None:
            return None, "ik_fail"
        if np.abs(q - qs[-1]).max() > jump:
            return None, "joint_jump"
        qs.append(q)
    for q in qs:
        c = rig.contacts(q, finger=0.0)
        if c:
            return None, "collision:%s-%s" % c[0]
    return qs, "ok"


def deck_route_forward(rig, region, cell):
    """v1 deck route planned FORWARD from a home-seeded high front waypoint: high_front -> (traverse at
    safe_z) -> above cell -> (vertical descent) -> grasp. Keeps the home-like IK branch. Returns
    (q_high_front, all_qs, why)."""
    best = "ik_fail"
    path = rig.approach_path(region, cell)              # above-cell -> grasp (vertical)
    ffy = rig.ws["shelf_interior"].origin[1] - 0.02
    above = path[0]; high_front = np.array([above[0], ffy - 0.06, above[2]])
    for quat in FRONT_QUATS:
        R = quat_to_R(quat)
        for seed in [P.Q_HOME] + seeds_for(rig, region, cell)[1:]:
            q_hf = rig.ik(high_front, R, seed)
            if q_hf is None:
                continue
            tq, why = cart_path(rig, high_front, above, R, q_hf)
            if tq is None:
                best = "traverse:" + why; continue
            dq, why = cart_path(rig, above, path[-1], R, tq[-1])
            if dq is None:
                best = "descent:" + why; continue
            for q in dq:                               # descent with fingers OPEN around the (absent) cube
                c = rig.contacts(q)
                if c:
                    why = "descent_open:%s-%s" % c[0]; break
            else:
                return q_hf, tq + dq[1:], "ok"
            best = why
    return None, None, best


def deck_transit(rig, region, cell, q_above):
    """v1 deck route, reversed for planning: from above the cell (q_above) traverse at safe_z to a high
    waypoint just outside the shelf front, then home <- that waypoint by lerp or RRT."""
    T = rig.site_T(q_above); R = T[:3, :3]; above = T[:3, 3]
    ffy = rig.ws["shelf_interior"].origin[1] - 0.02
    high_front = np.array([above[0], ffy - 0.06, above[2]])
    qs, why = cart_path(rig, above, high_front, R, q_above)
    return qs, why


def link0_clear(rig):
    """Enable link0's collision mesh temporarily and look for link0-vs-shelf contacts (floor ignored)."""
    m, d = rig.m, rig.d
    gids = [g for g in range(m.ngeom) if m.geom_bodyid[g] == rig.link0 and m.geom_group[g] == 3]
    saved = [(m.geom_contype[g], m.geom_conaffinity[g]) for g in gids]
    for g in gids:
        m.geom_conaffinity[g] = 1
    d.qpos[rig.jq] = P.Q_HOME; mujoco.mj_forward(m, d)
    bad = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[c.geom1 if m.geom_bodyid[c.geom2] == rig.link0 else c.geom2])
           for c in d.contact[:d.ncon]
           if rig.link0 in (m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]) and 0 not in (m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2])]
    for g, (ct, ca) in zip(gids, saved):
        m.geom_contype[g], m.geom_conaffinity[g] = ct, ca
    return sorted(set(bad))


def lerp_free(rig, qa, qb, step=0.02):
    n = max(2, int(np.ceil(np.abs(qb - qa).max() / step)) + 1)
    return all(not rig.contacts(qa + (qb - qa) * t, finger=0.0) for t in np.linspace(0, 1, n))


def main():
    kv = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a and not a.startswith("--"))
    out = next((sys.argv[i + 1] for i, a in enumerate(sys.argv) if a == "--json"), None)
    max_iter = int(next((sys.argv[i + 1] for i, a in enumerate(sys.argv) if a == "--max-iter"), 3000))
    p = V2Params()
    for k, v in kv.items():
        cur = getattr(p, k)
        setattr(p, k, tuple(float(x) for x in v.split(",")) if isinstance(cur, tuple) else type(cur)(v))
    rig = Rig(p)
    from tampanda.planners.rrt_star import RRTStar
    planner = RRTStar(rig.env, max_iterations=max_iter, step_size=0.1)
    res = {"params": p.as_dict(), "link0_vs_scene": link0_clear(rig), "cells": {}}
    print("link0 static contacts:", res["link0_vs_scene"], flush=True)
    for rname in ("shelf_interior", "shelf_top"):
        region = rig.ws[rname]
        for cell in region.cells():
            key = "%s__%d_%d" % (rname, cell.ix, cell.iy)
            rig.park_all()
            ok, why, qs = rig.try_grasp(region, cell, seeds_for(rig, region, cell))
            row = {"grasp_ok": ok}
            if ok:
                qst = qs[0]
                if rname == "shelf_top":                 # structured v1 route, planned forward from home
                    q_hf, allq, twhy = deck_route_forward(rig, region, cell)
                    row["traverse_ok"] = q_hf is not None; row["traverse_why"] = twhy
                    if q_hf is not None:
                        qst = q_hf                       # plan home -> high front waypoint
                row["lerp_free"] = lerp_free(rig, P.Q_HOME, qst)
                np.random.seed(0)
                t0 = time.time(); path = planner.plan(P.Q_HOME.copy(), qst.copy()); dt = time.time() - t0
                row.update(rrt_ok=path is not None, rrt_s=round(dt, 2), rrt_len=len(path) if path else 0)
            res["cells"][key] = row
            print("%-20s %s" % (key, row), flush=True)
    C = [v for v in res["cells"].values() if v["grasp_ok"]]
    t = [v["rrt_s"] for v in C if v.get("rrt_ok")]
    D = [v for k, v in res["cells"].items() if k.startswith("shelf_top") and v["grasp_ok"]]
    res["summary_deck_traverse"] = {"traverse_ok": sum(v.get("traverse_ok", False) for v in D), "n": len(D),
                                    "why": sorted({v.get("traverse_why") for v in D if not v.get("traverse_ok")})}
    print("DECK", res["summary_deck_traverse"])
    res["summary"] = {"grasp_ok": len(C), "lerp_free": sum(v["lerp_free"] for v in C),
                      "rrt_ok": sum(v.get("rrt_ok", False) for v in C),
                      "rrt_s_median": float(np.median(t)) if t else None, "rrt_s_p95": float(np.percentile(t, 95)) if t else None,
                      "rrt_s_max": max(t) if t else None}
    print("SUMMARY", res["summary"])
    if out:
        json.dump(res, open(out, "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
