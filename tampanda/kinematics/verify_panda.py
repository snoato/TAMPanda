"""Verify ``tampanda.kinematics.panda`` against the MuJoCo Panda that tampanda simulates, and inventory
which arm bodies can collide (2026-10-01, Access-19 v2 rebuild, step 1).

Checks
  1. Joint limits in the MJCF equal the datasheet limits.
  2. FK: for N random in-limit configurations, every link frame, the hand frame and the TCP from the
     analytic DH model match MuJoCo's body poses (position error, rotation angle error).
  3. Jacobian: the analytic TCP Jacobian matches MuJoCo's mj_jacSite-equivalent (mj_jac at the TCP point).
  4. Collision inventory: for every arm body, its geoms' contype / conaffinity / group in the raw MJCF and
     after FrankaEnvironment's runtime patch -- i.e. which links can generate contacts with the scene.

    MUJOCO_GL=egl python -m tampanda.kinematics.verify_panda [N=2000]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import mujoco
import numpy as np

from tampanda.kinematics import panda as P

_ASSETS = Path(__file__).resolve().parents[1] / "environments" / "assets" / "franka_emika_panda"
MJCF = _ASSETS / "panda.xml"
SCENE_MJCF = _ASSETS / "base_panda.xml"   # what scenes/builder.py (PANDA_BASE_XML) loads into every scene
ARM = ["link0", "link1", "link2", "link3", "link4", "link5", "link6", "link7", "hand",
       "left_finger", "right_finger"]


def body_T(m, d, name):
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
    T = np.eye(4); T[:3, :3] = d.xmat[bid].reshape(3, 3); T[:3, 3] = d.xpos[bid]
    return T


def rot_err(A, B):
    R = A[:3, :3].T @ B[:3, :3]
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def collision_inventory(m, label):
    rows = {}
    for gid in range(m.ngeom):
        b = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[gid])
        if b not in ARM:
            continue
        r = rows.setdefault(b, {"geoms": 0, "collision_group3": 0, "can_contact_scene": 0})
        r["geoms"] += 1
        r["collision_group3"] += int(m.geom_group[gid] == 3)
        # a geom can touch a default scene geom (contype=1, conaffinity=1) iff its contype or conaffinity has bit 1
        r["can_contact_scene"] += int(bool(m.geom_contype[gid] & 1) or bool(m.geom_conaffinity[gid] & 1))
    print("\n[%s] arm bodies: geoms / collision-class (group 3) / geoms able to contact the scene" % label)
    for b in ARM:
        r = rows.get(b, {"geoms": 0, "collision_group3": 0, "can_contact_scene": 0})
        flag = "" if r["can_contact_scene"] else "   <-- NO scene contact"
        print("  %-12s %3d %3d %3d%s" % (b, r["geoms"], r["collision_group3"], r["can_contact_scene"], flag))
    return rows


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    m = mujoco.MjModel.from_xml_path(str(MJCF)); d = mujoco.MjData(m)
    jids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "joint%d" % i) for i in range(1, 8)]
    qadr = [m.jnt_qposadr[j] for j in jids]

    # 1. limits
    lo = np.array([m.jnt_range[j][0] for j in jids]); hi = np.array([m.jnt_range[j][1] for j in jids])
    lim_ok = np.allclose(lo, P.Q_MIN, atol=1e-4) and np.allclose(hi, P.Q_MAX, atol=1e-4)
    print("joint limits MJCF == datasheet:", lim_ok)
    if not lim_ok:
        print("  MJCF lo", lo, "\n  MJCF hi", hi)

    # 2. FK
    rng = np.random.default_rng(0)
    worst = {"pos_mm": 0.0, "rot_deg": 0.0, "where": None}
    jac_err = 0.0
    for q in P.random_q(rng, n):
        d.qpos[:] = 0; d.qpos[qadr] = q
        mujoco.mj_kinematics(m, d); mujoco.mj_comPos(m, d)
        base = body_T(m, d, "link0")
        F = P.fk(q, base)
        pairs = [("link%d" % (i + 1), F.links[i]) for i in range(7)] + [("hand", F.hand)]
        for name, Ta in pairs:
            Tm = body_T(m, d, name)
            pe = 1000 * np.linalg.norm(Ta[:3, 3] - Tm[:3, 3]); re = rot_err(Ta, Tm)
            if pe > worst["pos_mm"] or re > worst["rot_deg"]:
                worst = {"pos_mm": max(pe, worst["pos_mm"]), "rot_deg": max(re, worst["rot_deg"]), "where": name}
        # TCP = hand origin + TCP_OFFSET along hand z, compared against MuJoCo's hand frame
        hand = body_T(m, d, "hand"); tcp_m = hand[:3, 3] + P.TCP_OFFSET * hand[:3, 2]
        pe = 1000 * np.linalg.norm(F.tcp[:3, 3] - tcp_m)
        if pe > worst["pos_mm"]:
            worst.update(pos_mm=pe, where="tcp")
        # 3. Jacobian at the TCP point (MuJoCo: mj_jac on the hand body at the TCP point)
        jp = np.zeros((3, m.nv)); jr = np.zeros((3, m.nv))
        mujoco.mj_jac(m, d, jp, jr, tcp_m, mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "hand"))
        dadr = [m.jnt_dofadr[j] for j in jids]
        Jm = np.vstack([jp[:, dadr], jr[:, dadr]])
        jac_err = max(jac_err, float(np.abs(P.jacobian(q, "tcp", base) - Jm).max()))
    print("FK over %d random configs: worst position error %.4f mm, worst rotation error %.4f deg (at %s)"
          % (n, worst["pos_mm"], worst["rot_deg"], worst["where"]))
    print("TCP Jacobian: worst abs element error %.2e" % jac_err)

    # reach sanity: max |tcp - joint2 origin| over the samples (datasheet reach 0.855 m is flange-based)
    F = P.fk(P.Q_HOME)
    print("home pose: TCP at %s (base frame), hand z axis %s" % (np.round(F.tcp[:3, 3], 4), np.round(F.hand[:3, 2], 3)))

    # 4. collisions
    raw = collision_inventory(m, "raw panda.xml")
    from tampanda.environments.franka_env import FrankaEnvironment
    env = FrankaEnvironment(str(MJCF))
    patched = collision_inventory(env.model, "panda.xml after FrankaEnvironment runtime patch")
    # the robot model actually used by the Access-19 scene
    ms = mujoco.MjModel.from_xml_path(str(SCENE_MJCF))
    raw_scene = collision_inventory(ms, "base_panda.xml raw (scene robot)")
    envs = FrankaEnvironment(str(SCENE_MJCF))
    patched_scene = collision_inventory(envs.model, "base_panda.xml after FrankaEnvironment runtime patch (= Access-19 v1)")
    # FK of the scene robot must match too (same kinematic tree?)
    ds = mujoco.MjData(ms); worst_s = 0.0
    js = [mujoco.mj_name2id(ms, mujoco.mjtObj.mjOBJ_JOINT, "joint%d" % i) for i in range(1, 8)]
    for q in P.random_q(np.random.default_rng(1), 200):
        ds.qpos[:] = 0; ds.qpos[[ms.jnt_qposadr[j] for j in js]] = q; mujoco.mj_kinematics(ms, ds)
        F = P.fk(q, body_T(ms, ds, "link0"))
        worst_s = max(worst_s, 1000 * np.linalg.norm(F.hand[:3, 3] - body_T(ms, ds, "hand")[:3, 3]))
    print("\nbase_panda.xml FK vs analytic: worst hand position error %.4f mm" % worst_s)
    out = {"n": n, "limits_match": bool(lim_ok), "fk_worst": worst, "jac_worst": jac_err,
           "collision_raw": raw, "collision_patched": patched,
           "scene_collision_raw": raw_scene, "scene_collision_patched": patched_scene, "scene_fk_worst_mm": worst_s}
    json.dump(out, open("verify_panda_report.json", "w"), indent=1, default=float)
    print("\nwrote verify_panda_report.json")


if __name__ == "__main__":
    main()
