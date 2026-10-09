"""Honest 3D-stacking check with the shelf libraries' IK (2026-10-05, Daniel: "it is an IK-branch problem -- MINK lands
wrist-down at the stack table, and the 180 deg-rotated, equivalent grasp would clear it").

Same as ``honest.py`` (VERSION 2026-10-04b, unchanged and still selectable) -- same scene (base_panda_v2 real link
meshes), same 2 mm full-arm rule, same dense checks of every committed motion incl. transits, same executor chain --
except the IK the executor plans with. tampanda's MinkIK (damped least squares from one seed: one solution, in the
seed's branch) is replaced by ``AnalyticIK``:

* closed-form frantik IK (He & Liu 2021) over ALL branches, joint 7 sampled on the libraries' grid
  (access19_v2.kinematics.q7_grid(3.0): Q_MIN[6] + 1 mrad ... Q_MAX[6] in 3 deg steps), joint-limit filtered
  (access19_v2.kinematics.ik_all, exactly as the Access-19 v2 / confined / M3 library builders);
* plus the 180 deg-flipped equivalent grasp (hand rotated about its approach axis), whenever that leaves the grasped
  block where it is: nothing held (a pick: the hand turns about the grasp point, the block does not move), or the
  held block centred on the approach axis (top-down grasps of cubes / flat blocks; a 180 deg turn of a box about a
  principal axis occupies the same space). Not for the off-axis front grasp of a held upright block (that would move
  the block by twice its offset); the executor's upright grasp set already contains both hand rolls there;
* among all solutions, nearest to the seed first (max-abs joint distance), the first that is collision-free under the
  honest rule AND whose straight joint motion from the seed passes the dense honest check (<= 5 mrad) is returned;
  if none, the first collision-free one; if none, the nearest (the executor then rejects it as before). So a
  motion is accepted if ANY branch / grasp passes the full-arm mesh check.
The executor's precomputed hand-off configurations are kept (computed with MinkIK before the swap).

Selected with MLB_CHECKER=honest_aik (rgnet plangolin/domains/checker_info + multilevel_blocks.py); logged as
mlblocks-honest-aik-2026-10-05. Needs frantik, located through A19V2_PYLIB (access19_v2._deps; no default location).
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import mujoco
import numpy as np

from tampanda.kinematics import panda as P
from tampanda.symbolic.domains.access19_v2 import kinematics as K
from tampanda.symbolic.domains.multilevel_blocks import honest as H
from tampanda.symbolic.domains.multilevel_blocks.env_builder import MultilevelBlocksConfig

VERSION = "2026-10-05"
Q7_STEP_DEG = 3.0                     # the libraries' joint-7 grid
FLIP = np.array([0.0, 0.0, 0.0, 1.0])  # 180 deg about the hand's local z (approach axis), (w, x, y, z)
MAX_LERP_CHECKS = 24                  # collision-free candidates whose joint motion from the seed is dense-checked


def _qmul(a, b):
    w1, x1, y1, z1 = a; w2, x2, y2, z2 = b
    return np.array([w1*w2 - x1*x2 - y1*y2 - z1*z2, w1*x2 + x1*w2 + y1*z2 - z1*y2,
                     w1*y2 - x1*z2 + y1*w2 + z1*x2, w1*z2 + x1*y2 - y1*x2 + z1*w2])


class _Config:
    def __init__(self, q):
        self.q = np.array(q, float)


class AnalyticIK:
    """Drop-in for tampanda's MinkIK as used by the multilevel-blocks executor and LinearIKPlanner:
    update_configuration(qpos), set_target_position(pos, quat), converge_ik(dt) -> bool, configuration.q."""

    def __init__(self, env, mink_ik):
        self.env = env; self.mink = mink_ik
        self.model, self.data = env.model, env.data
        self.max_iters = mink_ik.max_iters             # read/written by the executor; unused here
        self.pos_threshold = getattr(mink_ik, "pos_threshold", 0.002)
        self.ori_threshold = getattr(mink_ik, "ori_threshold", 1e-3)
        self.q7s = K.q7_grid(Q7_STEP_DEG)
        m = env.model
        sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
        body = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.site_bodyid[sid])
        off = np.asarray(m.site_pos[sid], float)
        if body != "hand" or abs(off[0]) > 1e-9 or abs(off[1]) > 1e-9 or np.any(np.asarray(m.site_quat[sid]) != [1, 0, 0, 0]):
            raise RuntimeError("attachment_site is not a pure z-offset on the hand (%s, %s)" % (body, off))
        self.site_to_tcp = P.TCP_OFFSET - float(off[2])
        l0 = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "link0")
        mujoco.mj_kinematics(m, env.data)
        self.base = np.eye(4); self.base[:3, :3] = env.data.xmat[l0].reshape(3, 3); self.base[:3, 3] = env.data.xpos[l0]
        self.configuration = _Config(env.data.qpos)
        self._pos = None; self._quat = None
        self.calls = 0

    def update_configuration(self, qpos):
        self.configuration = _Config(qpos)

    def set_target_position(self, pos, quat):
        self._pos = np.asarray(pos, float).copy()
        self._quat = np.asarray(quat, float) / np.linalg.norm(quat)
        self.mink.set_target_position(pos, quat)       # keeps the mocap target marker in sync (render only)

    def solutions(self, pos, quat):
        out = []
        quats = [quat]
        att = self.env._attached
        if att is None or np.linalg.norm(np.asarray(att["rel_pos"], float)[:2]) < 1e-3:
            quats.append(_qmul(quat, FLIP))
        for q4 in quats:
            T = np.eye(4); T[:3, :3] = K.quat_to_R(q4); T[:3, 3] = pos
            T[:3, 3] = T[:3, 3] + self.site_to_tcp * T[:3, 2]
            Tb = np.linalg.inv(self.base) @ T
            frantik = K._deps.load("frantik")
            for q7 in self.q7s:
                for s in frantik.ik(Tb, float(q7), P.Q_HOME):
                    if np.isfinite(s).all() and (s >= P.Q_MIN).all() and (s <= P.Q_MAX).all():
                        out.append(s)
        return np.array(out) if out else np.zeros((0, 7))

    def converge_ik(self, dt: float = 0.005) -> bool:
        self.calls += 1
        sols = self.solutions(self._pos, self._quat)
        if not len(sols):
            return False
        seed = self.configuration.q[:7].copy()
        order = np.argsort(np.abs(sols - seed).max(1), kind="stable")
        env = self.env
        free, checked = None, 0
        for i in order:
            s = sols[i]
            if not env.is_collision_free(s):
                continue
            if free is None:
                free = s
            if checked >= MAX_LERP_CHECKS:
                break
            checked += 1
            n = max(1, int(math.ceil(float(np.max(np.abs(s - seed))) / H.DQ)))
            if env.is_path_collision_free(seed, s, steps=n):
                self._set(s); return True
        self._set(free if free is not None else sols[order[0]])
        return True

    def _set(self, s):
        q = self.configuration.q.copy(); q[:7] = s
        self.configuration = _Config(q)


def build_honest_aik_env(scratch_dir: Path, config: Optional[MultilevelBlocksConfig] = None, rate: float = 10000.0):
    return H.build_honest_env(scratch_dir, config=config, rate=rate)


def make_honest_aik_executor(env, workspace, config):
    """The honest executor (hand-offs precomputed with MinkIK), then the IK swapped for AnalyticIK."""
    ex = H.make_honest_executor(env, workspace, config)
    env.ik = AnalyticIK(env, env.ik)
    return ex


check_action_honest_aik = H.check_action_honest
