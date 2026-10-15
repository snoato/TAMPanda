"""Closed-form inverse kinematics for the Panda (He & Liu 2021, via the `frantik` package).

A drop-in for `MinkIK` as the planners use it: `update_configuration(qpos)`,
`set_target_position(pos, quat)`, `converge_ik(dt) -> bool`, `configuration.q`.

MinkIK runs damped least squares from one seed and so returns one solution, in the seed's branch.
This solver enumerates the closed-form solutions over a grid of joint-7 values (every branch the
closed form yields per value), keeps those within the model's joint ranges, and returns the one
nearest the seed (max-abs joint distance) whose configuration is collision-free in the environment
(`env.is_collision_free`); if none is, the nearest one. Every returned solution is checked against
the target with MuJoCo's own forward kinematics on a scratch copy of the data.

Selected by `FrankaEnv` when the environment variable TAMPANDA_IK=analytic; `frantik` is imported
from sys.path or from the directories in TAMPANDA_FRANTIK_PYLIB (os.pathsep-separated, appended).
Frames and the joint-7 grid follow `access19_v2.kinematics` / `multilevel_blocks.honest_aik`
(manipulation, release/v1.1.0): the target is the `attachment_site`, a pure z-offset on `hand`;
frantik's end effector is hand + TCP_OFFSET along hand z, in the frame of `link0`.
"""
from __future__ import annotations

import importlib
import os
import sys

import mujoco
import numpy as np

TCP_OFFSET = 0.1034                  # flange -> TCP along hand z (Franka Hand), frantik's EE frame
Q_HOME = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853])
Q7_STEP_DEG = float(os.environ.get("TAMPANDA_IK_Q7_STEP_DEG", "3.0"))
POS_TOL, ORI_TOL = 1e-4, 1e-3        # forward-kinematics check of every returned solution


def _frantik():
    try:
        return importlib.import_module("frantik")
    except ImportError:
        for p in os.environ.get("TAMPANDA_FRANTIK_PYLIB", "").split(os.pathsep):
            if p and p not in sys.path:
                sys.path.append(p)
        return importlib.import_module("frantik")


def _quat_to_R(q):
    w, x, y, z = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


class _Config:
    def __init__(self, q):
        self.q = np.array(q, float)


class AnalyticIK:
    def __init__(self, model, data, env=None, target_name: str = "target"):
        self.model, self.data, self.env = model, data, env
        self.target_name = target_name
        self.max_iters = 1000                    # read/written by planners; unused here
        self.pos_threshold, self.ori_threshold = 0.002, 1e-3
        self._fr = _frantik()
        m = model
        self.site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
        body = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.site_bodyid[self.site])
        off = np.asarray(m.site_pos[self.site], float)
        if body != "hand" or abs(off[0]) > 1e-9 or abs(off[1]) > 1e-9 or np.any(np.asarray(m.site_quat[self.site]) != [1, 0, 0, 0]):
            raise RuntimeError("attachment_site is not a pure z-offset on the hand (%s, %s)" % (body, off))
        self.site_to_tcp = TCP_OFFSET - float(off[2])
        self.q_lo, self.q_hi = m.jnt_range[:7, 0].copy(), m.jnt_range[:7, 1].copy()
        self.q7s = np.arange(self.q_lo[6] + 1e-3, self.q_hi[6], np.radians(Q7_STEP_DEG))
        l0 = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "link0")
        scratch = mujoco.MjData(m)
        scratch.qpos[:] = data.qpos
        mujoco.mj_kinematics(m, scratch)
        self.base = np.eye(4)
        self.base[:3, :3] = scratch.xmat[l0].reshape(3, 3)
        self.base[:3, 3] = scratch.xpos[l0]
        self._scratch = scratch
        self.configuration = _Config(data.qpos)
        self._pos = self._quat = None
        self.calls = self.solutions_found = 0

    def update_configuration(self, qpos):
        self.configuration = _Config(qpos)

    def set_target_position(self, pos, quat):
        self._pos = np.asarray(pos, float).copy()
        self._quat = np.asarray(quat, float) / np.linalg.norm(quat)
        self.data.mocap_pos[0] = pos                  # the target marker, as MinkIK sets it
        self.data.mocap_quat[0] = quat

    def solutions(self, pos, quat):
        T = np.eye(4)
        T[:3, :3] = _quat_to_R(quat)
        T[:3, 3] = np.asarray(pos, float) + self.site_to_tcp * T[:3, 2]
        Tb = np.linalg.inv(self.base) @ T
        out = []
        for q7 in self.q7s:
            for s in self._fr.ik(Tb, float(q7), Q_HOME):
                s = np.asarray(s, float)
                if np.isfinite(s).all() and (s >= self.q_lo).all() and (s <= self.q_hi).all():
                    out.append(s)
        return np.array(out) if out else np.zeros((0, 7))

    def _reaches(self, s):
        d = self._scratch
        d.qpos[:] = self.configuration.q if len(self.configuration.q) == self.model.nq else d.qpos
        d.qpos[:7] = s
        mujoco.mj_kinematics(self.model, d)
        p = d.site_xpos[self.site]
        R = d.site_xmat[self.site].reshape(3, 3)
        Rt = _quat_to_R(self._quat)
        ang = np.arccos(np.clip((np.trace(Rt.T @ R) - 1.0) / 2.0, -1.0, 1.0))
        return np.linalg.norm(p - self._pos) <= POS_TOL and ang <= ORI_TOL

    def converge_ik(self, dt: float = 0.005) -> bool:
        self.calls += 1
        sols = self.solutions(self._pos, self._quat)
        if not len(sols):
            return False
        seed = self.configuration.q[:7].copy()
        order = np.argsort(np.abs(sols - seed).max(1), kind="stable")
        pick = None
        for i in order:
            if self.env is None or self.env.is_collision_free(sols[i]):
                pick = sols[i]
                break
        if pick is None:
            pick = sols[order[0]]
        if not self._reaches(pick):
            raise RuntimeError("analytic IK: the returned solution does not reach the target "
                               "(frame convention broken)")
        self.solutions_found += 1
        q = self.configuration.q.copy()
        q[:7] = pick
        self.configuration = _Config(q)
        return True
