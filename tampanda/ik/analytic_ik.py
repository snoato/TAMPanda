"""Closed-form inverse kinematics for the Panda (He & Liu 2021, via the `frantik` package).

A drop-in for `MinkIK` as the planners use it: `update_configuration(qpos)`,
`set_target_position(pos, quat)`, `converge_ik(dt) -> bool`, `configuration.q`.

MinkIK runs damped least squares from one seed and so returns one solution, in the seed's branch.
This solver enumerates the closed-form solutions over a grid of joint-7 values (every branch the
closed form yields per value), keeps those within the model's joint ranges that reach the target
under MuJoCo's own forward kinematics, and returns the one nearest the seed (max-abs joint distance)
whose configuration is collision-free in the environment (`env.is_collision_free`); if none is, the
nearest one.

The shoulder singularity. When the shoulder is within acos(0.999) = 2.56 deg of vertical
(|cos q2| > 0.999) the closed form does not solve for joints 1 and 2: it returns q1 = q_actual[0],
q2 = 0 exactly and derives q3 and q5 from those, so the configuration misses the target by up to a
few centimetres (q1, q3 are only interchangeable at q2 = 0 itself); and when q3 then leaves its
range the output is NaN, so the closed form is also asked with other q_actual[0] values (`_raw`).
Every candidate is checked with forward kinematics; one that fails is refined by Gauss-Newton on
joints 1, 2, 3, 5 (q4, q6, q7 are exact; MuJoCo's site Jacobian), and the refined configuration and
its shoulder mirror (q1 + pi, -q2, q3 + pi) are kept if they are within the joint ranges and pass the
same check. Candidates that still fail are dropped (counted in `n_dropped`).

After a failed solve. MinkIK leaves its last iterate in `configuration` and the planners read the
residual through `ik.ee_task.compute_error(configuration)` (rrt_star._get_ik_error_details): which
threshold failed, position or orientation, decides retries downstream (e.g. a lower place approach
when the position is out of reach). This solver does the same: when no checked candidate exists,
`configuration` is set to a best-effort configuration (damped least squares on all seven joints from
the seed, within the joint ranges) and `ee_task.compute_error` reports [position error, rotation
vector] of any configuration against the current target.

Not covered: the closed form yields one elbow family only, q4 below about -0.47 rad; the Panda's
q4 range reaches -0.07 rad (elbow over-extended). Targets there have a solution in the covered family
with the same shoulder-wrist distance, so reach is not lost, but those postures are never returned.

The IK `FrankaEnv` uses by default (TAMPANDA_IK=mink selects MinkIK); `frantik` is imported
from sys.path or from the directories in TAMPANDA_FRANTIK_PYLIB (os.pathsep-separated, appended).
TAMPANDA_IK_LOG=<path> appends one JSON line per `converge_ik` call to <path>.<pid> (target, seed,
the pick, its forward-kinematics error, the counts above, the pick the solver made before candidates
were checked with its error, the calling functions, and TAMPANDA_IK_LOG_TAG if set), for diagnosis.
Frames and the joint-7 grid follow `access19_v2.kinematics` / `multilevel_blocks.honest_aik`
(manipulation, release/v1.1.0): the target is the `attachment_site`, a pure z-offset on `hand`;
frantik's end effector is hand + TCP_OFFSET along hand z, in the frame of `link0`.
"""
from __future__ import annotations

import importlib
import json
import os
import sys
import time

import mujoco
import numpy as np

TCP_OFFSET = 0.1034                  # flange -> TCP along hand z (Franka Hand), frantik's EE frame
Q_HOME = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853])
Q7_STEP_DEG = float(os.environ.get("TAMPANDA_IK_Q7_STEP_DEG", "3.0"))
POS_TOL, ORI_TOL = 1e-4, 1e-3        # forward-kinematics check of every candidate (m, rad)
REFINE_ITERS, REFINE_DAMPING = 50, 1e-10
REFINE_JOINTS = (0, 1, 2, 4)          # joints 1, 2, 3, 5 (0-based): the ones the singular output gets wrong
SINGULAR_Q1 = (-2.8, -1.4, 1.4, 2.8)  # extra q_actual[0] values for the singular output (see _raw)
BEST_EFFORT_ITERS, BEST_EFFORT_DAMPING = 200, 1e-3   # after a failed solve (see _best_effort)


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


def _wrap(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


class _Config:
    def __init__(self, q):
        self.q = np.array(q, float)


class _EETask:
    """The part of MinkIK's end-effector task the planners read: `compute_error(configuration)`, the
    6-vector [site position error (m), rotation-vector error (rad)] against the current target."""

    def __init__(self, ik):
        self._ik = ik

    def compute_error(self, configuration):
        ik = self._ik
        return ik._residual(np.asarray(configuration.q, float)[:7], ik._pos, _quat_to_R(ik._quat))


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
        self._jp, self._jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        self.configuration = _Config(data.qpos)
        self._pos = self._quat = None
        self.calls = self.solutions_found = 0
        self.n_candidates = self.n_repaired = self.n_dropped = 0
        self._log = os.environ.get("TAMPANDA_IK_LOG") or None
        self.ee_task = _EETask(self)

    def update_configuration(self, qpos):
        self.configuration = _Config(qpos)

    def set_target_position(self, pos, quat):
        self._pos = np.asarray(pos, float).copy()
        self._quat = np.asarray(quat, float) / np.linalg.norm(quat)
        self.data.mocap_pos[0] = pos                  # the target marker, as MinkIK sets it
        self.data.mocap_quat[0] = quat

    # -- forward kinematics on the scratch data ------------------------------------------------
    def _fk(self, s):
        d = self._scratch
        if len(self.configuration.q) == self.model.nq:
            d.qpos[:] = self.configuration.q
        d.qpos[:7] = s
        mujoco.mj_kinematics(self.model, d)
        return d.site_xpos[self.site].copy(), d.site_xmat[self.site].reshape(3, 3).copy()

    def _error(self, s, pos, R):
        p, Rs = self._fk(s)
        ang = np.arccos(np.clip((np.trace(R.T @ Rs) - 1.0) / 2.0, -1.0, 1.0))
        return float(np.linalg.norm(p - pos)), float(ang)

    def _residual(self, s, pos, R):
        """[target - site position, rotation vector taking the site orientation to the target]."""
        p, Rs = self._fk(s)
        qe = np.zeros(4)
        mujoco.mju_mat2Quat(qe, (R @ Rs.T).flatten())
        w = np.zeros(3)
        mujoco.mju_quat2Vel(w, qe, 1.0)
        return np.r_[pos - p, w]

    def _gauss_newton(self, s, pos, R, joints, iters, damping, clip=False):
        m, d, q = self.model, self._scratch, np.array(s, float)
        free = list(joints)
        for _ in range(iters):
            e = self._residual(q, pos, R)            # leaves the scratch data at q
            if np.linalg.norm(e[:3]) < 1e-10 and np.linalg.norm(e[3:]) < 1e-10:
                break
            mujoco.mj_comPos(m, d)
            mujoco.mj_jacSite(m, d, self._jp, self._jr, self.site)
            J = np.r_[self._jp[:, free], self._jr[:, free]]
            q[free] += np.linalg.solve(J.T @ J + damping * np.eye(len(free)), J.T @ e)
            if clip:
                q[:7] = np.clip(q[:7], self.q_lo, self.q_hi)
        return q

    def _refine(self, s, pos, R):
        """Gauss-Newton (lightly damped) on joints 1, 2, 3 and 5 towards the target. q4, q6 and q7 are
        exact in the closed form's singular output (they depend on the wrist geometry only), so they stay."""
        return self._gauss_newton(s, pos, R, REFINE_JOINTS, REFINE_ITERS, REFINE_DAMPING)

    def _best_effort(self, seed, pos, R):
        """Where a failed solve leaves `configuration` (as MinkIK leaves its last iterate): damped least
        squares on all seven joints from the seed, kept within the joint ranges."""
        return self._gauss_newton(seed, pos, R, range(7), BEST_EFFORT_ITERS, BEST_EFFORT_DAMPING, clip=True)

    def _in_range(self, s):
        return bool((s >= self.q_lo).all() and (s <= self.q_hi).all())

    # -- candidates -------------------------------------------------------------------------------
    def _raw(self, pos, quat):
        """Every finite closed-form output over the joint-7 grid, as (q, q7 index, branch, q_actual index).

        q_actual index 0 is the call with Q_HOME. The others re-ask with q_actual[0] set to each of
        SINGULAR_Q1 and keep only singular outputs (q2 == 0 exactly): there the closed form sets
        q1 = q_actual[0] and q3 absorbs the rest of q1 + q3, which can push q3 out of its range and the
        output to NaN. Non-singular outputs do not depend on q_actual."""
        T = np.eye(4)
        T[:3, :3] = _quat_to_R(quat)
        T[:3, 3] = np.asarray(pos, float) + self.site_to_tcp * T[:3, 2]
        Tb = np.linalg.inv(self.base) @ T
        out = []
        for k, q7 in enumerate(self.q7s):
            for j, q1 in enumerate((Q_HOME[0],) + SINGULAR_Q1):
                qa = Q_HOME.copy()
                qa[0] = q1
                for b, s in enumerate(self._fr.ik(Tb, float(q7), qa)):
                    s = np.asarray(s, float)
                    if np.isfinite(s).all() and (j == 0 or s[1] == 0.0):
                        out.append((s, k, b, j))
        return out

    def _checked(self, raw, pos, R):
        """Candidates within the joint ranges that pass the forward-kinematics check (refined if needed)."""
        sols, meta, seen = [], [], set()
        for s, k, b, j in raw:
            ep, ea = self._error(s, pos, R)
            if ep <= POS_TOL and ea <= ORI_TOL:
                key = tuple(np.round(s, 6))
                if self._in_range(s) and key not in seen:
                    seen.add(key)
                    sols.append(s); meta.append((k, b, "exact"))
                continue
            r = self._refine(s, pos, R)
            mirror = r.copy()
            mirror[0] += np.pi; mirror[1] = -mirror[1]; mirror[2] += np.pi
            kept = 0
            for v in (r, mirror):
                v = v.copy()
                v[0], v[2] = _wrap(v[0]), _wrap(v[2])
                if self._in_range(v):
                    ep2, ea2 = self._error(v, pos, R)
                    if ep2 <= POS_TOL and ea2 <= ORI_TOL:
                        kept += 1
                        key = tuple(np.round(v, 6))
                        if key not in seen:
                            seen.add(key)
                            sols.append(v); meta.append((k, b, "refined"))
            if kept:
                self.n_repaired += 1
            else:
                self.n_dropped += 1
        self.n_candidates += len(raw)
        return (np.array(sols) if sols else np.zeros((0, 7))), meta

    def solutions(self, pos, quat):
        pos = np.asarray(pos, float)
        return self._checked(self._raw(pos, quat), pos, _quat_to_R(quat))[0]

    def _reaches(self, s):
        ep, ea = self._error(s, self._pos, _quat_to_R(self._quat))
        return ep <= POS_TOL and ea <= ORI_TOL

    def _nearest_free(self, sols, seed, free):
        """Nearest to the seed (max-abs) that is collision-free, else the nearest; index or None."""
        if not len(sols):
            return None
        order = np.argsort(np.abs(sols - seed).max(1), kind="stable")
        for i in order:
            key = sols[i].tobytes()
            if key not in free:
                free[key] = self.env is None or bool(self.env.is_collision_free(sols[i]))
            if free[key]:
                return int(i)
        return int(order[0])

    def converge_ik(self, dt: float = 0.005) -> bool:
        t0 = time.perf_counter()
        self.calls += 1
        pos, R = self._pos, _quat_to_R(self._quat)
        seed = self.configuration.q[:7].copy()
        rep0, drop0 = self.n_repaired, self.n_dropped
        raw = self._raw(pos, self._quat)
        sols, meta = self._checked(raw, pos, R)
        free = {}                                   # collision results shared by both picks below
        i = self._nearest_free(sols, seed, free)
        if i is not None and not self._reaches(sols[i]):
            raise RuntimeError("analytic IK: the returned solution does not reach the target "
                               "(frame convention broken)")
        if i is None:
            q = self.configuration.q.copy()
            q[:7] = self._best_effort(seed, pos, R)
            self.configuration = _Config(q)
        if self._log:
            self._write_log(raw, sols, meta, i, seed, pos, R, free, rep0, drop0, time.perf_counter() - t0)
        if i is None:
            return False
        self.solutions_found += 1
        q = self.configuration.q.copy()
        q[:7] = sols[i]
        self.configuration = _Config(q)
        return True

    def _write_log(self, raw, sols, meta, i, seed, pos, R, free, rep0, drop0, seconds):
        # The pick made before candidates were checked: the Q_HOME call's outputs within range, same rule.
        inr = [(s, k, b) for s, k, b, jq in raw if jq == 0 and self._in_range(s)]
        arr = np.array([s for s, _, _ in inr]) if inr else np.zeros((0, 7))
        j = self._nearest_free(arr, seed, free)
        fr, callers = sys._getframe(2), []
        while fr is not None and len(callers) < 4:
            callers.append(fr.f_code.co_name)
            fr = fr.f_back
        rec = {"tag": os.environ.get("TAMPANDA_IK_LOG_TAG"), "callers": callers, "call": self.calls,
               "pos": pos.tolist(), "quat": self._quat.tolist(), "seed": seed.tolist(), "n_raw": len(raw), "n_raw_in_range": len(inr), "n_checked": int(len(sols)),
               "n_refined_kept": self.n_repaired - rep0, "n_dropped": self.n_dropped - drop0,
               "seconds": seconds}
        if j is not None:
            s, k, b = inr[j]
            ep, ea = self._error(s, pos, R)
            rec["unchecked_pick"] = {"q": s.tolist(), "q7": float(self.q7s[k]), "branch": b,
                                     "pos_err": ep, "ang_err": ea, "passes": ep <= POS_TOL and ea <= ORI_TOL}
            rec["any_unchecked_passes"] = any(
                (lambda e: e[0] <= POS_TOL and e[1] <= ORI_TOL)(self._error(x, pos, R)) for x, _, _ in inr)
        if i is None:
            e = self._residual(self.configuration.q[:7], pos, R)
            rec["best_effort"] = {"q": self.configuration.q[:7].tolist(), "pos_err": float(np.linalg.norm(e[:3])),
                                  "ang_err": float(np.linalg.norm(e[3:]))}
        if i is not None:
            ep, ea = self._error(sols[i], pos, R)
            k, b, kind = meta[i]
            rec["pick"] = {"q": sols[i].tolist(), "q7": float(self.q7s[k]), "branch": b, "kind": kind,
                           "pos_err": ep, "ang_err": ea}
        with open("%s.%d" % (self._log, os.getpid()), "a") as fh:
            fh.write(json.dumps(rec) + "\n")
