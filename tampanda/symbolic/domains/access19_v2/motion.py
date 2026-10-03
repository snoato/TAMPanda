"""Access-19 v2 transit planning: VAMP proposes, exact MuJoCo geometry decides (local override of tampanda's RRT).

VAMP (vamp-planner 0.6.4) plans on a sphere model of the Panda against cuboids, in milliseconds. Every proposal is
re-checked densely (``FINE`` rad) against the exact collision meshes of ``shelf.Scene`` with clearance
>= the requested margin; a failed check is retried with more obstacle padding and other seeds.

The planned start of a transit is the outer end of a grasp path, which is often too close to the shelf for VAMP's
padded model; ``backoff`` first moves the hand straight out (fixed orientation, analytic IK nearest to the previous
configuration, exact checks) until VAMP accepts the configuration. ``shortcut`` then replaces stretches of a
validated path by straight joint-space segments that keep >= 1 cm clearance.
"""
from __future__ import annotations

import time

import numpy as np

from tampanda.kinematics import panda as P
from tampanda.symbolic.domains.access19_v2 import _deps
from tampanda.symbolic.domains.access19_v2 import kinematics as K
from tampanda.symbolic.domains.access19_v2.grasp import FINE, dense
from tampanda.symbolic.domains.access19_v2.shelf import centre

EE_FROM_SITE = 0.105 - K.ATTACH_OFFSET   # VAMP end effector (panda_grasptarget) = hand + 0.105 m along z
PAD = 0.01                               # VAMP obstacle inflation (m): proposals keep ~1 cm away
MIN_BACKOFF = 0.05                       # hand over to VAMP at least this far out of the grasp path start


def _vamp():
    return _deps.load("vamp")


def vamp_env(sc, occupied, pad=0.0):
    vamp = _vamp(); g = sc.g; env = vamp.Environment()
    for _, lo, hi in sc.boxes:           # shelf: exact (the robot base stands close to the shelf)
        env.add_cuboid(vamp.Cuboid(list((lo + hi) / 2), [0, 0, 0], list((hi - lo) / 2)))
    for k in occupied:
        c = centre(g, k)
        env.add_cuboid(vamp.Cuboid([c[0], c[1], c[2] + g.h / 2], [0, 0, 0], [g.r + pad, g.r + pad, g.h / 2 + pad]))
    return env


def attach(env, sc, held_local, R=None):
    """Held box as spheres (radius = its horizontal half-diagonal) in VAMP's end-effector frame."""
    vamp = _vamp(); g = sc.g; R = K.grasp_R(g) if R is None else R
    T = np.eye(4, dtype=np.float32); T[:3, 3] = np.asarray(held_local) - np.array([0, 0, EE_FROM_SITE])
    a = vamp.Attachment(np.asfortranarray(T))
    axis = R.T @ np.array([0, 0, 1.0])
    rad = g.r * np.sqrt(2)
    n = max(2, int(np.ceil(g.h / rad)) + 1)
    a.add_spheres([vamp.Sphere(list(axis * s), rad) for s in np.linspace(-g.h / 2 + rad, g.h / 2 - rad, n)])
    env.attach(a)


def mj_check(sc, Q, finger, held_local, margin, step=FINE):
    """Dense exact check against the current occupancy: (ok, smallest clearance)."""
    worst = 1.0
    for q in dense(Q, step):
        cl = sc.clearance(q, finger, held_local); worst = min(worst, cl)
        if cl < margin:
            return False, worst
    return True, worst


def backoff(sc, q, direction, finger, held_local, env, q7s, margin, R=None, step=0.01, max_d=0.30):
    VP = _vamp().panda; R = K.grasp_R(sc.g) if R is None else R
    seg = [np.asarray(q, float)]
    site0 = sc.site_T(q)[:3, 3]
    for k in range(1, int(max_d / step) + 1):
        if (k - 1) * step >= MIN_BACKOFF and VP.validate(np.asarray(seg[-1], np.float32), env):
            return seg
        T = np.eye(4); T[:3, :3] = R; T[:3, 3] = site0 + k * step * np.asarray(direction, float)
        Q = K.ik_all(sc.base, T, q7s)
        stop = not len(Q)
        if not stop:
            dq = np.abs(Q - seg[-1]).max(1); i = int(np.argmin(dq))
            stop = dq[i] > 0.2 or not mj_check(sc, [seg[-1], Q[i]], finger, held_local, margin)[0]
        if stop:   # cannot back off further: keep what we have if VAMP accepts it
            return seg if VP.validate(np.asarray(seg[-1], np.float32), env) else None
        seg.append(Q[i])
    return seg if VP.validate(np.asarray(seg[-1], np.float32), env) else None


def rrtc(start, goal, env, seed=0, tries=4):
    vamp = _vamp(); VP = vamp.panda
    rng = VP.halton() if seed == 0 else VP.xorshift()
    for _ in range(seed):
        if hasattr(rng, "next"):
            rng.next()
    s = vamp.RRTCSettings(); ss = vamp.SimplifySettings()
    for _ in range(tries):
        r = VP.rrtc(np.asarray(start, np.float32), np.asarray(goal, np.float32), env, s, rng)
        if r.solved:
            p = VP.simplify(r.path, env, ss, rng).path; p.interpolate_to_resolution(VP.resolution())
            return np.array(p.numpy(), float)
    return None


def plan_validated(sc, q_path_end, goal, occupied, held_local, finger, direction, q7s, margin, reverse=False):
    """Transit between a grasp path's outer end and ``goal`` in the scene ``occupied``.
    reverse=False: goal -> q_path_end; reverse=True: q_path_end -> goal. Returns the path or None."""
    for pad in (PAD, 2 * PAD, 3 * PAD):
        sc.occupy(occupied); env = vamp_env(sc, occupied, pad)
        if held_local is not None:
            attach(env, sc, held_local)
        b = None
        for dvec in (direction, (np.asarray(direction, float) + np.array([0, 0, 1.0])) / np.sqrt(2), [0, 0, 1],
                     [0, -1, 0]):
            b = backoff(sc, q_path_end, dvec, finger, held_local, env, q7s, margin)
            if b:
                break
        if not b:
            continue
        for seed in range(3):
            p = rrtc(b[-1], goal, env, seed) if reverse else rrtc(goal, b[-1], env, seed)
            if p is None:
                continue
            full = np.vstack([np.array(b), p]) if reverse else np.vstack([p, np.array(b[::-1])])
            sc.occupy(occupied)
            if mj_check(sc, full, finger, held_local, margin)[0]:
                return full
    return None


def shortcut(sc, path, occupied, held_local, finger, margin, tries=80, keep=0.01, seed=0):
    """Exact-geometry shortcutting: replace Q[i..j] by the straight segment when it keeps >= ``keep`` clearance;
    the result is re-validated at ``margin`` (falls back to the input path)."""
    Q = [np.asarray(x, float) for x in path]; rng = np.random.default_rng(seed); sc.occupy(occupied)
    for _ in range(tries):
        if len(Q) < 3:
            break
        i, j = sorted(rng.choice(len(Q), 2, replace=False))
        if j - i >= 2 and mj_check(sc, [Q[i], Q[j]], finger, held_local, keep)[0]:
            Q = Q[:i + 1] + Q[j:]
    Q = np.array(Q)
    return Q if mj_check(sc, Q, finger, held_local, margin)[0] else np.asarray(path, float)


def joint_length(Q) -> float:
    return float(np.linalg.norm(np.diff(np.asarray(Q), axis=0), axis=1).sum()) if len(Q) > 1 else 0.0


HOME = P.Q_HOME.copy()
