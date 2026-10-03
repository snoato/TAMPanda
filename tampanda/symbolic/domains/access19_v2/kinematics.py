"""Access-19 v2 kinematics: the one grasp frame and closed-form IK (local override of tampanda's numeric IK).

tampanda's generic IK (damped least squares from a seed) returns ONE solution near its seed, so a collision-free
arm configuration for a deep shelf cell is found only if the seed happens to be in the right branch. Here every
query returns ALL solutions: joint 7 is sampled on a grid and frantik's closed form (He & Liu 2021) yields up to
four branches per value, filtered by the joint limits.

Frames: ``site`` = hand + ATTACH_OFFSET along the hand z axis (tampanda's attachment_site); ``tcp`` = hand +
P.TCP_OFFSET (frantik's end-effector frame). The one grasp puts the site at
``(x_cell, y_cell - GRASP_CONTACT_OFFSET, box_top - grasp_depth)`` with orientation ``FRONT_QUAT`` (hand
horizontal, approach +y, palm facing the shelf).
"""
from __future__ import annotations

import numpy as np

from tampanda.kinematics import panda as P
from tampanda.symbolic.domains.access19_v2 import _deps

GRASP_CONTACT_OFFSET = 0.0137        # site -> object centre along +y at the grasp
ATTACH_OFFSET = 0.09                 # attachment_site: hand frame + 0.09 m along z
TCP_FROM_SITE = P.TCP_OFFSET - ATTACH_OFFSET
FRONT_QUAT = np.array([-0.5, 0.5, 0.5, 0.5])   # (w, x, y, z)
HELD_LIFT = 0.004                    # a held object rides 4 mm above its resting height


def quat_to_R(q) -> np.ndarray:
    w, x, y, z = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


R_FRONT = quat_to_R(FRONT_QUAT)
FRONT_QUAT_FLIPPED = np.array([-0.5, 0.5, -0.5, -0.5])   # the same grasp with the hand rotated 180 deg about the approach


def grasp_R(g) -> np.ndarray:
    """Hand orientation of the one grasp of geometry ``g`` (``g.flip`` selects the hand roll)."""
    return quat_to_R(FRONT_QUAT_FLIPPED if g.flip else FRONT_QUAT)


def site_T(q, base) -> np.ndarray:
    F = P.fk(q, base)
    T = F.hand.copy(); T[:3, 3] = F.hand[:3, 3] + ATTACH_OFFSET * F.hand[:3, 2]
    return T


def q7_grid(step_deg: float) -> np.ndarray:
    return np.arange(P.Q_MIN[6] + 1e-3, P.Q_MAX[6], np.radians(step_deg))


def ik_all(base, T_site, q7s) -> np.ndarray:
    """All joint-limit-respecting IK solutions (n x 7) for a site pose in the world frame."""
    frantik = _deps.load("frantik")
    T = np.array(T_site, float); T[:3, 3] = T[:3, 3] + TCP_FROM_SITE * T[:3, 2]
    Tb = np.linalg.inv(base) @ T
    out = []
    for q7 in q7s:
        for s in frantik.ik(Tb, float(q7), P.Q_HOME):
            if np.isfinite(s).all() and (s >= P.Q_MIN).all() and (s <= P.Q_MAX).all():
                out.append(s)
    return np.array(out) if out else np.zeros((0, 7))
