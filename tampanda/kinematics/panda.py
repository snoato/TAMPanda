"""Franka Emika Panda kinematics: the single source of truth for the Access-19 v2 rebuild (2026-10-01).

Everything here follows Franka's published "Robot and interface specifications" for the Panda (FER):
Denavit-Hartenberg parameters in Craig's (modified) convention, joint and Cartesian limits, and the
Franka Hand geometry. ``verify_panda.py`` checks this module against the MuJoCo model tampanda simulates
(``environments/assets/franka_emika_panda/panda.xml``), so the analytic model and the simulator are proven
to agree before the shelf is designed around it.

Frames
------
``base``    link0 frame (robot mounting flange on the table).
``link{i}`` frame i of the DH chain, i = 1..7 (coincides with the MJCF body of the same name).
``flange``  frame 8: link7 translated d = 0.107 m along z.
``hand``    flange rotated by -pi/4 about z (Franka Hand mounting; MJCF body ``hand``).
``tcp``     hand translated ``TCP_OFFSET`` along z: the point between the fingertips.

All lengths in metres, angles in radians, poses as 4x4 homogeneous matrices.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# --------------------------------------------------------------------------------------------------
# Denavit-Hartenberg (Craig convention):  T_{i-1,i} = RotX(alpha_i) TransX(a_i) RotZ(theta_i) TransZ(d_i)
# rows: (a, d, alpha); theta_i = q_i for the seven joints, 0 for the flange.
DH = np.array([
    [0.0,     0.333, 0.0],          # joint 1
    [0.0,     0.0,   -np.pi / 2],   # joint 2
    [0.0,     0.316, np.pi / 2],    # joint 3
    [0.0825,  0.0,   np.pi / 2],    # joint 4
    [-0.0825, 0.384, -np.pi / 2],   # joint 5
    [0.0,     0.0,   np.pi / 2],    # joint 6
    [0.088,   0.0,   np.pi / 2],    # joint 7
])
FLANGE_D = 0.107                    # flange: a = 0, d = 0.107, alpha = 0, theta = 0
HAND_YAW = -np.pi / 4               # Franka Hand is mounted rotated -45 deg about the flange z axis
TCP_OFFSET = 0.1034                 # flange -> TCP along hand z (Franka Hand default, fingertip centre)

# --------------------------------------------------------------------------------------------------
# Joint-space limits (Panda datasheet).
Q_MIN = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
Q_MAX = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
QD_MAX = np.array([2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100])          # rad/s
QDD_MAX = np.array([15.0, 7.5, 10.0, 12.5, 15.0, 20.0, 20.0])                         # rad/s^2
QDDD_MAX = np.array([7500.0, 3750.0, 5000.0, 6250.0, 7500.0, 10000.0, 10000.0])       # rad/s^3
TAU_MAX = np.array([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0])                        # N m
# Cartesian limits of the end effector (datasheet).
V_TRANS_MAX = 1.7       # m/s
V_ROT_MAX = 2.5         # rad/s
V_ELBOW_MAX = 2.175     # rad/s
REACH = 0.855           # m, maximum reach (datasheet)

# Franka Hand (gripper).
GRIPPER_MAX_WIDTH = 0.08            # m, finger opening
GRIPPER_FINGER_MAX = 0.04           # m per finger (MJCF slide joint range 0..0.04)

# Tampanda's arm home pose (FrankaEnvironment.HOME_QPOS, arm joints only).
Q_HOME = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853])


def _rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1.0]])


def _rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1.0]])


def _trans(x=0.0, y=0.0, z=0.0):
    T = np.eye(4); T[:3, 3] = (x, y, z)
    return T


def _dh(a, d, alpha, theta):
    return _rot_x(alpha) @ _trans(x=a) @ _rot_z(theta) @ _trans(z=d)


@dataclass
class Frames:
    """Poses of every frame of the chain, all expressed in the base (link0) frame."""
    links: list           # links[i] = T_base_link{i+1}, i = 0..6
    flange: np.ndarray
    hand: np.ndarray
    tcp: np.ndarray


def fk(q, base=np.eye(4)) -> Frames:
    """Forward kinematics. ``q``: 7 joint angles. ``base``: world pose of link0 (default identity)."""
    q = np.asarray(q, float).reshape(7)
    T = np.array(base, float)
    links = []
    for (a, d, alpha), qi in zip(DH, q):
        T = T @ _dh(a, d, alpha, qi)
        links.append(T)
    flange = T @ _trans(z=FLANGE_D)
    hand = flange @ _rot_z(HAND_YAW)
    tcp = hand @ _trans(z=TCP_OFFSET)
    return Frames(links, flange, hand, tcp)


def jacobian(q, frame="tcp", base=np.eye(4)) -> np.ndarray:
    """Geometric 6x7 Jacobian [v; w] of ``frame`` ('tcp', 'hand' or 'flange'), base-frame coordinates."""
    F = fk(q, base)
    p = getattr(F, frame)[:3, 3]
    J = np.zeros((6, 7))
    for i, T in enumerate(F.links):
        z, o = T[:3, 2], T[:3, 3]           # joint i+1 rotates about its own frame's z axis
        J[:3, i] = np.cross(z, p - o)
        J[3:, i] = z
    return J


def within_limits(q, margin=0.0) -> bool:
    q = np.asarray(q, float)
    return bool(np.all(q >= Q_MIN + margin) and np.all(q <= Q_MAX - margin))


def manipulability(q, frame="tcp") -> float:
    """Yoshikawa manipulability sqrt(det(J J^T)); 0 at singularities."""
    J = jacobian(q, frame)
    return float(np.sqrt(max(np.linalg.det(J @ J.T), 0.0)))


def random_q(rng, n=1, margin=0.0):
    """Uniform samples inside the joint limits (optionally shrunk by ``margin``)."""
    return rng.uniform(Q_MIN + margin, Q_MAX - margin, size=(n, 7))
