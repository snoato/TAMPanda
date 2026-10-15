"""AnalyticIK (TAMPANDA_IK=analytic): every candidate reaches its target under MuJoCo's forward kinematics,
including at the shoulder singularity, where the closed form returns q2 = 0 and misses the target."""
import os
from pathlib import Path

import mujoco
import numpy as np
import pytest

pytest.importorskip("tampanda.ik.analytic_ik")
from tampanda.ik import analytic_ik as A

try:
    A._frantik()
except ImportError:
    pytest.skip("frantik not importable (set TAMPANDA_FRANTIK_PYLIB)", allow_module_level=True)

XML = Path(A.__file__).resolve().parents[1] / "environments/assets/franka_emika_panda/scene.xml"


@pytest.fixture(scope="module")
def ik():
    m = mujoco.MjModel.from_xml_path(str(XML))
    return A.AnalyticIK(m, mujoco.MjData(m), None)


def _targets(ik, n, shoulder_sd, seed):
    rng = np.random.default_rng(seed)
    for _ in range(n):
        q = ik.q_lo + (ik.q_hi - ik.q_lo) * rng.random(7)
        if shoulder_sd is not None:
            q[1] = rng.normal(0.0, shoulder_sd)
        p, R = ik._fk(q)
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, R.flatten())
        yield q, p, quat, R


def test_raw_closed_form_misses_at_the_shoulder_singularity(ik):
    """The reason for the check: frantik's singular branch (q2 == 0 exactly) misses targets with |q2| < 2.56 deg."""
    missed = 0
    for q, p, quat, R in _targets(ik, 20, 0.02, 1):
        for s, _, _, _ in ik._raw(p, quat):
            ep, ea = ik._error(s, p, R)
            if ep > A.POS_TOL or ea > A.ORI_TOL:
                assert s[1] == 0.0
                missed += 1
    assert missed > 0


@pytest.mark.parametrize("shoulder_sd", [None, 0.02])
def test_every_solution_reaches_the_target(ik, shoulder_sd):
    for q, p, quat, R in _targets(ik, 30, shoulder_sd, 2):
        sols = ik.solutions(p, quat)
        assert len(sols)
        for s in sols:
            ep, ea = ik._error(s, p, R)
            assert ep <= A.POS_TOL and ea <= A.ORI_TOL
            assert (s >= ik.q_lo).all() and (s <= ik.q_hi).all()


def test_the_generating_configuration_is_recovered_near_the_singularity(ik):
    """A target produced by a configuration with q7 on the grid, the shoulder 0.3-2.5 deg from vertical and
    the elbow in the closed form's family (q4 < -0.6): that configuration is among the solutions (before the
    fix its branch came back with q2 = 0, off target, or as NaN). Closer to q2 = 0 the q1/q3 split is
    ill-conditioned (sub-micrometre FK error, 1e-3 rad joint differences), hence the margins."""
    rng = np.random.default_rng(4)
    for _ in range(40):
        q = ik.q_lo + (ik.q_hi - ik.q_lo) * rng.random(7)
        q[1] = rng.choice([-1, 1]) * rng.uniform(0.005, 0.044)
        q[3] = rng.uniform(ik.q_lo[3], -0.6)
        q[6] = ik.q7s[rng.integers(len(ik.q7s))]
        p, R = ik._fk(q)
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, R.flatten())
        assert np.abs(ik.solutions(p, quat) - q).max(1).min() < 1e-3


def test_overextended_elbow_targets_still_have_a_solution(ik):
    """The closed form never returns q4 above about -0.47 rad, but such targets are reached by its own family."""
    rng = np.random.default_rng(6)
    for _ in range(30):
        q = ik.q_lo + (ik.q_hi - ik.q_lo) * rng.random(7)
        q[3] = rng.uniform(-0.4, ik.q_hi[3])
        p, R = ik._fk(q)
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, R.flatten())
        sols = ik.solutions(p, quat)
        assert len(sols) and (sols[:, 3] < -0.4).all()
