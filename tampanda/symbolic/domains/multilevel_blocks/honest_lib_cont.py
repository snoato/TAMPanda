"""Grasp-continuous certification of 3D-stacking plans (2026-10-05, prototype; honest_lib 2026-10-05b is unchanged).

Problem it addresses: every per-action check so far (the original fast check, honest, honest_aik, honest_lib, and the
dataset generator's check_action_sequence(per_action_restore=True)) restores the symbolic state before each action and
poses a held block CANONICALLY (state._attach_held_block). The in-hand pose a plan actually produces differs:
- make-upright-from-x (a rigid wrist rotation of the top-down centre grasp) leaves the EE 14.2 mm from the block centre
  on the approach axis, the canonical held-upright puts the EE at the top cell (20.7 mm apart);
- turn-long-x-to-y keeps the fingers across the block's 30 mm width, the canonical held-flat-y puts the finger axis along
  the 90 mm long axis (fingers 23 mm inside the block: physically impossible, unseen because gripper vs held block is
  self-contact). So per-action certificates do not chain.

Continuity certificate: the plan is certified ONCE, in order, on one world -- no restore between actions. The world
(arm, gripper, every block, the in-hand pose = grasp point + orientation) is exactly what the previous certified motion
left. Per action:
  * pick-upright / put-upright (+long) / put-cube: the honest_lib library-style search (analytic all-branch IK, branch
    choice over the whole approach line, VAMP transits with exact dense re-checks, release check), but a put uses the
    ACTUAL in-hand pose (no canonical top-cell pose; the 8 hand orientations rotate hand + block rigidly);
  * re-orientation actions (make-upright-from-x/y, make-flat-*-from-upright, turn-*, long variants): the executor's own
    realisation, a rigid wrist rotation at the hand-off pose with the block attached, dense-checked (no teleport);
  * every other action: the honest executor chain (it measures the actual held offset);
  * after every action the arm must return to the neutral home (dense-checked), as in the honest check.
Same scene and rule as honest (base_panda_v2 link meshes, 2 mm arm, 1 mm carried block, <= 5 mrad dense checks).
"""
from __future__ import annotations

import numpy as np

from tampanda.symbolic.domains.multilevel_blocks import honest_lib as L
from tampanda.symbolic.domains.multilevel_blocks.executor import _upright_grasp_quats

VERSION = "2026-10-05-cont-a"


class HonestLibContExecutor(L.HonestLibExecutor):
    def _lib_put_upright(self, block, c_low, c_high):
        anchor = (self._cell_pose(c_low) + self._cell_pose(c_high)) / 2
        # the actual in-hand pose (no upright_top_grasp override): candidates rotate hand + block rigidly
        return self._lib_put(block, anchor, _upright_grasp_quats(), self._traverse_z(), ee_release_z=anchor[2] + 0.04)


def make_honest_lib_cont_executor(env, workspace, config):
    from tampanda.planners.rrt_star import RRTStar
    ex = HonestLibContExecutor(env, workspace, config, motion_planner=RRTStar(env, max_iterations=3000))
    env.ik.max_iters = 100
    ex._yaw_pool = None
    return ex
