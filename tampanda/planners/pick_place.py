"""Blocking pick-and-place executor built on FrankaEnvironment + RRTStar.

Provides a high-level, reusable interface for grasping and placing objects
using the GraspPlanner for grasp candidate generation, execute_path for smooth
gravity-compensated trajectory execution, and wait_idle for synchronous
blocking waits.

Typical usage:
    from tampanda import FrankaEnvironment, RRTStar, GraspPlanner
    from tampanda.planners import PickPlaceExecutor

    env = FrankaEnvironment(xml_path)
    planner = RRTStar(env)
    grasp_planner = GraspPlanner(table_z=0.27)
    executor = PickPlaceExecutor(env, planner, grasp_planner)

    block_pos  = env.get_object_position("block_0")
    half_size  = np.array([0.02, 0.02, 0.02])
    block_quat = env.get_object_orientation("block_0")

    success = executor.pick("block_0", block_pos, half_size, block_quat)
    if success:
        place_pos = np.array([0.5, 0.2, target_top_z + 0.02])
        executor.place(place_pos, np.array([0, 1, 0, 0]))
"""

import numpy as np
import mujoco
from typing import Any, Callable, Dict, List, Optional

from tampanda.planners.grasp_planner import (
    GraspPlanner, LouisGraspPlanner, GraspCandidate, GRASP_CONTACT_OFFSET,
    LINK7_CAPSULE_RADIUS, quat_to_rotmat,
)
from tampanda.controllers.position_controller import ControllerStatus




class PickPlaceExecutor:
    """Blocking pick-and-place state machine.

    All methods are synchronous: they run the simulation loop internally
    and return only when the action is complete (or has failed).

    Args:
        env: A FrankaEnvironment instance.
        planner: An RRTStar instance configured for this env.
        grasp_planner: A GraspPlanner instance.  If None, a default one is
            created (requires table_z to be passed per call).
        approach_step_size: Joint-space interpolation step for approach moves.
        grasp_step_size: Joint-space step for final grasp descent.
        lift_step_size: Joint-space step for lift (slow = less cube slip).
        place_step_size: Joint-space step for placement descent.
        retreat_step_size: Joint-space step for post-place retreat.
        max_plan_iters: RRT max iterations for each planning call.
        arm_settle_vel: qvel norm threshold — arm considered settled below this.
        gripper_start_vel: qvel[7] threshold — gripper started closing above this.
        gripper_stop_vel: qvel[7] threshold — gripper fully closed below this.
        settle_steps: Extra physics steps after wait_idle finishes.
        use_attachment: If True, kinematically attach the object to the EE during
            lift and place so it cannot slip.  Detached before gripper opens.
    """

    def __init__(
        self,
        env,
        planner,
        grasp_planner: Optional[GraspPlanner] = None,
        approach_step_size: float = 0.01,
        grasp_step_size: float = 0.003,
        lift_step_size: float = 0.003,
        place_step_size: float = 0.003,
        retreat_step_size: float = 0.01,
        max_plan_iters: int = 2000,
        arm_settle_vel: float = 0.01,
        gripper_start_vel: float = 0.0005,
        gripper_stop_vel: float = 0.0001,
        settle_steps: int = 80,
        use_attachment: bool = True,
        linear_ik_planner=None,
    ):
        self.env = env
        self.planner = planner
        self.grasp_planner = grasp_planner or GraspPlanner()
        self.approach_step_size = approach_step_size
        self.grasp_step_size    = grasp_step_size
        self.lift_step_size     = lift_step_size
        self.place_step_size    = place_step_size
        self.retreat_step_size  = retreat_step_size
        self.max_plan_iters     = max_plan_iters
        self.arm_settle_vel     = arm_settle_vel
        self.gripper_start_vel  = gripper_start_vel
        self.gripper_stop_vel   = gripper_stop_vel
        self.settle_steps       = settle_steps
        self.use_attachment     = use_attachment
        # Optional Linear-IK planner used as the *primary* planner for
        # short, low-clutter Cartesian moves (descent and lift).  RRT*
        # remains the fallback when linear-IK can't find a path
        # (collision in the joint-space transition).
        from tampanda.planners.linear_ik import LinearIKPlanner as _LIK
        self.linear_ik_planner = (
            linear_ik_planner if linear_ik_planner is not None
            else _LIK(env)
        )

        self._ee_site_id = mujoco.mj_name2id(
            env.model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site"
        )
        self._last_grasp_quat: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def pick(
        self,
        block_name: str,
        block_pos: np.ndarray,
        half_size: np.ndarray,
        block_quat: np.ndarray,
        candidates=None,
    ) -> bool:
        """Execute a full grasping sequence for the named block.

        Tries each GraspPlanner candidate in turn.  For each candidate:
          1. Plan and execute approach (above grasp position)
          2. Plan and execute grasp descent (collision-excepted)
          3. Wait for arm to settle, verify EE position
          4. Close gripper, wait for fingers to stop
          5. Optionally attach block to EE kinematically
          6. Execute lift

        Args:
            candidates: Optional list of GraspCandidate to try instead of
                generating them from the GraspPlanner.  Useful for forcing a
                specific grasp type (e.g. FRONT only).

        Returns True on success, False if all candidates were exhausted.
        """
        if candidates is None:
            candidates = self.grasp_planner.generate_candidates(
                block_pos, half_size, block_quat
            )
        if not candidates:
            print(f"[PickPlaceExecutor] No grasp candidates for {block_name}")
            return False

        for i, cand in enumerate(candidates):
            print(f"[PickPlaceExecutor] Trying candidate {i+1}/{len(candidates)}: "
                  f"{cand.grasp_type.value}")

            # Track everything we mutate so we can undo on a mid-
            # attempt failure and either try the next candidate from
            # a clean state, or return False with the arm back where
            # it started.
            executed_paths: List[List[np.ndarray]] = []
            attached_here = False
            gripper_closed_here = False

            def _rollback(reason: str) -> None:
                """Undo this candidate's partial execution."""
                if attached_here and self.env._attached is not None:
                    self.env.detach_object()
                if gripper_closed_here:
                    self.env.controller.open_gripper()
                    self._wait_gripper_open()
                self.env.remove_collision_exception(block_name)
                # Restore controller delta in case it was tightened.
                self.env.controller._advance_delta_override = 0.1
                for executed in reversed(executed_paths):
                    self.env.execute_path(list(reversed(executed)),
                                            self.planner,
                                            step_size=self.approach_step_size)
                    self.env.wait_idle(settle_steps=self.settle_steps)
                print(f"  {reason} — rolled back, next candidate")

            # 1. Approach.  When the home pose is already in the
            # target orientation (e.g., palm-+y staging home for
            # closed-top shelves), the approach is a same-orientation
            # translation — joint-lerp is the most reliable option.
            # When the orientation differs, joint-lerp's IK at the
            # target with the home as seed naturally finds a config
            # near the home, then we lerp.  Falls back to Cartesian
            # linear-IK and finally RRT* if joint-lerp can't find a
            # collision-free corridor.
            path = self.linear_ik_planner.plan_joint_lerp(
                cand.approach_pos, cand.grasp_quat, dt=0.005,
                n_substeps=20,
            )
            if path is None:
                path = self.linear_ik_planner.plan_to_pose(
                    cand.approach_pos, cand.grasp_quat,
                    dt=0.005, n_substeps=12,
                )
            if path is None:
                path = self.planner.plan_to_pose(
                    cand.approach_pos, cand.grasp_quat,
                    dt=0.005, max_iterations=self.max_plan_iters,
                )
            if path is None:
                print("  approach plan failed (joint-lerp + linear-IK + RRT*) — next candidate")
                continue
            self.env.execute_path(path, self.planner, step_size=self.approach_step_size)
            self.env.wait_idle(settle_steps=self.settle_steps)
            executed_paths.append(path)

            # 2. Grasp descent.  Strategy: joint-space lerp first
            # (IK once at the grasp pose seeded from the current
            # post-approach config, then straight line in joint
            # space).  This is the most reliable form of short-move
            # planning when the EE orientation doesn't change — no
            # Cartesian-substep IK basin-flips, no RRT* randomness.
            # Cartesian-IK chain second, RRT* third.
            #
            # Make sure the gripper is fully open before threading the
            # fingers around the target — a partially-closed gripper
            # carried over from a prior attempt clips the cylinder
            # going in (visible as the cylinder being shoved forward
            # by ~1 cell during the descent).
            self.env.controller.open_gripper()
            self._wait_gripper_open()
            # Tighten the controller's advance delta for the descent.
            # The default 0.1 rad (~5.7°) is much larger than the 3 mm
            # grasp step_size, so the controller blasts past every
            # waypoint without actually following the trajectory —
            # the arm overswings and shoves the target object during
            # entry.  0.01 rad (~0.6°) tracks the trajectory tightly.
            prev_delta = getattr(self.env.controller,
                                   "_advance_delta_override", 0.1)
            self.env.controller._advance_delta_override = 0.01
            self.env.add_collision_exception(block_name)
            path = self.linear_ik_planner.plan_joint_lerp(
                cand.grasp_pos, cand.grasp_quat, dt=0.005,
            )
            if path is None:
                path = self.linear_ik_planner.plan_to_pose(
                    cand.grasp_pos, cand.grasp_quat, dt=0.005,
                    slerp_orientation=False,
                )
            if path is None:
                path = self.planner.plan_to_pose(
                    cand.grasp_pos, cand.grasp_quat,
                    dt=0.005, max_iterations=self.max_plan_iters,
                )
            if path is None:
                _rollback("grasp descent plan failed (joint-lerp + linear-IK + RRT*)")
                continue
            self.env.execute_path(path, self.planner, step_size=self.grasp_step_size)
            self.env.wait_idle(settle_steps=self.settle_steps)
            executed_paths.append(path)

            # 3. Arm settle check + EE accuracy report
            self._wait_arm_settle()
            ee_pos = self.env.data.site_xpos[self._ee_site_id].copy()
            err_mm = np.linalg.norm(ee_pos - cand.grasp_pos) * 1000
            print(f"  EE error at grasp: {err_mm:.1f} mm")

            # 4. Close gripper, wait for full closure
            self.env.controller.close_gripper()
            self._wait_gripper_closed()
            gripper_closed_here = True

            # 5. Kinematic attachment (prevents slip during lift)
            if self.use_attachment:
                self.env.attach_object_to_ee(block_name)
                attached_here = True

            # 6. Lift — same strategy chain as descent.
            path = self.linear_ik_planner.plan_joint_lerp(
                cand.lift_pos, cand.grasp_quat, dt=0.005,
            )
            if path is None:
                path = self.linear_ik_planner.plan_to_pose(
                    cand.lift_pos, cand.grasp_quat, dt=0.005,
                    slerp_orientation=False,
                )
            if path is None:
                path = self.planner.plan_to_pose(
                    cand.lift_pos, cand.grasp_quat,
                    dt=0.005, max_iterations=self.max_plan_iters,
                )
            if path is None:
                _rollback("lift plan failed (joint-lerp + linear-IK + RRT*)")
                continue
            self.env.execute_path(path, self.planner, step_size=self.lift_step_size)
            self.env.wait_idle(settle_steps=self.settle_steps)

            self._last_grasp_quat = cand.grasp_quat.copy()
            # Restore the controller's default advance delta — only
            # the precision descent/lift wants the tight 0.01 rad
            # tracking; subsequent free-space moves should use the
            # default for speed.
            self.env.controller._advance_delta_override = 0.1
            print(f"  pick SUCCESS")
            return True

        print(f"[PickPlaceExecutor] All {len(candidates)} candidates exhausted for {block_name}")
        return False

    def place(
        self,
        block_name: str,
        place_block_center: np.ndarray,
        ee_quat: Optional[np.ndarray] = None,
        target_block_name: Optional[str] = None,
        approach_height: float = 0.15,
        place_clearance: float = 0.003,
        retreat_lift: Optional[float] = None,
    ) -> bool:
        """Execute a blocking place sequence.

        The EE descends to the position that puts the held block's centre at
        place_block_center + place_clearance above the target surface
        (GRASP_CONTACT_OFFSET is added automatically).  After the gripper opens
        physics settles the block the remaining clearance onto the surface.

        The kinematic attachment (if active) is released before the descent so
        that contact forces are handled by physics rather than fighting the
        forced position — this prevents the block from being pushed into the
        target.

        Args:
            block_name: Name of the held block (for collision exceptions).
            place_block_center: 3D target position for the held block's centre
                                when resting on the target surface.
            ee_quat: EE orientation during placement.
            target_block_name: If given, add collision exception for the target.
            approach_height: Extra height above the placement EE position to
                             approach from.
            place_clearance: Extra z added to ee_place so the block is released
                             this far above the target surface (default 3 mm).
                             Physics settles it down naturally.

        Returns True on success.
        """
        if ee_quat is None:
            if self._last_grasp_quat is None:
                raise ValueError(
                    "ee_quat must be provided if no prior pick has been executed"
                )
            ee_quat = self._last_grasp_quat

        # EE position so the held block ends up at ``place_block_center``,
        # offset by ``place_clearance`` in +z so the block can drop a few
        # millimetres into the surface when the gripper releases.
        # Mirror of ``GraspPlanner._build_candidate``'s formula:
        #     grasp_pos = block_pos - hand_z_world * GRASP_CONTACT_OFFSET
        # For TOP-DOWN grasps ``hand_z_world == (0, 0, -1)``, so the
        # contact term lands on +z (matching the previous behaviour); for
        # FRONT (palm-+y) ``hand_z_world == (0, 1, 0)`` and the offset
        # correctly shifts in -y, where the cylinder actually sits
        # relative to the gripper.  Without this, FRONT places hardcoded
        # a +z offset, putting the EE *above* the target rather than
        # in front of it — the IK either fails outright or grounds at a
        # config that drops the cylinder into the wrong cell.
        R_ee = quat_to_rotmat(np.asarray(ee_quat))
        hand_z_world = R_ee[:, 2]
        ee_place = (np.asarray(place_block_center, dtype=float)
                     - hand_z_world * GRASP_CONTACT_OFFSET)
        ee_place[2] += place_clearance

        # Mirror ``GraspPlanner``'s link7-floor safety: for side
        # grasps (palm-+y / palm-+x, hand z-axis roughly horizontal),
        # link7's 5.5 cm capsule sits well below the EE in world
        # frame.  At a deep cell — inside the cubicle where the floor
        # plate extends — this clips the floor with several mm of
        # penetration.  Pick avoids it because ``GraspPlanner``
        # raises ``grasp_pos[2]`` above ``table_z + LINK7_CAPSULE_RADIUS``;
        # place needs the same treatment.  Top-down grasps already
        # sit well above the table so the clamp is a no-op for them.
        if abs(float(hand_z_world[2])) < 0.7:  # not top-down
            min_safe_z = (self.grasp_planner.table_z
                           + LINK7_CAPSULE_RADIUS + 0.005)
            if ee_place[2] < min_safe_z:
                ee_place[2] = min_safe_z

        # Approach pose is ``approach_height`` away from ``ee_place``
        # along ``-hand_z_world``, i.e. mirror of ``GraspPlanner``'s
        # pick approach.  For TOP-DOWN grasps this lands the approach
        # +z above the place (matching the prior hardcoded behaviour);
        # for FRONT grasps the approach is -y of the place — backing
        # the gripper out the open face — instead of pushing the
        # gripper up into the closed shelf ceiling, which is what the
        # old hardcoded +z formula did.
        ee_approach = ee_place - hand_z_world * approach_height

        # The held block is already in ``_collision_body_ids`` (added by
        # ``attach_object_to_ee``), so its contacts with the gripper
        # count as self-collisions and are skipped automatically.  Do
        # NOT add it to the exception list — that would skip ALL its
        # contacts, including the ones we *want* the planner to see
        # (the held block bumping a parked cylinder during the
        # descent).  Only except the optional target block, used when
        # placing on top of a stack.
        if target_block_name is not None:
            self.env.add_collision_exception(target_block_name)

        # Snapshot the pre-place arm config + record paths we
        # actually executed so we can roll back to the starting state
        # if any later phase fails midway.  Without rollback, a
        # failed put leaves the gripper stuck at the partially-
        # executed approach pose — every subsequent put attempt then
        # fails because the new approach plan can't be made from
        # that intermediate state.
        executed_paths: List[List[np.ndarray]] = []

        def _rollback(reason: str) -> bool:
            """Replay every executed path in reverse so the arm
            returns to its pre-place configuration."""
            # Restore default controller delta first; rollback should
            # use the same loose tracking the approach used.
            self.env.controller._advance_delta_override = 0.1
            for executed in reversed(executed_paths):
                reversed_path = list(reversed(executed))
                self.env.execute_path(reversed_path, self.planner,
                                        step_size=self.approach_step_size)
                self.env.wait_idle(settle_steps=self.settle_steps)
            self._clear_exceptions(block_name, target_block_name)
            print(f"[PickPlaceExecutor] place rolled back: {reason}")
            return False

        # 1. Approach above placement (attachment still active — block rides along)
        path = self.planner.plan_to_pose(
            ee_approach, ee_quat,
            dt=0.005, max_iterations=self.max_plan_iters,
        )
        if path is None:
            print("[PickPlaceExecutor] place approach plan failed")
            self._clear_exceptions(block_name, target_block_name)
            return False  # nothing executed, no rollback needed
        self.env.execute_path(path, self.planner, step_size=self.approach_step_size)
        self.env.wait_idle(settle_steps=self.settle_steps)
        executed_paths.append(path)

        # Tighten the controller delta for the descent + open + retreat.
        # Same reason as in ``pick``: the default 0.1 rad lets the arm
        # blast through fine waypoints, overshoot the place pose, and
        # drag the held block several cm before the gripper opens.
        self.env.controller._advance_delta_override = 0.01

        # 2. Descend to placement position WITH the kinematic
        # attachment still active so the held object rides along.
        # Detaching here (the old behaviour) leaves the cylinder held
        # only by passive finger friction during the descent, and any
        # slight perturbation drops it before we reach the place
        # pose — visible in the simulator as the cylinder slipping
        # out and falling to the shelf floor.
        path = self.planner.plan_to_pose(
            ee_place, ee_quat,
            dt=0.005, max_iterations=self.max_plan_iters,
        )
        if path is None:
            return _rollback("descent plan failed")
        self.env.execute_path(path, self.planner, step_size=self.place_step_size)
        self.env.wait_idle(settle_steps=self.settle_steps)

        # 3. Detach the kinematic attachment NOW (after the descent
        # finishes), then open the gripper.  The brief window between
        # detach and open lets physics settle the block on the
        # surface; gripper-friction still holds the block until the
        # fingers actually move.
        if self.use_attachment:
            self.env.detach_object()

        # 4. Open gripper, let physics settle the block onto the surface
        self.env.controller.open_gripper()
        self._wait_gripper_open()

        # 5. Retreat.  The naive retreat target is ``ee_approach``,
        # which for a FRONT grasp sits -y of ``ee_place`` at the
        # same z.  Moving the *open* gripper straight -y from
        # ``ee_place`` sweeps the finger tips backward through the
        # just-placed block's y-range — and since the open jaws (at
        # x=±0.04 from EE) overlap the block in x, they drag the
        # block several cm with them.  Add a small +z lift to
        # ``ee_approach`` so the gripper clears the block height
        # before the lateral retreat sweeps past it.
        self._clear_exceptions(block_name, target_block_name)
        ee_retreat = ee_approach.copy()
        # +z lift on the retreat to clear the just-placed block's
        # top so the open finger pads (at x=±4 cm from EE on a FRONT
        # grasp) don't drag the block sideways as they sweep back.
        # Caller supplies the lift via ``retreat_lift``; default 15
        # cm is generous for an open tabletop but too tall for short
        # closed cubicles (interior height ~18 cm) where it puts the
        # retreat target above the closed ceiling and IK can't
        # converge.  Closed-top callers should pass a value tuned to
        # the held block's height — e.g. ``2 * half_z + 0.02``.
        ee_retreat[2] += 0.15 if retreat_lift is None else retreat_lift
        path = self.planner.plan_to_pose(
            ee_retreat, ee_quat,
            dt=0.005, max_iterations=self.max_plan_iters,
        )
        if path is not None:
            self.env.execute_path(path, self.planner, step_size=self.retreat_step_size)
            self.env.wait_idle(settle_steps=self.settle_steps)

        # Restore default controller delta for subsequent free-space moves.
        self.env.controller._advance_delta_override = 0.1
        print("[PickPlaceExecutor] place SUCCESS")
        return True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _wait_arm_settle(self, max_steps: int = 500):
        """Spin until joint velocity is below arm_settle_vel (or timeout)."""
        for _ in range(max_steps):
            self.env.controller.step()
            self.env.step()
            if np.linalg.norm(self.env.data.qvel[:7]) < self.arm_settle_vel:
                return

    def _wait_gripper_closed(self, max_steps: int = 1000):
        """Two-phase wait: fingers start moving, then stop (contact or limit)."""
        started = False
        for _ in range(max_steps):
            self.env.controller.step()
            self.env.step()
            vel = abs(self.env.data.qvel[7])
            if not started:
                if vel > self.gripper_start_vel:
                    started = True
            else:
                if vel < self.gripper_stop_vel:
                    return

    def _wait_gripper_open(self, steps: int = 300):
        """Run physics for a fixed number of steps after opening the gripper."""
        for _ in range(steps):
            self.env.controller.step()
            self.env.step()

    def _clear_exceptions(self, block_name: str, target_block_name: Optional[str]):
        self.env.remove_collision_exception(block_name)
        if target_block_name is not None:
            self.env.remove_collision_exception(target_block_name)



GRASP_ORDER = {
    "top_down_x": 0,
    "top_down_y": 1,
    "top_down_45": 2,
    "top_down_135": 3,
    "front": 4,
}


def _sort_candidates(candidates):
    return sorted(
        list(candidates),
        key=lambda cand: GRASP_ORDER.get(
            getattr(cand.grasp_type, "value", cand.grasp_type),
            len(GRASP_ORDER),
        ),
    )

class LouisPickPlaceExecutor:
    """Blocking pick-and-place state machine.

    All methods are synchronous: they run the simulation loop internally
    and return only when the action is complete (or has failed).

    Args:
        env: A FrankaEnvironment instance.
        planner: An RRTStar instance configured for this env.
        grasp_planner: A GraspPlanner instance.  If None, a default one is
            created (requires table_z to be passed per call).
        approach_step_size: Joint-space interpolation step for approach moves.
        grasp_step_size: Joint-space step for final grasp descent.
        lift_step_size: Joint-space step for lift (slow = less cube slip).
        place_step_size: Joint-space step for placement descent.
        retreat_step_size: Joint-space step for post-place retreat.
        max_plan_iters: RRT max iterations for each planning call.
        max_ik_retries: IK seed attempts for each end-effector pose.
        plan_dt: IK integration timestep passed into RRTStar.plan_to_pose.
        arm_settle_vel: qvel norm threshold — arm considered settled below this.
        gripper_start_vel: qvel[7] threshold — gripper started closing above this.
        gripper_stop_vel: qvel[7] threshold — gripper fully closed below this.
        settle_steps: Extra physics steps after wait_idle finishes.
        use_attachment: If True, kinematically attach the object to the EE during
            lift and place so it cannot slip.  Detached before gripper opens.
    """

    def __init__(
        self,
        env,
        planner,
        grasp_planner: Optional[LouisGraspPlanner] = None,
        approach_step_size: float = 0.01,
        grasp_step_size: float = 0.003,
        lift_step_size: float = 0.003,
        place_step_size: float = 0.003,
        retreat_step_size: float = 0.01,
        max_plan_iters: int = 2000,
        max_ik_retries: int = 8,
        plan_dt: float = 0.005,
        arm_settle_vel: float = 0.01,
        gripper_start_vel: float = 0.0005,
        gripper_stop_vel: float = 0.0001,
        settle_steps: int = 80,
        use_attachment: bool = True,
        planning_trace_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ):
        self.env = env
        self.planner = planner
        self.grasp_planner = grasp_planner or LouisGraspPlanner()
        self.approach_step_size = approach_step_size
        self.grasp_step_size    = grasp_step_size
        self.lift_step_size     = lift_step_size
        self.place_step_size    = place_step_size
        self.retreat_step_size  = retreat_step_size
        self.max_plan_iters     = max_plan_iters
        self.max_ik_retries     = max_ik_retries
        self.plan_dt            = plan_dt
        self.arm_settle_vel     = arm_settle_vel
        self.gripper_start_vel  = gripper_start_vel
        self.gripper_stop_vel   = gripper_stop_vel
        self.settle_steps       = settle_steps
        self.use_attachment     = use_attachment
        self.planning_trace_callback = planning_trace_callback
        self.planning_trace_context: Dict[str, Any] = {}
        self._planning_trace_index = 0
        self.last_failure = None
        self.last_warning = None
        self.failure_history = []

        # Scoped controller tolerance.  The shared PositionController
        # default for grasping_delta is 0.3; this executor was tuned
        # against 0.03, so set it here instead of changing the default
        # for every caller.
        env.controller.grasping_delta = 0.03

        self._ee_site_id = mujoco.mj_name2id(
            env.model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site"
        )
        self._last_grasp_quat: Optional[np.ndarray] = None
        self.last_grasp_ee_error_mm: Optional[float] = None
        self.grasp_ee_error_history = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_planning_trace_context(self, **context):
        self.planning_trace_context = dict(context)

    def pick(
        self,
        block_name: str,
        block_pos: np.ndarray,
        half_size: np.ndarray,
        block_quat: np.ndarray,
        candidates=None,
    ) -> bool:
        """Execute a full grasping sequence for the named block.

        Tries each GraspPlanner candidate in turn.  For each candidate:
          1. Plan and execute approach (above grasp position)
          2. Plan and execute grasp descent (collision-excepted)
          3. Wait for arm to settle, verify EE position
          4. Close gripper, wait for fingers to stop
          5. Optionally attach block to EE kinematically
          6. Execute lift

        Args:
            candidates: Optional list of GraspCandidate to try instead of
                generating them from the GraspPlanner.  Useful for forcing a
                specific grasp type (e.g. FRONT only).

        Returns True on success, False if all candidates were exhausted.
        """
        self._reset_diagnostics()
        self.last_grasp_ee_error_mm = None
        if candidates is None:
            object_shape = "sphere" if block_name.startswith("ball_") else None
            candidates = self.grasp_planner.generate_candidates(
                block_pos,
                half_size,
                block_quat,
                object_shape=object_shape,
            )
        candidates = _sort_candidates(candidates)
        if not candidates:
            print(f"[PickPlaceExecutor] No grasp candidates for {block_name}")
            self._set_failure(
                "pick",
                "grasp_generation",
                "no_grasp_candidates",
                block_name=block_name,
            )
            return False

        for i, cand in enumerate(candidates):
            print(f"[PickPlaceExecutor] Trying candidate {i+1}/{len(candidates)}: "
                  f"{cand.grasp_type.value}")

            # 1. Approach
            path = self._plan_to_pose_with_trace(
                cand.approach_pos, cand.grasp_quat,
                high_level_action="pick",
                phase="approach",
                candidate_index=i + 1,
                total_candidates=len(candidates),
                grasp_type=cand.grasp_type.value,
                block_name=block_name,
            )
            if path is None:
                print("  approach plan failed — next candidate")
                self._set_failure(
                    "pick",
                    "approach",
                    "approach_plan_failed",
                    block_name=block_name,
                    candidate_index=i + 1,
                    total_candidates=len(candidates),
                    grasp_type=cand.grasp_type.value,
                    planner_failure=getattr(self.planner, "last_plan_failure", None),
                )
                continue
            self.env.execute_path(path, self.planner, step_size=self.approach_step_size)
            self.env.wait_idle(settle_steps=self.settle_steps)

            # 2. Grasp descent (block not an obstacle)
            self.env.add_collision_exception(block_name)
            path = self._plan_to_pose_with_trace(
                cand.grasp_pos, cand.grasp_quat,
                high_level_action="pick",
                phase="grasp_descent",
                candidate_index=i + 1,
                total_candidates=len(candidates),
                grasp_type=cand.grasp_type.value,
                block_name=block_name,
            )
            if path is None:
                print("  grasp descent plan failed — next candidate")
                self.env.remove_collision_exception(block_name)
                self._set_failure(
                    "pick",
                    "grasp_descent",
                    "grasp_descent_plan_failed",
                    block_name=block_name,
                    candidate_index=i + 1,
                    total_candidates=len(candidates),
                    grasp_type=cand.grasp_type.value,
                    planner_failure=getattr(self.planner, "last_plan_failure", None),
                )
                continue
            self.env.execute_path(path, self.planner, step_size=self.grasp_step_size)
            self.env.wait_idle(settle_steps=self.settle_steps)

            # 3. Arm settle check + EE accuracy report
            self._wait_arm_settle()
            ee_pos = self.env.data.site_xpos[self._ee_site_id].copy()
            err_mm = np.linalg.norm(ee_pos - cand.grasp_pos) * 1000
            self.last_grasp_ee_error_mm = float(err_mm)
            self.grasp_ee_error_history.append({
                "candidate_index": i + 1,
                "grasp_type": cand.grasp_type.value,
                "ee_error_mm": float(err_mm),
            })
            print(f"  EE error at grasp: {err_mm:.1f} mm")

            # 4. Close gripper, wait for full closure
            self.env.controller.close_gripper()
            self._wait_gripper_closed()

            # 5. Kinematic attachment (prevents slip during lift)
            if self.use_attachment:
                self.env.attach_object_to_ee(block_name)

            # 6. Lift
            path = self._plan_to_pose_with_trace(
                cand.lift_pos, cand.grasp_quat,
                high_level_action="pick",
                phase="lift",
                candidate_index=i + 1,
                total_candidates=len(candidates),
                grasp_type=cand.grasp_type.value,
                block_name=block_name,
            )
            if path is None:
                print("  lift plan failed — next candidate")
                self._set_failure(
                    "pick",
                    "lift",
                    "lift_plan_failed",
                    block_name=block_name,
                    candidate_index=i + 1,
                    total_candidates=len(candidates),
                    grasp_type=cand.grasp_type.value,
                    planner_failure=getattr(self.planner, "last_plan_failure", None),
                )
                self.env.detach_object()
                self.env.remove_collision_exception(block_name)
                self.env.controller.open_gripper()
                self._wait_gripper_open()
                continue
            self.env.execute_path(path, self.planner, step_size=self.lift_step_size)
            self.env.wait_idle(settle_steps=self.settle_steps)

            self._last_grasp_quat = cand.grasp_quat.copy()
            self.last_failure = None
            print(f"  pick SUCCESS")
            return True

        last_candidate_failure = self.last_failure
        self._set_failure(
            "pick",
            "all_candidates",
            "all_candidates_exhausted",
            block_name=block_name,
            total_candidates=len(candidates),
            last_candidate_failure=last_candidate_failure,
        )
        print(f"[PickPlaceExecutor] All {len(candidates)} candidates exhausted for {block_name}")
        return False

    def place(
        self,
        block_name: str,
        place_block_center: np.ndarray,
        ee_quat: Optional[np.ndarray] = None,
        target_block_name: Optional[str] = None,
        approach_height: float = 0.15,
        place_clearance: float = 0.003,
        candidate_index: Optional[int] = None,
        total_candidates: Optional[int] = None,
        grasp_type: Optional[str] = None,
    ) -> bool:
        """Execute a blocking place sequence.

        The EE descends to the position that puts the held block's centre at
        place_block_center + place_clearance above the target surface
        (GRASP_CONTACT_OFFSET is added automatically).  After the gripper opens
        physics settles the block the remaining clearance onto the surface.

        The kinematic attachment (if active) is released before the descent so
        that contact forces are handled by physics rather than fighting the
        forced position — this prevents the block from being pushed into the
        target.

        Args:
            block_name: Name of the held block (for collision exceptions).
            place_block_center: 3D target position for the held block's centre
                                when resting on the target surface.
            ee_quat: EE orientation during placement.
            target_block_name: If given, add collision exception for the target.
            approach_height: Extra height above the placement EE position to
                             approach from.
            place_clearance: Extra z added to ee_place so the block is released
                             this far above the target surface (default 3 mm).
                             Physics settles it down naturally.

        Returns True on success.
        """
        self._reset_diagnostics()
        if ee_quat is None:
            if self._last_grasp_quat is None:
                self._set_failure(
                    "place",
                    "orientation",
                    "missing_ee_quat",
                    block_name=block_name,
                    target_block_name=target_block_name,
                )
                raise ValueError(
                    "ee_quat must be provided if no prior pick has been executed"
                )
            ee_quat = self._last_grasp_quat

        # EE position when block centre is at place_block_center + clearance
        ee_place = place_block_center.copy()
        ee_place[2] += GRASP_CONTACT_OFFSET + place_clearance

        ee_approach = ee_place.copy()
        ee_approach[2] += approach_height

        # Add exceptions for held block and (optionally) target
        self.env.add_collision_exception(block_name)
        if target_block_name is not None:
            self.env.add_collision_exception(target_block_name)

        # 1. Approach above placement (attachment still active — block rides along)
        path = self._plan_to_pose_with_trace(
            ee_approach, ee_quat,
            high_level_action="place",
            phase="approach",
            candidate_index=candidate_index,
            total_candidates=total_candidates,
            grasp_type=grasp_type,
            block_name=block_name,
            target_block_name=target_block_name,
        )
        if path is None:
            print("[PickPlaceExecutor] place approach plan failed")
            self._set_failure(
                "place",
                "approach",
                "place_approach_plan_failed",
                block_name=block_name,
                target_block_name=target_block_name,
                candidate_index=candidate_index,
                total_candidates=total_candidates,
                grasp_type=grasp_type,
                planner_failure=getattr(self.planner, "last_plan_failure", None),
            )
            self._clear_exceptions(block_name, target_block_name)
            return False
        self.env.execute_path(path, self.planner, step_size=self.approach_step_size)
        self.env.wait_idle(settle_steps=self.settle_steps)

        # 2. Detach before descent so contact forces are handled by physics
        if self.use_attachment:
            self.env.detach_object()

        # 3. Descend to placement position
        path = self._plan_to_pose_with_trace(
            ee_place, ee_quat,
            high_level_action="place",
            phase="descent",
            candidate_index=candidate_index,
            total_candidates=total_candidates,
            grasp_type=grasp_type,
            block_name=block_name,
            target_block_name=target_block_name,
        )
        if path is None:
            print("[PickPlaceExecutor] place descent plan failed")
            self._set_failure(
                "place",
                "descent",
                "place_descent_plan_failed",
                block_name=block_name,
                target_block_name=target_block_name,
                candidate_index=candidate_index,
                total_candidates=total_candidates,
                grasp_type=grasp_type,
                planner_failure=getattr(self.planner, "last_plan_failure", None),
            )
            self._clear_exceptions(block_name, target_block_name)
            return False
        self.env.execute_path(path, self.planner, step_size=self.place_step_size)
        self.env.wait_idle(settle_steps=self.settle_steps)

        # 4. Open gripper, let physics settle the block onto the surface
        self.env.controller.open_gripper()
        self._wait_gripper_open()

        # 5. Retreat upward
        self._clear_exceptions(block_name, target_block_name)
        path = self._plan_to_pose_with_trace(
            ee_approach, ee_quat,
            high_level_action="place",
            phase="retreat",
            candidate_index=candidate_index,
            total_candidates=total_candidates,
            grasp_type=grasp_type,
            block_name=block_name,
            target_block_name=target_block_name,
        )
        if path is not None:
            self.env.execute_path(path, self.planner, step_size=self.retreat_step_size)
            self.env.wait_idle(settle_steps=self.settle_steps)
        else:
            self._set_warning(
                "place",
                "retreat",
                "retreat_plan_failed",
                block_name=block_name,
                target_block_name=target_block_name,
                planner_failure=getattr(self.planner, "last_plan_failure", None),
            )

        print("[PickPlaceExecutor] place SUCCESS")
        return True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _plan_to_pose_with_trace(self, target_pos, target_quat, **trace_context):
        path = self.planner.plan_to_pose_traced(
            target_pos, target_quat,
            dt=self.plan_dt, max_iterations=self.max_plan_iters,
            max_ik_retries=self.max_ik_retries,
        )

        if self.planning_trace_callback is not None:
            self._planning_trace_index += 1
            event = {
                **self.planning_trace_context,
                **trace_context,
                "plan_call_index": self._planning_trace_index,
                "max_plan_iters": self.max_plan_iters,
                "rrt_max_iterations_default": getattr(self.planner, "max_iterations", None),
                "plan_dt": self.plan_dt,
                "max_ik_retries": self.max_ik_retries,
                "target_pos": target_pos.copy(),
                "target_quat": target_quat.copy(),
                "result": "success" if path is not None else "failure",
                "planner_stats": getattr(self.planner, "last_plan_stats", None),
                "planner_failure": getattr(self.planner, "last_plan_failure", None),
            }
            self.planning_trace_callback(event)

        return path

    def _reset_diagnostics(self):
        self.last_failure = None
        self.last_warning = None
        self.failure_history = []

    def _wait_arm_settle(self, max_steps: int = 500):
        """Spin until joint velocity is below arm_settle_vel (or timeout)."""
        for _ in range(max_steps):
            self.env.controller.step()
            self.env.step()
            if np.linalg.norm(self.env.data.qvel[:7]) < self.arm_settle_vel:
                return

    def _wait_gripper_closed(self, max_steps: int = 1000):
        """Two-phase wait: fingers start moving, then stop (contact or limit)."""
        started = False
        for _ in range(max_steps):
            self.env.controller.step()
            self.env.step()
            vel = abs(self.env.data.qvel[7])
            if not started:
                if vel > self.gripper_start_vel:
                    started = True
            else:
                if vel < self.gripper_stop_vel:
                    return

    def _wait_gripper_open(self, steps: int = 300):
        """Run physics for a fixed number of steps after opening the gripper."""
        for _ in range(steps):
            self.env.controller.step()
            self.env.step()

    def _clear_exceptions(self, block_name: str, target_block_name: Optional[str]):
        self.env.remove_collision_exception(block_name)
        if target_block_name is not None:
            self.env.remove_collision_exception(target_block_name)

    def _set_failure(self, action, phase, reason, **details):
        self.last_failure = {
            "action": action,
            "phase": phase,
            "reason": reason,
            **details,
        }
        self.failure_history.append(self.last_failure)

    def _set_warning(self, action, phase, reason, **details):
        self.last_warning = {
            "action": action,
            "phase": phase,
            "reason": reason,
            **details,
        }
