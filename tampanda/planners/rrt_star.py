"""RRT* motion planner implementation."""

import numpy as np
import mujoco
try:
    import mujoco.viewer
except ImportError:
    pass
from typing import List, Optional

from tampanda.core.base_mp import BaseMotionPlanner


class Node:
    """RRT* tree node."""
    
    def __init__(self, config: np.ndarray):
        self.config = config.copy()
        self.parent: Optional[Node] = None
        self.cost: float = 0.0
        self.children: List[Node] = []


class RRTStar(BaseMotionPlanner):
    """RRT* motion planner for robot manipulation."""

    def __init__(
        self,
        environment,
        max_iterations: int = 5000,
        step_size: float = 0.1,
        search_radius: float = 0.5,
        goal_threshold: float = 0.05
    ):
        self.env = environment
        self.model = environment.get_model()
        self.data = environment.get_data()
        self.ik = environment.get_ik()
        
        # RRT* parameters
        self.max_iterations = max_iterations
        self.goal_sample_rate = 0.1
        self.step_size = step_size
        self.search_radius = search_radius
        self.goal_threshold = goal_threshold
        # Number of intermediate configs checked per edge during collision
        # detection.  Higher values catch narrow obstacles but increase planning
        # time roughly linearly.  Default 5 matches original behaviour.
        self.collision_check_steps: int = 5
        
        # Joint limits for sampling
        self.joint_limits_low = self.model.jnt_range[:7, 0].copy()
        self.joint_limits_high = self.model.jnt_range[:7, 1].copy()
        # Diagnostics captured by the ``*_traced`` entry points only.  The
        # stock plan()/plan_to_pose() never touch these, so enabling tracing
        # costs nothing on the normal planning path.
        self.last_plan_failure = None
        self.plan_failure_history = []
        self.last_plan_stats = None
    
    def sample_random_config(self) -> np.ndarray:
        return np.random.uniform(self.joint_limits_low, self.joint_limits_high)
    
    def distance(self, config1: np.ndarray, config2: np.ndarray) -> float:
        return np.linalg.norm(config1 - config2)
    
    def nearest_node(self, tree: List[Node], config: np.ndarray) -> Node:
        return min(tree, key=lambda node: self.distance(node.config, config))
    
    def steer(self, from_config: np.ndarray, to_config: np.ndarray) -> np.ndarray:
        direction = to_config - from_config
        dist = np.linalg.norm(direction)
        if dist <= self.step_size:
            return to_config.copy()
        return from_config + (direction / dist) * self.step_size
    
    def is_path_collision_free(
        self,
        config1: np.ndarray,
        config2: np.ndarray,
        steps: int = None,
    ) -> bool:
        if steps is None:
            # Scale checks proportionally to path length so that smooth_path
            # shortcuts (which can span many step_size-length edges) are checked
            # at the same resolution as individual RRT edges.
            dist = float(np.linalg.norm(config2 - config1))
            steps = max(
                self.collision_check_steps,
                int(np.ceil(dist / self.step_size)) * self.collision_check_steps,
            )
        # Use optimized batch method from environment if available
        if hasattr(self.env, 'is_path_collision_free'):
            return self.env.is_path_collision_free(config1, config2, steps)
        # Fallback to original implementation
        for i in range(steps + 1):
            alpha = i / steps
            config = (1 - alpha) * config1 + alpha * config2
            if not self.env.is_collision_free(config):
                return False
        return True
    
    def near_nodes(self, tree: List[Node], config: np.ndarray, radius: float) -> List[Node]:
        return [node for node in tree if self.distance(node.config, config) < radius]
    
    def choose_parent(self, new_node: Node, near_nodes: List[Node]) -> Node:
        if not near_nodes:
            return new_node
        
        best_parent = new_node.parent
        min_cost = new_node.cost
        
        for near_node in near_nodes:
            potential_cost = near_node.cost + self.distance(near_node.config, new_node.config)
            if potential_cost < min_cost and self.is_path_collision_free(
                near_node.config, new_node.config
            ):
                best_parent = near_node
                min_cost = potential_cost
        
        if best_parent != new_node.parent:
            if new_node.parent:
                new_node.parent.children.remove(new_node)
            new_node.parent = best_parent
            new_node.cost = min_cost
            best_parent.children.append(new_node)
        
        return new_node
    
    def rewire(self, tree: List[Node], new_node: Node, near_nodes: List[Node]):
        for near_node in near_nodes:
            if near_node == new_node or near_node == new_node.parent:
                continue
            
            potential_cost = new_node.cost + self.distance(new_node.config, near_node.config)
            if potential_cost < near_node.cost and self.is_path_collision_free(
                new_node.config, near_node.config
            ):
                if near_node.parent:
                    near_node.parent.children.remove(near_node)
                near_node.parent = new_node
                new_node.children.append(near_node)
                self._update_costs_recursive(near_node, potential_cost)
    
    def _update_costs_recursive(self, node: Node, new_cost: float):
        cost_diff = new_cost - node.cost
        node.cost = new_cost
        for child in node.children:
            self._update_costs_recursive(child, child.cost + cost_diff)

    def plan(
        self,
        start_config: np.ndarray,
        goal_config: np.ndarray,
        max_iterations: Optional[int] = None
    ) -> Optional[List[np.ndarray]]:
        if max_iterations is None:
            max_iterations = self.max_iterations
        
        if not self.env.is_collision_free(start_config):
            return None

        if not self.env.is_collision_free(goal_config):
            return None

        start_node = Node(start_config)
        tree = [start_node]
        goal_node = None
        
        for iteration in range(max_iterations):
            # Sample configuration
            if np.random.random() < self.goal_sample_rate:
                random_config = goal_config.copy()
            else:
                random_config = self.sample_random_config()
            
            nearest = self.nearest_node(tree, random_config)
            new_config = self.steer(nearest.config, random_config)
            
            if not self.is_path_collision_free(nearest.config, new_config):
                continue
            
            new_node = Node(new_config)
            new_node.parent = nearest
            new_node.cost = nearest.cost + self.distance(nearest.config, new_config)
            nearest.children.append(new_node)
            
            # Adaptive radius
            radius = min(
                self.search_radius,
                self.search_radius * np.power(np.log(len(tree)) / len(tree), 1.0 / 7.0)
            )
            near = self.near_nodes(tree, new_config, radius)
            
            new_node = self.choose_parent(new_node, near)
            tree.append(new_node)
            self.rewire(tree, new_node, near)
            
            # Check goal
            if self.distance(new_config, goal_config) < self.goal_threshold:
                if self.is_path_collision_free(new_config, goal_config):
                    goal_node = Node(goal_config)
                    goal_node.parent = new_node
                    goal_node.cost = new_node.cost + self.distance(new_config, goal_config)
                    new_node.children.append(goal_node)
                    tree.append(goal_node)
                    break
        
        if goal_node is None:
            return None

        return self._extract_path(goal_node)
    
    def _extract_path(self, goal_node: Node) -> List[np.ndarray]:
        path = []
        current = goal_node
        while current is not None:
            path.append(current.config.copy())
            current = current.parent
        path.reverse()
        return path
    
    def smooth_path(
        self,
        path: List[np.ndarray],
        max_iterations: int = 100
    ) -> List[np.ndarray]:
        if len(path) <= 2:
            return path
        
        smoothed = [config.copy() for config in path]
        for _ in range(max_iterations):
            if len(smoothed) <= 2:
                break
            i = np.random.randint(0, len(smoothed) - 2)
            j = np.random.randint(i + 2, len(smoothed))
            if self.is_path_collision_free(smoothed[i], smoothed[j]):
                smoothed = smoothed[:i+1] + smoothed[j:]
        return smoothed
    
    def plan_to_pose(
        self,
        target_pos: np.ndarray,
        target_quat: np.ndarray,
        dt: float = 0.01,
        max_iterations: Optional[int] = None,
        max_ik_retries: int = 3
    ) -> Optional[List[np.ndarray]]:
        """Plan a path to reach a target end-effector pose.
        
        Args:
            target_pos: Target position
            target_quat: Target orientation (quaternion)
            dt: Time step for IK solver
            max_iterations: Maximum RRT iterations
            max_ik_retries: Number of times to retry IK from different starting configs
        
        Returns:
            Path as list of configurations, or None if planning failed
        """
        start_config = self.data.qpos[:7].copy()
        
        # Try IK with progressive fallback strategy
        for attempt in range(max_ik_retries):
            if attempt == 0:
                # First attempt: try from current configuration
                self.ik.update_configuration(self.data.qpos)
            else:
                # Subsequent attempts: move robot partway to a neutral config
                
                # Use a neutral configuration as fallback (middle of joint ranges)
                neutral_config = (self.joint_limits_low + self.joint_limits_high) / 2.0
                
                # Move progressively toward neutral config
                alpha = 0.3 * attempt  # Move 30%, 60%, 90% etc.
                fallback_config = (1 - alpha) * start_config + alpha * neutral_config
                
                # Temporarily move the robot in simulation
                self.data.qpos[:7] = fallback_config
                mujoco.mj_forward(self.model, self.data)
                
                # Update IK configuration
                self.ik.update_configuration(self.data.qpos)
            
            self.ik.set_target_position(target_pos, target_quat)
            
            if self.ik.converge_ik(dt):
                goal_config = self.ik.configuration.q[:7].copy()

                # Restore original configuration before planning
                self.data.qpos[:7] = start_config
                mujoco.mj_forward(self.model, self.data)

                # Plan from original start to IK solution; continue to next
                # IK seed if planning fails (different seeds reach different
                # joint-space configurations with very different reachability).
                path = self.plan(start_config, goal_config, max_iterations)
                if path is not None:
                    return path

        # Restore original configuration
        self.data.qpos[:7] = start_config
        mujoco.mj_forward(self.model, self.data)

        return None
    
    def get_path_cost(self, path: List[np.ndarray]) -> float:
        if len(path) < 2:
            return 0.0
        return sum(self.distance(path[i], path[i + 1]) for i in range(len(path) - 1))

    # ------------------------------------------------------------------
    # Traced planning
    #
    # ``plan_traced`` / ``plan_to_pose_traced`` are instrumented duplicates of
    # ``plan`` / ``plan_to_pose``.  They exist so that callers who want failure
    # diagnostics (LouisPickPlaceExecutor) get them without changing the
    # behaviour or the cost of the untraced path used by the symbolic domains.
    #
    # Deliberate behavioural differences from the untraced versions:
    #   * plan_to_pose_traced defaults to 8 IK retries (vs 3) and spreads the
    #     neutral-config blend evenly across them, instead of 0.3 * attempt.
    #   * collision diagnostics are enabled on the environment for the
    #     duration of the call.
    # Keep the search logic itself in sync with plan() when that changes.
    # ------------------------------------------------------------------

    def _reset_plan_diagnostics(self):
        self.last_plan_failure = None
        self.plan_failure_history = []
        self.last_plan_stats = None

    def _set_plan_failure(self, stage: str, reason: str, **details):
        self.last_plan_failure = {
            "stage": stage,
            "reason": reason,
            **details,
        }
        self.plan_failure_history.append(self.last_plan_failure)

    def _set_plan_success(self, stage: str, **details):
        self.last_plan_failure = None
        self.last_plan_stats = {
            "stage": stage,
            "reason": "success",
            **details,
        }

    def _get_ik_error_details(self):
        if getattr(self.ik, "ee_task", None) is None:
            return {}

        err = self.ik.ee_task.compute_error(self.ik.configuration)
        position_error = float(np.linalg.norm(err[:3]))
        orientation_error = float(np.linalg.norm(err[3:]))
        position_threshold = getattr(self.ik, "pos_threshold", None)
        orientation_threshold = getattr(self.ik, "ori_threshold", None)
        failed_thresholds = []

        if position_threshold is not None and position_error > position_threshold:
            failed_thresholds.append("position")
        if orientation_threshold is not None and orientation_error > orientation_threshold:
            failed_thresholds.append("orientation")

        return {
            "position_error": position_error,
            "orientation_error": orientation_error,
            "position_threshold": position_threshold,
            "orientation_threshold": orientation_threshold,
            "failed_thresholds": failed_thresholds,
        }

    def plan_traced(
        self,
        start_config: np.ndarray,
        goal_config: np.ndarray,
        max_iterations: Optional[int] = None
    ) -> Optional[List[np.ndarray]]:
        """Instrumented duplicate of :meth:`plan` — records why it failed."""
        if max_iterations is None:
            max_iterations = self.max_iterations

        prev_diag = getattr(self.env, "collision_diagnostics", False)
        self.env.collision_diagnostics = True
        try:
            if not self.env.is_collision_free(start_config):
                self._set_plan_failure(
                    "rrt",
                    "start_config_in_collision",
                    max_iterations=max_iterations,
                    iterations_used=0,
                    start_config=start_config.copy(),
                    collision=getattr(self.env, "last_collision", None),
                )
                return None

            if not self.env.is_collision_free(goal_config):
                self._set_plan_failure(
                    "rrt",
                    "goal_config_in_collision",
                    max_iterations=max_iterations,
                    iterations_used=0,
                    collision=getattr(self.env, "last_collision", None),
                )
                return None

            start_node = Node(start_config)
            tree = [start_node]
            goal_node = None
            collision_rejections = 0
            goal_connection_rejections = 0
            goal_samples = 0
            nodes_added = 0
            best_goal_distance = float(self.distance(start_config, goal_config))
            last_collision_rejection = None
            iterations_used = 0

            for iteration in range(max_iterations):
                iterations_used = iteration + 1
                # Sample configuration
                if np.random.random() < self.goal_sample_rate:
                    goal_samples += 1
                    random_config = goal_config.copy()
                else:
                    random_config = self.sample_random_config()

                nearest = self.nearest_node(tree, random_config)
                new_config = self.steer(nearest.config, random_config)

                if not self.is_path_collision_free(nearest.config, new_config):
                    collision_rejections += 1
                    last_collision_rejection = getattr(self.env, "last_collision", None)
                    continue

                new_node = Node(new_config)
                new_node.parent = nearest
                new_node.cost = nearest.cost + self.distance(nearest.config, new_config)
                nearest.children.append(new_node)

                # Adaptive radius
                radius = min(
                    self.search_radius,
                    self.search_radius * np.power(np.log(len(tree)) / len(tree), 1.0 / 7.0)
                )
                near = self.near_nodes(tree, new_config, radius)

                new_node = self.choose_parent(new_node, near)
                tree.append(new_node)
                nodes_added += 1
                self.rewire(tree, new_node, near)
                best_goal_distance = min(
                    best_goal_distance,
                    float(self.distance(new_config, goal_config)),
                )

                # Check goal
                if self.distance(new_config, goal_config) < self.goal_threshold:
                    if self.is_path_collision_free(new_config, goal_config):
                        goal_node = Node(goal_config)
                        goal_node.parent = new_node
                        goal_node.cost = new_node.cost + self.distance(new_config, goal_config)
                        new_node.children.append(goal_node)
                        tree.append(goal_node)
                        break
                    goal_connection_rejections += 1
                    last_collision_rejection = getattr(self.env, "last_collision", None)

            if goal_node is None:
                self._set_plan_failure(
                    "rrt",
                    "failed_to_connect_goal",
                    max_iterations=max_iterations,
                    iterations_used=iterations_used,
                    nodes_added=nodes_added,
                    tree_size=len(tree),
                    collision_rejections=collision_rejections,
                    goal_connection_rejections=goal_connection_rejections,
                    goal_samples=goal_samples,
                    start_goal_distance=float(self.distance(start_config, goal_config)),
                    best_goal_distance=best_goal_distance,
                    last_collision_rejection=last_collision_rejection,
                )
                return None

            path = self._extract_path(goal_node)
            self._set_plan_success(
                "rrt",
                max_iterations=max_iterations,
                iterations_used=iterations_used,
                nodes_added=nodes_added,
                tree_size=len(tree),
                collision_rejections=collision_rejections,
                goal_connection_rejections=goal_connection_rejections,
                goal_samples=goal_samples,
                path_length=len(path),
                path_cost=float(self.get_path_cost(path)),
            )
            return path
        finally:
            self.env.collision_diagnostics = prev_diag

    def plan_to_pose_traced(
        self,
        target_pos: np.ndarray,
        target_quat: np.ndarray,
        dt: float = 0.01,
        max_iterations: Optional[int] = None,
        max_ik_retries: int = 8
    ) -> Optional[List[np.ndarray]]:
        """Instrumented duplicate of :meth:`plan_to_pose`.

        Differs from the untraced version by design: 8 IK retries instead of 3,
        with the neutral-config blend spread evenly over them.
        """
        self._reset_plan_diagnostics()
        start_config = self.data.qpos[:7].copy()
        ik_failures = []
        rrt_failures = []
        ik_successes = 0

        # Try IK with progressive fallback strategy
        for attempt in range(max_ik_retries):
            if attempt == 0:
                # First attempt: use current configuration
                self.ik.update_configuration(self.data.qpos)
            else:
                # Subsequent attempts: blend toward the neutral configuration,
                # spread evenly across the retry budget.
                neutral_config = (self.joint_limits_low + self.joint_limits_high) / 2.0
                alpha = attempt / max(1, max_ik_retries - 1)
                fallback_config = (1 - alpha) * start_config + alpha * neutral_config

                # Temporarily move the robot in simulation
                self.data.qpos[:7] = fallback_config
                mujoco.mj_forward(self.model, self.data)
                self.ik.update_configuration(self.data.qpos)

            self.ik.set_target_position(target_pos, target_quat)

            if self.ik.converge_ik(dt):
                ik_successes += 1
                goal_config = self.ik.configuration.q[:7].copy()

                # Restore original configuration before planning
                self.data.qpos[:7] = start_config
                mujoco.mj_forward(self.model, self.data)

                path = self.plan_traced(start_config, goal_config, max_iterations)
                if path is not None:
                    self._set_plan_success(
                        "plan_to_pose",
                        ik_attempt=attempt,
                        ik_successes=ik_successes,
                        target_pos=target_pos.copy(),
                        target_quat=target_quat.copy(),
                        goal_config=goal_config.copy(),
                        path_length=len(path),
                        rrt_stats=self.last_plan_stats,
                    )
                    return path
                rrt_failures.append({
                    "attempt": attempt,
                    "goal_config": goal_config.copy(),
                    "planner_failure": (
                        self.last_plan_failure.copy()
                        if self.last_plan_failure is not None
                        else None
                    ),
                })
            else:
                ik_failures.append({
                    "attempt": attempt,
                    **self._get_ik_error_details(),
                })

        # Restore original configuration
        self.data.qpos[:7] = start_config
        mujoco.mj_forward(self.model, self.data)

        if ik_successes > 0:
            self._set_plan_failure(
                "plan_to_pose",
                "ik_succeeded_but_rrt_failed",
                attempts=max_ik_retries,
                ik_successes=ik_successes,
                ik_failures=ik_failures,
                rrt_failures=rrt_failures,
            )
        else:
            self._set_plan_failure(
                "ik",
                "ik_failed_all_attempts",
                attempts=max_ik_retries,
                ik_failures=ik_failures,
            )

        return None
