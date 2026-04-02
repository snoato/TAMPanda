"""A* navigation planner for differential-drive mobile robots.

Builds a 2D occupancy grid directly from scene geometry using Minkowski sum
expansion (robot footprint inflated around each obstacle), then runs A* to
find a shortest path between two positions. A line-of-sight string-pulling
pass prunes unnecessary waypoints.

Typical usage::

    from tampanda.planners import AStarNav

    planner = AStarNav(env, x_range=(-2, 2), y_range=(-2, 2), resolution=0.05)
    path = planner.plan((0.0, 0.0), (1.5, 1.5))   # list of (x, y) waypoints
    path = planner.smooth_path(path)
"""

from __future__ import annotations

import heapq
import warnings
from typing import List, Optional, Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from tampanda.core.mobile_robot_env import MobileRobotEnvironment


class AStarNav:
    """Grid-based A* planner for 2D navigation.

    The occupancy grid is built from MuJoCo scene geometry: each obstacle geom
    is expanded by the robot's circumscribed radius (Minkowski sum), then
    rasterized onto the grid.  This replaces the old collision-probe approach
    (which called ``mj_forward`` for every cell) with a direct geometry query,
    which is orders of magnitude faster.

    Supported geom types:
      - Sphere, Cylinder   → circle in XY
      - Capsule            → stadium (segment + radius) in XY
      - Box                → oriented rounded rectangle in XY
      - Mesh               → 2D convex hull of projected vertices (requires scipy;
                             falls back to bounding sphere without it)
      - Plane              → silently skipped (floor)
      - HeightField        → warning issued, skipped
      - Unknown            → warning issued, bounding sphere used

    Args:
        env:                Mobile environment (must implement
                            ``get_model()``, ``get_data()``,
                            ``get_robot_body_ids()``).
        x_range:            ``(x_min, x_max)`` workspace bounds in metres.
        y_range:            ``(y_min, y_max)`` workspace bounds in metres.
        resolution:         Grid cell size in metres.
        robot_radius:       Circumscribed robot radius (m).  ``None`` =
                            auto-detect from the robot's own geoms.
        robot_radius_buffer: Extra clearance added on top of the detected or
                            provided robot radius (m).
    """

    def __init__(
        self,
        env: "MobileRobotEnvironment",
        *,
        x_range: Tuple[float, float],
        y_range: Tuple[float, float],
        resolution: float = 0.05,
        robot_radius: Optional[float] = None,
        robot_radius_buffer: float = 0.0,
    ):
        self._env = env
        self._resolution = resolution
        self._x_min, self._x_max = x_range
        self._y_min, self._y_max = y_range

        self._nx = max(1, int(np.ceil((self._x_max - self._x_min) / resolution)) + 1)
        self._ny = max(1, int(np.ceil((self._y_max - self._y_min) / resolution)) + 1)

        if robot_radius is None:
            robot_radius = self._compute_robot_radius()
        self._robot_radius = robot_radius + robot_radius_buffer

        print(
            f"[AStarNav] Building {self._nx}×{self._ny} occupancy grid "
            f"({self._nx * self._ny} cells, resolution={resolution} m, "
            f"robot_radius={self._robot_radius:.3f} m) …"
        )
        self._grid = self._build_grid_from_geoms()

        occupied = int(self._grid.sum())
        print(
            f"[AStarNav] Grid ready — {occupied}/{self._nx * self._ny} cells occupied."
        )

    # ------------------------------------------------------------------
    # Robot radius auto-detection
    # ------------------------------------------------------------------

    def _compute_robot_radius(self) -> float:
        """Compute circumscribed robot radius from geom extents in XY."""
        import mujoco
        model = self._env.get_model()
        data = self._env.get_data()
        robot_body_ids = self._env.get_robot_body_ids()

        r_max = 0.0
        for gid in range(model.ngeom):
            if model.geom_bodyid[gid] not in robot_body_ids:
                continue
            cx = float(data.geom_xpos[gid, 0])
            cy = float(data.geom_xpos[gid, 1])
            gtype = model.geom_type[gid]
            size = model.geom_size[gid]

            if gtype == mujoco.mjtGeom.mjGEOM_SPHERE:
                r_geom = float(size[0])
            elif gtype == mujoco.mjtGeom.mjGEOM_CYLINDER:
                r_geom = float(size[0])
            elif gtype == mujoco.mjtGeom.mjGEOM_CAPSULE:
                r_geom = float(size[0])
            elif gtype == mujoco.mjtGeom.mjGEOM_BOX:
                r_geom = float(np.hypot(size[0], size[1]))
            else:
                r_geom = float(model.geom_rbound[gid])

            r_max = max(r_max, np.hypot(cx, cy) + r_geom)

        return r_max

    # ------------------------------------------------------------------
    # Geometry-based grid construction
    # ------------------------------------------------------------------

    def _build_grid_from_geoms(self) -> np.ndarray:
        """Build bool occupancy grid by rasterizing each obstacle geom."""
        import mujoco
        model = self._env.get_model()
        data = self._env.get_data()
        robot_body_ids = self._env.get_robot_body_ids()

        grid = np.zeros((self._nx, self._ny), dtype=bool)

        for gid in range(model.ngeom):
            body_id = int(model.geom_bodyid[gid])
            if body_id in robot_body_ids:
                continue

            gtype = int(model.geom_type[gid])

            if gtype == int(mujoco.mjtGeom.mjGEOM_PLANE):
                continue  # floor — skip silently

            pos2 = data.geom_xpos[gid, :2].copy()
            mat3 = data.geom_xmat[gid].reshape(3, 3)
            size = model.geom_size[gid]

            if gtype == int(mujoco.mjtGeom.mjGEOM_HFIELD):
                warnings.warn(
                    f"[AStarNav] Geom {gid} (body {body_id}) is a heightfield — "
                    "not supported for geometry-based grid, skipping."
                )
                continue

            elif gtype in (int(mujoco.mjtGeom.mjGEOM_SPHERE),
                           int(mujoco.mjtGeom.mjGEOM_CYLINDER)):
                self._mark_circle(grid, pos2, float(size[0]))

            elif gtype == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
                axis_w = mat3[:, 2]
                half_len = float(size[1])
                p1 = data.geom_xpos[gid, :2] + axis_w[:2] * half_len
                p2 = data.geom_xpos[gid, :2] - axis_w[:2] * half_len
                self._mark_stadium(grid, p1, p2, float(size[0]))

            elif gtype == int(mujoco.mjtGeom.mjGEOM_BOX):
                self._mark_rounded_box(grid, pos2, mat3[:2, :2], size[:2].copy())

            elif gtype == int(mujoco.mjtGeom.mjGEOM_MESH):
                mid = int(model.geom_dataid[gid])
                if mid < 0:
                    warnings.warn(
                        f"[AStarNav] Mesh geom {gid} has no mesh data — "
                        "using bounding sphere."
                    )
                    self._mark_circle(grid, pos2, float(model.geom_rbound[gid]))
                else:
                    adr = int(model.mesh_vertadr[mid])
                    num = int(model.mesh_vertnum[mid])
                    verts = model.mesh_vert[adr: adr + num]        # (N, 3)
                    world_verts = (mat3 @ verts.T).T + data.geom_xpos[gid]
                    self._mark_convex_hull(grid, world_verts[:, :2])

            else:
                warnings.warn(
                    f"[AStarNav] Geom {gid} (body {body_id}) has unknown type "
                    f"{gtype} — using bounding sphere."
                )
                self._mark_circle(grid, pos2, float(model.geom_rbound[gid]))

        return grid

    # ------------------------------------------------------------------
    # Rasterization helpers
    # ------------------------------------------------------------------

    def _cells_in_bbox(
        self, x_min: float, x_max: float, y_min: float, y_max: float
    ) -> Tuple[int, int, int, int]:
        """Return grid index ranges for a world-frame bounding box."""
        ix_min = max(0, int(np.floor((x_min - self._x_min) / self._resolution)))
        ix_max = min(self._nx, int(np.ceil((x_max - self._x_min) / self._resolution)) + 1)
        iy_min = max(0, int(np.floor((y_min - self._y_min) / self._resolution)))
        iy_max = min(self._ny, int(np.ceil((y_max - self._y_min) / self._resolution)) + 1)
        return ix_min, ix_max, iy_min, iy_max

    def _cell_meshgrid(
        self, ix_min: int, ix_max: int, iy_min: int, iy_max: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return world-coordinate meshgrids for a cell range."""
        cx = self._x_min + np.arange(ix_min, ix_max) * self._resolution
        cy = self._y_min + np.arange(iy_min, iy_max) * self._resolution
        return np.meshgrid(cx, cy, indexing="ij")

    def _mark_circle(self, grid: np.ndarray, center: np.ndarray, obstacle_radius: float):
        """Mark cells within (obstacle_radius + robot_radius) of center."""
        r = obstacle_radius + self._robot_radius
        ix_min, ix_max, iy_min, iy_max = self._cells_in_bbox(
            center[0] - r, center[0] + r, center[1] - r, center[1] + r
        )
        if ix_min >= ix_max or iy_min >= iy_max:
            return
        CX, CY = self._cell_meshgrid(ix_min, ix_max, iy_min, iy_max)
        grid[ix_min:ix_max, iy_min:iy_max] |= (
            (CX - center[0]) ** 2 + (CY - center[1]) ** 2 <= r ** 2
        )

    def _mark_stadium(
        self,
        grid: np.ndarray,
        p1: np.ndarray,
        p2: np.ndarray,
        capsule_radius: float,
    ):
        """Mark cells within (capsule_radius + robot_radius) of the segment p1–p2."""
        r = capsule_radius + self._robot_radius
        ix_min, ix_max, iy_min, iy_max = self._cells_in_bbox(
            min(p1[0], p2[0]) - r, max(p1[0], p2[0]) + r,
            min(p1[1], p2[1]) - r, max(p1[1], p2[1]) + r,
        )
        if ix_min >= ix_max or iy_min >= iy_max:
            return
        CX, CY = self._cell_meshgrid(ix_min, ix_max, iy_min, iy_max)
        seg = p2 - p1
        seg_len2 = float(np.dot(seg, seg))
        if seg_len2 < 1e-12:
            d2 = (CX - p1[0]) ** 2 + (CY - p1[1]) ** 2
        else:
            ax = CX - p1[0]
            ay = CY - p1[1]
            t = np.clip((ax * seg[0] + ay * seg[1]) / seg_len2, 0.0, 1.0)
            d2 = (CX - (p1[0] + t * seg[0])) ** 2 + (CY - (p1[1] + t * seg[1])) ** 2
        grid[ix_min:ix_max, iy_min:iy_max] |= (d2 <= r ** 2)

    def _mark_rounded_box(
        self,
        grid: np.ndarray,
        center: np.ndarray,
        R2: np.ndarray,
        half_extents: np.ndarray,
    ):
        """Mark cells within robot_radius of an oriented box (Minkowski sum)."""
        r = self._robot_radius
        bbox_r = float(np.hypot(half_extents[0], half_extents[1])) + r
        ix_min, ix_max, iy_min, iy_max = self._cells_in_bbox(
            center[0] - bbox_r, center[0] + bbox_r,
            center[1] - bbox_r, center[1] + bbox_r,
        )
        if ix_min >= ix_max or iy_min >= iy_max:
            return
        CX, CY = self._cell_meshgrid(ix_min, ix_max, iy_min, iy_max)
        dx = CX - center[0]
        dy = CY - center[1]
        # Transform to box local frame (R2.T @ [dx, dy])
        lx = R2[0, 0] * dx + R2[1, 0] * dy
        ly = R2[0, 1] * dx + R2[1, 1] * dy
        ex = np.maximum(np.abs(lx) - half_extents[0], 0.0)
        ey = np.maximum(np.abs(ly) - half_extents[1], 0.0)
        grid[ix_min:ix_max, iy_min:iy_max] |= (ex ** 2 + ey ** 2 <= r ** 2)

    def _mark_convex_hull(self, grid: np.ndarray, xy_verts: np.ndarray):
        """Mark cells within robot_radius of the 2D convex hull of xy_verts."""
        r = self._robot_radius
        try:
            from scipy.spatial import ConvexHull
            hull = ConvexHull(xy_verts)
            hull_verts = xy_verts[hull.vertices]  # ordered CCW
        except Exception:
            # scipy unavailable or degenerate hull — fall back to bounding circle
            center = xy_verts.mean(axis=0)
            r_bound = float(np.max(np.linalg.norm(xy_verts - center, axis=1)))
            self._mark_circle(grid, center, r_bound)
            return

        ix_min, ix_max, iy_min, iy_max = self._cells_in_bbox(
            hull_verts[:, 0].min() - r, hull_verts[:, 0].max() + r,
            hull_verts[:, 1].min() - r, hull_verts[:, 1].max() + r,
        )
        if ix_min >= ix_max or iy_min >= iy_max:
            return
        CX, CY = self._cell_meshgrid(ix_min, ix_max, iy_min, iy_max)
        pts = np.stack([CX.ravel(), CY.ravel()], axis=1)  # (N, 2)

        # Inside check via hull equations: a·x + b·y + c ≤ 0 for interior
        lhs = pts @ hull.equations[:, :2].T + hull.equations[:, 2]  # (N, n_facets)
        inside = np.all(lhs <= 0.0, axis=1)  # (N,)

        # Distance to nearest edge for outside points
        n = len(hull_verts)
        min_d2 = np.full(len(pts), np.inf)
        for i in range(n):
            a = hull_verts[i]
            b = hull_verts[(i + 1) % n]
            seg = b - a
            seg_len2 = float(np.dot(seg, seg))
            if seg_len2 < 1e-12:
                d2 = np.sum((pts - a) ** 2, axis=1)
            else:
                t = np.clip(((pts - a) @ seg) / seg_len2, 0.0, 1.0)
                closest = a + t[:, None] * seg
                d2 = np.sum((pts - closest) ** 2, axis=1)
            min_d2 = np.minimum(min_d2, d2)

        mask = (inside | (min_d2 <= r ** 2)).reshape(CX.shape)
        grid[ix_min:ix_max, iy_min:iy_max] |= mask

    # ------------------------------------------------------------------
    # Grid helpers
    # ------------------------------------------------------------------

    def _world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        ix = int(round((x - self._x_min) / self._resolution))
        iy = int(round((y - self._y_min) / self._resolution))
        return (
            max(0, min(self._nx - 1, ix)),
            max(0, min(self._ny - 1, iy)),
        )

    def _grid_to_world(self, ix: int, iy: int) -> Tuple[float, float]:
        return (
            self._x_min + ix * self._resolution,
            self._y_min + iy * self._resolution,
        )

    def is_free(self, ix: int, iy: int) -> bool:
        if ix < 0 or ix >= self._nx or iy < 0 or iy >= self._ny:
            return False
        return not self._grid[ix, iy]

    # ------------------------------------------------------------------
    # Path planning
    # ------------------------------------------------------------------

    def plan(
        self,
        start: Tuple[float, float],
        goal: Tuple[float, float],
    ) -> Optional[List[Tuple[float, float]]]:
        """Run A* from *start* to *goal*.

        Args:
            start: ``(x, y)`` start position in world frame.
            goal:  ``(x, y)`` goal position in world frame.

        Returns:
            List of ``(x, y)`` waypoints from start to goal (inclusive), or
            ``None`` if no path was found.
        """
        gs = self._world_to_grid(*start)
        gg = self._world_to_grid(*goal)

        if not self.is_free(*gs):
            print(f"[AStarNav] Start {start} is in collision (grid {gs}).")
            return None
        if not self.is_free(*gg):
            print(f"[AStarNav] Goal {goal} is in collision (grid {gg}).")
            return None

        # 8-connected neighbours with diagonal cost √2
        neighbours = [
            (-1,  0, 1.0), ( 1,  0, 1.0), ( 0, -1, 1.0), ( 0,  1, 1.0),
            (-1, -1, 1.414), (-1,  1, 1.414), ( 1, -1, 1.414), ( 1,  1, 1.414),
        ]

        def h(ix, iy):
            # Octile heuristic (consistent with 8-connectivity)
            dx = abs(ix - gg[0])
            dy = abs(iy - gg[1])
            return max(dx, dy) + (1.414 - 1.0) * min(dx, dy)

        open_heap: list = []
        heapq.heappush(open_heap, (h(*gs), 0.0, gs))
        g_cost: dict[Tuple[int, int], float] = {gs: 0.0}
        came_from: dict[Tuple[int, int], Optional[Tuple[int, int]]] = {gs: None}

        while open_heap:
            _, g, node = heapq.heappop(open_heap)
            if node == gg:
                return self._reconstruct(came_from, gg)
            if g > g_cost.get(node, float("inf")):
                continue  # stale entry
            ix, iy = node
            for ddx, ddy, cost in neighbours:
                nb = (ix + ddx, iy + ddy)
                if not self.is_free(*nb):
                    continue
                ng = g + cost
                if ng < g_cost.get(nb, float("inf")):
                    g_cost[nb] = ng
                    came_from[nb] = node
                    f = ng + h(*nb)
                    heapq.heappush(open_heap, (f, ng, nb))

        return None  # no path found

    def _reconstruct(
        self,
        came_from: dict,
        node: Tuple[int, int],
    ) -> List[Tuple[float, float]]:
        path_grid = []
        while node is not None:
            path_grid.append(node)
            node = came_from[node]
        path_grid.reverse()
        return [self._grid_to_world(ix, iy) for ix, iy in path_grid]

    # ------------------------------------------------------------------
    # Path smoothing (line-of-sight string pulling)
    # ------------------------------------------------------------------

    def smooth_path(
        self,
        path: List[Tuple[float, float]],
    ) -> List[Tuple[float, float]]:
        """Remove collinear and occluded waypoints using a greedy LoS pass.

        Works on the grid: a direct connection between two waypoints is kept
        only if every cell on the Bresenham line between them is free.

        Args:
            path: Raw A* path from :meth:`plan`.

        Returns:
            Pruned path (always includes start and goal).
        """
        if len(path) <= 2:
            return path

        smoothed = [path[0]]
        anchor = 0
        while anchor < len(path) - 1:
            reach = anchor + 1
            for candidate in range(len(path) - 1, anchor, -1):
                p1 = self._world_to_grid(*path[anchor])
                p2 = self._world_to_grid(*path[candidate])
                if self._los_clear(p1, p2):
                    reach = candidate
                    break
            smoothed.append(path[reach])
            anchor = reach

        return smoothed

    def _los_clear(self, p1: Tuple[int, int], p2: Tuple[int, int]) -> bool:
        """Bresenham line-of-sight check between two grid cells."""
        x0, y0 = p1
        x1, y1 = p2
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        x, y = x0, y0
        while True:
            if not self.is_free(x, y):
                return False
            if x == x1 and y == y1:
                return True
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy

    # ------------------------------------------------------------------
    # Occupancy grid access
    # ------------------------------------------------------------------

    @property
    def grid(self) -> np.ndarray:
        """Bool occupancy grid (nx × ny); True = obstacle."""
        return self._grid

    def grid_shape(self) -> Tuple[int, int]:
        return self._nx, self._ny

    def to_image(self) -> np.ndarray:
        """Return an (ny, nx) uint8 image: 255 = free, 0 = occupied."""
        return np.where(self._grid.T, 0, 255).astype(np.uint8)
