"""Grasp candidate generation for objects with unknown geometry.

Fits an oriented bounding box (OBB) to a segmented point cloud via PCA,
then delegates candidate generation to GraspPlanner.  Acts as a drop-in
replacement wherever GraspPlanner is used when true object geometry is not
available (unseen objects, real-sensor pipelines, etc.).

Algorithm
---------
1. Compute centroid and centre the cloud.
2. SVD of the centred cloud → principal axes sorted by variance descending.
3. Re-order axes so the one most aligned with world-Z becomes body-Z
   (correct for both flat boxes and upright objects).
4. Fix rotation handedness (det = +1).
5. Project cloud onto each axis → half-extents and geometric centre.
6. Pass (pos, half_size, quat) to GraspPlanner.generate_candidates().

References
----------
- Point Cloud Projective Analysis for Part-Based Grasp Planning (2020)
  https://www.researchgate.net/publication/342320540
  Direct precedent: PCA on segmented point clouds for grasp axis estimation.

- Improved PCA + ICP for workpiece point cloud pose estimation (2025)
  https://doi.org/10.1007/s00371-025-04154-7
  Validates PCA as the correct foundation for object axis estimation.

- QuickGrasp: Lightweight Antipodal Grasp Planning with Point Clouds
  arXiv:2504.19716 / IEEE ICRA 2025
  https://arxiv.org/abs/2504.19716
  Validates that analytical, no-ML grasp planning matches learning-based
  methods for tabletop manipulation.

Future upgrade path
-------------------
QuickGrasp's region-growing plane segmentation + force-closure quality
metric can replace or augment the PCA OBB step here for non-convex objects.
The interface (points → List[GraspCandidate]) would remain unchanged, so
PickPlaceExecutor and all callers are unaffected.  See generate_candidates()
for the natural extension point.
"""

from dataclasses import replace

import numpy as np
from typing import List

from tampanda.planners.grasp_planner import GraspCandidate, GraspPlanner, GraspType

# Minimum number of points required for a reliable PCA fit.
_MIN_POINTS: int = 20


class PointCloudGraspPlanner:
    """Generate grasp candidates from a segmented point cloud.

    Mirrors the GraspPlanner constructor so the two are interchangeable
    depending on whether object geometry is known:

        # known geometry
        planner = GraspPlanner(table_z=0.27)
        candidates = planner.generate_candidates(pos, half_size, quat)

        # unknown geometry — only the point cloud is needed
        planner = PointCloudGraspPlanner(table_z=0.27)
        candidates = planner.generate_candidates(points)

    An existing GraspPlanner can be injected to share its configuration:

        pc_planner = PointCloudGraspPlanner(grasp_planner=existing_planner)

    Point clouds are obtained from MujocoCamera (simulation) or any RGBD
    sensor pipeline (real hardware) — the interface is identical:

        clouds = camera.get_multi_camera_segmented_pointcloud(
            ["top_camera", "side_camera"], total_samples_per_object=2000
        )
        pts, _ = clouds["object_name"]
        candidates = pc_planner.generate_candidates(pts)
    """

    def __init__(
        self,
        grasp_planner: GraspPlanner | None = None,
        **kwargs,
    ) -> None:
        """
        Args:
            grasp_planner: Optional pre-configured GraspPlanner to delegate to.
                           If None, a new GraspPlanner is created from **kwargs.
            **kwargs:      Forwarded to GraspPlanner.__init__ when no planner
                           is provided (approach_dist, lift_height, table_z,
                           table_clearance).
        """
        self._planner = grasp_planner or GraspPlanner(**kwargs)

    @property
    def table_z(self) -> float:
        return self._planner.table_z

    def generate_candidates(self, points: np.ndarray) -> List[GraspCandidate]:
        """Return grasp candidates sorted best-first.

        Args:
            points: (N, 3) world-frame point cloud of the target object.
                    Returns [] if the cloud is too sparse for reliable fitting.

        Shape-adaptive behaviour
        ------------------------
        For tall objects (height > widest horizontal dimension):

        * The grasp target is shifted upward so top-down fingers close around
          the upper portion of the object rather than descending through the
          full body height — which would tip the object sideways.

        * FRONT (horizontal approach) candidates are boosted above TOP_DOWN in
          the ranking, since they never need to pass through the object at all.

        Future: replace or augment _fit_obb() with QuickGrasp-style plane
        segmentation + force-closure scoring for non-convex object support.
        """
        if len(points) < _MIN_POINTS:
            return []
        pos, half_size, quat = _fit_obb(points)

        # For tall objects, shift the grasp target toward the upper portion.
        # GraspPlanner places contact at block_pos, so shifting pos[2] up means
        # the fingers close higher on the object — avoiding a full descent
        # through the body.  The shift scales with how much taller the object
        # is than its widest horizontal dimension.
        horiz_max = max(half_size[0], half_size[1])
        if half_size[2] > horiz_max:
            pos = pos.copy()
            pos[2] += (half_size[2] - horiz_max) * 0.4

        candidates = self._planner.generate_candidates(pos, half_size, quat)
        return _boost_front_for_tall(candidates, half_size)


def _boost_front_for_tall(
    candidates: List[GraspCandidate],
    half_size: np.ndarray,
) -> List[GraspCandidate]:
    """Re-rank FRONT candidates above TOP_DOWN for tall objects.

    GraspPlanner._score() always adds +20 for TOP_DOWN, which works well for
    flat blocks but is wrong for tall objects where a horizontal FRONT approach
    is geometrically safer (fingers never pass through the top surface).

    When the object is taller than it is wide (aspect ratio > 1.1), FRONT
    candidates receive +30 so they rank above TOP_DOWN's baseline of +25.
    For squat/flat objects the ranking is unchanged.
    """
    horiz_max = max(half_size[0], half_size[1])
    if half_size[2] <= horiz_max * 1.1:
        return candidates

    reranked = [
        replace(c, score=c.score + 30.0) if c.grasp_type == GraspType.FRONT else c
        for c in candidates
    ]
    reranked.sort(key=lambda c: c.score, reverse=True)
    return reranked


def _fit_obb(
    points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit an oriented bounding box to a point cloud via PCA.

    Args:
        points: (N, 3) array of 3D points in world frame.

    Returns:
        pos:       geometric centre of the OBB in world frame.
        half_size: half-extents along each OBB axis.
        quat:      WXYZ quaternion of the OBB frame relative to world.
    """
    centroid = points.mean(axis=0)
    centered = points - centroid

    # PCA: columns of Vt.T are principal axes, sorted by variance descending.
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    R = Vt.T  # (3, 3) — columns are body-X, body-Y, body-Z candidate axes

    # Re-order so the axis most aligned with world-Z becomes body-Z.
    # SVD places the lowest-variance axis last, which for flat tabletop objects
    # is already the vertical axis.  For tall/upright objects the highest-
    # variance axis is vertical and needs to be moved to the last column.
    dots = np.abs(R.T @ np.array([0.0, 0.0, 1.0]))
    z_idx = int(np.argmax(dots))
    order = [i for i in range(3) if i != z_idx] + [z_idx]
    R = R[:, order]

    # Ensure proper rotation (det must be +1).
    if np.linalg.det(R) < 0:
        R[:, 2] *= -1

    # Project the centred cloud onto each OBB axis.
    proj = centered @ R                           # (N, 3)
    lo, hi = proj.min(axis=0), proj.max(axis=0)
    half_size = (hi - lo) / 2.0
    # Shift from mean centroid to geometric centre (matters for occluded clouds).
    pos = centroid + R @ ((hi + lo) / 2.0)

    quat = GraspPlanner._rotmat_to_quat(R)
    return pos, half_size, quat
