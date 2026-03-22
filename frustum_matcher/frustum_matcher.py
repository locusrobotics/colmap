"""
Frustum-Aware Spatial Matcher for COLMAP
=========================================
Generates match-pair lists using 6-DoF pose priors to eliminate
false matches in large, self-similar environments.

Instead of exhaustive or purely visual matching, this filters candidate
pairs by two geometric criteria derived from pose priors:

  1. Spatial proximity  — cameras must be within a configurable radius.
  2. Frustum overlap    — at least one camera must plausibly observe the
                          other camera's neighbourhood.

The output is a text file of image-name pairs that can be fed directly
into COLMAP's feature matcher via --match_list_path.

Requires: pycolmap, numpy, scipy
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.spatial import KDTree

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class FrustumMatcherConfig:
    """All tunables live here."""

    # --- spatial gate ---
    max_distance: float = 30.0
    """Maximum Euclidean distance (in prior coordinate units) between two
    cameras for them to be considered a candidate pair."""

    max_neighbors: int = 50
    """Cap on the number of spatial neighbours returned by the KD-tree
    query.  Keeps runtime bounded for dense trajectories."""

    # --- frustum gate ---
    horizontal_fov_deg: float = 90.0
    """Horizontal field-of-view of the camera (degrees).  Used to build
    the viewing-frustum cone.  Set conservatively wide if unsure."""

    frustum_margin_deg: float = 30.0
    """Extra angular margin added to the half-FOV cone.  Accounts for the
    fact that features near image borders can still be matched, and for
    moderate prior noise."""

    max_optical_axis_angle_deg: float = 150.0
    """Reject pairs whose optical axes diverge by more than this angle.
    Two cameras facing in nearly opposite directions can never share
    useful observations, regardless of proximity."""

    parallel_axis_threshold_deg: float = 45.0
    """Cameras whose optical axes are within this angle are considered
    'parallel' and accepted even if neither is in the other's frustum
    cone.  This handles the common trajectory case of side-by-side
    cameras looking in the same direction."""

    # --- covisibility scoring (optional ranking) ---
    enable_covisibility_scoring: bool = True
    """When True, pairs that pass the gate are ranked by a predicted
    covisibility score, and only the top-k per image are kept."""

    max_pairs_per_image: int = 30
    """If covisibility scoring is enabled, keep at most this many pairs
    per image (the highest-scoring ones)."""

    # --- prior quality ---
    min_prior_confidence: float = 0.0
    """Skip images whose max position-covariance eigenvalue exceeds this
    (metres).  Set to 0 to disable."""

    require_rotation_prior: bool = True
    """If True, images without a valid rotation prior are excluded from
    frustum filtering (they can still participate in position-only
    spatial matching with a wider gate)."""

    fallback_position_only_distance: float = 10.0
    """For images without rotation priors, use this tighter position-only
    radius so that the lack of frustum filtering doesn't let in too
    many false matches."""


# ---------------------------------------------------------------------------
# Per-image prior data (extracted from the database)
# ---------------------------------------------------------------------------

@dataclass
class ImagePrior:
    image_id: int
    image_name: str
    position: np.ndarray            # (3,) world-frame position
    optical_axis: Optional[np.ndarray] = None   # (3,) unit vector, world frame
    rotation: Optional[np.ndarray] = None       # (4,) quaternion wxyz
    position_covariance: Optional[np.ndarray] = None
    has_rotation: bool = False


# ---------------------------------------------------------------------------
# Core matcher
# ---------------------------------------------------------------------------

class FrustumAwareMatcher:
    """Builds a filtered match-pair list from 6-DoF pose priors."""

    def __init__(self, config: FrustumMatcherConfig | None = None):
        self.config = config or FrustumMatcherConfig()
        self._image_priors: dict[int, ImagePrior] = {}
        self._kdtree: Optional[KDTree] = None
        self._ordered_ids: list[int] = []

    # ------------------------------------------------------------------
    # 1.  Load priors from a COLMAP database
    # ------------------------------------------------------------------

    def load_from_database(self, database_path: str | Path) -> int:
        """Read pose priors from a COLMAP SQLite database via pycolmap.

        Returns the number of images with valid priors.
        """
        database_path = Path(database_path)
        if not database_path.exists():
            raise FileNotFoundError(f"Database not found: {database_path}")

        try:
            import pycolmap

            db = pycolmap.Database(str(database_path))
            images = db.read_all_images()

            num_loaded = 0
            for image_id, image in images.items():
                prior = db.read_pose_prior(image_id)
                if prior is None or not prior.is_valid():
                    continue

                pos = np.array(prior.position)
                if not np.all(np.isfinite(pos)):
                    continue

                ip = ImagePrior(
                    image_id=image_id,
                    image_name=image.name,
                    position=pos,
                )

                if prior.has_rotation():
                    rotvec = np.array(prior.rotation, dtype=np.float64)
                    ip.rotation = rotvec
                    ip.has_rotation = True

                    R_cw = _rotvec_to_rotation_matrix(rotvec)
                    optical_axis = R_cw[2, :]
                    ip.optical_axis = optical_axis / np.linalg.norm(optical_axis)

                if prior.is_covariance_valid() and prior.covariance.shape[0] >= 3:
                    ip.position_covariance = np.array(prior.covariance)[:3, :3]

                self._image_priors[image_id] = ip
                num_loaded += 1

            logger.info("Loaded %d images with valid pose priors via pycolmap.", num_loaded)
            return num_loaded

        except ImportError:
            logger.warning("pycolmap not available, falling back to direct SQLite parsing.")
            return self._load_from_sqlite(database_path)

    def _load_from_sqlite(self, database_path: Path) -> int:
        conn = sqlite3.connect(str(database_path))
        try:
            cursor = conn.execute(
                """
                SELECT images.image_id, images.name, pose_priors.position,
                       pose_priors.rotation, pose_priors.covariance
                FROM images
                JOIN pose_priors ON images.image_id = pose_priors.image_id
                """
            )

            num_loaded = 0
            for image_id, image_name, pos_blob, rot_blob, cov_blob in cursor:
                if pos_blob is None:
                    continue

                pos = np.frombuffer(pos_blob, dtype=np.float64)
                if pos.size != 3 or not np.all(np.isfinite(pos)):
                    continue

                ip = ImagePrior(
                    image_id=int(image_id),
                    image_name=str(image_name),
                    position=pos.copy(),
                )

                if rot_blob is not None:
                    rotvec = np.frombuffer(rot_blob, dtype=np.float64)
                    if rotvec.size == 3 and np.all(np.isfinite(rotvec)):
                        ip.rotation = rotvec.copy()
                        ip.has_rotation = True
                        R_cw = _rotvec_to_rotation_matrix(rotvec)
                        optical_axis = R_cw[2, :]
                        ip.optical_axis = optical_axis / np.linalg.norm(optical_axis)

                if cov_blob is not None:
                    cov_flat = np.frombuffer(cov_blob, dtype=np.float64)
                    if cov_flat.size == 9:
                        ip.position_covariance = cov_flat.reshape(3, 3).copy()
                    elif cov_flat.size == 36:
                        ip.position_covariance = cov_flat.reshape(6, 6)[:3, :3].copy()

                self._image_priors[int(image_id)] = ip
                num_loaded += 1

            logger.info("Loaded %d images with valid pose priors via SQLite fallback.", num_loaded)
            return num_loaded
        finally:
            conn.close()

    def load_from_arrays(
        self,
        image_ids: list[int],
        image_names: list[str],
        positions: np.ndarray,
        rotations: Optional[np.ndarray] = None,
    ) -> int:
        """Load priors from numpy arrays (useful for testing or non-COLMAP
        pipelines).

        Args:
            positions: (N, 3) world-frame positions.
            rotations: (N, 4) quaternions [w, x, y, z], or None.
        """
        num_images = len(image_ids)
        if len(image_names) != num_images:
            raise ValueError("image_ids and image_names must have the same length")

        if positions.shape != (num_images, 3):
            raise ValueError("positions must have shape (N, 3)")

        if rotations is not None and rotations.shape != (num_images, 4):
            raise ValueError("rotations must have shape (N, 4)")

        for i, (img_id, name) in enumerate(zip(image_ids, image_names)):
            ip = ImagePrior(
                image_id=img_id,
                image_name=name,
                position=positions[i],
            )
            if rotations is not None and np.all(np.isfinite(rotations[i])):
                qw, qx, qy, qz = rotations[i]
                ip.rotation = rotations[i]
                ip.has_rotation = True
                R_cw = _quat_to_rotation_matrix(qw, qx, qy, qz)
                optical_axis = R_cw[2, :]
                ip.optical_axis = optical_axis / np.linalg.norm(optical_axis)

            self._image_priors[img_id] = ip

        logger.info("Loaded %d images from arrays.", len(image_ids))
        return len(image_ids)

    # ------------------------------------------------------------------
    # 2.  Build spatial index
    # ------------------------------------------------------------------

    def _build_index(self) -> None:
        self._ordered_ids = list(self._image_priors.keys())
        positions = np.array(
            [self._image_priors[i].position for i in self._ordered_ids]
        )
        self._kdtree = KDTree(positions)
        logger.info("Built KD-tree over %d positions.", len(self._ordered_ids))

    # ------------------------------------------------------------------
    # 3.  Pair filtering
    # ------------------------------------------------------------------

    def compute_pairs(self) -> list[tuple[str, str, float]]:
        """Compute filtered match pairs.

        Returns a list of (image_name_i, image_name_j, score) tuples,
        sorted by descending score within each image's neighbourhood.
        """
        cfg = self.config
        self._build_index()

        half_cone_rad = np.radians(
            cfg.horizontal_fov_deg / 2.0 + cfg.frustum_margin_deg
        )
        max_axis_angle_rad = np.radians(cfg.max_optical_axis_angle_deg)

        # Collect candidates per image (we'll deduplicate later)
        per_image_candidates: dict[int, list[tuple[int, float]]] = {
            i: [] for i in self._ordered_ids
        }

        # Query KD-tree for each image
        positions = np.array(
            [self._image_priors[i].position for i in self._ordered_ids]
        )

        distances, indices = self._kdtree.query(
            positions,
            k=min(cfg.max_neighbors + 1, len(self._ordered_ids)),
            distance_upper_bound=cfg.max_distance,
        )

        for row_idx, img_id_i in enumerate(self._ordered_ids):
            prior_i = self._image_priors[img_id_i]

            for col_idx in range(distances.shape[1]):
                dist = distances[row_idx, col_idx]
                neighbor_row = indices[row_idx, col_idx]

                # KDTree pads with inf / len(data) for missing neighbors
                if not np.isfinite(dist) or neighbor_row >= len(self._ordered_ids):
                    continue

                img_id_j = self._ordered_ids[neighbor_row]
                if img_id_i >= img_id_j:
                    continue  # avoid self-pairs and duplicates

                prior_j = self._image_priors[img_id_j]
                both_have_rotation = prior_i.has_rotation and prior_j.has_rotation

                # --- Gate: no rotation priors → fall back to tighter radius ---
                if not both_have_rotation:
                    if dist <= cfg.fallback_position_only_distance:
                        score = 1.0 / (1.0 + dist)
                        per_image_candidates[img_id_i].append((img_id_j, score))
                        per_image_candidates[img_id_j].append((img_id_i, score))
                    continue

                # --- Gate 1: baseline-in-frustum ---
                # Check whether the baseline direction falls within either
                # camera's (expanded) frustum cone.  This is the primary
                # geometric filter and naturally handles both converging
                # cameras (anti-parallel axes) and parallel cameras.
                baseline = prior_j.position - prior_i.position
                baseline_norm = np.linalg.norm(baseline)
                if baseline_norm < 1e-9:
                    # Co-located cameras — always a candidate
                    score = 1.0
                    per_image_candidates[img_id_i].append((img_id_j, score))
                    per_image_candidates[img_id_j].append((img_id_i, score))
                    continue

                baseline_dir = baseline / baseline_norm

                # Is camera j inside camera i's frustum cone?
                angle_i = _angle_between(prior_i.optical_axis, baseline_dir)
                # Is camera i inside camera j's frustum cone?
                angle_j = _angle_between(prior_j.optical_axis, -baseline_dir)

                in_frustum_i = angle_i <= half_cone_rad
                in_frustum_j = angle_j <= half_cone_rad

                # --- Optical axis angle (used by multiple gates) ---
                axis_angle = _angle_between(prior_i.optical_axis, prior_j.optical_axis)

                # --- Acceptance logic (3 tiers) ---
                #
                # Tier 1 — MUTUAL frustum: both cameras see each other.
                #   Strongest evidence of covisibility.  Accept, unless
                #   the axes are anti-parallel AND the baseline is only
                #   marginally inside the cones (diagonal cross-corridor
                #   trap).  For truly converging cameras the baseline
                #   is tightly aligned with the optical axes.
                #
                # Tier 2 — PARALLEL nearby: optical axes nearly aligned.
                #   Handles side-by-side trajectory cameras where the
                #   baseline is perpendicular to both viewing directions.
                #
                # Tier 3 — SINGLE frustum: one camera sees the other,
                #   but not vice-versa.  Only accept if the axis angle
                #   is below the divergence cap.

                parallel_threshold_rad = np.radians(cfg.parallel_axis_threshold_deg)
                is_parallel_nearby = (axis_angle <= parallel_threshold_rad)

                # Convergence check: when axes diverge > 90°, the pair is
                # only valid if the baseline is well-aligned with both
                # optical axes (i.e. the cameras genuinely look at each
                # other).  We require the baseline to sit in the inner
                # half of both frustum cones.
                anti_parallel = axis_angle > np.pi / 2.0
                convergence_cone = half_cone_rad * 0.5

                accepted = False

                if in_frustum_i and in_frustum_j:
                    if anti_parallel:
                        # Tier 1a: mutual frustum + anti-parallel →
                        # only accept if baseline is tightly aligned
                        # (truly converging, not diagonal)
                        if angle_i <= convergence_cone and angle_j <= convergence_cone:
                            accepted = True
                    else:
                        # Tier 1b: mutual frustum + compatible axes → accept
                        accepted = True
                elif is_parallel_nearby:
                    # Tier 2: parallel nearby → accept
                    accepted = True
                elif (in_frustum_i or in_frustum_j):
                    # Tier 3: single frustum → accept only if axes
                    # aren't wildly divergent
                    if axis_angle <= max_axis_angle_rad:
                        accepted = True

                if not accepted:
                    continue

                # --- Covisibility score ---
                score = _covisibility_score(
                    dist, baseline_norm, axis_angle,
                    angle_i, angle_j,
                    half_cone_rad, cfg.max_distance,
                )

                per_image_candidates[img_id_i].append((img_id_j, score))
                per_image_candidates[img_id_j].append((img_id_i, score))

        # --- Top-k per image (if scoring enabled) ---
        if cfg.enable_covisibility_scoring:
            for img_id in per_image_candidates:
                candidates = per_image_candidates[img_id]
                candidates.sort(key=lambda x: x[1], reverse=True)
                per_image_candidates[img_id] = candidates[: cfg.max_pairs_per_image]

        # --- Deduplicate into a canonical pair set ---
        seen: set[tuple[int, int]] = set()
        result: list[tuple[str, str, float]] = []

        for img_id_i, candidates in per_image_candidates.items():
            for img_id_j, score in candidates:
                pair = (min(img_id_i, img_id_j), max(img_id_i, img_id_j))
                if pair not in seen:
                    seen.add(pair)
                    name_i = self._image_priors[pair[0]].image_name
                    name_j = self._image_priors[pair[1]].image_name
                    result.append((name_i, name_j, score))

        result.sort(key=lambda x: x[2], reverse=True)
        logger.info(
            "Generated %d candidate pairs from %d images (%.1f%% of exhaustive).",
            len(result),
            len(self._ordered_ids),
            100.0 * len(result) / max(1, len(self._ordered_ids) * (len(self._ordered_ids) - 1) / 2),
        )
        return result

    # ------------------------------------------------------------------
    # 4.  Export for COLMAP
    # ------------------------------------------------------------------

    def write_match_list(
        self,
        pairs: list[tuple[str, str, float]],
        output_path: str | Path,
    ) -> Path:
        """Write pairs to a text file for COLMAP's --match_list_path.

        Format: one pair per line, space-separated image names.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            for name_i, name_j, _ in pairs:
                f.write(f"{name_i} {name_j}\n")

        logger.info("Wrote %d pairs to %s", len(pairs), output_path)
        return output_path

    def get_pair_statistics(
        self, pairs: list[tuple[str, str, float]]
    ) -> dict:
        """Return summary statistics about the generated pairs."""
        if not pairs:
            return {"num_pairs": 0}

        num_images = len(self._image_priors)
        num_exhaustive = num_images * (num_images - 1) // 2
        scores = [s for _, _, s in pairs]

        pairs_per_image: dict[str, int] = {}
        for n_i, n_j, _ in pairs:
            pairs_per_image[n_i] = pairs_per_image.get(n_i, 0) + 1
            pairs_per_image[n_j] = pairs_per_image.get(n_j, 0) + 1

        counts = list(pairs_per_image.values())

        return {
            "num_images": num_images,
            "num_pairs": len(pairs),
            "num_exhaustive": num_exhaustive,
            "reduction_factor": num_exhaustive / max(1, len(pairs)),
            "score_min": float(np.min(scores)),
            "score_median": float(np.median(scores)),
            "score_max": float(np.max(scores)),
            "pairs_per_image_min": int(np.min(counts)) if counts else 0,
            "pairs_per_image_median": float(np.median(counts)) if counts else 0,
            "pairs_per_image_max": int(np.max(counts)) if counts else 0,
        }


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _quat_to_rotation_matrix(w: float, x: float, y: float, z: float) -> np.ndarray:
    """Convert quaternion (w, x, y, z) to a 3×3 rotation matrix."""
    n = np.sqrt(w * w + x * x + y * y + z * z)
    if n < 1e-12:
        return np.eye(3)
    w, x, y, z = w / n, x / n, y / n, z / n

    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ])


def _angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    """Angle in radians between two unit vectors (clamped for numerical safety)."""
    dot = np.clip(np.dot(v1, v2), -1.0, 1.0)
    return np.arccos(dot)


def _rotvec_to_rotation_matrix(rotvec: np.ndarray) -> np.ndarray:
    """Convert axis-angle rotation vector to a 3x3 rotation matrix."""
    theta = float(np.linalg.norm(rotvec))
    if theta < 1e-12:
        return np.eye(3)

    axis = rotvec / theta
    x, y, z = axis

    K = np.array([
        [0.0, -z, y],
        [z, 0.0, -x],
        [-y, x, 0.0],
    ])
    I = np.eye(3)
    return I + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def _covisibility_score(
    distance: float,
    baseline: float,
    axis_angle: float,
    angle_i: float,
    angle_j: float,
    half_cone: float,
    max_distance: float,
) -> float:
    """Heuristic covisibility score ∈ (0, 1].

    Rewards:
      - Small distance (nearby cameras share more content)
      - Small optical-axis divergence (similar viewing direction)
      - Baseline direction centred in both frustums
    """
    # Distance component: Gaussian fall-off
    sigma_d = max_distance / 3.0
    s_dist = np.exp(-0.5 * (distance / sigma_d) ** 2)

    # Axis alignment: cosine similarity, mapped to [0, 1]
    s_axis = max(0.0, np.cos(axis_angle))

    # Frustum centrality: how centred is the baseline in each frustum?
    # 1.0 when on-axis, 0.0 at the frustum edge
    s_frust_i = max(0.0, 1.0 - angle_i / half_cone) if half_cone > 0 else 0.0
    s_frust_j = max(0.0, 1.0 - angle_j / half_cone) if half_cone > 0 else 0.0
    s_frust = max(s_frust_i, s_frust_j)

    # Weighted combination
    return 0.4 * s_dist + 0.3 * s_axis + 0.3 * s_frust
