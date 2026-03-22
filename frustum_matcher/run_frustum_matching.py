#!/usr/bin/env python3
"""
run_frustum_matching.py
========================
CLI tool that generates a frustum-filtered match-pair list and optionally
runs COLMAP feature matching with it.

Usage
-----
  # Generate match list only:
    python -m frustum_matcher.run_frustum_matching \
      --database_path /path/to/database.db \
      --output_path /path/to/match_list.txt

  # Generate and immediately run COLMAP matching + reconstruction:
    python -m frustum_matcher.run_frustum_matching \
      --database_path /path/to/database.db \
      --output_path /path/to/match_list.txt \
      --run_matching \
          --run_reconstruction \
      --image_path /path/to/images \
      --sparse_path /path/to/sparse

  # Tune for a tight warehouse environment:
    python -m frustum_matcher.run_frustum_matching \
      --database_path /path/to/database.db \
      --output_path /path/to/match_list.txt \
      --max_distance 15.0 \
      --horizontal_fov 100 \
      --max_pairs_per_image 20 \
      --max_optical_axis_angle 120
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

from frustum_matcher import FrustumAwareMatcher, FrustumMatcherConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_frustum_matching")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate frustum-aware match pairs for COLMAP.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required
    p.add_argument("--database_path", type=str, required=True,
                    help="Path to COLMAP SQLite database.")
    p.add_argument("--output_path", type=str, required=True,
                    help="Output match-list text file.")

    # Spatial parameters
    p.add_argument("--max_distance", type=float, default=30.0,
                    help="Max distance between cameras (prior coord units).")
    p.add_argument("--max_neighbors", type=int, default=50,
                    help="Max KD-tree neighbours per query.")

    # Frustum parameters
    p.add_argument("--horizontal_fov", type=float, default=90.0,
                    help="Camera horizontal FOV in degrees.")
    p.add_argument("--frustum_margin", type=float, default=30.0,
                    help="Extra angular margin on the frustum cone (degrees).")
    p.add_argument("--max_optical_axis_angle", type=float, default=150.0,
                    help="Max angle between optical axes (degrees).")
    p.add_argument("--parallel_threshold", type=float, default=45.0,
                    help="Axis angle under which cameras are 'parallel' (degrees).")

    # Scoring / top-k
    p.add_argument("--max_pairs_per_image", type=int, default=30,
                    help="Keep top-k scoring pairs per image.")
    p.add_argument("--no_scoring", action="store_true",
                    help="Disable covisibility scoring (keep all passing pairs).")

    # Fallback for images without rotation priors
    p.add_argument("--fallback_distance", type=float, default=10.0,
                    help="Position-only radius for images without rotation priors.")

    # Optional: run COLMAP matching
    p.add_argument("--run_matching", action="store_true",
                    help="Run COLMAP feature matching with the generated list.")
    p.add_argument("--image_path", type=str, default=None,
                    help="Path to images (required if --run_matching).")
    p.add_argument("--colmap_bin", type=str, default="colmap",
                    help="Path to COLMAP binary.")

    # Optional: run full reconstruction
    p.add_argument("--run_reconstruction", action="store_true",
                    help="Run COLMAP mapper after matching.")
    p.add_argument("--sparse_path", type=str, default=None,
                    help="Output path for sparse reconstruction.")

    # Diagnostics
    p.add_argument("--stats_path", type=str, default=None,
                    help="Write pair statistics to this JSON file.")
    p.add_argument("--verbose", "-v", action="store_true")

    return p.parse_args()


def run_colmap_matching(
    colmap_bin: str,
    database_path: str,
    match_list_path: str,
) -> None:
    """Run COLMAP image_pairs matcher with the generated match list."""
    cmd = [
        colmap_bin, "matches_importer",
        "--database_path", database_path,
        "--match_list_path", match_list_path,
        "--match_type", "pairs",
    ]
    logger.info("Running COLMAP matches_importer: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)

    # Run feature matching on the imported pairs
    cmd = [
        colmap_bin, "feature_matcher",
        "--database_path", database_path,
        "--SiftMatching.use_gpu", "1",
        "--SiftMatching.max_num_matches", "32768",
    ]
    logger.info("Running COLMAP feature_matcher: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def run_colmap_mapper(
    colmap_bin: str,
    database_path: str,
    image_path: str,
    sparse_path: str,
) -> None:
    """Run COLMAP mapper for reconstruction."""
    Path(sparse_path).mkdir(parents=True, exist_ok=True)
    cmd = [
        colmap_bin, "mapper",
        "--database_path", database_path,
        "--image_path", image_path,
        "--output_path", sparse_path,
    ]
    logger.info("Running COLMAP mapper: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()

    database_path = Path(args.database_path)
    if not database_path.exists():
        logger.error("Database path does not exist: %s", database_path)
        sys.exit(1)

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.run_matching and not args.image_path and args.run_reconstruction:
        logger.error("--image_path is required when --run_reconstruction is set.")
        sys.exit(1)

    if args.run_reconstruction and not args.sparse_path:
        logger.error("--sparse_path is required when --run_reconstruction is set.")
        sys.exit(1)

    if args.run_reconstruction and args.image_path:
        image_path = Path(args.image_path)
        if not image_path.exists():
            logger.error("Image path does not exist: %s", image_path)
            sys.exit(1)

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Build config
    config = FrustumMatcherConfig(
        max_distance=args.max_distance,
        max_neighbors=args.max_neighbors,
        horizontal_fov_deg=args.horizontal_fov,
        frustum_margin_deg=args.frustum_margin,
        max_optical_axis_angle_deg=args.max_optical_axis_angle,
        parallel_axis_threshold_deg=args.parallel_threshold,
        enable_covisibility_scoring=not args.no_scoring,
        max_pairs_per_image=args.max_pairs_per_image,
        fallback_position_only_distance=args.fallback_distance,
    )

    # Load and compute
    matcher = FrustumAwareMatcher(config)
    num_loaded = matcher.load_from_database(database_path)

    if num_loaded == 0:
        logger.error("No images with valid pose priors found. Exiting.")
        sys.exit(1)

    pairs = matcher.compute_pairs()

    if not pairs:
        logger.error("No valid pairs generated. Check your parameters.")
        sys.exit(1)

    # Write match list
    matcher.write_match_list(pairs, output_path)

    # Statistics
    stats = matcher.get_pair_statistics(pairs)
    logger.info("--- Pair Statistics ---")
    logger.info("  Images with priors:   %d", stats["num_images"])
    logger.info("  Candidate pairs:      %d", stats["num_pairs"])
    logger.info("  Exhaustive pairs:     %d", stats["num_exhaustive"])
    logger.info("  Reduction factor:     %.1fx", stats["reduction_factor"])
    logger.info("  Score range:          [%.3f, %.3f]  median=%.3f",
                stats["score_min"], stats["score_max"], stats["score_median"])
    logger.info("  Pairs/image:          [%d, %d]  median=%.0f",
                stats["pairs_per_image_min"], stats["pairs_per_image_max"],
                stats["pairs_per_image_median"])

    if args.stats_path:
        Path(args.stats_path).parent.mkdir(parents=True, exist_ok=True)
        with open(args.stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        logger.info("Wrote statistics to %s", args.stats_path)

    # Optionally run COLMAP
    if args.run_matching:
        run_colmap_matching(args.colmap_bin, str(database_path), str(output_path))

    if args.run_reconstruction:
        if not args.image_path or not args.sparse_path:
            logger.error("--image_path and --sparse_path required for reconstruction.")
            sys.exit(1)
        run_colmap_mapper(
            args.colmap_bin, str(database_path),
            args.image_path, args.sparse_path,
        )


if __name__ == "__main__":
    main()
