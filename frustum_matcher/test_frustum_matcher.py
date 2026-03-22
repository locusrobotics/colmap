#!/usr/bin/env python3
"""
test_frustum_matcher.py
========================
Validates the frustum-aware matcher geometry and filtering logic
using synthetic camera layouts.  Does not require pycolmap.
"""

import numpy as np

from frustum_matcher.frustum_matcher import (
    FrustumAwareMatcher,
    FrustumMatcherConfig,
    _angle_between,
    _quat_to_rotation_matrix,
)


def _axis_angle_to_quat(axis: np.ndarray, angle_deg: float) -> np.ndarray:
    """Helper: axis-angle → quaternion [w, x, y, z]."""
    axis = axis / np.linalg.norm(axis)
    half = np.radians(angle_deg) / 2.0
    w = np.cos(half)
    xyz = axis * np.sin(half)
    return np.array([w, xyz[0], xyz[1], xyz[2]])


def _identity_quat() -> np.ndarray:
    return np.array([1.0, 0.0, 0.0, 0.0])


# -------------------------------------------------------------------
# Test 1: Two cameras facing each other — should match
# -------------------------------------------------------------------
def test_facing_cameras():
    """Cameras 10m apart, facing each other along the X axis."""
    config = FrustumMatcherConfig(
        max_distance=20.0,
        horizontal_fov_deg=90.0,
        frustum_margin_deg=10.0,
        max_optical_axis_angle_deg=150.0,
        enable_covisibility_scoring=False,
    )
    matcher = FrustumAwareMatcher(config)

    # Camera A at origin, looking along +X (identity rotation means +Z is
    # optical axis in cam frame; we need to rotate so +Z world = optical axis?
    # Actually: with identity quaternion, R_cw = I, optical axis = R_cw[2,:] = [0,0,1]
    # So camera A looks along +Z.  Camera B at (0, 0, 10) also looks along +Z → parallel.
    # For *facing* cameras: A at origin looking +Z, B at (0,0,10) looking -Z.

    q_identity = _identity_quat()  # looks along +Z
    # Rotate 180° around Y to look along -Z
    q_flipped = _axis_angle_to_quat(np.array([0, 1, 0]), 180.0)

    matcher.load_from_arrays(
        image_ids=[1, 2],
        image_names=["cam_a.jpg", "cam_b.jpg"],
        positions=np.array([[0, 0, 0], [0, 0, 10.0]]),
        rotations=np.array([q_identity, q_flipped]),
    )

    pairs = matcher.compute_pairs()
    pair_names = {(a, b) for a, b, _ in pairs}

    assert ("cam_a.jpg", "cam_b.jpg") in pair_names or \
           ("cam_b.jpg", "cam_a.jpg") in pair_names, \
        f"Facing cameras should match. Got pairs: {pair_names}"
    print("PASS: test_facing_cameras")


# -------------------------------------------------------------------
# Test 2: Two cameras back-to-back — should NOT match
# -------------------------------------------------------------------
def test_back_to_back():
    """Cameras 5m apart, both looking in the same direction away from each other."""
    config = FrustumMatcherConfig(
        max_distance=20.0,
        horizontal_fov_deg=90.0,
        frustum_margin_deg=10.0,
        max_optical_axis_angle_deg=150.0,
        enable_covisibility_scoring=False,
    )
    matcher = FrustumAwareMatcher(config)

    # Both looking along +Z, but B is behind A.
    # A at origin looking +Z, B at (0, 0, -5) also looking +Z.
    # Baseline from A→B is [0, 0, -5], angle with A's optical axis (+Z) = 180°
    # Baseline from B→A is [0, 0, +5], angle with B's optical axis (+Z) = 0°
    # B→A is in B's frustum, but A→B is not in A's.  One is enough to pass.
    # Actually, this *should* pass because B can see A's region.
    # Let's instead make them face AWAY from each other:
    # A at origin looking +Z, B at (0, 0, -5) looking -Z.

    q_identity = _identity_quat()          # looks +Z
    q_flipped = _axis_angle_to_quat(np.array([0, 1, 0]), 180.0)  # looks -Z

    matcher.load_from_arrays(
        image_ids=[1, 2],
        image_names=["cam_a.jpg", "cam_b.jpg"],
        positions=np.array([[0, 0, 0], [0, 0, -5.0]]),
        rotations=np.array([q_identity, q_flipped]),
    )

    pairs = matcher.compute_pairs()

    # Optical axis angle = 180° > 150° threshold → rejected
    assert len(pairs) == 0, \
        f"Back-to-back cameras should NOT match. Got: {pairs}"
    print("PASS: test_back_to_back")


# -------------------------------------------------------------------
# Test 3: Cameras too far apart — should NOT match
# -------------------------------------------------------------------
def test_too_far():
    config = FrustumMatcherConfig(max_distance=10.0)
    matcher = FrustumAwareMatcher(config)

    q = _identity_quat()
    matcher.load_from_arrays(
        image_ids=[1, 2],
        image_names=["a.jpg", "b.jpg"],
        positions=np.array([[0, 0, 0], [0, 0, 50.0]]),
        rotations=np.array([q, q]),
    )

    pairs = matcher.compute_pairs()
    assert len(pairs) == 0, f"Distant cameras should not match. Got: {pairs}"
    print("PASS: test_too_far")


# -------------------------------------------------------------------
# Test 4: Self-similar corridor — key scenario
# -------------------------------------------------------------------
def test_self_similar_corridor():
    """
    Simulate two parallel corridors 20m apart.  Cameras in corridor A
    should match other corridor-A cameras, NOT corridor-B cameras,
    even though the corridors are visually identical.
    """
    config = FrustumMatcherConfig(
        max_distance=15.0,       # corridors are 20m apart → spatial gate rejects
        horizontal_fov_deg=90.0,
        frustum_margin_deg=15.0,
        max_optical_axis_angle_deg=120.0,
        enable_covisibility_scoring=False,
    )
    matcher = FrustumAwareMatcher(config)

    # Corridor A: cameras at y=0, spaced along x, looking along +Z
    # Corridor B: cameras at y=20, spaced along x, also looking along +Z
    q_fwd = _identity_quat()

    ids = []
    names = []
    positions = []
    rotations = []

    for i in range(5):
        # Corridor A
        ids.append(100 + i)
        names.append(f"corridor_a_{i}.jpg")
        positions.append([i * 3.0, 0.0, 0.0])
        rotations.append(q_fwd)

        # Corridor B
        ids.append(200 + i)
        names.append(f"corridor_b_{i}.jpg")
        positions.append([i * 3.0, 20.0, 0.0])
        rotations.append(q_fwd)

    matcher.load_from_arrays(
        image_ids=ids,
        image_names=names,
        positions=np.array(positions),
        rotations=np.array(rotations),
    )

    pairs = matcher.compute_pairs()
    pair_names = {(a, b) for a, b, _ in pairs}

    # No cross-corridor pairs should exist (distance > 15m)
    for a, b, _ in pairs:
        a_is_corr_a = "corridor_a" in a
        b_is_corr_a = "corridor_a" in b
        assert a_is_corr_a == b_is_corr_a, \
            f"Cross-corridor match found: {a} <-> {b}"

    # But intra-corridor pairs should exist
    assert len(pairs) > 0, "Should have intra-corridor matches"
    print(f"PASS: test_self_similar_corridor ({len(pairs)} intra-corridor pairs)")


# -------------------------------------------------------------------
# Test 4b: Close parallel corridors — frustum must do the rejection
# -------------------------------------------------------------------
def test_close_parallel_corridors():
    """
    Two corridors only 8m apart (within spatial gate range).
    Cameras in corridor A look along +X, corridor B look along -X.
    The spatial gate cannot reject cross-corridor pairs, so the frustum
    filter must do it: the cameras face opposite directions.
    """
    config = FrustumMatcherConfig(
        max_distance=15.0,
        horizontal_fov_deg=90.0,
        frustum_margin_deg=15.0,
        max_optical_axis_angle_deg=150.0,
        parallel_axis_threshold_deg=45.0,
        enable_covisibility_scoring=False,
    )
    matcher = FrustumAwareMatcher(config)

    # Corridor A: cameras looking along +X
    q_pos_x = _axis_angle_to_quat(np.array([0, 1, 0]), -90.0)  # optical axis → +X
    # Corridor B: cameras looking along -X
    q_neg_x = _axis_angle_to_quat(np.array([0, 1, 0]), 90.0)   # optical axis → -X

    ids, names, positions, rotations = [], [], [], []

    for i in range(5):
        # Corridor A at y=0
        ids.append(100 + i)
        names.append(f"corr_a_{i}.jpg")
        positions.append([i * 2.0, 0.0, 0.0])
        rotations.append(q_pos_x)

        # Corridor B at y=8 (within max_distance)
        ids.append(200 + i)
        names.append(f"corr_b_{i}.jpg")
        positions.append([i * 2.0, 8.0, 0.0])
        rotations.append(q_neg_x)

    matcher.load_from_arrays(
        image_ids=ids,
        image_names=names,
        positions=np.array(positions),
        rotations=np.array(rotations),
    )

    pairs = matcher.compute_pairs()

    # Cross-corridor pairs: opposite optical axes (180°) and baseline is
    # perpendicular to both axes → fails frustum AND parallel checks.
    cross_corridor = [(a, b) for a, b, _ in pairs
                      if ("corr_a" in a) != ("corr_a" in b)]
    intra_corridor = [(a, b) for a, b, _ in pairs
                      if ("corr_a" in a) == ("corr_a" in b)]

    assert len(cross_corridor) == 0, \
        f"Cross-corridor pairs should be rejected by frustum filter: {cross_corridor}"
    assert len(intra_corridor) > 0, \
        "Intra-corridor pairs should still be accepted"
    print(f"PASS: test_close_parallel_corridors "
          f"({len(intra_corridor)} intra, {len(cross_corridor)} cross)")



# -------------------------------------------------------------------
# Test 5: Perpendicular cameras — marginal case
# -------------------------------------------------------------------
def test_perpendicular_cameras():
    """Two cameras at the same spot, looking at 90° to each other.
    Should match — they share a frustum edge region."""
    config = FrustumMatcherConfig(
        max_distance=20.0,
        horizontal_fov_deg=90.0,
        frustum_margin_deg=30.0,   # 45° + 30° = 75° half-cone
        max_optical_axis_angle_deg=150.0,
        enable_covisibility_scoring=False,
    )
    matcher = FrustumAwareMatcher(config)

    q_z = _identity_quat()  # looks +Z
    q_x = _axis_angle_to_quat(np.array([0, 1, 0]), -90.0)  # looks +X

    matcher.load_from_arrays(
        image_ids=[1, 2],
        image_names=["a.jpg", "b.jpg"],
        positions=np.array([[0, 0, 0], [2, 0, 2]]),
        rotations=np.array([q_z, q_x]),
    )

    pairs = matcher.compute_pairs()
    assert len(pairs) > 0, \
        f"Perpendicular close cameras should match. Got: {pairs}"
    print("PASS: test_perpendicular_cameras")


# -------------------------------------------------------------------
# Test 6: Statistics and top-k filtering
# -------------------------------------------------------------------
def test_topk_filtering():
    """Verify that top-k limits the number of pairs per image."""
    config = FrustumMatcherConfig(
        max_distance=100.0,
        horizontal_fov_deg=120.0,
        frustum_margin_deg=30.0,
        enable_covisibility_scoring=True,
        max_pairs_per_image=3,
    )
    matcher = FrustumAwareMatcher(config)

    q = _identity_quat()
    n = 20
    positions = np.column_stack([
        np.linspace(0, 50, n),
        np.zeros(n),
        np.zeros(n),
    ])

    matcher.load_from_arrays(
        image_ids=list(range(n)),
        image_names=[f"img_{i}.jpg" for i in range(n)],
        positions=positions,
        rotations=np.tile(q, (n, 1)),
    )

    pairs = matcher.compute_pairs()
    stats = matcher.get_pair_statistics(pairs)

    # The median pairs per image should respect the top-k cap (approximately)
    assert stats["pairs_per_image_median"] <= config.max_pairs_per_image + 2, \
        f"Top-k not respected: median={stats['pairs_per_image_median']}"
    print(f"PASS: test_topk_filtering (median pairs/image={stats['pairs_per_image_median']:.0f})")


# -------------------------------------------------------------------
# Test 7: Rotation matrix conversion sanity check
# -------------------------------------------------------------------
def test_rotation_matrix():
    """Identity quaternion should give identity matrix."""
    R = _quat_to_rotation_matrix(1, 0, 0, 0)
    assert np.allclose(R, np.eye(3)), f"Identity quat should give I. Got:\n{R}"

    # 90° around Z: [0,0,1] axis, 90° → q = [cos45, 0, 0, sin45]
    q = _axis_angle_to_quat(np.array([0, 0, 1]), 90.0)
    R = _quat_to_rotation_matrix(*q)
    # Should map [1,0,0] → [0,1,0]
    v = R @ np.array([1, 0, 0])
    assert np.allclose(v, [0, 1, 0], atol=1e-9), f"90° Z rotation failed: {v}"
    print("PASS: test_rotation_matrix")


# -------------------------------------------------------------------
# Run all
# -------------------------------------------------------------------
if __name__ == "__main__":
    test_rotation_matrix()
    test_facing_cameras()
    test_back_to_back()
    test_too_far()
    test_self_similar_corridor()
    test_close_parallel_corridors()
    test_perpendicular_cameras()
    test_topk_filtering()
    print("\n=== ALL TESTS PASSED ===")
