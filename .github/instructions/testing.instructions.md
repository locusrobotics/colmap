---
applyTo: "{**/*test*,**/*_test.cc,**/test_*}"
---

# Testing

## C++ Tests (Google Test)

- Test files live alongside source as `*_test.cc` (e.g. `pose_prior_test.cc`).
- Every new field or method on `PosePrior` needs a corresponding test.
- Use `EXPECT_EQ` / `EXPECT_NE` for `PosePrior` comparison (relies on our custom `operator==`).
- Use `EXPECT_NEAR` or `Eigen::isApprox` for floating-point comparisons — never exact equality.

## Mandatory Test Scenarios

Always cover these edge cases for pose-prior-related code:

- **Quaternion sign ambiguity**: `q` and `-q` must be treated as equal.
- **NaN-as-unset**: two priors with NaN rotation should compare equal. A prior with NaN rotation should not crash any code path.
- **Zero quaternion**: `Quaterniond(0,0,0,0)` is invalid. `HasRotation()` must return false.
- **Identity rotation**: a valid, common case. Must not be confused with "no rotation set".
- **Denormalised quaternion**: a quaternion with `norm() != 1` should be handled gracefully (normalise, don't crash).
- **Degenerate covariance**: zero matrix, near-singular matrix, negative eigenvalues. These come from bad sensor data and must be caught by validation.
- **Co-located cameras**: zero baseline between two cameras. Matching and scoring logic must handle division-by-zero.
- **Axis-aligned trajectories**: cameras moving along a single axis (e.g. a straight corridor). Tests should verify this doesn't create degenerate geometry.

## Self-Similarity Scenarios (for matching tests)

These are first-class test cases, not edge cases:

- **Far parallel corridors**: two corridors far apart (beyond spatial gate). Verify zero cross-corridor matches.
- **Close parallel corridors**: two corridors within spatial range but with cameras facing opposite directions. The frustum filter (not spatial distance) must reject cross-corridor pairs.
- **Diagonal cross-corridor**: cameras in close corridors at a diagonal angle where wide FOV cones overlap. The convergence check must reject these.
- **Revisit loop closure**: a robot returns to a previously visited area. Priors should allow matching between the original and revisit images.
- **Multi-robot overlap**: two robots observe the same area from different trajectories. Priors from different robots in the same region should produce valid pairs.

## Python Tests

- Use `assert` statements with descriptive messages (not bare `assert`).
- Tests should be runnable without `pycolmap` installed — use `load_from_arrays()` to inject synthetic priors for geometry tests.
- Print `PASS: test_name` for each test. The test runner expects `=== ALL TESTS PASSED ===` at the end.

## Debugging Failed Tests

- If a geometric test fails unexpectedly, print the intermediate values: baseline direction, frustum angles, axis angle, which tier accepted/rejected the pair. The geometry is subtle enough that reading code alone won't reveal the bug.
- If a round-trip test fails, check quaternion component ordering first — it's the most common cause.
