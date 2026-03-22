---
applyTo: "{src/colmap/feature/**,src/colmap/retrieval/**,frustum_matcher/**}"
---

# Feature Matching & Pair Selection

## The Self-Similarity Problem

In warehouse/logistics environments, standard matching fails because:
- Exhaustive matching produces false cross-instance correspondences between visually identical aisles/shelves.
- Vocabulary tree retrieval ranks the wrong instance of a repeated structure highest.
- Epipolar geometry verification can pass for false matches when the relative pose between repeated structures is consistent.

The only reliable disambiguation signal is the metric pose prior. All matching strategies must incorporate spatial filtering.

## Frustum Matcher (Python, out-of-tree)

The `frustum_matcher/` module generates filtered match-pair lists using 6-DoF priors. It uses three acceptance tiers:

1. **Mutual frustum + convergence**: both cameras include the other in their frustum cone. For anti-parallel axes (>90° apart), require the baseline to sit in the inner half of both cones — this rejects diagonal cross-corridor false matches while accepting truly converging cameras.
2. **Parallel nearby**: optical axes within `parallel_axis_threshold_deg` — handles side-by-side trajectory cameras where the baseline is perpendicular to the viewing direction.
3. **Single frustum + axis check**: one camera sees the other, accepted only if axis divergence is below `max_optical_axis_angle_deg`.

When modifying this logic:
- Always test with the `test_close_parallel_corridors` scenario — it's the hardest case (corridors within spatial range but facing opposite directions).
- The covisibility score is a heuristic for ranking, not a hard gate. Don't over-tune it.
- Output is a text file of image-name pairs for COLMAP's `matches_importer`. Never modify the COLMAP database from this module.

## COLMAP Matching Integration

- Use `matches_importer` with `--match_type pairs` to feed filtered pair lists. This is the cleanest integration point and doesn't require C++ changes.
- If modifying C++ matchers in `src/colmap/feature/`, follow the existing `FeatureMatcherThread` pattern. Spatial matching already uses a KD-tree — extend it, don't replace it.
- `SiftMatching.max_num_matches` caps matches per pair. For self-similar environments, 32768 is a reasonable default — too low risks losing valid matches in texture-rich regions.

## Vocabulary Tree Re-ranking

If using COLMAP's vocab tree retrieval, re-rank results by multiplying the visual similarity score with a spatial plausibility score:
```
score_final = score_visual * exp(-d² / (2σ²))
```
where `d` is Euclidean distance between prior positions and `σ` is scene-scale-dependent (typically 10–30m for indoor environments).

## Geometric Verification

After matching, COLMAP's geometric verification estimates an essential/fundamental matrix via RANSAC. To tighten verification with priors:
- Use the relative pose from priors as an initial estimate.
- Reject pairs where the verified relative pose disagrees with the prior-predicted relative pose by more than 3σ of the combined covariance.
- This is a post-matching filter — implement it after `matches_importer`, before `mapper`.

## Validation

Always validate the output pair graph:
- **Coverage**: every image with a prior should appear in at least one pair.
- **Connectivity**: the pair graph should be a single connected component.
- **Reduction factor**: typically 5×–500× fewer pairs than exhaustive. Below 5× means filtering isn't working. Above 500× risks fragmentation.
- **Pairs per image**: target median 15–30. Below 3 risks registration failure.
