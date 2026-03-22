---
applyTo: "{src/colmap/controllers/**,src/colmap/sfm/**}"
---

# Reconstruction & Pipeline

## Incremental SfM with Priors

COLMAP's incremental mapper registers images one at a time, triangulates new points, and runs BA. With pose priors:

- Priors are consumed in `PosePriorBundleAdjuster`, which runs during the BA step of incremental reconstruction.
- The prior loss function must be robust (Cauchy/Huber) — incremental registration is particularly vulnerable to a single bad prior derailing the growing model.
- Images without priors can still be registered via standard PnP, but they won't have prior constraints in BA. This is fine for a small fraction of images; if >30% lack priors, the self-similarity defence is weakened.

## Debugging Failed Reconstructions

Investigate in this order (most-likely cause first):

1. **Match graph**: are there cross-instance matches? Export the match graph, colour by prior position. False matches between visually identical but spatially distant areas are the #1 failure mode.
2. **Geometric verification**: did false matches survive RANSAC? Check the inlier ratio and the estimated relative pose vs. prior-predicted relative pose.
3. **BA prior residuals**: large residuals mean the optimiser moved a camera far from its prior. Either the prior is an outlier, or a false match is pulling the camera away.
4. **Trajectory visualisation**: plot reconstructed camera positions vs. prior positions. Systematic drift = prior covariance too loose. Sudden jumps = false matches or prior outliers.

## Multi-Model Reconstructions

- COLMAP may produce multiple reconstruction components if the pair graph is disconnected. For self-similar environments, check whether multiple components represent genuinely separate areas or a fragmented single area.
- Use pose priors to diagnose: if two components have spatially overlapping priors, they should be a single model — the matching filter was too aggressive.
- Merging components: use `model_aligner` with priors as alignment targets, then `model_merger`. Verify the merged model's prior residuals don't spike.

## Sub-map Strategy for Large Scenes

For very large environments (>10,000 images):
- Partition spatially using prior positions. K-means or grid-based partitioning with overlap regions.
- Reconstruct each sub-map independently (self-similarity is less problematic at smaller scale).
- Merge using prior-based alignment.
- This is not natively supported — orchestrate via Python scripts calling COLMAP CLI.

## Registration Failure Triage

If many images fail to register:
- Check `pairs_per_image` in the match list. Below 3 is unreliable.
- Check feature extraction: are enough features being detected? Self-similar environments with uniform textures (plain warehouse walls) may need lower SIFT contrast thresholds.
- Check the image ordering: COLMAP's incremental mapper picks a seed pair and grows from there. If the seed is in a self-similar region, early registrations may be wrong. Consider forcing a seed pair from a region with distinctive features.
