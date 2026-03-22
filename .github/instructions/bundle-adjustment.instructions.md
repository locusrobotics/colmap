---
applyTo: "src/colmap/estimators/**"
---

# Bundle Adjustment & Cost Functions

## Ceres Solver Rules

- Use `AutoDiffCostFunction` unless you have a proven performance reason for analytic derivatives.
- Always attach `EigenQuaternionManifold` to quaternion parameter blocks. Eigen stores quaternions as `[x, y, z, w]` which requires this specific manifold, not the default `QuaternionManifold` (which expects `[w, x, y, z]`).
- Before adding a residual block, always check `problem->HasParameterBlock(ptr)`. The parameter may have been excluded from the problem by the image selection logic.
- Prior residuals only touch camera parameters (rotation + translation). They must not break the Schur-complement structure — 3D points are eliminated, camera params stay in the reduced system.

## Prior Residuals

- The `AbsolutePosePriorCostFunctor` handles full 6-DoF priors (position + rotation). The `AbsolutePosePositionPriorCostFunctor` handles position-only fallback. Always check `prior.HasRotation()` to choose.
- The 6×6 covariance is block-diagonal: position (top-left 3×3) and rotation (bottom-right 3×3). If the prior source provides cross-correlation terms, this is where they'd go — but currently they're zero.
- Rotation residuals must use a minimal 3-parameter representation (tangent space / Lie algebra). A raw 4-component quaternion difference is poorly conditioned near the identity. If the existing cost functor uses quaternion coefficient differences, replace with `(q_prior.inverse() * q_est).vec()` or a log-map.
- Pre-compute the information matrix square root (`L` where `L * L^T = Σ^{-1}`) at construction time. The `CovarianceWeightedCostFunctor` should never perform Cholesky decomposition inside `operator()`.

## Loss Functions

- Always use a robust loss (Cauchy or Huber) on prior residuals via `prior_loss_function_`. A single outlier prior with tight covariance and no loss function can corrupt the entire reconstruction.
- Set the loss threshold from the prior covariance: 3× the maximum standard deviation (square root of the max eigenvalue of the position covariance) is a reasonable default.
- Never use `nullptr` (trivial / squared loss) for prior residuals in production configurations.

## Coordinate Conversion

- Pose priors arrive as world-frame position `C_w` and cam-from-world rotation `R_cw`.
- The cam_from_world translation for the prior is `t_cw = -R_cw * C_w`. This conversion happens in `AddPosePriorToProblem`. Always normalise the quaternion before computing this.
- The metric-to-normalised scaling (`normalized_from_metric_`) applies to `C_w`, not to `t_cw`.

## Solver Configuration

- Default solver is `ITERATIVE_SCHUR` with `SCHUR_JACOBI` preconditioner. This works well when priors add moderate coupling.
- If adding rotation priors causes convergence issues, profile and consider switching to `SPARSE_NORMAL_CHOLESKY` for the augmented problem.
- Monitor convergence: if the solver takes >2× more iterations after adding prior residuals, the covariance weighting or residual parameterisation likely needs adjustment.
