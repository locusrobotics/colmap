---
applyTo: "src/colmap/geometry/**"
---

# Geometry & Pose Priors

## PosePrior Struct

- Optional fields use NaN-as-unset semantics: `position`, `position_covariance`, and `rotation_covariance` default to `quiet_NaN()`. Check with `allFinite()`.
- `rotation` defaults to `Quaterniond::Identity()`. Use `HasRotation()` (checks `allFinite() && norm() > 0`) to distinguish "has a rotation prior" from "no rotation prior".
- When adding new fields, always give them a NaN or sentinel default and add a corresponding validity check method.

## Constructor Design

- Avoid adding more constructor overloads. The struct already has many — prefer the default constructor plus direct member assignment (e.g. `prior.rotation = q;`). This is more readable, less error-prone, and matches COLMAP's idiom for similar structs.
- If a new constructor is truly needed, ensure it can't be confused with an existing one through implicit conversions (e.g. two overloads that differ only in `Matrix3d` argument position).

## Quaternion Handling

- Normalise quaternions before use. Never assume an incoming quaternion is unit-norm — validate with `q.norm() > 0` then call `q.normalize()`.
- Sign ambiguity: `q` and `-q` represent the same rotation. All equality checks must handle this — compare both `q1.coeffs().isApprox(q2.coeffs())` and `q1.coeffs().isApprox(-q2.coeffs())`, or use the geodesic distance `2 * acos(|q1.dot(q2)|)`.
- When computing angular differences, clamp the dot product to `[-1, 1]` before `acos` to avoid NaN from floating-point overshoot.

## Covariance Matrices

- Position covariance is a 3×3 symmetric positive-definite matrix in the world frame.
- Rotation covariance is a 3×3 SPD matrix in the tangent space of SO(3) at the prior rotation. It represents uncertainty in the axis-angle perturbation.
- Always validate SPD-ness when consuming external covariances: check `allFinite()`, check symmetry, check positive eigenvalues. Degenerate covariances (zero eigenvalues) will cause Cholesky failures downstream.

## Equality Operators

- Use approximate comparison with a tolerance (currently `1e-9`). Exact floating-point equality on poses is almost never correct.
- NaN fields are considered equal if both are NaN (both "unset"). This is intentional — it means two priors with no rotation are equal regardless of the NaN bit pattern.
- When adding new fields to `PosePrior`, update `operator==` and `operator!=`. Missing a field silently breaks comparison logic.

## Rigid3d and Transforms

- `Rigid3d` stores `(Quaterniond rotation, Vector3d translation)` representing the cam_from_world transform.
- To get the world-frame camera position: `C_w = -(rotation.inverse() * translation)` or equivalently `C_w = -(rotation.conjugate() * translation)`.
- Composing transforms: `a_from_c = a_from_b * b_from_c`. Read right to left.
