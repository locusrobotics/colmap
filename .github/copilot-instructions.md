# COLMAP Pose-Prior Fork

## Project Overview

A fork of [COLMAP](https://github.com/colmap/colmap) extended with **6-DoF pose priors** for Structure-from-Motion in large-scale, self-similar environments (warehouses, fulfilment centres, corridors). Robot swarms capture images with VSLAM/VIO providing position + orientation priors per frame.

Standard SfM fails here because visually identical regions produce geometrically consistent but spatially incorrect matches. Pose priors filter matches geometrically, constrain bundle adjustment, and validate loop closures.

## Tech Stack

- **C++17** + CMake, **Ceres Solver** (BA), **Eigen** (linear algebra/quaternions), **PoseLib** (minimal solvers), **SQLite** (COLMAP database)
- **Python 3.10+** (out-of-tree tooling), **pybind11** (`src/pycolmap/`), **Google Test** (C++ unit tests)

## Build & Test Workflow

```bash
# Configure (tests off by default — enable explicitly)
cmake -B build -DTESTS_ENABLED=ON -DCUDA_ENABLED=OFF -DGUI_ENABLED=OFF

# Build
cmake --build build -j$(nproc)

# Run all C++ tests
cd build && ctest --output-on-failure

# Run a specific test binary directly
./build/src/colmap/geometry/pose_prior_test

# Install Python bindings (from repo root)
pip install . --break-system-packages
```

Key CMake flags: `TESTS_ENABLED` (OFF by default), `CUDA_ENABLED`, `GUI_ENABLED`, `ASAN_ENABLED`, `TSAN_ENABLED`, `UBSAN_ENABLED`.

## Code Style

- 2-space indent, `snake_case` for variables/functions, `CamelCase` for classes, `#pragma once` for headers.
- `THROW_CHECK` / `THROW_CHECK_*` for preconditions — never raw `assert`.
- `LOG(INFO)` / `LOG(WARNING)` / `LOG(ERROR)` for logging — **no `std::cout` in library code**.
- Python: type hints, `dataclass` for config structs.

## Critical Coordinate Conventions

- **`cam_from_world` (`Rigid3d`)**: `translation = -rotation * C_w`. World-frame camera position: `C_w = -(rotation.inverse() * translation)`.
- **Pose priors store world-frame position** `C_w`. Convert to cam_from_world translation: `t_cw = -R_cw * C_w` (done in `AddPosePriorToProblem`).
- **Eigen quaternion layout**: internal `[x, y, z, w]` via `coeffs().data()`; constructor takes `(w, x, y, z)`. Ceres parameter blocks use `coeffs().data()` layout — attach `EigenQuaternionManifold`, not `QuaternionManifold`.
- `q` and `-q` represent the same rotation — all comparison and residual code must handle sign ambiguity: `q1.coeffs().isApprox(q2.coeffs()) || q1.coeffs().isApprox(-q2.coeffs())`.

## Core Design Principle

**Prefer false negatives over false positives.** A missing match reduces completeness. A false cross-instance match can destroy the entire reconstruction. When uncertain, reject the match or tighten the threshold.

## Key Modified Files (Our Fork vs Upstream)

| File | What changed |
|------|-------------|
| `src/colmap/geometry/pose_prior.h/.cc` | Extended `PosePrior` with `rotation`, `rotation_covariance` |
| `src/colmap/estimators/bundle_adjustment.cc` | `PosePriorBundleAdjuster` with 6-DoF prior residuals |
| `src/colmap/scene/database_sqlite.cc` | DB read/write for rotation prior columns |
| `src/pycolmap/geometry/pose_prior.cc` | Python bindings for new fields |
| `frustum_matcher/` | Out-of-tree Python spatial match-pair filter |

Minimise diffs to upstream — use the same variable names, formatting, and patterns as surrounding code.

## PosePrior Data Model

- Optional fields use **NaN-as-unset**: `position`, `position_covariance`, `rotation_covariance` default to `quiet_NaN()`. Check with `allFinite()`.
- `rotation` defaults to `Quaterniond::Identity()`. Use `HasRotation()` to distinguish a valid rotation prior from unset.
- When adding new fields: give them a NaN/sentinel default, add a validity check method, update `operator==`, and update the Python binding in the same PR.
- Always round-trip test DB serialisation: write a prior, read it back, compare. NaN values must survive the round-trip.

## Bundle Adjustment Rules

- Always attach `EigenQuaternionManifold` (not `QuaternionManifold`) to quaternion parameter blocks.
- Check `problem->HasParameterBlock(ptr)` before adding a residual block.
- **Always** use a robust loss (`CauchyLoss` / `HuberLoss`) on prior residuals — never `nullptr` for priors in production.
- Rotation residuals must use tangent-space (3-parameter) representation: `(q_prior.inverse() * q_est).vec()`. Raw 4-component quaternion differences are poorly conditioned.
- Pre-compute information matrix square root at construction time; never run Cholesky inside `operator()`.
- `AbsolutePosePriorCostFunctor` handles full 6-DoF priors; `AbsolutePosePositionPriorCostFunctor` handles position-only fallback. Check `prior.HasRotation()` to choose.

## Matching & Self-Similarity

`frustum_matcher/` generates filtered pair lists from 6-DoF priors using three acceptance tiers (mutual frustum + convergence → parallel nearby → single frustum). Output is a text file fed to `colmap matches_importer --match_type pairs` — this module never writes to the COLMAP database.

Critical test scenario: `test_close_parallel_corridors` — two corridors within spatial range but facing opposite directions. The frustum filter (not spatial distance) must reject cross-corridor pairs.

## Testing Conventions

C++ test files live alongside source as `*_test.cc`. Always cover:
- Quaternion sign ambiguity (`q` and `-q` equal)
- NaN-as-unset: two priors with NaN rotation compare equal; NaN rotation must not crash any code path
- `Quaterniond(0,0,0,0)` → `HasRotation()` must return `false`
- Degenerate covariance (zero/near-singular/negative eigenvalues)

Python tests: `assert` with descriptive messages, print `PASS: test_name`, end with `=== ALL TESTS PASSED ===`. Must run without `pycolmap` installed using synthetic data via `load_from_arrays()`.

## Debugging Failed Reconstructions

Investigate in this order:
1. **Match graph**: cross-instance matches between visually identical but spatially distant areas — #1 failure mode.
2. **Geometric verification**: false matches surviving RANSAC. Check inlier ratio vs prior-predicted relative pose.
3. **BA prior residuals**: large residuals indicate a camera pulled away from its prior by a false match.
4. **Trajectory plot**: reconstructed positions vs prior positions. Systematic drift = covariance too loose; sudden jumps = false matches.
