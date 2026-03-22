---
applyTo: "{src/colmap/scene/database*,src/pycolmap/scene/database*}"
---

# Database & Persistence

## Schema

- COLMAP uses SQLite. The schema is defined inline in `database_sqlite.cc` as SQL strings.
- Pose priors are stored per-image. Our fork adds columns for rotation (`qw, qx, qy, qz`) and rotation covariance (9 doubles, row-major) alongside the existing position and position covariance columns.
- When adding columns, check whether COLMAP has a schema migration mechanism. If not, handle legacy databases gracefully — detect missing columns and treat absent values as NaN/unset.

## Read/Write Pattern

- Follow the existing `ReadPosePrior` / `WritePosePrior` pattern. These methods serialise `PosePrior` structs to/from SQLite rows.
- Always round-trip test: write a prior, read it back, compare with `operator==`. This catches serialisation bugs, especially quaternion component ordering.
- NaN values should survive the round-trip. SQLite stores them as IEEE 754 NaN — verify this on your platform.

## Python Bindings

- The `src/pycolmap/scene/database.cc` bindings must expose all fields that the C++ struct exposes. When adding a field to `PosePrior`, always update the binding in the same PR.
- Use pybind11's `.def_readwrite()` for public fields. Don't add Python-only accessor methods unless there's a conversion needed.
- Test the Python bindings independently — don't assume that passing C++ tests means Python works correctly. Quaternion component ordering is a common source of C++/Python mismatches.
