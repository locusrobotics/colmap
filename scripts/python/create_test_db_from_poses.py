#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import sqlite3
from pathlib import Path

import numpy as np
from PIL import Image


def _array_to_blob(array: np.ndarray) -> bytes:
    return np.asarray(array, dtype=np.float64).tobytes(order="C")


def _quat_wxyz_to_rotvec(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    norm = np.linalg.norm(q)
    if norm < 1e-12:
        return np.zeros(3, dtype=np.float64)
    q = q / norm

    if q[0] < 0.0:
        q = -q

    w = float(np.clip(q[0], -1.0, 1.0))
    xyz = q[1:]
    xyz_norm = float(np.linalg.norm(xyz))
    if xyz_norm < 1e-12:
        return np.zeros(3, dtype=np.float64)

    angle = 2.0 * np.arctan2(xyz_norm, w)
    axis = xyz / xyz_norm
    return axis * angle


def _create_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS cameras (
            camera_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
            model INTEGER NOT NULL,
            width INTEGER NOT NULL,
            height INTEGER NOT NULL,
            params BLOB,
            prior_focal_length INTEGER NOT NULL
        );

        CREATE TABLE IF NOT EXISTS images (
            image_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
            name TEXT NOT NULL UNIQUE,
            camera_id INTEGER NOT NULL,
            CONSTRAINT image_id_check CHECK(image_id >= 0 and image_id < 2147483647),
            FOREIGN KEY(camera_id) REFERENCES cameras(camera_id)
        );

        CREATE UNIQUE INDEX IF NOT EXISTS index_name ON images(name);

        CREATE TABLE IF NOT EXISTS pose_priors (
            image_id INTEGER PRIMARY KEY NOT NULL,
            position BLOB,
            coordinate_system INTEGER NOT NULL,
            covariance BLOB,
            rotation BLOB,
            FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE
        );
        """
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a COLMAP test DB from poses.csv + images folder."
    )
    parser.add_argument(
        "--poses_csv",
        type=Path,
        default=Path("data/processed/poses.csv"),
        help="CSV file with tx,ty,tz,qx,qy,qz,qw,image columns.",
    )
    parser.add_argument(
        "--images_dir",
        type=Path,
        default=Path("data/processed/images"),
        help="Directory containing images referenced by poses.csv.",
    )
    parser.add_argument(
        "--database_path",
        type=Path,
        default=Path("data/processed/test_colmap.db"),
        help="Output SQLite database path.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output DB if it already exists.",
    )
    parser.add_argument(
        "--camera_model",
        type=int,
        default=0,
        help="COLMAP camera model id (default: SIMPLE_PINHOLE=0).",
    )
    parser.add_argument(
        "--focal_length",
        type=float,
        default=0.0,
        help="Focal length in pixels. If <= 0, infer as 1.2 * max(width,height).",
    )
    parser.add_argument(
        "--position_std",
        type=float,
        default=1.0,
        help="Position prior standard deviation (meters) for covariance.",
    )
    parser.add_argument(
        "--rotation_std_deg",
        type=float,
        default=10.0,
        help="Rotation prior standard deviation (degrees) for covariance.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.poses_csv.exists():
        raise FileNotFoundError(f"poses.csv not found: {args.poses_csv}")
    if not args.images_dir.exists():
        raise FileNotFoundError(f"images dir not found: {args.images_dir}")

    args.database_path.parent.mkdir(parents=True, exist_ok=True)
    if args.database_path.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Database already exists: {args.database_path}. Use --overwrite to replace."
            )
        args.database_path.unlink()

    first_image_path = next(args.images_dir.glob("*"), None)
    if first_image_path is None:
        raise RuntimeError(f"No images found in {args.images_dir}")

    with Image.open(first_image_path) as image:
        width, height = image.size

    focal_length = (
        args.focal_length
        if args.focal_length > 0
        else 1.2 * float(max(width, height))
    )
    cx = width / 2.0
    cy = height / 2.0
    camera_params = np.array([focal_length, cx, cy], dtype=np.float64)

    coord_system_cartesian = 1
    rot_std_rad = np.deg2rad(args.rotation_std_deg)
    cov6 = np.diag(
        [
            args.position_std**2,
            args.position_std**2,
            args.position_std**2,
            rot_std_rad**2,
            rot_std_rad**2,
            rot_std_rad**2,
        ]
    ).astype(np.float64)

    rows_loaded = 0
    rows_skipped_missing_image = 0

    conn = sqlite3.connect(str(args.database_path))
    try:
        _create_schema(conn)

        cursor = conn.execute(
            "INSERT INTO cameras(model, width, height, params, prior_focal_length) VALUES(?, ?, ?, ?, ?)",
            (
                args.camera_model,
                width,
                height,
                _array_to_blob(camera_params),
                1,
            ),
        )
        camera_id = int(cursor.lastrowid)

        with args.poses_csv.open("r", newline="") as f:
            reader = csv.DictReader(f)
            required = {"tx", "ty", "tz", "qx", "qy", "qz", "qw", "image"}
            missing = required - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"Missing required CSV columns: {sorted(missing)}")

            for row in reader:
                image_name = row["image"].strip()
                image_path = args.images_dir / image_name
                if not image_path.exists():
                    rows_skipped_missing_image += 1
                    continue

                tx = float(row["tx"])
                ty = float(row["ty"])
                tz = float(row["tz"])
                qx = float(row["qx"])
                qy = float(row["qy"])
                qz = float(row["qz"])
                qw = float(row["qw"])

                rotvec = _quat_wxyz_to_rotvec(qw, qx, qy, qz)
                position = np.array([tx, ty, tz], dtype=np.float64)

                image_cursor = conn.execute(
                    "INSERT INTO images(name, camera_id) VALUES(?, ?)",
                    (image_name, camera_id),
                )
                image_id = int(image_cursor.lastrowid)

                conn.execute(
                    "INSERT INTO pose_priors(image_id, position, coordinate_system, covariance, rotation) VALUES(?, ?, ?, ?, ?)",
                    (
                        image_id,
                        _array_to_blob(position),
                        coord_system_cartesian,
                        _array_to_blob(cov6),
                        _array_to_blob(rotvec),
                    ),
                )
                rows_loaded += 1

        conn.commit()

        print(f"Created database: {args.database_path}")
        print(f"Images + priors inserted: {rows_loaded}")
        print(f"Rows skipped (missing image file): {rows_skipped_missing_image}")
        print(f"Camera: id={camera_id}, model={args.camera_model}, width={width}, height={height}, f={focal_length:.2f}")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
