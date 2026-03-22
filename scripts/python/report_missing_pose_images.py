#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Report rows in poses.csv whose image file is missing."
    )
    parser.add_argument(
        "--poses_csv",
        type=Path,
        default=Path("data/processed/poses.csv"),
        help="Path to poses.csv",
    )
    parser.add_argument(
        "--images_dir",
        type=Path,
        default=Path("data/processed/images"),
        help="Directory containing images",
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        default=Path("data/processed/missing_pose_images.csv"),
        help="Output CSV path for missing rows",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.poses_csv.exists():
        raise FileNotFoundError(f"poses.csv not found: {args.poses_csv}")
    if not args.images_dir.exists():
        raise FileNotFoundError(f"images_dir not found: {args.images_dir}")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    missing_rows = []

    with args.poses_csv.open("r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("poses.csv has no header")

        if "image" not in reader.fieldnames:
            raise ValueError("poses.csv must contain an 'image' column")

        for idx, row in enumerate(reader, start=2):
            image_name = row["image"].strip()
            image_path = args.images_dir / image_name
            if not image_path.exists():
                row_out = {"line_number": idx, **row}
                missing_rows.append(row_out)

    fieldnames = ["line_number"]
    with args.poses_csv.open("r", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames.extend(reader.fieldnames or [])

    with args.output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(missing_rows)

    print(f"Scanned: {args.poses_csv}")
    print(f"Images dir: {args.images_dir}")
    print(f"Missing rows: {len(missing_rows)}")
    print(f"Report: {args.output_csv}")


if __name__ == "__main__":
    main()
