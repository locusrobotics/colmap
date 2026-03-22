#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter poses.csv to rows whose image file exists in images_dir."
    )
    parser.add_argument(
        "--poses_csv",
        type=Path,
        default=Path("data/processed/poses.csv"),
        help="Input poses CSV path.",
    )
    parser.add_argument(
        "--images_dir",
        type=Path,
        default=Path("data/processed/images"),
        help="Directory containing image files.",
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        default=Path("data/processed/poses_filtered.csv"),
        help="Output CSV path.",
    )
    parser.add_argument(
        "--write_missing_csv",
        action="store_true",
        help="Also write a CSV of missing rows next to output.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.poses_csv.exists():
        raise FileNotFoundError(f"poses.csv not found: {args.poses_csv}")
    if not args.images_dir.exists():
        raise FileNotFoundError(f"images_dir not found: {args.images_dir}")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    image_names = {p.name for p in args.images_dir.iterdir() if p.is_file()}

    kept_rows = []
    missing_rows = []

    with args.poses_csv.open("r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("poses.csv has no header")
        if "image" not in reader.fieldnames:
            raise ValueError("poses.csv must contain an 'image' column")

        fieldnames = reader.fieldnames

        for row in reader:
            image_name = row["image"].strip()
            if image_name in image_names:
                kept_rows.append(row)
            else:
                missing_rows.append(row)

    with args.output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kept_rows)

    print(f"Input rows: {len(kept_rows) + len(missing_rows)}")
    print(f"Kept rows: {len(kept_rows)}")
    print(f"Dropped rows: {len(missing_rows)}")
    print(f"Filtered CSV: {args.output_csv}")

    if args.write_missing_csv:
        missing_path = args.output_csv.with_name(
            args.output_csv.stem + "_missing" + args.output_csv.suffix
        )
        with missing_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(missing_rows)
        print(f"Missing CSV: {missing_path}")


if __name__ == "__main__":
    main()
