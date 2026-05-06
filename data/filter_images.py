#!/usr/bin/env python3
"""
Filter frames based on a COCO-style annotations JSON.

Given an annotations file and a frames directory, create an output directory that
contains only the images referenced in annotations["images"].
"""

from __future__ import annotations

import argparse
import json
import os
import shutil


def _candidate_sources(frames_dir: str, file_name: str) -> list[str]:
    """Return likely source paths for a COCO image file_name."""
    normalized = file_name.replace("\\", "/")
    basename = os.path.basename(normalized)
    return [
        os.path.join(frames_dir, normalized),
        os.path.join(frames_dir, basename),
    ]


def copy_images_from_annotations(
    anns_json_path: str,
    frames_dir: str,
    output_dir: str,
) -> tuple[int, int]:
    """Copy images referenced by anns_json_path into output_dir."""
    with open(anns_json_path, encoding="utf-8") as f:
        data = json.load(f)

    images = data.get("images", [])
    os.makedirs(output_dir, exist_ok=True)

    copied, missing = 0, 0
    seen_destinations: set[str] = set()

    for image in images:
        file_name = image.get("file_name")
        if not file_name:
            continue

        dst_name = os.path.basename(file_name.replace("\\", "/"))
        dst_path = os.path.join(output_dir, dst_name)

        # Avoid duplicate copies when annotations include duplicate file names.
        if dst_path in seen_destinations:
            continue

        src_path = None
        for candidate in _candidate_sources(frames_dir, file_name):
            if os.path.isfile(candidate):
                src_path = candidate
                break

        if src_path is None:
            print(f"Missing image: {file_name}")
            missing += 1
            continue

        shutil.copy2(src_path, dst_path)
        seen_destinations.add(dst_path)
        copied += 1

    return copied, missing


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create an output directory containing only images referenced "
            "in a COCO-style annotations JSON."
        )
    )
    parser.add_argument(
        "--anns",
        required=True,
        help="Path to annotations JSON file.",
    )
    parser.add_argument(
        "--frames-dir",
        required=True,
        help="Directory containing source frame images.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory to write filtered images into.",
    )
    args = parser.parse_args()

    anns_json_path = os.path.abspath(args.anns)
    frames_dir = os.path.abspath(args.frames_dir)
    output_dir = os.path.abspath(args.output)

    if not os.path.isfile(anns_json_path):
        raise SystemExit(f"Annotations file not found: {anns_json_path}")
    if not os.path.isdir(frames_dir):
        raise SystemExit(f"Frames directory not found: {frames_dir}")

    # Create output directory (and parents) only when needed.
    os.makedirs(output_dir, exist_ok=True)

    copied, missing = copy_images_from_annotations(anns_json_path, frames_dir, output_dir)

    print(f"Annotations: {anns_json_path}")
    print(f"Frames dir:   {frames_dir}")
    print(f"Output dir:   {output_dir}")
    print(f"Copied:       {copied}")
    print(f"Missing:      {missing}")


if __name__ == "__main__":
    main()
