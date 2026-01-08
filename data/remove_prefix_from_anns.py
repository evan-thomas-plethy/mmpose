#!/usr/bin/env python3
"""
Filter COCO annotations to only include images that exist in the target directory.

Usage:
    python remove_from_anns_except_prefix.py [--source SOURCE_JSON] [--output OUTPUT_JSON]

If --source is not provided, reads from coco/annotations/person_keypoints_val2017.json
If --output is not provided, overwrites the source file.
"""

import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Filter annotations to only include images present in val2017 directory"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="coco/annotations/person_keypoints_val2017.json",
        help="Source annotation file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output annotation file (defaults to overwriting source)",
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        default="coco/val2017",
        help="Directory containing the images to keep",
    )
    args = parser.parse_args()

    # Resolve paths relative to this script's directory
    script_dir = Path(__file__).parent
    source_path = script_dir / args.source
    images_dir = script_dir / args.images_dir
    output_path = script_dir / args.output if args.output else source_path

    print(f"Source annotations: {source_path}")
    print(f"Images directory: {images_dir}")
    print(f"Output: {output_path}")

    # Get list of image files in the target directory
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    existing_images = set(
        f.name for f in images_dir.iterdir() 
        if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    print(f"Found {len(existing_images)} images in {images_dir}")

    # Load annotations
    if not source_path.exists():
        raise FileNotFoundError(f"Source annotation file not found: {source_path}")

    with open(source_path, "r") as f:
        data = json.load(f)

    original_images = len(data.get("images", []))
    original_annotations = len(data.get("annotations", []))

    # Filter images to only those that exist in the directory
    filtered_images = [
        img for img in data.get("images", [])
        if img["file_name"] in existing_images
    ]

    # Get the IDs of images we're keeping
    kept_image_ids = {img["id"] for img in filtered_images}

    # Filter annotations to only those for kept images
    filtered_annotations = [
        ann for ann in data.get("annotations", [])
        if ann["image_id"] in kept_image_ids
    ]

    # Update the data
    data["images"] = filtered_images
    data["annotations"] = filtered_annotations

    print(f"Images: {original_images} -> {len(filtered_images)}")
    print(f"Annotations: {original_annotations} -> {len(filtered_annotations)}")

    # Save filtered annotations
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f)

    print(f"Saved filtered annotations to {output_path}")


if __name__ == "__main__":
    main()

