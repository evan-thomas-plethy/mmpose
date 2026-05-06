#!/usr/bin/env python3
"""Check whether two COCO annotation JSONs share any video ID prefixes (train/val leakage risk).

Prefix rule matches filter_val_images.get_video_prefixes: basename of file_name, text before
``_frame_`` if present, else the whole basename.

Usage:
  python train_val_leakage.py path/to/train.json path/to/val.json
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Set


def get_video_prefixes(annotations_path: str) -> Set[str]:
    """Extract unique video prefixes from a COCO images list (same logic as filter_val_images)."""
    with open(annotations_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    prefixes: Set[str] = set()
    for img in data["images"]:
        filename = os.path.basename(img["file_name"])
        if "_frame_" in filename:
            prefix = filename.split("_frame_")[0]
        else:
            prefix = filename
        prefixes.add(prefix)
    return prefixes


def main() -> None:
    parser = argparse.ArgumentParser(
        description="List video prefixes that appear in both annotation JSONs."
    )
    parser.add_argument(
        "ann_json_a",
        help="First COCO annotations JSON (e.g. person_keypoints_train2017.json)",
    )
    parser.add_argument(
        "ann_json_b",
        help="Second COCO annotations JSON (e.g. person_keypoints_val2017.json)",
    )
    args = parser.parse_args()

    for path, label in (args.ann_json_a, "arg1"), (args.ann_json_b, "arg2"):
        if not os.path.isfile(path):
            raise SystemExit(f"Not a file ({label}): {path}")

    p1 = get_video_prefixes(args.ann_json_a)
    p2 = get_video_prefixes(args.ann_json_b)
    overlap = p1 & p2

    print(f"A: {args.ann_json_a}")
    print(f"   unique prefixes: {len(p1)}")
    print(f"B: {args.ann_json_b}")
    print(f"   unique prefixes: {len(p2)}")
    print()

    if not overlap:
        print("No shared video prefixes between A and B.")
        return

    print(f"SHARED prefixes ({len(overlap)}) — possible train/val leakage:")
    for prefix in sorted(overlap):
        print(f"  {prefix}")


if __name__ == "__main__":
    main()
