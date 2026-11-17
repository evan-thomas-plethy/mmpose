#!/usr/bin/env python3
"""
Merge two COCO annotation JSON files into one new JSON.

Image and annotation ids from the second file are remapped so they do not
collide with the first. The first file's categories are kept. info becomes
source1_info, source2_info, and so on, including sources already stored that
way in either input. Neither input file is modified.

Usage:
    python data/merge_jsons.py \
        --json-a data/App_Vids_11-06-2026/v3/App_Vids_11-06-2026_v3.json \
        --json-b data/Ground_Based_Exercises_App_Vids/v5/Ground_Based_Exercises_App_Vids_v5.json \
        --output data/App_Vids_11-06-2026/v3_gbe_v5/App_Vids_v3_gbe_v5.json
"""

from __future__ import annotations

import argparse
import json
import os

from merge_datasets import merge_coco


def file_basenames(data: dict) -> set[str]:
    names = set()
    for image in data.get("images", []):
        file_name = str(image.get("file_name", "")).replace("\\", "/")
        names.add(os.path.basename(file_name))
    return names


def merge_jsons(json_a: str, json_b: str, output_path: str) -> None:
    with open(json_a, encoding="utf-8") as handle:
        data_a = json.load(handle)
    with open(json_b, encoding="utf-8") as handle:
        data_b = json.load(handle)

    overlap = sorted(file_basenames(data_a) & file_basenames(data_b))
    if overlap:
        preview = ", ".join(overlap[:10])
        extra = f" (+{len(overlap) - 10} more)" if len(overlap) > 10 else ""
        raise ValueError(
            f"{len(overlap)} file names appear in both JSON files ({preview}{extra})"
        )

    merged = merge_coco(data_a, data_b)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(merged, handle, indent=4)
        handle.write("\n")

    print(f"First JSON images: {len(data_a.get('images', []))}")
    print(f"Second JSON images: {len(data_b.get('images', []))}")
    print(f"Output images: {len(merged['images'])}")
    print(f"Output annotations: {len(merged['annotations'])}")
    print(f"Wrote {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge two COCO JSON files into one JSON, remapping ids from the second file"
    )
    parser.add_argument("--json-a", required=True, help="First COCO JSON. Its ids are kept.")
    parser.add_argument(
        "--json-b",
        required=True,
        help="Second COCO JSON. Its image and annotation ids are remapped.",
    )
    parser.add_argument("--output", required=True, help="Path of the merged JSON to write.")
    args = parser.parse_args()
    merge_jsons(args.json_a, args.json_b, args.output)


if __name__ == "__main__":
    main()
