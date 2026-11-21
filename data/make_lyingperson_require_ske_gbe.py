#!/usr/bin/env python3
"""Build App_Vids_v2_GBE_v4_val10_subject25_lyingperson_require_ske_gbe.

Copies App_Vids_v2_GBE_v4_val10_subject25_require_ske_gbe. Val, mirrored val,
held-out val, and videos_removed.json stay as copied. Lying_Person_Dataset_Single_Ann.json
is merged into the training JSON only. Its image and annotation ids are remapped,
and its info block is appended as the next sourceN_info, the same way
merge_datasets.merge_coco records a component source. The train config is kept,
with output set to this dataset.

Usage:
    python data/make_lyingperson_require_ske_gbe.py
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from copy import deepcopy

from merge_datasets import merge_coco

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE_NAME = "App_Vids_v2_GBE_v4_val10_subject25_require_ske_gbe"
OUTPUT_NAME = "App_Vids_v2_GBE_v4_val10_subject25_lyingperson_require_ske_gbe"
LYING_JSON = "Lying_Person_Dataset_Single_Ann/Lying_Person_Dataset_Single_Ann.json"
TRAIN_JSON = "person_keypoints_train2017.json"


def dataset_path(*parts: str) -> str:
    return os.path.join(REPO, "data", *parts)


def load_json(path: str) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=4)
        handle.write("\n")


def image_name(image: dict) -> str:
    return os.path.basename(str(image.get("file_name", "")).replace("\\", "/"))


def colliding_names(train: dict, lying: dict) -> list[str]:
    train_names = {image_name(image) for image in train.get("images", [])}
    return sorted(
        image_name(image)
        for image in lying.get("images", [])
        if image_name(image) in train_names
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy the require-ske-gbe dataset and add the lying-person "
            "annotations to its training JSON."
        )
    )
    parser.add_argument("--source", default=dataset_path(SOURCE_NAME))
    parser.add_argument("--lying-json", default=dataset_path(LYING_JSON))
    parser.add_argument("--output", default=dataset_path(OUTPUT_NAME))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_path = os.path.join(args.source, "annotations", TRAIN_JSON)
    for path in (train_path, args.lying_json):
        if not os.path.isfile(path):
            raise SystemExit(f"Missing file: {path}")
    if os.path.exists(args.output):
        raise SystemExit(f"Output already exists: {args.output}")

    train = load_json(train_path)
    lying = load_json(args.lying_json)
    collisions = colliding_names(train, lying)
    if collisions:
        preview = ", ".join(collisions[:10])
        extra = f" (+{len(collisions) - 10} more)" if len(collisions) > 10 else ""
        raise SystemExit(f"Train already has {len(collisions)} lying-person frames: {preview}{extra}")

    merged = merge_coco(train, lying)
    info = merged.setdefault("info", {})
    config = deepcopy((train.get("info") or {}).get("config") or {})
    config["output"] = os.path.abspath(args.output)
    info["config"] = config

    shutil.copytree(args.source, args.output)
    out_train = os.path.join(args.output, "annotations", TRAIN_JSON)
    write_json(out_train, merged)

    source_keys = [key for key in info if key.startswith("source") and key.endswith("_info")]
    print(f"Copied {args.source} to {args.output}")
    print(
        f"Added {len(lying.get('images', []))} images and "
        f"{len(lying.get('annotations', []))} annotations to train"
    )
    print(f"Train images: {len(merged['images'])}")
    print(f"Train annotations: {len(merged['annotations'])}")
    print(f"Sources: {', '.join(sorted(source_keys, key=lambda key: int(key[6:-5])))}")
    print(f"Wrote {out_train}")


if __name__ == "__main__":
    main()
