#!/usr/bin/env python3
"""Build the InfiniteForm736, 1472, and 2944 lying-person datasets.

Copies App_Vids_v2_GBE_v4_val10_subject25_lyingperson_require_ske_gbe three
times. Val, mirrored val, held-out val, and videos_removed.json stay as
copied. One seeded shuffle of InfiniteForm_Single_Ann.json picks images, and
each size takes a prefix of that order, so 736 is inside 1472 and 1472 is
inside 2944. Those images and their annotations are merged into the training
JSON only. Ids are remapped the same way as merge_datasets.merge_coco.

Usage:
    python data/make_lyingperson_infiniteform_require_ske_gbe.py
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from copy import deepcopy

from merge_datasets import merge_coco

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE_NAME = "App_Vids_v2_GBE_v4_val10_subject25_lyingperson_require_ske_gbe"
OUTPUT_TEMPLATE = "App_Vids_v2_GBE_v4_val10_subject25_lyingperson_InfiniteForm{n}_require_ske_gbe"
INFINITEFORM_JSON = "InfiniteForm_Single_Ann/InfiniteForm_Single_Ann.json"
TRAIN_JSON = "person_keypoints_train2017.json"
COUNTS = (736, 1472, 2944)


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


def colliding_names(train: dict, sample: dict) -> list[str]:
    train_names = {image_name(image) for image in train.get("images", [])}
    return sorted(
        image_name(image)
        for image in sample.get("images", [])
        if image_name(image) in train_names
    )


def sample_prefix(infiniteform: dict, images: list[dict]) -> dict:
    """Images in shuffle order, plus every annotation that belongs to them."""
    image_ids = {image["id"] for image in images}
    info = deepcopy(infiniteform.get("info") or {})
    return {
        "info": info,
        "licenses": infiniteform.get("licenses", []),
        "categories": infiniteform.get("categories", []),
        "images": images,
        "annotations": [
            ann
            for ann in infiniteform.get("annotations", [])
            if ann.get("image_id") in image_ids
        ],
    }


def output_dir(n: int) -> str:
    return dataset_path(OUTPUT_TEMPLATE.format(n=n))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy the lying-person require-ske-gbe dataset and add nested "
            "random InfiniteForm samples of 736, 1472, and 2944 images."
        )
    )
    parser.add_argument("--source", default=dataset_path(SOURCE_NAME))
    parser.add_argument("--infiniteform-json", default=dataset_path(INFINITEFORM_JSON))
    parser.add_argument("--counts", type=int, nargs="+", default=list(COUNTS))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    counts = sorted(args.counts)
    if not counts or any(count < 1 for count in counts):
        raise SystemExit(f"--counts must be positive, got {args.counts}")
    train_path = os.path.join(args.source, "annotations", TRAIN_JSON)
    for path in (train_path, args.infiniteform_json):
        if not os.path.isfile(path):
            raise SystemExit(f"Missing file: {path}")
    outputs = [output_dir(count) for count in counts]
    existing = [path for path in outputs if os.path.exists(path)]
    if existing:
        raise SystemExit("Output already exists: " + ", ".join(existing))

    train = load_json(train_path)
    infiniteform = load_json(args.infiniteform_json)
    images = list(infiniteform.get("images", []))
    if len(images) < counts[-1]:
        raise SystemExit(
            f"InfiniteForm has {len(images)} images, fewer than {counts[-1]}"
        )
    random.Random(args.seed).shuffle(images)

    for count, dest in zip(counts, outputs):
        sample = sample_prefix(infiniteform, images[:count])
        sample["info"]["sample_images"] = count
        sample["info"]["sample_seed"] = args.seed
        collisions = colliding_names(train, sample)
        if collisions:
            preview = ", ".join(collisions[:10])
            extra = f" (+{len(collisions) - 10} more)" if len(collisions) > 10 else ""
            raise SystemExit(
                f"Train already has {len(collisions)} InfiniteForm frames: {preview}{extra}"
            )
        merged = merge_coco(train, sample)
        info = merged.setdefault("info", {})
        config = deepcopy((train.get("info") or {}).get("config") or {})
        config["output"] = os.path.abspath(dest)
        config["infiniteform-images"] = count
        config["infiniteform-seed"] = args.seed
        info["config"] = config

        shutil.copytree(args.source, dest)
        out_train = os.path.join(dest, "annotations", TRAIN_JSON)
        write_json(out_train, merged)
        print(
            f"{os.path.basename(dest)}: added {len(sample['images'])} images and "
            f"{len(sample['annotations'])} annotations; "
            f"train images {len(merged['images'])}"
        )
        print(f"Wrote {out_train}")


if __name__ == "__main__":
    main()
