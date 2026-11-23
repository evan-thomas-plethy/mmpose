#!/usr/bin/env python3
"""Build the lying-person and InfiniteForm736 fold datasets used by v6.

Copies data/App_Vids_v2_GBE_v4_folds/rishi_jamie and tim_nina. Val stays the
copied fold split. Lying_Person_Dataset_Single_Ann.json is merged into each
training JSON, the same full set make_lyingperson_require_ske_gbe.py adds
(that merge has no seed). A second copy of each lying-person fold then gets
the first 736 images from one Random(seed=42) shuffle of
InfiniteForm_Single_Ann.json. That shuffle matches
make_lyingperson_infiniteform_require_ske_gbe.py, so these 736 images are
the InfiniteForm736 sample used by sweep_config_app_vids_gbe_v5p1.yaml.

Usage:
    python data/make_fold_lyingperson_infiniteform.py
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
FOLDS_DIR = "App_Vids_v2_GBE_v4_folds"
FOLDS = ("rishi_jamie", "tim_nina")
LYING_JSON = "Lying_Person_Dataset_Single_Ann/Lying_Person_Dataset_Single_Ann.json"
INFINITEFORM_JSON = "InfiniteForm_Single_Ann/InfiniteForm_Single_Ann.json"
TRAIN_JSON = "person_keypoints_train2017.json"
INFINITEFORM_COUNT = 736
INFINITEFORM_SEED = 42


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


def colliding_names(train: dict, extra: dict) -> list[str]:
    train_names = {image_name(image) for image in train.get("images", [])}
    return sorted(
        image_name(image)
        for image in extra.get("images", [])
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


def fold_dir(name: str) -> str:
    return dataset_path(FOLDS_DIR, name)


def lying_dir(name: str) -> str:
    return dataset_path(FOLDS_DIR, f"{name}_lyingperson")


def infiniteform_dir(name: str, count: int) -> str:
    return dataset_path(FOLDS_DIR, f"{name}_lyingperson_InfiniteForm{count}")


def merged_train(train: dict, extra: dict, output: str, extra_config: dict | None = None) -> dict:
    merged = merge_coco(train, extra)
    info = merged.setdefault("info", {})
    config = deepcopy((train.get("info") or {}).get("config") or {})
    config["output"] = os.path.abspath(output)
    if extra_config:
        config.update(extra_config)
    info["config"] = config
    return merged


def write_copy(source: str, dest: str, train: dict) -> str:
    shutil.copytree(source, dest)
    out_train = os.path.join(dest, "annotations", TRAIN_JSON)
    write_json(out_train, train)
    return out_train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy the rishi_jamie and tim_nina folds, merge the lying-person "
            "set into train, and merge the same InfiniteForm736 sample."
        )
    )
    parser.add_argument("--folds", nargs="+", default=list(FOLDS))
    parser.add_argument("--lying-json", default=dataset_path(LYING_JSON))
    parser.add_argument("--infiniteform-json", default=dataset_path(INFINITEFORM_JSON))
    parser.add_argument("--count", type=int, default=INFINITEFORM_COUNT)
    parser.add_argument("--seed", type=int, default=INFINITEFORM_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.count < 1:
        raise SystemExit(f"--count must be positive, got {args.count}")
    for path in (args.lying_json, args.infiniteform_json):
        if not os.path.isfile(path):
            raise SystemExit(f"Missing file: {path}")

    outputs = []
    for name in args.folds:
        source = fold_dir(name)
        train_path = os.path.join(source, "annotations", TRAIN_JSON)
        if not os.path.isfile(train_path):
            raise SystemExit(f"Missing file: {train_path}")
        outputs.extend((lying_dir(name), infiniteform_dir(name, args.count)))
    existing = [path for path in outputs if os.path.exists(path)]
    if existing:
        raise SystemExit("Output already exists: " + ", ".join(existing))

    lying = load_json(args.lying_json)
    infiniteform = load_json(args.infiniteform_json)
    images = list(infiniteform.get("images", []))
    if len(images) < args.count:
        raise SystemExit(
            f"InfiniteForm has {len(images)} images, fewer than {args.count}"
        )
    random.Random(args.seed).shuffle(images)
    sample = sample_prefix(infiniteform, images[: args.count])
    sample["info"]["sample_images"] = args.count
    sample["info"]["sample_seed"] = args.seed
    extra_config = {"infiniteform-images": args.count, "infiniteform-seed": args.seed}

    for name in args.folds:
        source = fold_dir(name)
        train = load_json(os.path.join(source, "annotations", TRAIN_JSON))
        collisions = colliding_names(train, lying)
        if collisions:
            preview = ", ".join(collisions[:10])
            extra = f" (+{len(collisions) - 10} more)" if len(collisions) > 10 else ""
            raise SystemExit(
                f"{name} already has {len(collisions)} lying-person frames: {preview}{extra}"
            )

        lying_dest = lying_dir(name)
        lying_train = merged_train(train, lying, lying_dest)
        lying_collisions = colliding_names(lying_train, sample)
        if lying_collisions:
            preview = ", ".join(lying_collisions[:10])
            extra = f" (+{len(lying_collisions) - 10} more)" if len(lying_collisions) > 10 else ""
            raise SystemExit(
                f"{name} already has {len(lying_collisions)} InfiniteForm frames: {preview}{extra}"
            )

        if_dest = infiniteform_dir(name, args.count)
        if_train = merged_train(lying_train, sample, if_dest, extra_config)
        write_copy(source, lying_dest, lying_train)
        write_copy(lying_dest, if_dest, if_train)
        print(
            f"{os.path.basename(lying_dest)}: train images {len(lying_train['images'])} "
            f"(+{len(lying.get('images', []))} lying-person)"
        )
        print(
            f"{os.path.basename(if_dest)}: train images {len(if_train['images'])} "
            f"(+{len(sample['images'])} InfiniteForm, seed {args.seed})"
        )


if __name__ == "__main__":
    main()
