#!/usr/bin/env python3
"""Build App_Vids_v2_GBE_v4_val10_subject25_require_ske_gbe_lyingperson.

Copies App_Vids_v2_GBE_v4_val10, including its val and mirrored val. Those
files are the same as App_Vids_v2_GBE_v4_val10_subject25. The subject cap
then runs on the copied train and protects already_2b_annotated_gbe.txt,
seated_knee_extension.txt, and ground_based_exercises.txt. After that cap,
Lying_Person_Dataset_Single_Ann.json is merged into the training JSON only.
Its image and annotation ids are remapped the same way as
merge_datasets.merge_coco.

Usage:
    python data/make_subject25_require_ske_gbe_lyingperson.py
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from copy import deepcopy

from filter_train_data import MAX_SUBJECT_FRACTION, REMOVED_JSON_NAME, filter_train_json
from merge_datasets import merge_coco

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TXT = (
    "/Users/evanthomas/Plethy/visionAI/Vision-ML/"
    "app_vids_dataset/training_dataset/outputs_txt"
)
SOURCE_NAME = "App_Vids_v2_GBE_v4_val10"
OUTPUT_NAME = "App_Vids_v2_GBE_v4_val10_subject25_require_ske_gbe_lyingperson"
LYING_JSON = "Lying_Person_Dataset_Single_Ann/Lying_Person_Dataset_Single_Ann.json"
TRAIN_JSON = "person_keypoints_train2017.json"
VAL_JSON = "person_keypoints_val2017.json"
MIRRORED_VAL_JSON = "person_keypoints_val2017_mirrored.json"


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
            "Copy the val10 dataset, cap its train while protecting seated "
            "knee extension and ground-based videos, then add lying-person frames."
        )
    )
    parser.add_argument("--source", default=dataset_path(SOURCE_NAME))
    parser.add_argument("--lying-json", default=dataset_path(LYING_JSON))
    parser.add_argument("--output", default=dataset_path(OUTPUT_NAME))
    parser.add_argument(
        "--subjects-path",
        default=os.path.join(TXT, "users.csv"),
    )
    parser.add_argument(
        "--already-required",
        default=os.path.join(TXT, "already_2b_annotated_gbe.txt"),
    )
    parser.add_argument(
        "--seated-knee-extension",
        default=os.path.join(TXT, "seated_knee_extension.txt"),
    )
    parser.add_argument(
        "--ground-based-exercises",
        default=os.path.join(TXT, "ground_based_exercises.txt"),
    )
    parser.add_argument(
        "--min-fraction-train-max-subject",
        type=float,
        default=MAX_SUBJECT_FRACTION,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_train = os.path.join(args.source, "annotations", TRAIN_JSON)
    source_val = os.path.join(args.source, "annotations", VAL_JSON)
    source_mirrored = os.path.join(args.source, "annotations", MIRRORED_VAL_JSON)
    required = [
        args.already_required,
        args.seated_knee_extension,
        args.ground_based_exercises,
    ]
    for path in (source_train, source_val, source_mirrored, args.lying_json, args.subjects_path, *required):
        if not os.path.isfile(path):
            raise SystemExit(f"Missing file: {path}")
    if os.path.exists(args.output):
        raise SystemExit(f"Output already exists: {args.output}")

    source_data = load_json(source_train)
    source_config = (source_data.get("info") or {}).get("config") or {}
    lying = load_json(args.lying_json)
    collisions = colliding_names(source_data, lying)
    if collisions:
        preview = ", ".join(collisions[:10])
        extra = f" (+{len(collisions) - 10} more)" if len(collisions) > 10 else ""
        raise SystemExit(
            f"Train already has {len(collisions)} lying-person frames: {preview}{extra}"
        )
    del source_data

    shutil.copytree(args.source, args.output)
    print(f"Copied {args.source} to {args.output}")

    train_path = os.path.join(args.output, "annotations", TRAIN_JSON)
    filter_train_json(
        train_path,
        train_path,
        args.min_fraction_train_max_subject,
        required,
        args.subjects_path,
        None,
        None,
    )

    train = load_json(train_path)
    info = dict(train.get("info") or {})
    config = deepcopy(info.get("config") or {})
    if source_config.get("input") is not None:
        config["input"] = source_config["input"]
    if source_config.get("annotation") is not None:
        config["annotation"] = source_config["annotation"]
    config["output"] = os.path.abspath(args.output)
    info["config"] = config
    train["info"] = info

    merged = merge_coco(train, lying)
    merged.setdefault("info", {})["config"] = config
    write_json(train_path, merged)

    removed_path = os.path.join(args.output, "annotations", REMOVED_JSON_NAME)
    if os.path.isfile(removed_path):
        removed = load_json(removed_path)
        removed["config"] = config
        write_json(removed_path, removed)

    source_keys = [
        key for key in merged["info"] if key.startswith("source") and key.endswith("_info")
    ]
    print(
        f"Added {len(lying.get('images', []))} lying-person images and "
        f"{len(lying.get('annotations', []))} annotations"
    )
    print(f"Train images: {len(merged['images'])}")
    print(f"Train annotations: {len(merged['annotations'])}")
    print(f"Sources: {', '.join(sorted(source_keys, key=lambda key: int(key[6:-5])))}")
    print(f"Wrote {train_path}")


if __name__ == "__main__":
    main()
