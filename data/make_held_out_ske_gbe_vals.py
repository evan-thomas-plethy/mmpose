#!/usr/bin/env python3
"""Split each single-subject held-out val into ske, gbe, and remainder vals.

Reads the eight held-out vals under data/App_Vids_v2_GBE_v4_folds/
<subject>_held_out/held_out/. Videos listed in seated_knee_extension.txt are
written to held_out_<subject>_ske.json. Videos listed in
ground_based_exercises.txt are written to held_out_<subject>_gbe.json.
Every other video is written to held_out_<subject>_non_ske_gbe.json. A video
listed in both files is written only to the ske val. The original
person_keypoints_val2017.json is left unchanged. A subject with no videos
on one side still gets that file, with empty image and annotation lists.

Usage:
    python data/make_held_out_ske_gbe_vals.py
"""

from __future__ import annotations

import argparse
import json
import os
from copy import deepcopy

from filter_train_data import video_prefix
from split_dataset import load_id_list

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FOLDS_DIR = "App_Vids_v2_GBE_v4_folds"
SUBJECTS = ("evan", "yordanos", "rishi", "jamie", "tim", "nina", "anthony", "joey")
VAL_JSON = "person_keypoints_val2017.json"
TXT = (
    "/Users/evanthomas/Plethy/visionAI/Vision-ML/"
    "app_vids_dataset/training_dataset/outputs_txt"
)
SKE_TXT = os.path.join(TXT, "seated_knee_extension.txt")
GBE_TXT = os.path.join(TXT, "ground_based_exercises.txt")


def dataset_path(*parts: str) -> str:
    return os.path.join(REPO, "data", *parts)


def ann_dir(subject: str) -> str:
    return dataset_path(FOLDS_DIR, f"{subject}_held_out", "held_out", "annotations")


def load_json(path: str) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=4)
        handle.write("\n")


def video_id(image: dict) -> str:
    return video_prefix(image.get("file_name", ""))


def subset(
    data: dict,
    keep_ids: set[str] | None,
    split: str,
    ske_videos: str,
    gbe_videos: str,
) -> dict:
    """keep_ids None keeps every image. Otherwise keep images in that set."""
    images = [
        image for image in data.get("images", [])
        if keep_ids is None or video_id(image) in keep_ids
    ]
    image_ids = {image["id"] for image in images}
    annotations = [
        ann for ann in data.get("annotations", [])
        if ann.get("image_id") in image_ids
    ]
    out = {
        key: deepcopy(data[key])
        for key in ("licenses", "categories")
        if key in data
    }
    info = deepcopy(data.get("info") or {})
    config = deepcopy(info.get("config") or {})
    config["held-out-split"] = split
    config["ske-videos"] = ske_videos
    config["gbe-videos"] = gbe_videos
    info["config"] = config
    out["info"] = info
    out["images"] = images
    out["annotations"] = annotations
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split each held-out subject val into seated-knee-extension, "
            "ground-based-exercise, and remaining frames."
        )
    )
    parser.add_argument("--subjects", nargs="+", default=list(SUBJECTS))
    parser.add_argument("--ske-videos", default=SKE_TXT)
    parser.add_argument("--gbe-videos", default=GBE_TXT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in (args.ske_videos, args.gbe_videos):
        if not os.path.isfile(path):
            raise SystemExit(f"Missing file: {path}")
    ske_ids = set(load_id_list(args.ske_videos))
    gbe_ids = set(load_id_list(args.gbe_videos)) - ske_ids
    overlap = set(load_id_list(args.gbe_videos)) & ske_ids
    if overlap:
        print(f"warning: {len(overlap)} videos are in both lists; they go to the ske val")

    planned = []
    for subject in args.subjects:
        source = os.path.join(ann_dir(subject), VAL_JSON)
        if not os.path.isfile(source):
            raise SystemExit(f"Missing file: {source}")
        outputs = [
            os.path.join(ann_dir(subject), f"held_out_{subject}_ske.json"),
            os.path.join(ann_dir(subject), f"held_out_{subject}_gbe.json"),
            os.path.join(ann_dir(subject), f"held_out_{subject}_non_ske_gbe.json"),
        ]
        existing = [path for path in outputs if os.path.exists(path)]
        if existing:
            raise SystemExit("Output already exists: " + ", ".join(existing))
        planned.append((subject, source, outputs))

    for subject, source, (ske_path, gbe_path, rest_path) in planned:
        data = load_json(source)
        present = {video_id(image) for image in data.get("images", [])}
        ske = subset(data, present & ske_ids, "ske", args.ske_videos, args.gbe_videos)
        gbe = subset(data, present & gbe_ids, "gbe", args.ske_videos, args.gbe_videos)
        rest_ids = present - ske_ids - gbe_ids
        rest = subset(data, rest_ids, "non_ske_gbe", args.ske_videos, args.gbe_videos)
        write_json(ske_path, ske)
        write_json(gbe_path, gbe)
        write_json(rest_path, rest)
        print(
            f"{subject}: ske {len(ske['images'])} images, "
            f"gbe {len(gbe['images'])} images, "
            f"non_ske_gbe {len(rest['images'])} images "
            f"(source {len(data.get('images', []))})"
        )
        if not ske["images"] or not gbe["images"] or not rest["images"]:
            print(f"  warning: {subject} has an empty split")


if __name__ == "__main__":
    main()
