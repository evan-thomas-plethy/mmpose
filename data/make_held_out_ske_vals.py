#!/usr/bin/env python3
"""Split each single-subject held-out val into non-ske and ske vals.

Reads the eight held-out vals under data/App_Vids_v2_GBE_v4_folds/
<subject>_held_out/held_out/. Videos listed in seated_knee_extension.txt are
written to held_out_<subject>_ske.json. Every other video in that held-out
val, including ground-based exercises, is written to
held_out_<subject>_non_ske.json. The original person_keypoints_val2017.json
is left unchanged. A subject with no videos on one side still gets that
file, with empty image and annotation lists.

Usage:
    python data/make_held_out_ske_vals.py
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
    exercise_ids: set[str],
    keep_exercise: bool,
    split: str,
    ske_videos: str,
) -> dict:
    images = [
        image for image in data.get("images", [])
        if (video_id(image) in exercise_ids) == keep_exercise
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
    info["config"] = config
    out["info"] = info
    out["images"] = images
    out["annotations"] = annotations
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split each held-out subject val into a non-ske val and a val of "
            "seated_knee_extension.txt videos."
        )
    )
    parser.add_argument("--subjects", nargs="+", default=list(SUBJECTS))
    parser.add_argument("--ske-videos", default=SKE_TXT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not os.path.isfile(args.ske_videos):
        raise SystemExit(f"Missing file: {args.ske_videos}")
    exercise_ids = set(load_id_list(args.ske_videos))

    planned = []
    for subject in args.subjects:
        source = os.path.join(ann_dir(subject), VAL_JSON)
        if not os.path.isfile(source):
            raise SystemExit(f"Missing file: {source}")
        outputs = [
            os.path.join(ann_dir(subject), f"held_out_{subject}_non_ske.json"),
            os.path.join(ann_dir(subject), f"held_out_{subject}_ske.json"),
        ]
        existing = [path for path in outputs if os.path.exists(path)]
        if existing:
            raise SystemExit("Output already exists: " + ", ".join(existing))
        planned.append((subject, source, outputs))

    for subject, source, (non_ske_path, ske_path) in planned:
        data = load_json(source)
        non_ske = subset(data, exercise_ids, False, "non_ske", args.ske_videos)
        ske = subset(data, exercise_ids, True, "ske", args.ske_videos)
        write_json(non_ske_path, non_ske)
        write_json(ske_path, ske)
        print(
            f"{subject}: non_ske {len(non_ske['images'])} images, "
            f"ske {len(ske['images'])} images "
            f"(source {len(data.get('images', []))})"
        )
        if not non_ske["images"] or not ske["images"]:
            print(f"  warning: {subject} has an empty split")


if __name__ == "__main__":
    main()
