#!/usr/bin/env python3
"""Write each subject's full ske and gbe vals from App_Vids_v2_GBE_v4.

For every single-subject held-out directory under
data/App_Vids_v2_GBE_v4_folds/, collect every image in
data/App_Vids_v2_GBE_v4.json whose subject is that held-out person and whose
video is listed in seated_knee_extension.txt or ground_based_exercises.txt.
Those images and their annotations are written to
held_out/annotations/full_held_out_<subject>_ske.json and
full_held_out_<subject>_gbe.json. This is the full v2_v4 frame set for those
videos, not the frame-limited held-out val. A video listed in both files is
written only to the ske val. A subject with no videos on one list skips that
file. An existing full_held_out file is overwritten. The source JSON and the
other held-out files are not changed.

Usage:
    python data/make_full_held_out_ske_gbe.py
"""

from __future__ import annotations

import argparse
import json
import os
from copy import deepcopy

from filter_train_data import video_prefix
from split_dataset import load_id_list, load_video_subjects

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FOLDS_DIR = "App_Vids_v2_GBE_v4_folds"
SOURCE_JSON = "App_Vids_v2_GBE_v4.json"
VAL_JSON = "person_keypoints_val2017.json"
TXT = (
    "/Users/evanthomas/Plethy/visionAI/Vision-ML/"
    "app_vids_dataset/training_dataset/outputs_txt"
)
SKE_TXT = os.path.join(TXT, "seated_knee_extension.txt")
GBE_TXT = os.path.join(TXT, "ground_based_exercises.txt")
SUBJECTS_CSV = os.path.join(TXT, "users.csv")


def dataset_path(*parts: str) -> str:
    return os.path.join(REPO, "data", *parts)


def load_json(path: str) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=4)
        handle.write("\n")


def video_id(image: dict) -> str:
    return video_prefix(image.get("file_name", ""))


def single_subject_dirs(folds_dir: str) -> list[tuple[str, str, str]]:
    """Return (short name, full subject, annotations dir) for one-person held-outs."""
    found = []
    for name in sorted(os.listdir(folds_dir)):
        ann_dir = os.path.join(folds_dir, name, "held_out", "annotations")
        val_path = os.path.join(ann_dir, VAL_JSON)
        if not os.path.isfile(val_path):
            continue
        config = ((load_json(val_path).get("info") or {}).get("config") or {})
        subjects = config.get("held-out-subjects") or []
        if len(subjects) != 1:
            continue
        short = name[: -len("_held_out")] if name.endswith("_held_out") else name
        found.append((short, subjects[0], ann_dir))
    return found


def videos_by_subject(
    video_ids: set[str],
    video_subjects: dict[str, str],
    label: str,
) -> dict[str, set[str]]:
    by_subject: dict[str, set[str]] = {}
    missing = []
    for video in sorted(video_ids):
        subject = video_subjects.get(video)
        if subject is None:
            missing.append(video)
            continue
        by_subject.setdefault(subject, set()).add(video)
    if missing:
        preview = ", ".join(missing[:10])
        raise SystemExit(f"{len(missing)} {label} videos have no subject: {preview}")
    return by_subject


def subset_for_subject(
    source: dict,
    video_ids: set[str],
    subject: str,
    split: str,
    ske_videos: str,
    gbe_videos: str,
    source_json: str,
) -> dict:
    images = [
        image for image in source.get("images", [])
        if video_id(image) in video_ids
    ]
    image_ids = {image["id"] for image in images}
    annotations = [
        ann for ann in source.get("annotations", [])
        if ann.get("image_id") in image_ids
    ]
    out = {
        key: deepcopy(source[key])
        for key in ("licenses", "categories")
        if key in source
    }
    info = deepcopy(source.get("info") or {})
    config = deepcopy(info.get("config") or {})
    config["held-out-subjects"] = [subject]
    config["held-out-split"] = split
    config["ske-videos"] = ske_videos
    config["gbe-videos"] = gbe_videos
    config["source-json"] = source_json
    info["config"] = config
    out["info"] = info
    out["images"] = images
    out["annotations"] = annotations
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Write full_held_out_<subject>_ske.json and "
            "full_held_out_<subject>_gbe.json from every App_Vids_v2_GBE_v4 "
            "frame of that subject's listed videos."
        )
    )
    parser.add_argument("--source-json", default=dataset_path(SOURCE_JSON))
    parser.add_argument("--folds-dir", default=dataset_path(FOLDS_DIR))
    parser.add_argument("--ske-videos", default=SKE_TXT)
    parser.add_argument("--gbe-videos", default=GBE_TXT)
    parser.add_argument("--subjects-path", default=SUBJECTS_CSV)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in (args.source_json, args.ske_videos, args.gbe_videos, args.subjects_path):
        if not os.path.isfile(path):
            raise SystemExit(f"Missing file: {path}")
    if not os.path.isdir(args.folds_dir):
        raise SystemExit(f"Missing directory: {args.folds_dir}")

    held_outs = single_subject_dirs(args.folds_dir)
    if not held_outs:
        raise SystemExit(f"No single-subject held-out dirs in {args.folds_dir}")

    video_subjects = load_video_subjects(args.subjects_path)
    ske_ids = set(load_id_list(args.ske_videos))
    gbe_listed = set(load_id_list(args.gbe_videos))
    overlap = gbe_listed & ske_ids
    if overlap:
        print(f"warning: {len(overlap)} videos are in both lists; they go to the ske val")
    gbe_ids = gbe_listed - ske_ids
    ske_by_subject = videos_by_subject(ske_ids, video_subjects, "seated-knee-extension")
    gbe_by_subject = videos_by_subject(gbe_ids, video_subjects, "ground-based-exercise")

    planned = []
    for short, subject, ann_dir in held_outs:
        splits = (
            ("full_ske", "ske", ske_by_subject.get(subject, set())),
            ("full_gbe", "gbe", gbe_by_subject.get(subject, set())),
        )
        outputs = []
        for split, label, video_ids in splits:
            if not video_ids:
                print(f"{short}: no {label} videos, skipped")
                continue
            output = os.path.join(ann_dir, f"full_held_out_{short}_{label}.json")
            outputs.append((split, label, output, video_ids))
        planned.append((short, subject, outputs))

    source = load_json(args.source_json)
    source_json = os.path.abspath(args.source_json)
    for short, subject, outputs in planned:
        for split, label, output, video_ids in outputs:
            payload = subset_for_subject(
                source,
                video_ids,
                subject,
                split,
                args.ske_videos,
                args.gbe_videos,
                source_json,
            )
            write_json(output, payload)
            print(
                f"{short} {label}: {len(payload['images'])} images from "
                f"{len(video_ids)} videos, wrote {output}"
            )


if __name__ == "__main__":
    main()
