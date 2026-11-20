#!/usr/bin/env python3
"""Build App_Vids_v2_GBE_v4_val10_subject25_require_ske_gbe.

Copies App_Vids_v2_GBE_v4_val10, including its val and mirrored val. From
seated_knee_extension.txt and ground_based_exercises.txt, moves 10% of the
listed train frames into a separate val JSON, one file per list. A subject-day
moves together: videos whose id is {name}_{exercise}_{date}_{id} and that
share the name and date stay in one split. Any other id is its own group.
Each held-out JSON config records this run: input is the copied dataset,
annotation is its train JSON, output is the new dataset directory, and seed
is the seed used to choose the 10%. held-out-val is the require-videos file
that produced that JSON, stored as a string the same way one require-videos
path is stored. After both held-out sets are written, every training video
that shares a moved subject-day is dropped. It is not added to a held-out
JSON. Those dropped videos are added to videos_removed.json.

The subject cap then runs on the remaining train. It protects
already_2b_annotated_gbe.txt and both lists, so a required video that stayed
in train is not dropped.

Usage:
    python data/make_subject25_require_ske_gbe.py
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from collections import defaultdict

from filter_train_data import (
    MAX_SUBJECT_FRACTION,
    REMOVED_JSON_NAME,
    filter_train_json,
    require_videos_config,
    video_prefix,
)
from split_dataset import load_id_list, parsed_name_date

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TXT = (
    "/Users/evanthomas/Plethy/visionAI/Vision-ML/"
    "app_vids_dataset/training_dataset/outputs_txt"
)
VAL10 = "App_Vids_v2_GBE_v4_val10"
OUTPUT_NAME = "App_Vids_v2_GBE_v4_val10_subject25_require_ske_gbe"
TRAIN_JSON = "person_keypoints_train2017.json"
VAL_JSON = "person_keypoints_val2017.json"

HELD_OUT = (
    ("seated_knee_extension_val.json", "seated_knee_extension.txt"),
    ("ground_based_exercises_val.json", "ground_based_exercises.txt"),
)


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


def frames_for_videos(data: dict, video_ids: set[str]) -> tuple[list[dict], list[dict]]:
    images = [image for image in data.get("images", []) if video_id(image) in video_ids]
    image_ids = {image["id"] for image in images}
    annotations = [
        ann for ann in data.get("annotations", []) if ann.get("image_id") in image_ids
    ]
    return images, annotations


def subject_day_key(video: str) -> tuple:
    """Same key as split_dataset.build_user_day_groups."""
    parsed = parsed_name_date(video)
    if parsed:
        return ("day", parsed[0], parsed[1])
    return ("video", video)


def subject_day_groups(frame_counts: dict[str, int]) -> list[dict]:
    """Group listed videos that share a parsed name and date."""
    buckets = defaultdict(list)
    for video, frames in frame_counts.items():
        buckets[subject_day_key(video)].append((video, frames))
    groups = []
    for key, items in buckets.items():
        groups.append({
            "key": key,
            "videos": [video for video, _frames in items],
            "frames": sum(frames for _video, frames in items),
        })
    return groups


def select_groups(groups: list[dict], fraction: float, seed: int) -> list[dict]:
    """Take whole subject-day groups until the frame target is met."""
    total = sum(group["frames"] for group in groups)
    if total <= 0:
        return []
    target = fraction * total
    order = list(groups)
    random.Random(seed).shuffle(order)
    chosen = []
    count = 0
    for group in order:
        if count >= target:
            break
        chosen.append(group)
        count += group["frames"]
    return chosen


def move_held_out_fraction(
    train: dict,
    val_videos: set[str],
    list_path: str,
    filename: str,
    annotations_dir: str,
    source_dir: str,
    output_dir: str,
    fraction: float,
    seed: int,
) -> set[str]:
    listed = load_id_list(list_path)
    listed_ids = set(listed)
    frame_counts = defaultdict(int)
    for image in train.get("images", []):
        video = video_id(image)
        if video in listed_ids:
            frame_counts[video] += 1
    in_primary_val = sorted(listed_ids & val_videos)
    missing = sorted(listed_ids - set(frame_counts) - val_videos)
    if not frame_counts:
        raise RuntimeError(f"No train frames for {os.path.basename(list_path)}")

    groups = subject_day_groups(frame_counts)
    chosen = select_groups(groups, fraction, seed)
    move_ids = {video for group in chosen for video in group["videos"]}
    images, annotations = frames_for_videos(train, move_ids)
    payload = {
        "info": held_out_info(train, list_path, source_dir, output_dir, fraction, seed),
        "images": images,
        "annotations": annotations,
        "categories": train.get("categories", []),
    }
    write_json(os.path.join(annotations_dir, filename), payload)
    moved_frames = len(images)
    train_frames = sum(frame_counts.values())
    day_groups = sum(1 for group in chosen if group["key"][0] == "day")
    print(
        f"{filename}: moved {len(move_ids)} videos, {moved_frames} images "
        f"({moved_frames / train_frames:.1%} of {train_frames} listed train images) "
        f"in {day_groups} subject-day groups and "
        f"{len(chosen) - day_groups} single videos"
    )
    if in_primary_val:
        print(
            f"  {len(in_primary_val)} listed videos stay in the primary val"
        )
    if missing:
        preview = ", ".join(missing[:10])
        extra = f" (+{len(missing) - 10} more)" if len(missing) > 10 else ""
        print(f"  {len(missing)} listed videos are not in this dataset: {preview}{extra}")
    return move_ids


def held_out_info(
    train: dict,
    list_path: str,
    source_dir: str,
    output_dir: str,
    fraction: float,
    seed: int,
) -> dict:
    """Config for a held-out JSON: where it came from and which list made it."""
    info = dict(train.get("info") or {})
    info["config"] = {
        "input": os.path.abspath(source_dir),
        "annotation": TRAIN_JSON,
        "output": os.path.abspath(output_dir),
        "seed": seed,
        "held-out-val": require_videos_config([os.path.abspath(list_path)]),
        "held-out-fraction": fraction,
    }
    return info


def videos_sharing_moved_days(
    train: dict,
    moved: set[str],
    required_ids: set[str],
) -> set[str]:
    """Drop every train video that shares a subject-day with a held-out video."""
    moved_days = {
        subject_day_key(video)
        for video in moved
        if subject_day_key(video)[0] == "day"
    }
    dropped = set()
    frames = 0
    listed_videos = set()
    listed_frames = 0
    for image in train.get("images", []):
        video = video_id(image)
        if video in moved or subject_day_key(video) not in moved_days:
            continue
        dropped.add(video)
        frames += 1
        if video in required_ids:
            listed_videos.add(video)
            listed_frames += 1
    print(
        f"Subject-day atomicity: dropped {len(dropped)} videos, {frames} frames "
        "from train"
    )
    print(
        f"Subject-day atomicity: {len(listed_videos)} videos, {listed_frames} frames "
        "of those were on either require list"
    )
    return dropped


def remove_videos(train: dict, video_ids: set[str]) -> None:
    train["images"] = [
        image for image in train.get("images", []) if video_id(image) not in video_ids
    ]
    kept_ids = {image["id"] for image in train["images"]}
    train["annotations"] = [
        ann for ann in train.get("annotations", []) if ann.get("image_id") in kept_ids
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy the val10 dataset, hold out 10% of two required-video lists "
            "by subject-day, then cap the largest training subject."
        )
    )
    parser.add_argument(
        "--source",
        default=dataset_path(VAL10),
        help="Dataset directory to copy",
    )
    parser.add_argument(
        "--output",
        default=dataset_path(OUTPUT_NAME),
    )
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
        default=os.path.join(TXT, HELD_OUT[0][1]),
    )
    parser.add_argument(
        "--ground-based-exercises",
        default=os.path.join(TXT, HELD_OUT[1][1]),
    )
    parser.add_argument(
        "--held-out-fraction",
        type=float,
        default=0.10,
        help="Fraction of each list's train frames to move, by whole subject-day",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--min-fraction-train-max-subject",
        type=float,
        default=MAX_SUBJECT_FRACTION,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 < args.held_out_fraction < 1:
        raise SystemExit(
            "--held-out-fraction must be between 0 and 1, "
            f"got {args.held_out_fraction}"
        )
    if os.path.exists(args.output):
        raise SystemExit(f"Output already exists: {args.output}")

    held_out_lists = [
        (HELD_OUT[0][0], args.seated_knee_extension),
        (HELD_OUT[1][0], args.ground_based_exercises),
    ]
    required = [
        args.already_required,
        args.seated_knee_extension,
        args.ground_based_exercises,
    ]
    source_train = os.path.join(args.source, "annotations", TRAIN_JSON)
    source_val = os.path.join(args.source, "annotations", VAL_JSON)
    for path in (source_train, source_val, args.subjects_path, *required):
        if not os.path.isfile(path):
            raise SystemExit(f"Missing file: {path}")

    shutil.copytree(args.source, args.output)
    print(f"Copied {args.source} to {args.output}")

    annotations_dir = os.path.join(args.output, "annotations")
    train_path = os.path.join(annotations_dir, TRAIN_JSON)
    train = load_json(train_path)
    val = load_json(os.path.join(annotations_dir, VAL_JSON))
    val_videos = {video_id(image) for image in val.get("images", [])}

    moved = set()
    held_out_names = []
    for filename, list_path in held_out_lists:
        moved |= move_held_out_fraction(
            train,
            val_videos,
            list_path,
            filename,
            annotations_dir,
            args.source,
            args.output,
            args.held_out_fraction,
            args.seed,
        )
        held_out_names.append(filename)
    required_ids = set()
    for _filename, list_path in held_out_lists:
        required_ids.update(load_id_list(list_path))
    dropped = videos_sharing_moved_days(train, moved, required_ids)
    remove_videos(train, moved | dropped)
    info = dict(train.get("info") or {})
    config = dict(info.get("config") or {})
    config["held-out-val"] = held_out_names
    config["held-out-fraction"] = args.held_out_fraction
    info["config"] = config
    train["info"] = info
    write_json(train_path, train)
    print(f"Train images after the held-out move: {len(train['images'])}")

    filter_train_json(
        train_path,
        train_path,
        args.min_fraction_train_max_subject,
        required,
        args.subjects_path,
        None,
        None,
    )
    record_atomicity_drops(annotations_dir, dropped)


def record_atomicity_drops(annotations_dir: str, dropped: set[str]) -> None:
    """Add subject-day drops to the videos_removed.json written by the cap."""
    path = os.path.join(annotations_dir, REMOVED_JSON_NAME)
    if not os.path.isfile(path):
        raise SystemExit(f"Missing {path}")
    payload = load_json(path)
    listed = {str(video) for video in payload.get("videos", [])}
    listed.update(dropped)
    payload["videos"] = sorted(listed)
    write_json(path, payload)
    print(f"Recorded {len(dropped)} subject-day drops in {path}")


if __name__ == "__main__":
    main()
