#!/usr/bin/env python3
"""
Drop individual videos from one subject in a training COCO JSON until that
subject is no more than 25% of the remaining images, or one more video would
put them below the subject with the second-most frames. The subject is the one
with the most frames at the start. Other subjects are not trimmed.

Each pass drops one random video from that subject, then counts again.
Required videos stay. The subject for each video comes from the subjects
CSV, the same lookup as data/split_dataset.py. A video missing from the CSV
uses the name in its id, or the id itself when the name and date are absent.

Usage:
    python data/filter_train_data.py \
        --train-json annotations/person_keypoints_train2017.json \
        --subjects-path users.csv

    python data/filter_train_data.py \
        --train-json annotations/person_keypoints_train2017.json \
        --subjects-path users.csv \
        --output annotations/person_keypoints_train2017_subject25.json

The input file is not modified. When the training JSON sits beside
person_keypoints_val2017.json, that validation file is copied next to the
output training JSON. Removed video ids are written to train_videos_removed.txt
beside the output JSON. Pass that file as --removed-txt to drop those videos
only, with no subject cap.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from collections import defaultdict

from split_dataset import (
    load_id_list,
    load_video_subjects,
    required_video_set,
    subject_for_video,
)

MAX_SUBJECT_FRACTION = 0.25
VAL_JSON_NAME = "person_keypoints_val2017.json"


def video_prefix(file_name: str) -> str:
    filename = os.path.basename(file_name.replace("\\", "/"))
    if "_frame_" in filename:
        return filename.split("_frame_")[0]
    return filename


def over_fraction(count: int, total: int, fraction: float) -> bool:
    if total <= 0:
        return count > 0
    return count > fraction * total


def build_videos(
    images: list[dict],
    required_ids: set[str],
    video_subjects: dict[str, str] | None,
) -> list[dict]:
    videos = defaultdict(list)
    for image in images:
        videos[video_prefix(image.get("file_name", ""))].append(image)

    if video_subjects is not None:
        missing = [video_id for video_id in videos if video_id not in video_subjects]
        if missing:
            preview = ", ".join(missing[:10])
            extra = f" (+{len(missing) - 10} more)" if len(missing) > 10 else ""
            print(
                f"⚠️ {len(missing)} videos are not in the subjects CSV "
                f"({preview}{extra}). Each uses the name in its id, or the id "
                "itself when the id has no name and date."
            )

    records = []
    for video_id, video_images in videos.items():
        image_ids = [image["id"] for image in video_images]
        records.append({
            "video_id": video_id,
            "subject": subject_for_video(video_id, video_subjects or {}),
            "image_ids": image_ids,
            "frames": len(image_ids),
            "protected": video_id in required_ids,
        })
    return records


def cap_subjects(videos: list[dict], fraction: float) -> list[dict]:
    """Drop videos from the single largest subject until that subject is within fraction.

    Also stop before a removal that would put that subject below the subject
    with the second-most frames.
    """
    kept = list(videos)
    totals = defaultdict(int)
    total = 0
    for video in kept:
        total += video["frames"]
        totals[video["subject"]] += video["frames"]
    if total <= 0 or not totals:
        print("Dropped 0 videos")
        return kept
    subject = max(totals, key=lambda name: totals[name])
    print(f"Capping {subject}")
    dropped = 0
    max_steps = len(videos) + 1
    for _ in range(max_steps):
        totals = defaultdict(int)
        total = 0
        for video in kept:
            total += video["frames"]
            totals[video["subject"]] += video["frames"]
        if total <= 0 or not over_fraction(totals[subject], total, fraction):
            print(f"Dropped {dropped} videos")
            return kept
        removable = [
            video
            for video in kept
            if video["subject"] == subject and not video["protected"]
        ]
        if not removable:
            print(
                "Subject cap cannot be met without dropping required videos: "
                f"{subject} {totals[subject] / total:.1%}"
            )
            print(f"Dropped {dropped} videos")
            return kept
        choice = random.choice(removable)
        others = [frames for name, frames in totals.items() if name != subject]
        second = max(others) if others else 0
        new_total = total - choice["frames"]
        new_subject = totals[subject] - choice["frames"]
        if new_total > 0 and new_subject / new_total < second / new_total:
            second_name = max(
                (name for name in totals if name != subject),
                key=lambda name: totals[name],
            )
            print(
                f"Stopping: dropping one more {subject} video would put them at "
                f"{new_subject / new_total:.1%}, below {second_name} "
                f"{second / new_total:.1%}"
            )
            print(f"Dropped {dropped} videos")
            return kept
        kept.remove(choice)
        dropped += 1
    print(f"Subject cap stopped after {dropped} drops")
    return kept


def write_removed_videos(output_path: str, video_ids: list[str]) -> str:
    path = os.path.join(os.path.dirname(os.path.abspath(output_path)) or ".", "train_videos_removed.txt")
    with open(path, "w", encoding="utf-8") as handle:
        for video_id in video_ids:
            handle.write(f"{video_id}\n")
    return path


def drop_listed_videos(videos: list[dict], remove_ids: set[str]) -> list[dict]:
    """Drop every frame of the listed videos. Other videos stay."""
    present = {video["video_id"] for video in videos}
    missing = sorted(remove_ids - present)
    if missing:
        preview = ", ".join(missing[:10])
        extra = f" (+{len(missing) - 10} more)" if len(missing) > 10 else ""
        print(
            f"⚠️ {len(missing)} listed videos are not in the training JSON "
            f"({preview}{extra})"
        )
    kept = [video for video in videos if video["video_id"] not in remove_ids]
    print(f"Dropped {len(present & remove_ids)} videos")
    return kept


def default_output_path(train_json: str) -> str:
    directory, filename = os.path.split(train_json)
    stem, ext = os.path.splitext(filename)
    return os.path.join(directory, f"{stem}_subject25{ext or '.json'}")


def print_subject_shares(videos: list[dict]) -> None:
    totals = defaultdict(int)
    total = 0
    for video in videos:
        totals[video["subject"]] += video["frames"]
        total += video["frames"]
    if total <= 0:
        print("Subject frames: empty")
        return
    ranked = sorted(totals.items(), key=lambda item: (-item[1], item[0]))
    shares = ", ".join(f"{subject} {frames / total:.1%}" for subject, frames in ranked)
    print(f"Subject frames {total}: {shares}")


def filter_train_json(
    train_json: str,
    output_path: str,
    fraction: float,
    require_videos: str | None,
    subjects_path: str,
    removed_txt: str | None,
) -> None:
    with open(train_json, encoding="utf-8") as handle:
        data = json.load(handle)

    images = data.get("images", [])
    annotations = data.get("annotations", [])
    required_ids = set()
    video_subjects = load_video_subjects(subjects_path)
    if removed_txt:
        videos = build_videos(images, required_ids, video_subjects)
        remove_ids = set(load_id_list(removed_txt))
        kept = drop_listed_videos(videos, remove_ids)
    else:
        if require_videos:
            listed_required = load_id_list(require_videos)
            required_ids = required_video_set(listed_required)
            present = {
                video_prefix(image.get("file_name", ""))
                for image in images
                if video_prefix(image.get("file_name", "")) in required_ids
            }
            print(
                f"Required videos present: {len(present)} "
                f"(list has {len(listed_required)} ids)"
            )
        videos = build_videos(images, required_ids, video_subjects)
        random.seed(42)
        kept = cap_subjects(videos, fraction)
    removed_ids = sorted(
        video["video_id"] for video in videos if video not in kept
    )
    kept_ids = {image_id for video in kept for image_id in video["image_ids"]}

    filtered = dict(data)
    filtered["images"] = [image for image in images if image.get("id") in kept_ids]
    filtered["annotations"] = [
        ann for ann in annotations if ann.get("image_id") in kept_ids
    ]
    info = dict(filtered.get("info") or {})
    existing_config = info.get("config")
    config = dict(existing_config) if isinstance(existing_config, dict) else {}
    config["max-subject-train-fraction"] = fraction
    config["require-videos"] = require_videos
    info["config"] = config
    filtered["info"] = info

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(filtered, handle, indent=4)
        handle.write("\n")

    print(f"Input images: {len(images)}")
    print(f"Output images: {len(filtered['images'])}")
    print(f"Output annotations: {len(filtered['annotations'])}")
    print(f"Wrote {output_path}")
    removed_path = write_removed_videos(output_path, removed_ids)
    print(f"Wrote {removed_path}")
    print_subject_shares(kept)

    val_source = os.path.join(os.path.dirname(os.path.abspath(train_json)), VAL_JSON_NAME)
    val_dest = os.path.join(os.path.dirname(os.path.abspath(output_path)), VAL_JSON_NAME)
    if os.path.isfile(val_source) and os.path.abspath(val_source) != os.path.abspath(val_dest):
        shutil.copy2(val_source, val_dest)
        print(f"Wrote {val_dest}")
    elif not os.path.isfile(val_source):
        print(f"⚠️ No validation JSON at {val_source}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cap the largest subject at 25% of a training COCO JSON by dropping that subject's videos"
    )
    parser.add_argument(
        "--train-json",
        required=True,
        help="Training COCO JSON to read. The file is not modified.",
    )
    parser.add_argument(
        "--subjects-path",
        required=True,
        help="CSV of video_id,subject (e.g. AnthonyGeorge_SeatedStraightLegRaise_072426_oN2pDT,AnthonyGeorge)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON path. Defaults to <train-json stem>_subject25.json",
    )
    parser.add_argument(
        "--require-videos",
        default=None,
        help=(
            "Text file of video ids that must stay in the dataset. "
            "A line may be video_id or video_id: date"
        ),
    )
    parser.add_argument(
        "--removed-txt",
        default=None,
        help=(
            "Text file of video ids to remove. When set, those videos are "
            "dropped and the subject cap is not run. A line may be video_id "
            "or video_id: date"
        ),
    )
    args = parser.parse_args()
    output_path = args.output or default_output_path(args.train_json)
    filter_train_json(
        args.train_json,
        output_path,
        MAX_SUBJECT_FRACTION,
        args.require_videos,
        args.subjects_path,
        args.removed_txt,
    )


if __name__ == "__main__":
    main()
