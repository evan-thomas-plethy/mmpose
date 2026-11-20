#!/usr/bin/env python3
"""
Drop individual videos from one subject in a training COCO JSON. Stop before
a removal that would put that subject at or under 25% of the remaining
images, or below the subject with the second-most frames. The subject is the
one with the most frames at the start. Other subjects are not trimmed.

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
output training JSON. Removed video ids and the info.config block are written
to videos_removed.json beside the output JSON. Pass that file as
--removed-json to drop those videos only, with no subject cap. --frame-limit
skips the share cap and, for every subject, drops videos until the next one
would put that subject under the limit. The saved config is copied onto the
output, except input, annotation, output, subjects-path,
max-subject-train-fraction, min-fraction-train-max-subject, frame-limit, and
require-videos, which follow this call.
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
SECOND_SUBJECT_RULE = "equal to second highest frame-count subject"
VAL_JSON_NAME = "person_keypoints_val2017.json"
REMOVED_JSON_NAME = "videos_removed.json"


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
    """Drop videos from the single largest subject.

    Stop before a removal that would put that subject at or under fraction, or
    below the subject with the second-most frames.
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
        if new_total > 0 and not over_fraction(new_subject, new_total, fraction):
            print(
                f"Stopping: dropping one more {subject} video would put them at "
                f"{new_subject / new_total:.1%}, at or under {fraction:.0%}"
            )
            print(f"Dropped {dropped} videos")
            return kept
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


def cap_subjects_to_frame_limit(videos: list[dict], frame_limit: int) -> list[dict]:
    """Drop videos from every subject above frame_limit.

    A video is removed only when that subject would still have at least
    frame_limit frames afterward. Required videos stay.
    """
    kept = list(videos)
    subjects = sorted({video["subject"] for video in kept})
    dropped = 0
    for subject in subjects:
        frames = sum(video["frames"] for video in kept if video["subject"] == subject)
        if frames <= frame_limit:
            continue
        print(f"Capping {subject} ({frames} frames)")
        subject_dropped = 0
        max_steps = len(videos) + 1
        for _ in range(max_steps):
            frames = sum(video["frames"] for video in kept if video["subject"] == subject)
            if frames <= frame_limit:
                break
            removable = [
                video
                for video in kept
                if video["subject"] == subject and not video["protected"]
            ]
            if not removable:
                print(
                    "Frame limit cannot be met without dropping required videos: "
                    f"{subject} {frames} frames"
                )
                break
            fits = [
                video
                for video in removable
                if frames - video["frames"] >= frame_limit
            ]
            if not fits:
                smallest = min(video["frames"] for video in removable)
                print(
                    f"Stopping: dropping one more {subject} video would put them at "
                    f"{frames - smallest} frames, under {frame_limit}"
                )
                break
            choice = random.choice(fits)
            kept.remove(choice)
            subject_dropped += 1
            dropped += 1
        print(f"Dropped {subject_dropped} {subject} videos")
    print(f"Dropped {dropped} videos")
    return kept


def dataset_dir_for_train_json(train_json: str) -> str:
    parent = os.path.dirname(os.path.abspath(train_json))
    if os.path.basename(parent) == "annotations":
        return os.path.dirname(parent)
    return parent


def annotation_name_for_dataset(dataset_dir: str) -> str:
    name = os.path.basename(dataset_dir)
    for suffix in ("_val10_subject25", "_subject25", "_val10"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return f"{name}.json"


def require_videos_config(paths: list[str] | None) -> str | list[str] | None:
    """One path stays a string. Two or more are stored as a list."""
    if not paths:
        return None
    if len(paths) == 1:
        return paths[0]
    return list(paths)


def config_for_this_call(
    base: dict | None,
    train_json: str,
    fraction: float,
    require_videos: list[str] | None,
    subjects_path: str,
    frame_limit: int | None,
) -> dict:
    """Reuse a saved config, then set the fields that belong to this dataset."""
    config = dict(base) if isinstance(base, dict) else {}
    dataset_dir = dataset_dir_for_train_json(train_json)
    annotation = annotation_name_for_dataset(dataset_dir)
    parent = os.path.dirname(dataset_dir)
    if os.path.isfile(os.path.join(parent, annotation)):
        config["input"] = parent
    elif "input" not in config:
        config["input"] = parent
    config["annotation"] = annotation
    config["output"] = dataset_dir
    config["subjects-path"] = subjects_path
    kept_require = config.pop("require-videos", None)
    config.pop("max-subject-train-fraction", None)
    config.pop("min-fraction-train-max-subject", None)
    config.pop("frame-limit", None)
    if frame_limit is None:
        config["max-subject-train-fraction"] = SECOND_SUBJECT_RULE
        config["min-fraction-train-max-subject"] = fraction
    else:
        config["frame-limit"] = frame_limit
    stored_require = require_videos_config(require_videos)
    if stored_require is not None:
        config["require-videos"] = stored_require
    elif kept_require is not None:
        config["require-videos"] = kept_require
    return config


def write_removed_videos(output_path: str, video_ids: list[str], config: dict) -> str:
    path = os.path.join(
        os.path.dirname(os.path.abspath(output_path)) or ".",
        REMOVED_JSON_NAME,
    )
    payload = {"config": config, "videos": video_ids}
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=4)
        handle.write("\n")
    return path


def load_removed_file(path: str) -> tuple[list[str], dict | None]:
    """Read video ids and the info.config object from videos_removed.json."""
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must be a JSON object with config and videos")
    raw_videos = payload.get("videos")
    if not isinstance(raw_videos, list) or not raw_videos:
        raise ValueError(f"No video ids found in {path}")
    video_ids = []
    for entry in raw_videos:
        video_id = str(entry).split(":", 1)[0].strip()
        if video_id:
            video_ids.append(video_id)
    if not video_ids:
        raise ValueError(f"No video ids found in {path}")
    config = payload.get("config")
    if config is not None and not isinstance(config, dict):
        raise ValueError(f"Config in {path} must be a JSON object")
    return video_ids, config


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
    require_videos: list[str] | None,
    subjects_path: str,
    removed_json: str | None,
    frame_limit: int | None,
) -> None:
    with open(train_json, encoding="utf-8") as handle:
        data = json.load(handle)

    images = data.get("images", [])
    annotations = data.get("annotations", [])
    required_ids = set()
    video_subjects = load_video_subjects(subjects_path)
    saved_config = None
    if removed_json:
        listed_removed, saved_config = load_removed_file(removed_json)
        videos = build_videos(images, required_ids, video_subjects)
        remove_ids = set(listed_removed)
        kept = drop_listed_videos(videos, remove_ids)
    else:
        if require_videos:
            listed_required = []
            for path in require_videos:
                listed_required.extend(load_id_list(path))
            required_ids = required_video_set(listed_required)
            present = {
                video_prefix(image.get("file_name", ""))
                for image in images
                if video_prefix(image.get("file_name", "")) in required_ids
            }
            print(
                f"Required videos present: {len(present)} "
                f"({len(required_ids)} ids from {len(require_videos)} files)"
            )
        videos = build_videos(images, required_ids, video_subjects)
        random.seed(42)
        if frame_limit is None:
            kept = cap_subjects(videos, fraction)
        else:
            kept = cap_subjects_to_frame_limit(videos, frame_limit)
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
    if not isinstance(existing_config, dict):
        existing_config = None
    base_config = saved_config if saved_config is not None else existing_config
    config = config_for_this_call(
        base_config,
        train_json,
        fraction,
        require_videos,
        subjects_path,
        frame_limit,
    )
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
    removed_path = write_removed_videos(output_path, removed_ids, config)
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
        description="Cap the largest subject in a training COCO JSON by dropping that subject's videos"
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
        "--min-fraction-train-max-subject",
        type=float,
        default=MAX_SUBJECT_FRACTION,
        help=(
            "Stop before a removal that would put the largest subject at or "
            "under this share of the remaining train images. Default: 0.25"
        ),
    )
    parser.add_argument(
        "--require-videos",
        nargs="+",
        default=None,
        help=(
            "One or more text files of video ids that must stay in the dataset. "
            "A line may be video_id or video_id: date. One path is stored as a "
            "string; two or more are stored as a list."
        ),
    )
    parser.add_argument(
        "--removed-json",
        default=None,
        help=(
            "videos_removed.json from a previous run. Those videos are "
            "dropped and the subject cap is not run. Its config object is "
            "copied onto the output, with this dataset's annotation, output, "
            "and related paths filled in."
        ),
    )
    parser.add_argument(
        "--frame-limit",
        type=int,
        default=None,
        help=(
            "For every subject, drop videos until the next one would put that "
            "subject under this many frames. Skips the share cap."
        ),
    )
    args = parser.parse_args()
    if args.frame_limit is not None and args.removed_json:
        raise SystemExit("--frame-limit cannot be combined with --removed-json")
    if args.frame_limit is not None and args.frame_limit < 1:
        raise SystemExit(
            f"--frame-limit must be a positive integer, got {args.frame_limit}"
        )
    if args.frame_limit is None and not 0 < args.min_fraction_train_max_subject < 1:
        raise SystemExit(
            "--min-fraction-train-max-subject must be between 0 and 1, "
            f"got {args.min_fraction_train_max_subject}"
        )
    output_path = args.output or default_output_path(args.train_json)
    filter_train_json(
        args.train_json,
        output_path,
        args.min_fraction_train_max_subject,
        args.require_videos,
        args.subjects_path,
        args.removed_json,
        args.frame_limit,
    )


if __name__ == "__main__":
    main()
