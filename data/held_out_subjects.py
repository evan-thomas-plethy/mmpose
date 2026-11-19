#!/usr/bin/env python
"""Pull held-out subjects out of a COCO JSON, then leave the rest for split_dataset.

Every frame of each --held-out-subjects name is written to a val JSON. Those
videos are not in the train-pool JSON, so a later split_dataset.py run draws
its inner val only from the remaining subjects.

The held-out JSON is annotations/person_keypoints_val2017.json under
held_out/. The finetune config reads it with CustomDatasetHook.

Example:
  python data/held_out_subjects.py \\
    --json data/App_Vids_v3_GBE_v5.json \\
    --subjects-path /path/to/users.csv \\
    --held-out-subjects NinaVani JamieDoe JoeyReyna YordanosWoldebirhan \\
    --output data/subject_fold

  python data/split_dataset.py \\
    --input data/App_Vids_v3_GBE_v5_subject_fold/train_pool \\
    --annotation App_Vids_v3_GBE_v5_train_pool.json \\
    --output data/App_Vids_v3_GBE_v5_subject_fold/inner \\
    --no-frames \\
    --subjects-path /path/to/users.csv \\
    --val-ratio 0.1 \\
    --max-subject-val-fraction 0.25 \\
    --seed 42
"""

import argparse
import json
import os
import sys
from collections import defaultdict

from split_dataset import load_video_subjects, subject_for_video


def video_of_image(image):
    video_id = image.get("video_id")
    if video_id:
        return str(video_id)
    filename = os.path.basename(str(image.get("file_name", "")).replace("\\", "/"))
    if "_frame_" in filename:
        return filename.split("_frame_")[0]
    return filename


def write_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=4)
        handle.write("\n")


def subset(data, images, info):
    image_ids = {image["id"] for image in images}
    return {
        "info": info,
        "licenses": data.get("licenses", []),
        "images": images,
        "annotations": [
            ann for ann in data.get("annotations", [])
            if ann.get("image_id") in image_ids
        ],
        "categories": data.get("categories", []),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Write a held-out subject val JSON and a train pool for split_dataset.py"
    )
    parser.add_argument("--json", required=True, help="Base COCO JSON. It is not modified.")
    parser.add_argument("--subjects-path", required=True, help="CSV of video_id,subject")
    parser.add_argument(
        "--held-out-subjects",
        nargs="+",
        required=True,
        help="Subjects whose frames all go to the held-out val JSON",
    )
    parser.add_argument("--output", required=True, help="New output directory")
    return parser.parse_args()


def main():
    args = parse_args()
    if not os.path.isfile(args.json):
        raise SystemExit(f"JSON not found: {args.json}")
    if os.path.exists(args.output):
        raise SystemExit(f"Output already exists: {args.output}")

    held_out = []
    for value in args.held_out_subjects:
        held_out.extend(part.strip() for part in value.split(",") if part.strip())
    if not held_out:
        raise SystemExit("No held-out subjects")
    held_out_set = set(held_out)
    if len(held_out_set) != len(held_out):
        raise SystemExit(f"Duplicate held-out subject in {held_out}")

    video_subjects = load_video_subjects(args.subjects_path)
    with open(args.json, encoding="utf-8") as handle:
        data = json.load(handle)

    groups = defaultdict(list)
    subject_frames = defaultdict(int)
    missing_csv = []
    for image in data.get("images", []):
        video_id = video_of_image(image)
        if video_id not in video_subjects:
            missing_csv.append(video_id)
        subject = subject_for_video(video_id, video_subjects)
        groups[subject].append(image)
        subject_frames[subject] += 1

    absent = sorted(held_out_set.difference(groups))
    if absent:
        raise SystemExit(
            "Held-out subjects with no frames in this JSON: " + ", ".join(absent)
        )
    if missing_csv:
        preview = ", ".join(sorted(set(missing_csv))[:8])
        extra = len(set(missing_csv)) - 8
        suffix = f" (+{extra} more)" if extra > 0 else ""
        print(
            f"Warning: {len(set(missing_csv))} videos are not in the subjects CSV "
            f"({preview}{suffix}). Each uses the parsed name, or the video id."
        )

    held_images = [image for subject in held_out for image in groups[subject]]
    train_images = [
        image
        for subject, images in groups.items()
        if subject not in held_out_set
        for image in images
    ]
    info = dict(data.get("info") or {})
    config = dict(info.get("config") or {})
    config["held-out-subjects"] = held_out
    config["subjects-path"] = os.path.abspath(args.subjects_path)
    config["source-json"] = os.path.abspath(args.json)
    info["config"] = config

    stem = os.path.splitext(os.path.basename(args.json))[0]
    train_name = f"{stem}_train_pool.json"
    train_dir = os.path.join(args.output, "train_pool")
    held_dir = os.path.join(args.output, "held_out")
    ann_relpath = "annotations/person_keypoints_val2017.json"
    train_path = os.path.join(train_dir, train_name)
    held_path = os.path.join(held_dir, ann_relpath)
    write_json(train_path, subset(data, train_images, info))
    write_json(held_path, subset(data, held_images, info))

    print(f"Held-out images: {len(held_images)}")
    for subject in held_out:
        print(f"  {subject}: {subject_frames[subject]}")
    print(f"Train-pool images: {len(train_images)}")
    print(f"Wrote {held_path}")
    print(f"Wrote {train_path}")
    print()
    print("Inner split (held-out subjects are already absent, so val is drawn from the rest):")
    print(
        "python data/split_dataset.py \\\n"
        f"  --input {train_dir} \\\n"
        f"  --annotation {train_name} \\\n"
        f"  --output {os.path.join(args.output, 'inner')} \\\n"
        "  --no-frames \\\n"
        f"  --subjects-path {args.subjects_path} \\\n"
        "  --val-ratio 0.1 \\\n"
        "  --max-subject-val-fraction 0.25 \\\n"
        "  --seed 42"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
