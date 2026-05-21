#!/usr/bin/env python3
"""
Split a target COCO JSON into train vs val using reference val video names.

Video name rule (same as historical filter_val_images): basename of file_name,
text before ``_frame_`` if present, otherwise the whole basename.

Usage:
    python filter_val_data.py \
        --ref-val coco_ground_based_exercises_v2/annotations/person_keypoints_val2017.json \
        --target-anns Ground_Based_Exercises/v5/Ground_Based_Exercises_v5.json \
        --frames-dir Ground_Based_Exercises/v1/frames \
        --output coco_ground_based_exercises_v5/

Output (COCO layout):
    output_dir/
    ├── annotations/
    │   ├── person_keypoints_train2017.json
    │   └── person_keypoints_val2017.json
    ├── train2017/
    └── val2017/

Only ``--output`` is removed/recreated; inputs are read and copied from, not edited.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from typing import Any


def video_prefix_from_file_name(file_name: str) -> str:
    """Basename of file_name; prefix before ``_frame_`` when that substring exists."""
    filename = os.path.basename(file_name.replace("\\", "/"))
    if "_frame_" in filename:
        return filename.split("_frame_")[0]
    return filename


def get_video_prefixes(annotations_json_path: str) -> set[str]:
    """Extract unique video prefixes from a COCO annotation file (path)."""
    with open(annotations_json_path, encoding="utf-8") as f:
        data = json.load(f)

    prefixes: set[str] = set()
    for img in data.get("images", []):
        fn = img.get("file_name")
        if fn:
            prefixes.add(video_prefix_from_file_name(fn))
    return prefixes


def _build_split_payload(
    full_data: dict[str, Any],
    images_subset: list[dict[str, Any]],
) -> dict[str, Any]:
    ids = {img["id"] for img in images_subset}
    annotations = [
        ann for ann in full_data.get("annotations", []) if ann["image_id"] in ids
    ]
    normalized_images: list[dict[str, Any]] = []
    for img in images_subset:
        row = dict(img)
        row["file_name"] = os.path.basename(img["file_name"].replace("\\", "/"))
        normalized_images.append(row)
    return {
        "info": full_data.get("info", {}),
        "images": normalized_images,
        "annotations": annotations,
        "categories": full_data.get("categories", []),
    }


def _print_missing_ref_videos_in_target(
    ref_val_prefixes: set[str],
    target_prefixes: set[str],
) -> None:
    """Report ref-val video prefixes with no images in target anns."""
    missing = sorted(ref_val_prefixes - target_prefixes)
    if not missing:
        return
    print(f"\n⚠️ Ref val videos missing from target ({len(missing)}):")
    for prefix in missing:
        print(f"   {prefix}")


def _copy_images_for_split(
    images: list[dict[str, Any]],
    frames_dir: str,
    dest_img_dir: str,
    label: str,
) -> tuple[int, int]:
    os.makedirs(dest_img_dir, exist_ok=True)
    ok, missing = 0, 0
    seen_dst: set[str] = set()
    for img in images:
        base = os.path.basename(img["file_name"].replace("\\", "/"))
        dst_path = os.path.join(dest_img_dir, base)
        if dst_path in seen_dst:
            continue
        seen_dst.add(dst_path)
        src_path = os.path.join(frames_dir, base)
        if os.path.isfile(src_path):
            shutil.copy2(src_path, dst_path)
            ok += 1
        else:
            missing += 1
    print(f"   ✓ {label}: copied {ok} images" + (f", missing {missing}" if missing else ""))
    return ok, missing


def split_dataset_by_reference_val(
    ref_val_json: str,
    target_anns_json: str,
    frames_dir: str,
    output_dir: str,
) -> None:
    ref_val_abs = os.path.abspath(ref_val_json)
    target_abs = os.path.abspath(target_anns_json)
    frames_abs = os.path.abspath(frames_dir)
    out_abs = os.path.abspath(output_dir)

    if not os.path.isfile(ref_val_abs):
        raise SystemExit(f"Reference val JSON not found: {ref_val_abs}")
    if not os.path.isfile(target_abs):
        raise SystemExit(f"Target annotations JSON not found: {target_abs}")
    if not os.path.isdir(frames_abs):
        raise SystemExit(f"Frames directory not found: {frames_abs}")

    print(f"📂 Reference val: {ref_val_abs}")
    print(f"📂 Target anns:   {target_abs}")
    print(f"📂 Frames:       {frames_abs}")
    print(f"📂 Output:       {out_abs}")

    val_prefixes = get_video_prefixes(ref_val_abs)

    with open(target_abs, encoding="utf-8") as f:
        data = json.load(f)

    train_images: list[dict[str, Any]] = []
    val_images: list[dict[str, Any]] = []
    skipped_no_file_name = 0
    target_video_prefixes: set[str] = set()

    for img in data.get("images", []):
        fn = img.get("file_name")
        if not fn:
            skipped_no_file_name += 1
            continue
        prefix = video_prefix_from_file_name(fn)
        target_video_prefixes.add(prefix)
        if prefix in val_prefixes:
            val_images.append(img)
        else:
            train_images.append(img)

    ref_video_count = len(val_prefixes)
    ref_videos_in_target = len(val_prefixes & target_video_prefixes)
    print(
        f"\n📌 Reference val distinct videos: {ref_video_count} | "
        f"found in target anns: {ref_videos_in_target}"
    )
    _print_missing_ref_videos_in_target(val_prefixes, target_video_prefixes)

    print("\n📊 Split (target):")
    if skipped_no_file_name:
        print(f"   ⚠️ Skipped {skipped_no_file_name} images with missing file_name")
    print(
        f"   Images: {len(data.get('images', []))} total → "
        f"{len(train_images)} train, {len(val_images)} val"
    )
    train_ids = {img["id"] for img in train_images}
    val_ids = {img["id"] for img in val_images}
    n_ann = len(data.get("annotations", []))
    n_ann_train = sum(
        1 for a in data.get("annotations", []) if a["image_id"] in train_ids
    )
    n_ann_val = sum(
        1 for a in data.get("annotations", []) if a["image_id"] in val_ids
    )
    print(f"   Annotations: {n_ann} total → {n_ann_train} train, {n_ann_val} val")

    train_dir = os.path.join(out_abs, "train2017")
    val_dir = os.path.join(out_abs, "val2017")
    annotations_dir = os.path.join(out_abs, "annotations")
    train_json = os.path.join(annotations_dir, "person_keypoints_train2017.json")
    val_json = os.path.join(annotations_dir, "person_keypoints_val2017.json")

    if os.path.exists(out_abs):
        shutil.rmtree(out_abs)
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(annotations_dir, exist_ok=True)

    print("\n📁 Copying images...")
    _copy_images_for_split(train_images, frames_abs, train_dir, "train2017")
    _copy_images_for_split(val_images, frames_abs, val_dir, "val2017")

    train_payload = _build_split_payload(data, train_images)
    val_payload = _build_split_payload(data, val_images)

    with open(train_json, "w", encoding="utf-8") as f:
        json.dump(train_payload, f, indent=4)
        f.write("\n")
    with open(val_json, "w", encoding="utf-8") as f:
        json.dump(val_payload, f, indent=4)
        f.write("\n")

    print(f"\n✅ Dataset saved to: {out_abs}")
    print(f"   Train: {train_json}")
    print(f"   Val:   {val_json}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Split target COCO annotations into train/val: any video present in "
            "--ref-val goes to val; all other videos from --target-anns go to train. "
            "Frames for both splits are copied from --frames-dir."
        )
    )
    parser.add_argument(
        "--ref-val",
        required=True,
        help="Reference validation COCO JSON (defines which video prefixes are val)",
    )
    parser.add_argument(
        "--target-anns",
        required=True,
        help="Target COCO JSON to split (single file listing all images/annotations)",
    )
    parser.add_argument(
        "--frames-dir",
        required=True,
        help="Directory containing frame files; must include every image basename "
        "referenced in either JSON",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output COCO dataset root (train2017/, val2017/, annotations/)",
    )
    args = parser.parse_args()

    split_dataset_by_reference_val(
        ref_val_json=args.ref_val,
        target_anns_json=args.target_anns,
        frames_dir=args.frames_dir,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()
