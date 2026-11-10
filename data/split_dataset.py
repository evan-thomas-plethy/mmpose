import csv
import json
import os
import random
import shutil
from collections import defaultdict

"""
python split_dataset.py --input Heel_Slides_v1/ --output coco/ --val-ratio 0.19 --seed 46
python split_dataset.py --input Heel_Slides_v1/ --output coco/ \
    --subjects-path subjects.csv --subjects-val AnthonyGeorge \
    --subjects-exclude JaneDoe
"""


def load_video_subjects(subjects_path):
    """Load a video_id,subject CSV. A header row is skipped when present."""
    video_subjects = {}
    with open(subjects_path, newline="") as f:
        for row_number, row in enumerate(csv.reader(f), start=1):
            if not row or not row[0].strip():
                continue
            if len(row) < 2 or not row[1].strip():
                raise ValueError(
                    f"{subjects_path}:{row_number} must be video_id,subject"
                )
            video_id = row[0].strip()
            subject = row[1].strip()
            if row_number == 1 and video_id.lower() in {"video_id", "video"}:
                continue
            if video_id in video_subjects:
                raise ValueError(
                    f"Duplicate video_id in {subjects_path}: {video_id}"
                )
            video_subjects[video_id] = subject
    if not video_subjects:
        raise ValueError(f"No video_id,subject rows found in {subjects_path}")
    return video_subjects


def select_val_videos(
    video_to_images,
    val_ratio,
    subjects_path,
    subjects_val,
    subjects_exclude=None,
):
    """Return val video prefixes and video prefixes left out of both splits.

    Subject mode puts every remaining video whose subject is listed in
    subjects_val into val and every other remaining video into train. The frame
    ratio is not used. Excluded subjects are omitted from both splits.
    """
    if subjects_val and not subjects_path:
        raise ValueError("--subjects-val requires --subjects-path")
    if subjects_exclude and not subjects_path:
        raise ValueError("--subjects-exclude requires --subjects-path")
    if subjects_path and not subjects_val and not subjects_exclude:
        raise ValueError(
            "--subjects-path requires --subjects-val or --subjects-exclude"
        )

    videos = list(video_to_images.keys())
    video_subjects = load_video_subjects(subjects_path) if subjects_path else None
    if video_subjects is not None:
        missing = [video for video in videos if video not in video_subjects]
        if missing:
            preview = ", ".join(missing[:10])
            extra = f" (+{len(missing) - 10} more)" if len(missing) > 10 else ""
            raise ValueError(
                f"{len(missing)} videos have no subject in {subjects_path}: "
                f"{preview}{extra}"
            )

    excluded_videos = set()
    if subjects_exclude:
        exclude_subjects = set(subjects_exclude)
        overlap = sorted(exclude_subjects & set(subjects_val or []))
        if overlap:
            print(
                "⚠️ Subjects in both --subjects-val and --subjects-exclude "
                f"are excluded: {', '.join(overlap)}"
            )
        present_subjects = {video_subjects[video] for video in videos}
        unused = sorted(exclude_subjects - present_subjects)
        if unused:
            print(
                "⚠️ Excluded subjects with no videos in this dataset: "
                f"{', '.join(unused)}"
            )
        excluded_videos = {
            video
            for video in videos
            if video_subjects[video] in exclude_subjects
        }
        if len(excluded_videos) == len(videos):
            raise ValueError("Subject exclude removed every video")
        if excluded_videos:
            excluded_names = sorted(
                {video_subjects[video] for video in excluded_videos}
            )
            excluded_frames = sum(
                len(video_to_images[video]) for video in excluded_videos
            )
            print(
                f"🚫 Excluding {', '.join(excluded_names)}: "
                f"{len(excluded_videos)} videos, {excluded_frames} frames"
            )

    pool = [video for video in videos if video not in excluded_videos]
    if subjects_val is not None:
        val_subjects = set(subjects_val)
        val_videos = {
            video
            for video in pool
            if video_subjects[video] in val_subjects
        }
        present_subjects = {video_subjects[video] for video in pool}
        unused = sorted(val_subjects - present_subjects)
        if unused:
            print(f"⚠️ Val subjects with no videos left in this dataset: {', '.join(unused)}")
        if not val_videos:
            raise ValueError("Subject split produced an empty val set")
        if len(val_videos) == len(pool):
            raise ValueError("Subject split produced an empty train set")

        assigned = sorted(present_subjects & val_subjects)
        print(f"👤 Val subjects: {', '.join(assigned)}")
        print("   --val-ratio ignored")
        return val_videos, excluded_videos

    random.shuffle(pool)
    pool_frames = sum(len(video_to_images[video]) for video in pool)
    target_val_frames = int(val_ratio * pool_frames)
    val_videos = set()
    val_frame_count = 0
    for video in pool:
        if val_frame_count >= target_val_frames:
            break
        val_videos.add(video)
        val_frame_count += len(video_to_images[video])
    return val_videos, excluded_videos


def split_dataset(
    input_dir,
    output_dir,
    annotation_name=None,
    val_ratio=0.15,
    seed=42,
    subjects_path=None,
    subjects_val=None,
    subjects_exclude=None,
):
    """
    Splits a combined dataset into train and val sets by video.
    
    All frames from the same video (based on filename prefix before '_frame_')
    are kept together in either train or val to prevent data leakage.
    
    Input structure:
        input_dir/
        ├── frames/
        └── <annotation_name>.json
    
    Output structure (COCO format):
        output_dir/
        ├── annotations/
        │   ├── person_keypoints_train2017.json
        │   └── person_keypoints_val2017.json
        ├── train2017/
        └── val2017/
    
    Args:
        input_dir: Input directory containing frames/ and annotation JSON
        output_dir: Output directory for COCO-structured dataset
        annotation_name: Name of annotation JSON file (auto-detected if None)
        val_ratio: Fraction of frames to use for validation (default 0.15).
            Ignored when subjects_path and subjects_val are set.
        seed: Random seed for reproducible ratio splits (default 42)
        subjects_path: CSV of video_id,subject. Requires subjects_val or
            subjects_exclude.
        subjects_val: Subject names assigned entirely to val. Every other
            subject is assigned entirely to train.
        subjects_exclude: Subject names omitted from both train and val.
            Requires subjects_path.
    """
    random.seed(seed)
    
    # Define input paths
    input_frames_dir = os.path.join(input_dir, "frames")
    
    if annotation_name:
        json_path = os.path.join(input_dir, annotation_name)
    else:
        # Auto-detect JSON file in input_dir
        json_files = [f for f in os.listdir(input_dir) if f.endswith('.json')]
        if len(json_files) == 0:
            raise FileNotFoundError(f"No JSON file found in {input_dir}")
        elif len(json_files) > 1:
            raise ValueError(f"Multiple JSON files found in {input_dir}: {json_files}. Specify --annotation")
        json_path = os.path.join(input_dir, json_files[0])
    
    print(f"📂 Input: {input_dir}")
    print(f"   Frames: {input_frames_dir}")
    print(f"   Annotations: {json_path}")

    # Load combined dataset
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Group images by video prefix
    video_to_images = defaultdict(list)
    for img in data["images"]:
        filename = os.path.basename(img["file_name"])
        # Extract video prefix (everything before '_frame_')
        if "_frame_" in filename:
            video_prefix = filename.split("_frame_")[0]
        else:
            video_prefix = filename  # fallback: treat each image as its own group
        video_to_images[video_prefix].append(img)

    videos = list(video_to_images.keys())
    total_frames = sum(len(imgs) for imgs in video_to_images.values())
    val_videos, excluded_videos = select_val_videos(
        video_to_images,
        val_ratio,
        subjects_path,
        subjects_val,
        subjects_exclude,
    )

    # Define COCO output structure
    annotations_dir = os.path.join(output_dir, "annotations")
    train_dir = os.path.join(output_dir, "train2017")
    val_dir = os.path.join(output_dir, "val2017")
    train_json_out = os.path.join(annotations_dir, "person_keypoints_train2017.json")
    val_json_out = os.path.join(annotations_dir, "person_keypoints_val2017.json")

    # Create output directories
    os.makedirs(annotations_dir, exist_ok=True)

    if os.path.exists(train_dir):
        shutil.rmtree(train_dir)
    os.makedirs(train_dir, exist_ok=True)

    if os.path.exists(val_dir):
        shutil.rmtree(val_dir)
    os.makedirs(val_dir, exist_ok=True)

    # Split images into train and val
    train_images = []
    val_images = []
    
    for video, images in video_to_images.items():
        if video in excluded_videos:
            continue
        if video in val_videos:
            val_images.extend(images)
        else:
            train_images.extend(images)

    # Get corresponding annotations
    train_ids = {img["id"] for img in train_images}
    val_ids = {img["id"] for img in val_images}
    
    train_annotations = [ann for ann in data["annotations"] if ann["image_id"] in train_ids]
    val_annotations = [ann for ann in data["annotations"] if ann["image_id"] in val_ids]

    # Copy images to respective directories
    print(f"📁 Copying {len(train_images)} training images...")
    for img in train_images:
        src = os.path.join(input_frames_dir, os.path.basename(img["file_name"]))
        dst = os.path.join(train_dir, os.path.basename(img["file_name"]))
        if os.path.exists(src):
            shutil.copy(src, dst)
        else:
            print(f"⚠️ Missing: {src}")

    print(f"📁 Copying {len(val_images)} validation images...")
    for img in val_images:
        src = os.path.join(input_frames_dir, os.path.basename(img["file_name"]))
        dst = os.path.join(val_dir, os.path.basename(img["file_name"]))
        if os.path.exists(src):
            shutil.copy(src, dst)
        else:
            print(f"⚠️ Missing: {src}")

    # Prepare output JSON structure
    info = data.get("info", {})
    categories = data.get("categories", [])

    # Normalize file_name to just basename (remove any path prefixes like 'frames/')
    def normalize_images(images):
        normalized = []
        for img in images:
            normalized_img = dict(img)
            normalized_img["file_name"] = os.path.basename(img["file_name"])
            normalized.append(normalized_img)
        return normalized

    # Save train JSON
    train_data = {
        "info": info,
        "images": normalize_images(train_images),
        "annotations": train_annotations,
        "categories": categories
    }
    with open(train_json_out, "w") as f:
        json.dump(train_data, f, indent=4)

    # Save val JSON
    val_data = {
        "info": info,
        "images": normalize_images(val_images),
        "annotations": val_annotations,
        "categories": categories
    }
    with open(val_json_out, "w") as f:
        json.dump(val_data, f, indent=4)

    # Print summary
    kept_frames = len(train_images) + len(val_images)
    train_videos = len(videos) - len(val_videos) - len(excluded_videos)
    print(f"\n✅ Dataset split complete!")
    print(f"   Output: {output_dir}")
    print(
        f"   Videos: {len(videos)} total → {train_videos} train, "
        f"{len(val_videos)} val, {len(excluded_videos)} excluded"
    )
    print(
        f"   Images: {total_frames} total → {len(train_images)} train, "
        f"{len(val_images)} val, {total_frames - kept_frames} excluded"
    )
    print(f"   Annotations: {len(train_annotations)} train, {len(val_annotations)} val")
    print(f"   Actual val ratio: {len(val_images) / kept_frames:.2%}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Split dataset into train/val by video (COCO format)")
    parser.add_argument("--input", required=True, help="Input directory (contains frames/ and .json)")
    parser.add_argument("--output", required=True, help="Output directory for COCO-structured dataset")
    parser.add_argument("--annotation", default=None, help="Annotation JSON filename (auto-detected if omitted)")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation ratio (default: 0.15). Ignored with --subjects-path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument(
        "--subjects-path",
        default=None,
        help="CSV of video_id,subject (e.g. AnthonyGeorge_SeatedStraightLegRaise_072426_oN2pDT,AnthonyGeorge)",
    )
    parser.add_argument(
        "--subjects-val",
        nargs="+",
        default=None,
        help="Subjects to put entirely in val. Remaining subjects go to train. Requires --subjects-path",
    )
    parser.add_argument(
        "--subjects-exclude",
        nargs="+",
        default=None,
        help="Subjects to omit from both train and val. Requires --subjects-path",
    )

    args = parser.parse_args()

    def parse_subjects(values):
        if not values:
            return None
        subjects = [
            subject.strip()
            for value in values
            for subject in value.split(",")
            if subject.strip()
        ]
        return subjects or None

    split_dataset(
        input_dir=args.input,
        output_dir=args.output,
        annotation_name=args.annotation,
        val_ratio=args.val_ratio,
        seed=args.seed,
        subjects_path=args.subjects_path,
        subjects_val=parse_subjects(args.subjects_val),
        subjects_exclude=parse_subjects(args.subjects_exclude),
    )
