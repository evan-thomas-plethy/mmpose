import json
import os
import random
import shutil
from collections import defaultdict

"""
python split_dataset.py --input Heel_Slides_v1/ --output coco/ --val-ratio 0.19 --seed 46
"""

def split_dataset(
    input_dir,
    output_dir,
    annotation_name=None,
    val_ratio=0.15,
    seed=42
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
        val_ratio: Fraction of frames to use for validation (default 0.15)
        seed: Random seed for reproducible splits (default 42)
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

    # Shuffle videos and split by cumulative frame count
    videos = list(video_to_images.keys())
    random.shuffle(videos)
    
    total_frames = sum(len(imgs) for imgs in video_to_images.values())
    target_val_frames = int(val_ratio * total_frames)
    
    val_videos = set()
    val_frame_count = 0
    
    for video in videos:
        if val_frame_count < target_val_frames:
            val_videos.add(video)
            val_frame_count += len(video_to_images[video])
        else:
            break

    # Split images into train and val
    train_images = []
    val_images = []
    
    for video, images in video_to_images.items():
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
    print(f"\n✅ Dataset split complete!")
    print(f"   Output: {output_dir}")
    print(f"   Videos: {len(videos)} total → {len(videos) - len(val_videos)} train, {len(val_videos)} val")
    print(f"   Images: {total_frames} total → {len(train_images)} train, {len(val_images)} val")
    print(f"   Annotations: {len(train_annotations)} train, {len(val_annotations)} val")
    print(f"   Actual val ratio: {len(val_images) / total_frames:.2%}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Split dataset into train/val by video (COCO format)")
    parser.add_argument("--input", required=True, help="Input directory (contains frames/ and .json)")
    parser.add_argument("--output", required=True, help="Output directory for COCO-structured dataset")
    parser.add_argument("--annotation", default=None, help="Annotation JSON filename (auto-detected if omitted)")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation ratio (default: 0.15)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    
    args = parser.parse_args()
    
    split_dataset(
        input_dir=args.input,
        output_dir=args.output,
        annotation_name=args.annotation,
        val_ratio=args.val_ratio,
        seed=args.seed
    )
