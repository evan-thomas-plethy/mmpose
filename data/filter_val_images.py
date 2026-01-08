import json
import os
import shutil
from collections import defaultdict

"""
Filters out validation videos from a dataset directory.

Usage:
python filter_val_images.py \
    --ref-val coco_dedup_param_1.0/annotations/person_keypoints_val2017.json \
    --target Heel_Slides/v4 \
    --output coco_dedup_param_0.995

Output structure (COCO format):
    output_dir/
    ├── annotations/
    │   ├── person_keypoints_train2017.json
    │   ├── person_keypoints_val2017.json (copied from ref-val dir)
    │   └── person_keypoints_val2017_mirrored.json (copied if exists)
    ├── train2017/
    ├── val2017/ (copied from ref-val dir)
    └── val2017_mirrored/ (copied if exists)

This will:
1. Read video prefixes from the reference val annotations
2. Copy the target directory to output (excluding val videos)
3. Copy val data from the reference directory
"""


def get_video_prefixes(annotations_json):
    """Extract unique video prefixes from annotation file."""
    with open(annotations_json, 'r') as f:
        data = json.load(f)
    
    prefixes = set()
    for img in data["images"]:
        filename = os.path.basename(img["file_name"])
        if "_frame_" in filename:
            prefix = filename.split("_frame_")[0]
        else:
            prefix = filename
        prefixes.add(prefix)
    
    return prefixes


def filter_val_images(
    ref_val_json,
    target_dir,
    output_dir,
    annotation_name=None
):
    """
    Creates a filtered copy of target_dir with validation videos removed.
    
    Output is in COCO format:
        output_dir/train2017/
        output_dir/annotations/person_keypoints_train2017.json
    
    Args:
        ref_val_json: Path to reference validation annotations JSON
        target_dir: Target directory with structure: dir/frames/ and dir/*.json
        output_dir: Output directory for COCO-structured filtered dataset
        annotation_name: Name of annotation JSON file (auto-detected if None)
    """
    # Define input paths
    target_frames_dir = os.path.join(target_dir, "frames")
    
    if annotation_name:
        target_json = os.path.join(target_dir, annotation_name)
    else:
        # Auto-detect JSON file in target_dir
        json_files = [f for f in os.listdir(target_dir) if f.endswith('.json')]
        if len(json_files) == 0:
            raise FileNotFoundError(f"No JSON file found in {target_dir}")
        elif len(json_files) > 1:
            raise ValueError(f"Multiple JSON files found in {target_dir}: {json_files}. Specify --annotation")
        annotation_name = json_files[0]
        target_json = os.path.join(target_dir, annotation_name)
    
    print(f"📂 Reference val: {ref_val_json}")
    print(f"📂 Target: {target_dir}")
    print(f"   Frames: {target_frames_dir}")
    print(f"   Annotations: {target_json}")
    
    # Get video prefixes to exclude
    val_prefixes = get_video_prefixes(ref_val_json)
    print(f"\n🚫 Videos to exclude ({len(val_prefixes)}):")
    for prefix in sorted(val_prefixes):
        print(f"   - {prefix}")
    
    # Load target annotations
    with open(target_json, 'r') as f:
        data = json.load(f)
    
    # Separate images into keep vs exclude
    keep_images = []
    exclude_images = []
    
    for img in data["images"]:
        filename = os.path.basename(img["file_name"])
        if "_frame_" in filename:
            prefix = filename.split("_frame_")[0]
        else:
            prefix = filename
        
        if prefix in val_prefixes:
            exclude_images.append(img)
        else:
            keep_images.append(img)
    
    # Get annotations for kept images
    keep_ids = {img["id"] for img in keep_images}
    keep_annotations = [ann for ann in data["annotations"] if ann["image_id"] in keep_ids]
    
    print(f"\n📊 Filtering results:")
    print(f"   Images: {len(data['images'])} total → {len(keep_images)} kept, {len(exclude_images)} excluded")
    print(f"   Annotations: {len(data['annotations'])} total → {len(keep_annotations)} kept")
    
    # Create COCO output directory structure
    train_dir = os.path.join(output_dir, "train2017")
    annotations_dir = os.path.join(output_dir, "annotations")
    output_json = os.path.join(annotations_dir, "person_keypoints_train2017.json")
    
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(annotations_dir, exist_ok=True)
    
    # Copy only kept images
    print(f"\n📁 Copying {len(keep_images)} images to {train_dir}...")
    for img in keep_images:
        src = os.path.join(target_frames_dir, os.path.basename(img["file_name"]))
        dst = os.path.join(train_dir, os.path.basename(img["file_name"]))
        if os.path.exists(src):
            shutil.copy(src, dst)
        else:
            print(f"⚠️ Missing: {src}")
    
    # Normalize file_name to just basename (remove any path prefixes like 'frames/')
    normalized_images = []
    for img in keep_images:
        normalized_img = dict(img)
        normalized_img["file_name"] = os.path.basename(img["file_name"])
        normalized_images.append(normalized_img)
    
    # Save filtered annotations
    filtered_data = {
        "info": data.get("info", {}),
        "images": normalized_images,
        "annotations": keep_annotations,
        "categories": data.get("categories", [])
    }
    
    with open(output_json, "w") as f:
        json.dump(filtered_data, f, indent=4)
    
    # Copy validation data from reference directory
    # Extract ref directory (e.g., coco_dedup_param_1.0 from coco_dedup_param_1.0/annotations/person_keypoints_val2017.json)
    ref_dir = os.path.dirname(os.path.dirname(ref_val_json))
    
    print(f"\n📁 Copying validation data from {ref_dir}...")
    
    # Copy val2017 directory
    ref_val_dir = os.path.join(ref_dir, "val2017")
    output_val_dir = os.path.join(output_dir, "val2017")
    if os.path.exists(ref_val_dir):
        shutil.copytree(ref_val_dir, output_val_dir)
        print(f"   ✓ Copied val2017/")
    else:
        print(f"   ⚠️ val2017/ not found in {ref_dir}")
    
    # Copy val2017_mirrored directory (if exists)
    ref_val_mirrored_dir = os.path.join(ref_dir, "val2017_mirrored")
    output_val_mirrored_dir = os.path.join(output_dir, "val2017_mirrored")
    if os.path.exists(ref_val_mirrored_dir):
        shutil.copytree(ref_val_mirrored_dir, output_val_mirrored_dir)
        print(f"   ✓ Copied val2017_mirrored/")
    
    # Copy person_keypoints_val2017.json
    ref_val_json_file = os.path.join(ref_dir, "annotations", "person_keypoints_val2017.json")
    output_val_json = os.path.join(annotations_dir, "person_keypoints_val2017.json")
    if os.path.exists(ref_val_json_file):
        shutil.copy(ref_val_json_file, output_val_json)
        print(f"   ✓ Copied person_keypoints_val2017.json")
    
    # Copy person_keypoints_val2017_mirrored.json (if exists)
    ref_val_mirrored_json = os.path.join(ref_dir, "annotations", "person_keypoints_val2017_mirrored.json")
    output_val_mirrored_json = os.path.join(annotations_dir, "person_keypoints_val2017_mirrored.json")
    if os.path.exists(ref_val_mirrored_json):
        shutil.copy(ref_val_mirrored_json, output_val_mirrored_json)
        print(f"   ✓ Copied person_keypoints_val2017_mirrored.json")
    
    print(f"\n✅ Filtered dataset saved to: {output_dir}")
    print(f"   Train images: {train_dir}")
    print(f"   Train annotations: {output_json}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Filter out validation videos from a dataset")
    parser.add_argument("--ref-val", required=True, help="Reference validation annotations JSON")
    parser.add_argument("--target", required=True, help="Target directory (contains frames/ and .json)")
    parser.add_argument("--output", required=True, help="Output directory for COCO-structured filtered dataset")
    parser.add_argument("--annotation", default=None, help="Annotation JSON filename (auto-detected if omitted)")
    
    args = parser.parse_args()
    
    filter_val_images(
        ref_val_json=args.ref_val,
        target_dir=args.target,
        output_dir=args.output,
        annotation_name=args.annotation
    )
