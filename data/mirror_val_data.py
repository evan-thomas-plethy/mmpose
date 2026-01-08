import json
import os
from PIL import Image

"""
Creates mirrored (horizontally flipped) versions of validation images and annotations.

Usage:
python mirror_val_data.py \
    --annotations coco/annotations/person_keypoints_val2017.json \
    --images coco/val2017 \
    --output-dir coco/val2017_mirrored \
    --output-json coco/annotations/person_keypoints_val2017_mirrored.json
"""

# COCO keypoint swap pairs (left ↔ right)
# 0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear,
# 5: left_shoulder, 6: right_shoulder, 7: left_elbow, 8: right_elbow,
# 9: left_wrist, 10: right_wrist, 11: left_hip, 12: right_hip,
# 13: left_knee, 14: right_knee, 15: left_ankle, 16: right_ankle
KEYPOINT_SWAP_PAIRS = [
    (1, 2),   # left_eye ↔ right_eye
    (3, 4),   # left_ear ↔ right_ear
    (5, 6),   # left_shoulder ↔ right_shoulder
    (7, 8),   # left_elbow ↔ right_elbow
    (9, 10),  # left_wrist ↔ right_wrist
    (11, 12), # left_hip ↔ right_hip
    (13, 14), # left_knee ↔ right_knee
    (15, 16), # left_ankle ↔ right_ankle
]


def mirror_keypoints(keypoints, image_width):
    """
    Mirror keypoints horizontally.
    
    Args:
        keypoints: List of [x, y, v, x, y, v, ...] (COCO format)
        image_width: Width of the image
    
    Returns:
        Mirrored keypoints list
    """
    # Convert to list of (x, y, v) tuples
    num_keypoints = len(keypoints) // 3
    kpts = []
    for i in range(num_keypoints):
        x = keypoints[i * 3]
        y = keypoints[i * 3 + 1]
        v = keypoints[i * 3 + 2]
        kpts.append([x, y, v])
    
    # Flip x coordinates
    for kpt in kpts:
        if kpt[2] > 0:  # Only flip visible keypoints
            kpt[0] = image_width - kpt[0]
    
    # Swap left/right pairs
    for left_idx, right_idx in KEYPOINT_SWAP_PAIRS:
        if left_idx < len(kpts) and right_idx < len(kpts):
            kpts[left_idx], kpts[right_idx] = kpts[right_idx], kpts[left_idx]
    
    # Flatten back to COCO format
    mirrored = []
    for kpt in kpts:
        mirrored.extend(kpt)
    
    return mirrored


def mirror_bbox(bbox, image_width):
    """
    Mirror bounding box horizontally.
    
    Args:
        bbox: [x, y, width, height] (COCO format)
        image_width: Width of the image
    
    Returns:
        Mirrored bbox
    """
    x, y, w, h = bbox
    new_x = image_width - x - w
    return [new_x, y, w, h]


def mirror_segmentation(segmentation, image_width):
    """
    Mirror segmentation polygons horizontally.
    
    Args:
        segmentation: List of polygon coordinate lists
        image_width: Width of the image
    
    Returns:
        Mirrored segmentation
    """
    if not segmentation:
        return segmentation
    
    mirrored_seg = []
    for polygon in segmentation:
        # Polygon format: [x1, y1, x2, y2, ...]
        mirrored_poly = []
        for i in range(0, len(polygon), 2):
            x = polygon[i]
            y = polygon[i + 1]
            mirrored_poly.extend([image_width - x, y])
        mirrored_seg.append(mirrored_poly)
    
    return mirrored_seg


def mirror_val_data(
    annotations_path,
    images_dir,
    output_dir,
    output_json
):
    """
    Create mirrored versions of validation images and annotations.
    
    Args:
        annotations_path: Path to person_keypoints_val2017.json
        images_dir: Path to val2017 images directory
        output_dir: Output directory for mirrored images
        output_json: Output path for mirrored annotations JSON
    """
    print(f"📂 Loading annotations from: {annotations_path}")
    with open(annotations_path, 'r') as f:
        data = json.load(f)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    
    # Build image id to image info mapping
    image_id_to_info = {img["id"]: img for img in data["images"]}
    
    # Process images
    mirrored_images = []
    print(f"🔄 Mirroring {len(data['images'])} images...")
    
    for img_info in data["images"]:
        src_path = os.path.join(images_dir, os.path.basename(img_info["file_name"]))
        
        # Create mirrored filename
        base_name = os.path.basename(img_info["file_name"])
        name_parts = os.path.splitext(base_name)
        mirrored_name = f"{name_parts[0]}_mirrored{name_parts[1]}"
        dst_path = os.path.join(output_dir, mirrored_name)
        
        if os.path.exists(src_path):
            # Load and mirror image
            img = Image.open(src_path)
            mirrored_img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mirrored_img.save(dst_path)
            
            # Create new image info
            new_img_info = dict(img_info)
            new_img_info["file_name"] = mirrored_name
            mirrored_images.append(new_img_info)
        else:
            print(f"⚠️ Missing: {src_path}")
    
    # Process annotations
    mirrored_annotations = []
    print(f"🔄 Mirroring {len(data['annotations'])} annotations...")
    
    for ann in data["annotations"]:
        img_info = image_id_to_info.get(ann["image_id"])
        if not img_info:
            continue
        
        image_width = img_info["width"]
        
        new_ann = dict(ann)
        
        # Mirror keypoints
        if "keypoints" in ann:
            new_ann["keypoints"] = mirror_keypoints(ann["keypoints"], image_width)
        
        # Mirror bbox
        if "bbox" in ann:
            new_ann["bbox"] = mirror_bbox(ann["bbox"], image_width)
        
        # Mirror segmentation
        if "segmentation" in ann and isinstance(ann["segmentation"], list):
            new_ann["segmentation"] = mirror_segmentation(ann["segmentation"], image_width)
        
        mirrored_annotations.append(new_ann)
    
    # Create output JSON
    output_info = dict(data.get("info", {}))
    output_info["notice"] = "These images and annotations have been generated by mirroring across the image vertical center, and are not stored remotely"
    mirrored_data = {
        "info": output_info,
        "images": mirrored_images,
        "annotations": mirrored_annotations,
        "categories": data.get("categories", [])
    }
    
    with open(output_json, "w") as f:
        json.dump(mirrored_data, f, indent=4)
    
    print(f"\n✅ Mirroring complete!")
    print(f"   Images: {len(mirrored_images)} saved to {output_dir}")
    print(f"   Annotations: {len(mirrored_annotations)} saved to {output_json}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create mirrored validation images and annotations")
    parser.add_argument("--annotations", default="coco/annotations/person_keypoints_val2017.json",
                        help="Path to validation annotations JSON")
    parser.add_argument("--images", default="coco/val2017",
                        help="Path to validation images directory")
    parser.add_argument("--output-dir", default="coco/val2017_mirrored",
                        help="Output directory for mirrored images")
    parser.add_argument("--output-json", default="coco/annotations/person_keypoints_val2017_mirrored.json",
                        help="Output path for mirrored annotations JSON")
    
    args = parser.parse_args()
    
    mirror_val_data(
        annotations_path=args.annotations,
        images_dir=args.images,
        output_dir=args.output_dir,
        output_json=args.output_json
    )

