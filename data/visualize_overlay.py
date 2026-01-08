#!/usr/bin/env python3
"""
Script to overlay ground truth keypoint annotations on validation images.

Usage:
    python data/visualize_overlay.py --data-root data/coco_general_val/
    python data/visualize_overlay.py --data-root data/coco/ --ann-file annotations/person_keypoints_val2017.json
    python data/visualize_overlay.py --data-root data/coco_general_val/ --output general_val_gt.mp4 --max-images 100
"""

import argparse
import json
import cv2
import numpy as np
import shutil
from pathlib import Path
from collections import defaultdict


# COCO keypoint skeleton connections
COCO_SKELETON = [
    (0, 1), (0, 2),      # nose to eyes
    (1, 3), (2, 4),      # eyes to ears
    (5, 6),              # shoulders
    (5, 7), (7, 9),      # left arm
    (6, 8), (8, 10),     # right arm
    (5, 11), (6, 12),    # torso
    (11, 12),            # hips
    (11, 13), (13, 15),  # left leg
    (12, 14), (14, 16),  # right leg
]

# Colors for different body parts (BGR)
COLORS = {
    'face': (255, 200, 100),      # light blue
    'left_arm': (0, 255, 0),      # green
    'right_arm': (0, 0, 255),     # red
    'left_leg': (255, 255, 0),    # cyan
    'right_leg': (255, 0, 255),   # magenta
    'torso': (0, 255, 255),       # yellow
}

# Keypoint names for COCO
KEYPOINT_NAMES = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]


def get_keypoint_color(idx):
    """Get color for a keypoint based on its index."""
    if idx <= 4:
        return COLORS['face']
    elif idx in [5, 7, 9]:
        return COLORS['left_arm']
    elif idx in [6, 8, 10]:
        return COLORS['right_arm']
    elif idx in [11, 13, 15]:
        return COLORS['left_leg']
    elif idx in [12, 14, 16]:
        return COLORS['right_leg']
    else:
        return COLORS['torso']


def get_skeleton_color(start_idx, end_idx):
    """Get color for a skeleton connection."""
    if start_idx <= 4 or end_idx <= 4:
        return COLORS['face']
    elif start_idx in [5, 7, 9] or end_idx in [5, 7, 9]:
        return COLORS['left_arm']
    elif start_idx in [6, 8, 10] or end_idx in [6, 8, 10]:
        return COLORS['right_arm']
    elif start_idx in [11, 13, 15] or end_idx in [11, 13, 15]:
        return COLORS['left_leg']
    elif start_idx in [12, 14, 16] or end_idx in [12, 14, 16]:
        return COLORS['right_leg']
    else:
        return COLORS['torso']


def draw_keypoints(image, keypoints, radius=4):
    """
    Draw keypoints with index labels on an image.
    
    Args:
        image: BGR image
        keypoints: List of [x, y, visibility] for each keypoint
        radius: Keypoint circle radius
    
    Returns:
        Annotated image
    """
    # Convert keypoints to numpy array
    kpts = np.array(keypoints).reshape(-1, 3)
    
    # Draw keypoints with index labels
    for i, (x, y, v) in enumerate(kpts):
        if v > 0:  # Only draw visible keypoints
            x, y = int(x), int(y)
            color = get_keypoint_color(i)
            
            # Draw filled circle
            cv2.circle(image, (x, y), radius, color, -1)
            # Draw white border
            cv2.circle(image, (x, y), radius, (255, 255, 255), 1)
            
            # Draw keypoint index label
            label = str(i)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.4
            font_thickness = 1
            
            # Get text size for background
            (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
            
            # Position label to the right of keypoint
            label_x = x + radius + 2
            label_y = y + text_h // 2
            
            # Draw black background for better visibility
            cv2.rectangle(image, 
                         (label_x - 1, label_y - text_h - 1), 
                         (label_x + text_w + 1, label_y + 2), 
                         (0, 0, 0), -1)
            
            # Draw white text
            cv2.putText(image, label, (label_x, label_y), 
                       font, font_scale, (255, 255, 255), font_thickness)
    
    return image


def draw_bbox(image, bbox, color=(0, 255, 0), thickness=2):
    """Draw bounding box on image."""
    x, y, w, h = [int(v) for v in bbox]
    cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)
    return image


def clear_output_directory(output_dir):
    """Clear all files in the output directory."""
    if output_dir.exists():
        print(f"Clearing output directory: {output_dir}")
        for item in output_dir.iterdir():
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)
        print("✓ Output directory cleared")


def create_video_from_images(images_dir, fps=10, output_path=None):
    """Create a video from all annotated images."""
    if output_path is None:
        output_path = images_dir / "result.mp4"
    
    image_files = sorted(images_dir.glob("*_gt.jpg"))
    
    if not image_files:
        print("No annotated images found to create video")
        return
    
    print(f"Found {len(image_files)} images for video")
    
    # Find max dimensions
    max_width, max_height = 0, 0
    for image_file in image_files:
        img = cv2.imread(str(image_file))
        if img is not None:
            h, w = img.shape[:2]
            max_width = max(max_width, w)
            max_height = max(max_height, h)
    
    # Round to nearest 16 for codec compatibility
    target_width = ((max_width + 15) // 16) * 16
    target_height = ((max_height + 15) // 16) * 16
    
    print(f"Video dimensions: {target_width}x{target_height}")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (target_width, target_height))
    
    if not video_writer.isOpened():
        print("Could not open video writer")
        return
    
    for i, image_file in enumerate(image_files):
        if (i + 1) % 50 == 0:
            print(f"Adding frame {i+1}/{len(image_files)}")
        
        image = cv2.imread(str(image_file))
        if image is not None:
            resized = cv2.resize(image, (target_width, target_height))
            video_writer.write(resized)
    
    video_writer.release()
    print(f"✓ Video saved to: {output_path}")


def visualize_annotations(
    data_root,
    ann_file='annotations/person_keypoints_val2017.json',
    images_subdir='val2017',
    output_dir='data/visualization_gt',
    output_video=None,
    max_images=None,
    draw_bbox_flag=True,
    fps=10,
):
    """
    Overlay ground truth annotations on images.
    
    Args:
        data_root: Root directory (e.g., data/coco_general_val/)
        ann_file: Annotation file relative to data_root
        images_subdir: Images subdirectory relative to data_root
        output_dir: Output directory for annotated images
        output_video: Output video path (optional)
        max_images: Maximum number of images to process (None for all)
        draw_bbox_flag: Whether to draw bounding boxes
        fps: Video FPS
    """
    data_root = Path(data_root)
    output_dir = Path(output_dir)
    ann_path = data_root / ann_file
    images_dir = data_root / images_subdir
    
    print(f"Data root: {data_root}")
    print(f"Annotation file: {ann_path}")
    print(f"Images directory: {images_dir}")
    print(f"Output directory: {output_dir}")
    
    # Validate paths
    if not ann_path.exists():
        print(f"Error: Annotation file not found: {ann_path}")
        return
    
    if not images_dir.exists():
        print(f"Error: Images directory not found: {images_dir}")
        return
    
    # Clear and create output directory
    clear_output_directory(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load annotations
    print(f"\nLoading annotations...")
    with open(ann_path, 'r') as f:
        coco_data = json.load(f)
    
    # Create lookup dictionaries
    images_dict = {img['id']: img for img in coco_data['images']}
    
    # Group annotations by image
    anns_by_image = defaultdict(list)
    for ann in coco_data['annotations']:
        if 'keypoints' in ann and ann.get('num_keypoints', 0) > 0:
            anns_by_image[ann['image_id']].append(ann)
    
    print(f"Found {len(coco_data['images'])} images")
    print(f"Found {len(coco_data['annotations'])} annotations")
    print(f"Images with keypoints: {len(anns_by_image)}")
    
    # Process images
    image_ids = list(anns_by_image.keys())
    if max_images:
        image_ids = image_ids[:max_images]
    
    print(f"\nProcessing {len(image_ids)} images...")
    
    processed = 0
    for i, image_id in enumerate(image_ids):
        if (i + 1) % 50 == 0:
            print(f"Processing {i+1}/{len(image_ids)}...")
        
        img_info = images_dict.get(image_id)
        if not img_info:
            continue
        
        img_path = images_dir / img_info['file_name']
        if not img_path.exists():
            continue
        
        # Load image
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        
        # Draw all annotations for this image
        for ann in anns_by_image[image_id]:
            # Draw bounding box
            if draw_bbox_flag and 'bbox' in ann:
                image = draw_bbox(image, ann['bbox'], color=(100, 100, 100))
            
            # Draw keypoints
            if 'keypoints' in ann:
                image = draw_keypoints(image, ann['keypoints'])
        
        # Save annotated image
        output_filename = f"{img_info['file_name'].replace('.jpg', '').replace('.png', '')}_gt.jpg"
        output_path = output_dir / output_filename
        cv2.imwrite(str(output_path), image)
        processed += 1
    
    print(f"\n✓ Processed {processed} images")
    print(f"✓ Annotated images saved to: {output_dir}")
    
    # Create video
    if output_video or processed > 0:
        print("\nCreating video...")
        video_path = Path(output_video) if output_video else output_dir / "gt_annotations.mp4"
        create_video_from_images(output_dir, fps=fps, output_path=video_path)


def main():
    parser = argparse.ArgumentParser(
        description='Visualize ground truth keypoint annotations on images'
    )
    parser.add_argument(
        '--data-root', type=str, required=True,
        help='Root directory containing annotations and images (e.g., data/coco_general_val/)'
    )
    parser.add_argument(
        '--ann-file', type=str, default='annotations/person_keypoints_val2017.json',
        help='Annotation file path relative to data-root'
    )
    parser.add_argument(
        '--images-subdir', type=str, default='val2017',
        help='Images subdirectory relative to data-root'
    )
    parser.add_argument(
        '--output-dir', type=str, default='data/visualization_gt',
        help='Output directory for annotated images'
    )
    parser.add_argument(
        '--output', type=str, default=None,
        help='Output video file path'
    )
    parser.add_argument(
        '--max-images', type=int, default=None,
        help='Maximum number of images to process (default: all)'
    )
    parser.add_argument(
        '--no-bbox', action='store_true',
        help='Do not draw bounding boxes'
    )
    parser.add_argument(
        '--fps', type=float, default=10,
        help='Video FPS (default: 10)'
    )
    
    args = parser.parse_args()
    
    visualize_annotations(
        data_root=args.data_root,
        ann_file=args.ann_file,
        images_subdir=args.images_subdir,
        output_dir=args.output_dir,
        output_video=args.output,
        max_images=args.max_images,
        draw_bbox_flag=not args.no_bbox,
        fps=args.fps,
    )


if __name__ == "__main__":
    main()

