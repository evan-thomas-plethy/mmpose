#!/usr/bin/env python3
"""
Overlay COCO keypoint annotations on images from a dataset directory.

Usage:
    python overlay_dataset.py Heel_Slides/v1/frames Heel_Slides/v1/HeelSlides_V1.json --max-images 50 --fps 15
    python overlay_dataset.py Heel_Slides/v1/frames Heel_Slides/v1/HeelSlides_V1.json --output-dir ./overlay_output
"""

import argparse
import json
import re
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

VIDEO_NAME_RE = re.compile(r"^(.*)_frame_\d+$", re.IGNORECASE)


def video_name_from_file_name(file_name: str) -> str | None:
    """Return the video/session stem before ``_frame_XXXX``."""
    stem = Path(file_name).stem
    match = VIDEO_NAME_RE.match(stem)
    if match:
        return match.group(1)
    return None


# Colors for different body parts (BGR)
COLORS = {
    'face': (255, 200, 100),      # light blue
    'left_arm': (0, 255, 0),      # green
    'right_arm': (0, 0, 255),     # red
    'left_leg': (255, 255, 0),    # cyan
    'right_leg': (255, 0, 255),   # magenta
    'torso': (0, 255, 255),       # yellow
}


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


def draw_skeleton(image, keypoints, thickness=2):
    """Draw skeleton connections between keypoints."""
    kpts = np.array(keypoints).reshape(-1, 3)
    
    for start_idx, end_idx in COCO_SKELETON:
        if start_idx < len(kpts) and end_idx < len(kpts):
            x1, y1, v1 = kpts[start_idx]
            x2, y2, v2 = kpts[end_idx]
            
            if v1 > 0 and v2 > 0:
                color = get_skeleton_color(start_idx, end_idx)
                cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
    
    return image


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
    kpts = np.array(keypoints).reshape(-1, 3)
    
    for i, (x, y, v) in enumerate(kpts):
        if v > 0:
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
            
            (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
            
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


def draw_video_label(image, label: str):
    """Draw the video/session name along the top of the frame."""
    if not label:
        return image

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    font_thickness = 1
    padding_x = 8
    padding_y = 6

    (text_w, text_h), baseline = cv2.getTextSize(
        label, font, font_scale, font_thickness
    )
    box_h = text_h + baseline + padding_y * 2
    box_w = min(text_w + padding_x * 2, image.shape[1])

    cv2.rectangle(image, (0, 0), (box_w, box_h), (0, 0, 0), -1)
    cv2.putText(
        image,
        label,
        (padding_x, text_h + padding_y),
        font,
        font_scale,
        (255, 255, 255),
        font_thickness,
        cv2.LINE_AA,
    )
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
    
    image_files = sorted(images_dir.glob("*_overlay.jpg"))
    
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


def overlay_dataset(
    frames_dir,
    ann_file,
    output_dir=None,
    output_video=None,
    max_images=None,
    draw_bbox_flag=True,
    draw_skeleton_flag=True,
    fps=10,
):
    """
    Overlay COCO annotations on images from a dataset.
    
    Args:
        frames_dir: Directory containing frame images
        ann_file: Path to COCO annotation JSON file
        output_dir: Output directory for annotated images (default: ./)
        output_video: Output video path (optional)
        max_images: Maximum number of images to process (None for all)
        draw_bbox_flag: Whether to draw bounding boxes
        draw_skeleton_flag: Whether to draw skeleton connections
        fps: Video FPS
    """
    frames_dir = Path(frames_dir)
    ann_file = Path(ann_file)
    
    if output_dir is None:
        output_dir = Path(".")
    else:
        output_dir = Path(output_dir)
    
    print(f"Frames directory: {frames_dir}")
    print(f"Annotation file: {ann_file}")
    print(f"Output directory: {output_dir}")
    
    # Validate paths
    if not ann_file.exists():
        print(f"Error: Annotation file not found: {ann_file}")
        return
    
    if not frames_dir.exists():
        print(f"Error: Frames directory not found: {frames_dir}")
        return
    
    # Clear and create output directory
    clear_output_directory(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load annotations
    print(f"\nLoading annotations...")
    with open(ann_file, 'r') as f:
        coco_data = json.load(f)
    
    # Create lookup dictionaries
    images_dict = {img['id']: img for img in coco_data['images']}
    
    # Also create filename-based lookup for flexibility
    images_by_filename = {img['file_name']: img for img in coco_data['images']}
    
    # Group annotations by image
    anns_by_image = defaultdict(list)
    for ann in coco_data['annotations']:
        if 'keypoints' in ann:
            # Check if annotation has keypoints (even if num_keypoints is missing)
            keypoints = ann['keypoints']
            # Count visible keypoints
            kpts = np.array(keypoints).reshape(-1, 3)
            num_visible = np.sum(kpts[:, 2] > 0)
            if num_visible > 0:
                anns_by_image[ann['image_id']].append(ann)
    
    print(f"Found {len(coco_data['images'])} images in annotations")
    print(f"Found {len(coco_data['annotations'])} annotations")
    print(f"Images with visible keypoints: {len(anns_by_image)}")
    
    # Get list of actual frame files
    frame_files = sorted(list(frames_dir.glob("*.jpg")) + list(frames_dir.glob("*.png")))
    print(f"Found {len(frame_files)} frame files in directory")
    
    # Process images
    image_ids = list(anns_by_image.keys())
    if max_images:
        image_ids = image_ids[:max_images]
    
    print(f"\nProcessing {len(image_ids)} images with annotations...")
    
    processed = 0
    for i, image_id in enumerate(image_ids):
        if (i + 1) % 50 == 0:
            print(f"Processing {i+1}/{len(image_ids)}...")
        
        img_info = images_dict.get(image_id)
        if not img_info:
            continue
        
        img_path = frames_dir / img_info['file_name']
        if not img_path.exists():
            # Try without path prefix if file_name includes subdirectory
            img_path = frames_dir / Path(img_info['file_name']).name
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
            
            # Draw skeleton first (so keypoints are on top)
            if draw_skeleton_flag and 'keypoints' in ann:
                image = draw_skeleton(image, ann['keypoints'])
            
            # Draw keypoints
            if 'keypoints' in ann:
                image = draw_keypoints(image, ann['keypoints'])

        video_label = video_name_from_file_name(img_info['file_name'])
        if video_label:
            image = draw_video_label(image, video_label)

        # Save annotated image
        output_filename = f"{Path(img_info['file_name']).stem}_overlay.jpg"
        output_path = output_dir / output_filename
        cv2.imwrite(str(output_path), image)
        processed += 1
    
    print(f"\n✓ Processed {processed} images")
    print(f"✓ Annotated images saved to: {output_dir}")
    
    # Create video
    if output_video or processed > 0:
        print("\nCreating video...")
        video_path = Path(output_video) if output_video else output_dir / "overlay.mp4"
        create_video_from_images(output_dir, fps=fps, output_path=video_path)


def main():
    parser = argparse.ArgumentParser(
        description='Overlay COCO keypoint annotations on images from a dataset directory'
    )
    parser.add_argument(
        'frames_dir', type=str,
        help='Directory containing frame images (e.g., Heel_Slides/v1/frames)'
    )
    parser.add_argument(
        'ann_file', type=str,
        help='Path to COCO annotation JSON file (e.g., Heel_Slides/v1/HeelSlides_V1.json)'
    )
    parser.add_argument(
        '--output-dir', type=str, default=None,
        help='Output directory for annotated images (default: ./)',
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
        '--no-skeleton', action='store_true',
        help='Do not draw skeleton connections'
    )
    parser.add_argument(
        '--fps', type=float, default=10,
        help='Video FPS (default: 10)'
    )
    
    args = parser.parse_args()
    
    # Resolve paths relative to the data directory if not absolute
    script_dir = Path(__file__).parent
    
    frames_dir = args.frames_dir
    if not Path(frames_dir).is_absolute():
        frames_dir = script_dir / frames_dir
    
    ann_file = args.ann_file
    if not Path(ann_file).is_absolute():
        ann_file = script_dir / ann_file
    
    output_dir = args.output_dir
    if output_dir and not Path(output_dir).is_absolute():
        output_dir = script_dir / output_dir
    
    overlay_dataset(
        frames_dir=frames_dir,
        ann_file=ann_file,
        output_dir=output_dir,
        output_video=args.output,
        max_images=args.max_images,
        draw_bbox_flag=not args.no_bbox,
        draw_skeleton_flag=not args.no_skeleton,
        fps=args.fps,
    )


if __name__ == "__main__":
    main()

