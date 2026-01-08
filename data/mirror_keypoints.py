#!/usr/bin/env python3
"""
Script to swap left/right keypoint pairs in COCO annotations.

Use this to fix annotations where left and right sides are mislabeled.

Usage:
    # Preview changes (dry run)
    python mirror_keypoints.py --data-root coco_general_val/ --dry-run

    # Apply changes (creates backup, modifies in place)
    python mirror_keypoints.py --data-root coco_standing_general_val/ --ann-file coco_standing_general_val/annotations/standing_general_val.json

    # Apply to specific annotation file
    python mirror_keypoints.py --ann-file coco/annotations/person_keypoints_train2017.json
"""

import argparse
import json
import shutil
from pathlib import Path
from datetime import datetime


# COCO keypoint indices
# 0: nose (no pair - center)
# 1: left_eye      <-> 2: right_eye
# 3: left_ear      <-> 4: right_ear
# 5: left_shoulder <-> 6: right_shoulder
# 7: left_elbow    <-> 8: right_elbow
# 9: left_wrist    <-> 10: right_wrist
# 11: left_hip     <-> 12: right_hip
# 13: left_knee    <-> 14: right_knee
# 15: left_ankle   <-> 16: right_ankle

LEFT_RIGHT_PAIRS = [
    (1, 2),   # eyes
    (3, 4),   # ears
    (5, 6),   # shoulders
    (7, 8),   # elbows
    (9, 10),  # wrists
    (11, 12), # hips
    (13, 14), # knees
    (15, 16), # ankles
]

KEYPOINT_NAMES = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]


def swap_lr_keypoints(keypoints):
    """
    Swap left/right keypoint pairs in a keypoints list.
    
    Args:
        keypoints: List of [x, y, v, x, y, v, ...] for 17 keypoints (51 values)
    
    Returns:
        New keypoints list with L/R pairs swapped
    """
    # Make a copy
    new_kpts = list(keypoints)
    
    # Swap each L/R pair
    for left_idx, right_idx in LEFT_RIGHT_PAIRS:
        # Each keypoint has 3 values: x, y, visibility
        left_start = left_idx * 3
        right_start = right_idx * 3
        
        # Swap the 3 values for each keypoint
        new_kpts[left_start:left_start+3], new_kpts[right_start:right_start+3] = \
            new_kpts[right_start:right_start+3], new_kpts[left_start:left_start+3]
    
    return new_kpts


def mirror_annotations(ann_file, dry_run=False, backup=True):
    """
    Swap L/R keypoints in all annotations in a COCO annotation file.
    
    Args:
        ann_file: Path to annotation JSON file
        dry_run: If True, only preview changes without modifying
        backup: If True, create backup before modifying
    
    Returns:
        Number of annotations modified
    """
    ann_path = Path(ann_file)
    
    if not ann_path.exists():
        print(f"Error: Annotation file not found: {ann_path}")
        return 0
    
    print(f"Loading annotations from: {ann_path}")
    
    with open(ann_path, 'r') as f:
        coco_data = json.load(f)
    
    # Count annotations with keypoints
    total_anns = len(coco_data.get('annotations', []))
    anns_with_kpts = sum(1 for ann in coco_data.get('annotations', []) 
                         if 'keypoints' in ann and len(ann['keypoints']) == 51)
    
    print(f"Total annotations: {total_anns}")
    print(f"Annotations with keypoints: {anns_with_kpts}")
    
    if dry_run:
        print("\n[DRY RUN] Preview of changes:")
        print("-" * 60)
    
    # Process annotations
    modified_count = 0
    for i, ann in enumerate(coco_data.get('annotations', [])):
        if 'keypoints' not in ann or len(ann['keypoints']) != 51:
            continue
        
        original_kpts = ann['keypoints']
        swapped_kpts = swap_lr_keypoints(original_kpts)
        
        # Show preview for first few annotations
        if dry_run and modified_count < 3:
            print(f"\nAnnotation ID: {ann.get('id', i)}")
            for left_idx, right_idx in LEFT_RIGHT_PAIRS:
                left_name = KEYPOINT_NAMES[left_idx]
                right_name = KEYPOINT_NAMES[right_idx]
                
                orig_left = original_kpts[left_idx*3:left_idx*3+3]
                orig_right = original_kpts[right_idx*3:right_idx*3+3]
                
                new_left = swapped_kpts[left_idx*3:left_idx*3+3]
                new_right = swapped_kpts[right_idx*3:right_idx*3+3]
                
                if orig_left[2] > 0 or orig_right[2] > 0:  # Only show visible keypoints
                    print(f"  {left_name} ({left_idx}): {orig_left[:2]} -> {new_left[:2]}")
                    print(f"  {right_name} ({right_idx}): {orig_right[:2]} -> {new_right[:2]}")
        
        if not dry_run:
            ann['keypoints'] = swapped_kpts
        
        modified_count += 1
    
    if dry_run:
        print("-" * 60)
        print(f"\n[DRY RUN] Would modify {modified_count} annotations")
        print("Run without --dry-run to apply changes")
        return modified_count
    
    # Create backup
    if backup:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = ann_path.with_suffix(f'.backup_{timestamp}.json')
        print(f"\nCreating backup: {backup_path}")
        shutil.copy(ann_path, backup_path)
    
    # Save modified annotations
    print(f"Saving modified annotations to: {ann_path}")
    with open(ann_path, 'w') as f:
        json.dump(coco_data, f)
    
    print(f"✓ Modified {modified_count} annotations")
    
    return modified_count


def main():
    parser = argparse.ArgumentParser(
        description='Swap left/right keypoint pairs in COCO annotations'
    )
    parser.add_argument(
        '--data-root', type=str, default=None,
        help='Data root directory (will process train and val annotations)'
    )
    parser.add_argument(
        '--ann-file', type=str, default=None,
        help='Specific annotation file to process'
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Preview changes without modifying files'
    )
    parser.add_argument(
        '--no-backup', action='store_true',
        help='Do not create backup before modifying'
    )
    
    args = parser.parse_args()
    
    if not args.data_root and not args.ann_file:
        print("Error: Must specify either --data-root or --ann-file")
        parser.print_help()
        return
    
    ann_files = []
    
    if args.ann_file:
        ann_files.append(Path(args.ann_file))
    
    if args.data_root:
        data_root = Path(args.data_root)
        ann_dir = data_root / 'annotations'
        
        if ann_dir.exists():
            # Find all keypoint annotation files
            for ann_file in ann_dir.glob('*keypoints*.json'):
                ann_files.append(ann_file)
        else:
            print(f"Warning: Annotations directory not found: {ann_dir}")
    
    if not ann_files:
        print("No annotation files found to process")
        return
    
    print(f"Found {len(ann_files)} annotation file(s) to process:")
    for f in ann_files:
        print(f"  - {f}")
    print()
    
    total_modified = 0
    for ann_file in ann_files:
        print(f"\n{'='*60}")
        print(f"Processing: {ann_file.name}")
        print('='*60)
        
        modified = mirror_annotations(
            ann_file,
            dry_run=args.dry_run,
            backup=not args.no_backup
        )
        total_modified += modified
    
    print(f"\n{'='*60}")
    print(f"Total annotations {'would be ' if args.dry_run else ''}modified: {total_modified}")
    print('='*60)


if __name__ == "__main__":
    main()

