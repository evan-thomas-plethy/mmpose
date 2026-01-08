#!/usr/bin/env python3
"""
Script to create an augmented dataset from the original COCO data.
This script copies the original data to a new directory and applies augmentations,
WITHOUT modifying the original data at all.

Usage:
    python make_augmented_dataset.py --output_name coco_augmented
    python make_augmented_dataset.py --output_name coco_augmented --augmentations_yaml pair_augmentations.yaml
    python make_augmented_dataset.py --output_name coco_augmented --train_only
    python make_augmented_dataset.py --source_dir coco_dedup_param_0.995 --output_name coco_dedup_param_0.995_augmented_v1.1 --augmentations_yaml pair_augmentations.yaml --train_only
"""

import argparse
import json
import os
import shutil
from datetime import datetime
from pathlib import Path

import yaml

from augment import run_all_augmentations


def make_augmented_dataset(
    source_dir: str = "coco",
    output_name: str = "coco_augmented",
    augmentations_yaml: str = "augmentations.yaml",
    train_only: bool = False,
    val_only: bool = False,
):
    """
    Create an augmented dataset by:
    1. Copying original data from source_dir to output_name directory
    2. Applying augmentations to the copied data (NOT the original)
    
    Args:
        source_dir: Path to the source COCO dataset (default: "coco")
        output_name: Name for the output augmented dataset directory
        augmentations_yaml: Path to the augmentations configuration file
        train_only: If True, only augment training data
        val_only: If True, only augment validation data
    """
    # Get the directory where this script is located
    script_dir = Path(__file__).parent.resolve()
    
    # Resolve paths relative to script directory
    source_path = script_dir / source_dir
    output_path = script_dir / output_name
    yaml_path = script_dir / "augmentations" / augmentations_yaml
    
    if not source_path.exists():
        raise FileNotFoundError(f"Source directory not found: {source_path}")
    
    if not yaml_path.exists():
        raise FileNotFoundError(f"Augmentations YAML not found: {yaml_path}")
    
    print(f"📂 Source directory: {source_path}")
    print(f"📂 Output directory: {output_path}")
    print(f"📄 Augmentations config: {yaml_path}")
    
    # Check if output directory already exists
    if output_path.exists():
        response = input(f"⚠️  Output directory '{output_path}' already exists. Delete and recreate? [y/N]: ")
        if response.lower() != 'y':
            print("Aborting.")
            return
        shutil.rmtree(output_path)
    
    # Create output directory structure
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "annotations").mkdir(exist_ok=True)
    (output_path / "train2017").mkdir(exist_ok=True)
    (output_path / "val2017").mkdir(exist_ok=True)
    
    # Define datasets to process
    datasets = []
    if not val_only:
        datasets.append(("train2017", "person_keypoints_train2017.json"))
    if not train_only:
        datasets.append(("val2017", "person_keypoints_val2017.json"))
    
    for img_subdir, ann_file in datasets:
        print(f"\n{'='*60}")
        print(f"Processing {img_subdir}...")
        print(f"{'='*60}")
        
        source_img_dir = source_path / img_subdir
        source_ann_file = source_path / "annotations" / ann_file
        output_img_dir = output_path / img_subdir
        output_ann_file = output_path / "annotations" / ann_file
        
        if not source_img_dir.exists():
            print(f"⚠️  Skipping {img_subdir}: source directory not found")
            continue
        
        if not source_ann_file.exists():
            print(f"⚠️  Skipping {img_subdir}: annotation file not found")
            continue
        
        # Step 1: Load source annotation file to get counts
        with open(source_ann_file, 'r') as f:
            source_data = json.load(f)
        original_num_images = len(source_data.get("images", []))
        original_num_annotations = len(source_data.get("annotations", []))
        print(f"📊 Original dataset: {original_num_images} images, {original_num_annotations} annotations")
        
        # Step 2: Copy all images from source to output
        print(f"📋 Copying images from {source_img_dir} to {output_img_dir}...")
        image_count = 0
        for img_file in source_img_dir.iterdir():
            if img_file.is_file() and img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                shutil.copy2(img_file, output_img_dir / img_file.name)
                image_count += 1
        print(f"✅ Copied {image_count} original images")
        
        # Step 3: Copy annotation file
        print(f"📋 Copying annotation file...")
        shutil.copy2(source_ann_file, output_ann_file)
        print(f"✅ Copied annotation file")
        
        # Step 4: Apply augmentations to the COPIED data (adds new images + annotations)
        print(f"🔄 Applying augmentations...")
        run_all_augmentations(
            augmentations_yaml=str(yaml_path),
            json_path=str(output_ann_file),
            img_dir=str(output_img_dir)
        )
        
        # Step 5: Add augmentations info to the JSON "info" section
        print(f"📝 Adding augmentations metadata to annotations...")
        with open(yaml_path, 'r') as f:
            augmentations_config = yaml.safe_load(f)
        
        with open(output_ann_file, 'r') as f:
            ann_data = json.load(f)
        
        # Ensure "info" section exists
        if "info" not in ann_data:
            ann_data["info"] = {}
        
        # Add augmentations metadata
        ann_data["info"]["augmentations"] = {
            "config_file_name": augmentations_yaml,
            "settings": augmentations_config
        }
        
        with open(output_ann_file, 'w') as f:
            json.dump(ann_data, f, indent=2)
        print(f"✅ Added augmentations metadata to info section")
        
        # Step 6: Verify final counts
        with open(output_ann_file, 'r') as f:
            final_data = json.load(f)
        final_num_images = len(final_data.get("images", []))
        final_num_annotations = len(final_data.get("annotations", []))
        augmented_images = final_num_images - original_num_images
        augmented_annotations = final_num_annotations - original_num_annotations
        print(f"📊 Final dataset: {final_num_images} images ({original_num_images} original + {augmented_images} augmented)")
        print(f"📊 Final annotations: {final_num_annotations} ({original_num_annotations} original + {augmented_annotations} augmented)")
    
    # If train_only, copy over val directories and annotations without augmentation
    if train_only:
        # Find all directories starting with "val2017"
        val_dirs = sorted([d for d in source_path.iterdir() if d.is_dir() and d.name.startswith("val2017")])
        # Find all annotation files starting with "person_keypoints_val2017"
        val_ann_files = sorted([f for f in (source_path / "annotations").glob("person_keypoints_val2017*.json")])
        
        if not val_dirs:
            print(f"\n⚠️  No val2017* directories found in {source_path}")
        
        # Copy val directories
        for source_img_dir in val_dirs:
            img_subdir = source_img_dir.name
            output_img_dir = output_path / img_subdir
            
            print(f"\n{'='*60}")
            print(f"Copying {img_subdir} (no augmentation)...")
            print(f"{'='*60}")
            
            # Create output directory
            output_img_dir.mkdir(exist_ok=True)
            
            # Copy images
            print(f"📋 Copying images from {source_img_dir} to {output_img_dir}...")
            image_count = 0
            for img_file in source_img_dir.iterdir():
                if img_file.is_file() and img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    shutil.copy2(img_file, output_img_dir / img_file.name)
                    image_count += 1
            print(f"✅ Copied {image_count} images")
        
        # Copy val annotation files
        for source_ann_file in val_ann_files:
            output_ann_file = output_path / "annotations" / source_ann_file.name
            print(f"📋 Copying annotation file: {source_ann_file.name}...")
            shutil.copy2(source_ann_file, output_ann_file)
            print(f"✅ Copied {source_ann_file.name}")
    
    # Print summary
    print(f"\n{'='*60}")
    print("📊 FINAL SUMMARY")
    print(f"{'='*60}")
    
    for img_subdir, ann_file in datasets:
        source_ann_file = source_path / "annotations" / ann_file
        output_ann_file = output_path / "annotations" / ann_file
        output_img_dir = output_path / img_subdir
        
        if output_ann_file.exists() and source_ann_file.exists():
            # Get original counts
            with open(source_ann_file, 'r') as f:
                orig_data = json.load(f)
            orig_images = len(orig_data.get("images", []))
            orig_annotations = len(orig_data.get("annotations", []))
            
            # Get final counts
            with open(output_ann_file, 'r') as f:
                final_data = json.load(f)
            final_images = len(final_data.get("images", []))
            final_annotations = len(final_data.get("annotations", []))
            
            # Count actual image files
            num_files = sum(1 for f in output_img_dir.iterdir() 
                          if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp'])
            
            print(f"\n{img_subdir}:")
            print(f"  Original:  {orig_images} images, {orig_annotations} annotations")
            print(f"  Final:     {final_images} images, {final_annotations} annotations")
            print(f"  Added:     {final_images - orig_images} augmented images, {final_annotations - orig_annotations} augmented annotations")
            print(f"  Files:     {num_files} image files on disk")
            
            # Verify consistency
            if num_files != final_images:
                print(f"  ⚠️  WARNING: Image file count ({num_files}) doesn't match JSON ({final_images})")
    
    print(f"\n✅ Augmented dataset created at: {output_path}")
    print(f"   Original data in '{source_dir}' was NOT modified.")


def main():
    parser = argparse.ArgumentParser(
        description="Create an augmented dataset from original COCO data without modifying the original."
    )
    parser.add_argument(
        "--source_dir",
        type=str,
        default="coco",
        help="Source COCO dataset directory (default: coco)"
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default="coco_augmented",
        help="Name for the output augmented dataset directory (default: coco_augmented)"
    )
    parser.add_argument(
        "--augmentations_yaml",
        type=str,
        default="augmentations.yaml",
        help="Path to augmentations YAML config file (default: augmentations.yaml)"
    )
    parser.add_argument(
        "--train_only",
        action="store_true",
        help="Only augment training data"
    )
    parser.add_argument(
        "--val_only",
        action="store_true",
        help="Only augment validation data"
    )
    parser.add_argument(
        "-y", "--yes",
        action="store_true",
        help="Automatically confirm overwrite of existing output directory"
    )
    
    args = parser.parse_args()
    
    if args.train_only and args.val_only:
        parser.error("Cannot specify both --train_only and --val_only")
    
    # Handle auto-confirm for non-interactive usage
    if args.yes:
        # Monkey-patch input to return 'y'
        import builtins
        original_input = builtins.input
        builtins.input = lambda _: 'y'
        try:
            make_augmented_dataset(
                source_dir=args.source_dir,
                output_name=args.output_name,
                augmentations_yaml=args.augmentations_yaml,
                train_only=args.train_only,
                val_only=args.val_only,
            )
        finally:
            builtins.input = original_input
    else:
        make_augmented_dataset(
            source_dir=args.source_dir,
            output_name=args.output_name,
            augmentations_yaml=args.augmentations_yaml,
            train_only=args.train_only,
            val_only=args.val_only,
        )


if __name__ == "__main__":
    main()

