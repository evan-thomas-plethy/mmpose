#!/usr/bin/env python3
"""
Script to load results from results.pkl, overlay keypoints on images using OpenCV,
and save annotated images to the visualization folder.
"""

import pickle
import os
import sys
import cv2
import numpy as np
import shutil
import argparse
from pathlib import Path

def clear_output_directory(output_dir):
    """Clear all files in the output directory before processing."""
    if output_dir.exists():
        print(f"Clearing output directory: {output_dir}")
        # Remove all files and subdirectories
        for item in output_dir.iterdir():
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)
        print(f"✓ Output directory cleared")
    else:
        print(f"Output directory does not exist, will be created: {output_dir}")

def draw_keypoints(image, keypoints, keypoints_visible=None, color=(0, 255, 0), radius=3):
    """Draw keypoints on an image."""
    if keypoints is None or len(keypoints) == 0:
        return image
    
    # Ensure keypoints is 2D array
    if len(keypoints.shape) == 3:
        keypoints = keypoints[0]  # Take first instance if multiple
    
    # Draw each keypoint
    for i, kpt in enumerate(keypoints):
        if len(kpt) >= 2:
            x, y = int(kpt[0]), int(kpt[1])
            
            # Draw all keypoints regardless of visibility
            
            # Draw circle for keypoint
            cv2.circle(image, (x, y), radius, color, -1)
            
            # Draw keypoint index
            cv2.putText(image, str(i), (x + 5, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return image

def overlay_keypoints_on_images(results_path, images_dir, output_dir, output_video_path=None):
    """Load results and overlay keypoints on images."""
    
    # Convert string paths to Path objects if needed
    results_path = Path(results_path)
    images_dir = Path(images_dir)
    output_dir = Path(output_dir)
    if output_video_path is not None:
        output_video_path = Path(output_video_path)
    
    print(f"Looking for results file at: {results_path}")
    print(f"Images directory: {images_dir}")
    print(f"Output directory: {output_dir}")
    
    # Clear output directory before processing
    clear_output_directory(output_dir)
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not results_path.exists():
        print(f"Error: File not found at {results_path}")
        return
    
    if not images_dir.exists():
        print(f"Error: Images directory not found at {images_dir}")
        return
    
    try:
        # Load the pickle file
        print(f"Loading results from: {results_path}")
        with open(results_path, 'rb') as f:
            results = pickle.load(f)
        
        print(f"Loaded {len(results)} results")
        
        # Process each result
        for i, result in enumerate(results):
            try:
                # Extract image path and keypoints
                if 'img_path' in result:
                    img_path = result['img_path']
                    # Remove 'data/coco/val2017/' prefix if present
                    if img_path.startswith('data/coco/val2017/'):
                        img_path = img_path[18:]  # Remove 'data/coco/val2017/' prefix
                    
                    full_img_path = images_dir / img_path
                    
                    if not full_img_path.exists():
                        print(f"Warning: Image not found: {full_img_path}")
                        continue
                    
                    # Load image
                    image = cv2.imread(str(full_img_path))
                    if image is None:
                        print(f"Warning: Could not load image: {full_img_path}")
                        continue
                    
                    print(f"Processing image {i+1}/{len(results)}: {img_path}")
                    
                    # Extract keypoints from different possible locations
                    keypoints = None
                    keypoints_visible = None
                    
                    # Check pred_instances first (predicted keypoints)
                    if 'pred_instances' in result and 'keypoints' in result['pred_instances']:
                        keypoints = result['pred_instances']['keypoints']
                        if 'keypoints_visible' in result['pred_instances']:
                            keypoints_visible = result['pred_instances']['keypoints_visible']
                        print(f"  Using predicted keypoints: {keypoints.shape if hasattr(keypoints, 'shape') else 'no shape'}")
                    
                    # Check gt_instances (ground truth keypoints)
                    elif 'gt_instances' in result and 'keypoints' in result['gt_instances']:
                        keypoints = result['gt_instances']['keypoints']
                        if 'keypoints_visible' in result['gt_instances']:
                            keypoints_visible = result['gt_instances']['keypoints_visible']
                        print(f"  Using ground truth keypoints: {keypoints.shape if hasattr(keypoints, 'shape') else 'no shape'}")
                    
                    if keypoints is not None:
                        # Draw keypoints on image
                        image_with_keypoints = draw_keypoints(image.copy(), keypoints, keypoints_visible)
                        
                        # Generate output filename
                        img_name = Path(img_path).stem
                        output_filename = f"{img_name}_keypoints.jpg"
                        output_path = output_dir / output_filename
                        
                        # Save annotated image
                        cv2.imwrite(str(output_path), image_with_keypoints)
                        print(f"  Saved annotated image: {output_filename}")
                    else:
                        print(f"  No keypoints found in result")
                
            except Exception as e:
                print(f"Error processing result {i}: {e}")
                continue
        
        print(f"\nProcessing complete! Annotated images saved to: {output_dir}")
        
        # Create video from all annotated images
        print("\nCreating video from annotated images...")
        video_output_path = output_video_path if output_video_path is not None else output_dir / "result.mp4"
        create_video_from_images(output_dir, fps=10, output_path=video_output_path)
        
    except Exception as e:
        print(f"Error loading pickle file: {e}")
        import traceback
        traceback.print_exc()

def create_video_from_images(images_dir, fps=10, output_path=None):
    """Create a video from all images in the directory."""
    if output_path is None:
        output_path = images_dir.parent / "result.mp4"
    
    # Get all image files
    image_files = sorted([f for f in images_dir.glob("*_keypoints.jpg")])
    
    if not image_files:
        print("No annotated images found to create video")
        return
    
    print(f"Found {len(image_files)} images to create video")
    
    # Read first image to get dimensions
    first_image = cv2.imread(str(image_files[0]))
    if first_image is None:
        print("Could not read first image")
        return
    
    # Find the maximum dimensions across all images to ensure compatibility
    max_width, max_height = 0, 0
    for image_file in image_files:
        img = cv2.imread(str(image_file))
        if img is not None:
            h, w = img.shape[:2]
            max_width = max(max_width, w)
            max_height = max(max_height, h)
    
    # Use a standard resolution that can accommodate all images
    # Round up to nearest multiple of 16 for better codec compatibility
    target_width = ((max_width + 15) // 16) * 16
    target_height = ((max_height + 15) // 16) * 16
    
    print(f"Maximum image dimensions: {max_width}x{max_height}")
    print(f"Target video dimensions: {target_width}x{target_height}")
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (target_width, target_height))
    
    if not video_writer.isOpened():
        print("Could not open video writer")
        return
    
    # Add each image to video
    for i, image_file in enumerate(image_files):
        print(f"Adding frame {i+1}/{len(image_files)}: {image_file.name}")
        
        image = cv2.imread(str(image_file))
        if image is not None:
            # Resize image to target dimensions
            resized_image = cv2.resize(image, (target_width, target_height))
            video_writer.write(resized_image)
        else:
            print(f"Warning: Could not read image {image_file}")
    
    # Release video writer
    video_writer.release()
    print(f"Video saved to: {output_path}")


def main():
    """Main function with command line argument support."""
    parser = argparse.ArgumentParser(description='Visualize pose estimation results by overlaying keypoints on images')
    parser.add_argument('--results', type=str, 
                       default='results.pkl',
                       help='Path to results.pkl file')
    parser.add_argument('--images-dir', type=str,
                       default='data/coco/val2017',
                       help='Path to directory containing images')
    parser.add_argument('--output-dir', type=str,
                       default='visualization',
                       help='Path to output directory for annotated images')
    parser.add_argument('--output', type=str,
                       default=None,
                       help='Path to output video file (default: output_dir/result.mp4)')
    parser.add_argument('--fps', type=float, default=10,
                       help='FPS for output video')
    
    args = parser.parse_args()
    
    # Get script directory for relative path resolution
    script_dir = Path(__file__).parent
    
    # Resolve paths relative to script directory if they are relative
    results_path = args.results if Path(args.results).is_absolute() else script_dir.parent / args.results
    images_dir = args.images_dir if Path(args.images_dir).is_absolute() else script_dir.parent / args.images_dir
    output_dir = args.output_dir if Path(args.output_dir).is_absolute() else script_dir.parent / args.output_dir
    output_video_path = None
    if args.output is not None:
        output_video_path = args.output if Path(args.output).is_absolute() else script_dir.parent / args.output
    
    # Run the visualization
    overlay_keypoints_on_images(results_path, images_dir, output_dir, output_video_path)


if __name__ == "__main__":
    main()
