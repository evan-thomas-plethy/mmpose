#!/usr/bin/env python3
"""
Script to convert pose estimation results from pickle format to JSON format.
Creates a dictionary with image filenames as keys and keypoint predictions as values.
Keypoints are formatted as arrays of shape (17, 3) with [x, y, confidence].

This script properly extracts both keypoint coordinates and confidence scores from the pkl file,
combining them into the standard [x, y, confidence] format for each keypoint.
"""

import pickle
import json
import numpy as np
from pathlib import Path
import argparse


def extract_image_filename(img_path):
    """
    Extract the filename from the image path.
    
    Args:
        img_path: Full path to the image
    
    Returns:
        str: Just the filename (e.g., "image001.jpg")
    """
    if isinstance(img_path, str):
        return Path(img_path).name
    else:
        return str(img_path)


def process_keypoints_and_scores(keypoints, keypoint_scores=None):
    """
    Process keypoints and confidence scores to create the correct format (17, 3).
    
    Args:
        keypoints: Raw keypoints from the pickle file (shape: N, K, 2)
        keypoint_scores: Raw confidence scores from the pickle file (shape: N, K)
    
    Returns:
        list: Keypoints array of shape (17, 3) with [x, y, confidence]
    """
    if keypoints is None:
        return None
    
    # Convert to numpy array if not already
    keypoints = np.array(keypoints)
    if keypoint_scores is not None:
        keypoint_scores = np.array(keypoint_scores)
    
    # Handle different input shapes for keypoints
    if len(keypoints.shape) == 3:
        # If shape is (1, 17, 2) or similar, take the first instance
        keypoints = keypoints[0]
        if keypoint_scores is not None and len(keypoint_scores.shape) == 2:
            keypoint_scores = keypoint_scores[0]
    elif len(keypoints.shape) == 2:
        # If shape is (17, 2), use as is
        pass
    else:
        print(f"Warning: Unexpected keypoint shape {keypoints.shape}")
        return None
    
    # Ensure we have exactly 17 keypoints with 2 coordinates each
    if keypoints.shape[0] != 17:
        print(f"Warning: Expected 17 keypoints, got {keypoints.shape[0]}")
        # Try to pad or truncate if possible
        if keypoints.shape[0] < 17:
            # Pad with zeros
            padded_kpts = np.zeros((17, 2))
            padded_kpts[:keypoints.shape[0]] = keypoints
            keypoints = padded_kpts
            if keypoint_scores is not None:
                padded_scores = np.zeros(17)
                padded_scores[:keypoint_scores.shape[0]] = keypoint_scores
                keypoint_scores = padded_scores
        elif keypoints.shape[0] > 17:
            # Truncate
            keypoints = keypoints[:17]
            if keypoint_scores is not None:
                keypoint_scores = keypoint_scores[:17]
    
    # Ensure we have 2 coordinates per keypoint
    if keypoints.shape[1] != 2:
        print(f"Warning: Expected 2 coordinates per keypoint, got {keypoints.shape[1]}")
        if keypoints.shape[1] < 2:
            # Pad with zeros
            padded = np.zeros((keypoints.shape[0], 2))
            padded[:, :keypoints.shape[1]] = keypoints
            keypoints = padded
        elif keypoints.shape[1] > 2:
            # Truncate to first 2 values
            keypoints = keypoints[:, :2]
    
    # Handle confidence scores
    if keypoint_scores is not None:
        # Ensure keypoint_scores has the right shape
        if len(keypoint_scores.shape) == 1:
            # Shape is (17,)
            if len(keypoint_scores) != 17:
                print(f"Warning: Expected 17 confidence scores, got {len(keypoint_scores)}")
                if len(keypoint_scores) < 17:
                    # Pad with zeros
                    padded_scores = np.zeros(17)
                    padded_scores[:len(keypoint_scores)] = keypoint_scores
                    keypoint_scores = padded_scores
                else:
                    # Truncate
                    keypoint_scores = keypoint_scores[:17]
    else:
        # If no confidence scores provided, create default ones (1.0)
        print("Warning: No confidence scores found, using default value 1.0")
        keypoint_scores = np.ones(17)
    
    # Combine keypoints and confidence scores into (17, 3) format
    result = np.zeros((17, 3))
    result[:, :2] = keypoints  # x, y coordinates
    result[:, 2] = keypoint_scores  # confidence scores
    
    # Ensure all values are Python floats (not numpy types)
    result = result.astype(float)
    
    return result.tolist()  # Convert to Python list for JSON serialization


def process_pkl_results(results_path):
    """
    Process pickle results and extract keypoint predictions.
    
    Args:
        results_path: Path to the pickle results file
    
    Returns:
        dict: Mapping from image filename to keypoint predictions
    """
    print(f"Loading results from: {results_path}")
    
    with open(results_path, 'rb') as f:
        results = pickle.load(f)
    
    print(f"Loaded {len(results)} results")
    
    image_keypoints = {}
    
    for i, result in enumerate(results):
        try:
            # Debug: Print available keys in result (first few results only)
            if i < 3:
                print(f"Result {i} keys: {list(result.keys())}")
                if 'pred_instances' in result:
                    pred_keys = list(result['pred_instances'].keys())
                    print(f"  pred_instances keys: {pred_keys}")
                if 'gt_instances' in result:
                    gt_keys = list(result['gt_instances'].keys())
                    print(f"  gt_instances keys: {gt_keys}")
            
            # Extract image path/filename
            img_path = None
            if 'img_path' in result:
                img_path = result['img_path']
            elif 'image_path' in result:
                img_path = result['image_path']
            else:
                print(f"Warning: No image path found in result {i}")
                continue
            
            # Extract filename
            filename = extract_image_filename(img_path)
            
            # Extract keypoints and confidence scores from different possible locations
            keypoints = None
            keypoint_scores = None
            
            # Check pred_instances first (predicted keypoints)
            if 'pred_instances' in result:
                pred_instances = result['pred_instances']
                if 'keypoints' in pred_instances:
                    keypoints = pred_instances['keypoints']
                    print(f"Image {filename}: Using predicted keypoints")
                    
                    # Extract confidence scores if available
                    if 'keypoint_scores' in pred_instances:
                        keypoint_scores = pred_instances['keypoint_scores']
                        print(f"  Found keypoint confidence scores")
                    else:
                        print(f"  No keypoint confidence scores found in pred_instances")
            
            # Check gt_instances (ground truth keypoints)
            elif 'gt_instances' in result:
                gt_instances = result['gt_instances']
                if 'keypoints' in gt_instances:
                    keypoints = gt_instances['keypoints']
                    print(f"Image {filename}: Using ground truth keypoints")
                    
                    # Extract confidence scores if available (usually 1.0 for ground truth)
                    if 'keypoint_scores' in gt_instances:
                        keypoint_scores = gt_instances['keypoint_scores']
                        print(f"  Found ground truth keypoint confidence scores")
                    else:
                        print(f"  No keypoint confidence scores found in gt_instances")
            
            # Check direct keypoints field
            elif 'keypoints' in result:
                keypoints = result['keypoints']
                print(f"Image {filename}: Using direct keypoints")
                
                # Check for direct confidence scores
                if 'keypoint_scores' in result:
                    keypoint_scores = result['keypoint_scores']
                    print(f"  Found direct keypoint confidence scores")
            
            if keypoints is not None:
                # Process keypoints and confidence scores to ensure correct format
                processed_keypoints = process_keypoints_and_scores(keypoints, keypoint_scores)
                
                if processed_keypoints is not None:
                    image_keypoints[filename] = processed_keypoints
                    print(f"  Processed keypoints shape: {len(processed_keypoints)}x{len(processed_keypoints[0])}")
                    # Show some confidence score examples
                    if len(processed_keypoints) > 0 and len(processed_keypoints[0]) > 2:
                        confidences = [kp[2] for kp in processed_keypoints[:5]]  # First 5 keypoints
                        print(f"  Sample confidence scores: {confidences}")
                else:
                    print(f"  Failed to process keypoints for {filename}")
            else:
                print(f"Image {filename}: No keypoints found")
                
        except Exception as e:
            print(f"Error processing result {i}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"Processed {len(image_keypoints)} images with keypoints")
    return image_keypoints


def save_json_results(image_keypoints, output_path):
    """
    Save the keypoint predictions to a JSON file.
    
    Args:
        image_keypoints: Dict mapping image filename to keypoint predictions
        output_path: Path to save the JSON file
    """
    print(f"Saving keypoint predictions to: {output_path}")
    
    # Create output directory if it doesn't exist
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save as JSON
    with open(output_path, 'w') as f:
        json.dump(image_keypoints, f, indent=2)
    
    print(f"Saved {len(image_keypoints)} image keypoint predictions to: {output_path}")


def main():
    """Main function with command line argument support."""
    parser = argparse.ArgumentParser(description='Convert pose estimation pickle results to JSON format')
    parser.add_argument('--results', type=str, 
                       default='results.pkl',
                       help='Path to results.pkl file')
    parser.add_argument('--output', type=str,
                       default='results.json',
                       help='Path to save JSON file')
    
    args = parser.parse_args()
    
    # Process pickle results
    image_keypoints = process_pkl_results(args.results)
    
    # Save JSON results
    save_json_results(image_keypoints, args.output)
    
    print("Conversion completed successfully!")
    print(f"Output format: Each keypoint is [x, y, confidence] where confidence is the model's confidence score for that keypoint detection.")


if __name__ == "__main__":
    main()
