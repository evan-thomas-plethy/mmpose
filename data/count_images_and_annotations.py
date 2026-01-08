#!/usr/bin/env python3
"""
Count images and annotations in a COCO annotation file.

Usage:
    python count_images_and_annotations.py path/to/annotations.json
    python count_images_and_annotations.py coco/annotations/person_keypoints_train2017.json
"""

import argparse
import json
import sys


def count_coco(json_path: str):
    """Count images and annotations in a COCO JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    num_images = len(data.get("images", []))
    num_annotations = len(data.get("annotations", []))
    
    print(f"Images: {num_images}")
    print(f"Annotations: {num_annotations}")


def main():
    parser = argparse.ArgumentParser(description="Count images and annotations in a COCO annotation file.")
    parser.add_argument("json_path", type=str, help="Path to the COCO annotation JSON file")
    
    args = parser.parse_args()
    
    try:
        count_coco(args.json_path)
    except FileNotFoundError:
        print(f"Error: File not found: {args.json_path}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

