import json
import os

"""
Removes path prefixes from file_name in annotation JSON.

Usage:
    python remove_filename_prefix_annotations.py Heel_Slides/v4/HeelSlides_V4.json

This will create: Heel_Slides/v2/HeelSlides_V2_normalized.json
"""


def remove_filename_prefix(json_path):
    """
    Removes path prefixes from file_name values in an annotation JSON.
    
    Args:
        json_path: Path to the annotation JSON file
    
    Creates:
        A new JSON file with '_normalized' suffix in the same directory
    """
    print(f"📂 Loading: {json_path}")
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Check if normalization is needed
    needs_normalization = False
    for img in data.get("images", []):
        if os.path.basename(img["file_name"]) != img["file_name"]:
            needs_normalization = True
            break
    
    if not needs_normalization:
        print("✅ No normalization needed - file_name values are already basenames")
        return
    
    # Normalize file_name to just basename
    normalized_count = 0
    for img in data["images"]:
        original = img["file_name"]
        normalized = os.path.basename(original)
        if original != normalized:
            img["file_name"] = normalized
            normalized_count += 1
    
    # Create output path
    base, ext = os.path.splitext(json_path)
    output_path = f"{base}_normalized{ext}"
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=4)
    
    print(f"✅ Normalized {normalized_count} file_name entries")
    print(f"   Saved to: {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Remove path prefixes from file_name in annotation JSON")
    parser.add_argument("json_path", help="Path to annotation JSON file")
    
    args = parser.parse_args()
    
    remove_filename_prefix(args.json_path)

