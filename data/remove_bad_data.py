#!/usr/bin/env python3
"""Remove bad video-prefix frames (and linked annotations) from anns JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


VIDEONAMES = [
    "Prone_Ball_Exercise_Prone_Scapula_W_to_Y",
    "Prone_Bilateral_Scapular_Retraction_with_Band",
    "Prone_Prone_Hip_Abduction",
    "Side_Lying_Subscapularis_Release",
    "Side_Lying_Table_sidelying_IT_band_stretch",
    "Supine_Doorway_Hamstring_Stretch",
    "Supine_Supine_Cross_Body_Shoulder_Stretch",
    "Supine_Supine_Elbow_Extension",
    "Supine_Supine_Hip_Abduction",
    "Supine_Supine_Shoulder_Circles_Alphabet",
    "Supine_Supine_Shoulder_Internal_Rotation",
    "Supine_Upper_Cervical_Deep_Neck_Flexor_Strengthening",
]

def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def _output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_cleaned{input_path.suffix}")


def _matching_prefix(filename: str, prefixes: list[str]) -> str | None:
    base_name = Path(filename).name
    for prefix in prefixes:
        frame_prefix = f"{prefix}_frame_"
        if base_name.startswith(frame_prefix):
            return prefix
    return None


def remove_bad_frames(anns: dict[str, Any], prefixes: list[str]) -> tuple[dict[str, Any], set[str]]:
    remove_image_ids: set[int] = set()
    matched_prefixes: set[str] = set()

    for image in anns.get("images", []):
        image_id = image.get("id")
        file_name = image.get("file_name")
        if not isinstance(image_id, int) or not isinstance(file_name, str):
            continue

        matched = _matching_prefix(file_name, prefixes)
        if matched is not None:
            remove_image_ids.add(image_id)
            matched_prefixes.add(matched)

    filtered = dict(anns)
    filtered["images"] = [
        image for image in anns.get("images", []) if image.get("id") not in remove_image_ids
    ]
    filtered["annotations"] = [
        ann
        for ann in anns.get("annotations", [])
        if ann.get("image_id") not in remove_image_ids
    ]
    return filtered, matched_prefixes


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Remove images/annotations for frames that start with one of the "
            "bad videoname prefixes."
        )
    )
    parser.add_argument("anns_json", help="Path to annotations JSON to clean")
    args = parser.parse_args()

    anns_path = Path(args.anns_json)
    anns = _load_json(anns_path)
    cleaned, matched_prefixes = remove_bad_frames(anns, VIDEONAMES)
    out_path = _output_path(anns_path)
    _save_json(out_path, cleaned)

    print(f"Input images: {len(anns.get('images', []))}")
    print(f"Input annotations: {len(anns.get('annotations', []))}")
    print(f"Output images: {len(cleaned.get('images', []))}")
    print(f"Output annotations: {len(cleaned.get('annotations', []))}")
    print(
        f"Found and removed at least one matching frame for "
        f"{len(matched_prefixes)}/{len(VIDEONAMES)} listed video names."
    )
    print(f"Wrote cleaned annotations to: {out_path}")


if __name__ == "__main__":
    main()