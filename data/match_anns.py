#!/usr/bin/env python3
"""Filter a base annotations JSON by images present in a reference JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def _image_ids(anns: dict[str, Any]) -> set[int]:
    images = anns.get("images", [])
    ids: set[int] = set()
    for image in images:
        image_id = image.get("id")
        if isinstance(image_id, int):
            ids.add(image_id)
    return ids


def filter_base_by_reference(
    reference_anns: dict[str, Any], base_anns: dict[str, Any]
) -> dict[str, Any]:
    keep_ids = _image_ids(reference_anns)

    filtered = dict(base_anns)
    filtered["images"] = [
        image
        for image in base_anns.get("images", [])
        if image.get("id") in keep_ids
    ]
    filtered["annotations"] = [
        ann
        for ann in base_anns.get("annotations", [])
        if ann.get("image_id") in keep_ids
    ]
    return filtered


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create a filtered copy of base annotations, keeping only images "
            "present in reference annotations."
        )
    )
    parser.add_argument(
        "--reference",
        required=True,
        help="Reference annotations JSON (drives which image ids are kept)",
    )
    parser.add_argument(
        "--base",
        required=True,
        help="Base annotations JSON to copy and filter",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output path for filtered JSON",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).parent

    reference_path = Path(args.reference)
    if not reference_path.is_absolute():
        reference_path = script_dir / reference_path

    base_path = Path(args.base)
    if not base_path.is_absolute():
        base_path = script_dir / base_path

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = script_dir / output_path

    reference = _load_json(reference_path)
    base = _load_json(base_path)

    filtered = filter_base_by_reference(reference, base)
    _save_json(output_path, filtered)

    print(f"Reference image ids: {len(_image_ids(reference))}")
    print(f"Base images: {len(base.get('images', []))}")
    print(f"Base annotations: {len(base.get('annotations', []))}")
    print(f"Filtered images: {len(filtered.get('images', []))}")
    print(f"Filtered annotations: {len(filtered.get('annotations', []))}")
    print(f"Wrote filtered annotations to: {output_path}")


if __name__ == "__main__":
    main()
