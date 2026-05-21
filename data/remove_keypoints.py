#!/usr/bin/env python3
"""Zero out selected COCO keypoints in a copy of an annotations JSON file.

Each listed keypoint index is set to [0, 0, 0] for [x, y, visibility] in every
annotation. The input file is not modified.

Usage:
    python remove_keypoints.py path/to/anns.json --kp-indices 0 1 2 3 4
    python remove_keypoints.py path/to/anns.json --kp-indices 0,1,2,3,4
"""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any


def _parse_kp_indices(values: list[str]) -> list[int]:
    indices: list[int] = []
    for value in values:
        for part in value.split(","):
            part = part.strip()
            if not part:
                continue
            idx = int(part)
            if idx < 0:
                raise ValueError(f"Keypoint index must be non-negative, got {idx}")
            indices.append(idx)
    if not indices:
        raise ValueError("At least one keypoint index is required")
    return sorted(set(indices))


def _output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_no_head_kps{input_path.suffix}")


def _zero_keypoints(
    keypoints: list[float],
    kp_indices: list[int],
    num_kpts: int = 17,
) -> list[float]:
    expected_len = num_kpts * 3
    if len(keypoints) != expected_len:
        raise ValueError(
            f"Expected {expected_len} keypoint values ({num_kpts} keypoints), "
            f"got {len(keypoints)}"
        )

    new_kpts = list(keypoints)
    for idx in kp_indices:
        if idx >= num_kpts:
            raise ValueError(
                f"Keypoint index {idx} out of range for {num_kpts} keypoints"
            )
        start = idx * 3
        new_kpts[start : start + 3] = [0.0, 0.0, 0.0]
    return new_kpts


def _count_visible_keypoints(keypoints: list[float]) -> int:
    return sum(1 for i in range(2, len(keypoints), 3) if keypoints[i] > 0)


def remove_keypoints(
    anns: dict[str, Any],
    kp_indices: list[int],
    num_kpts: int = 17,
) -> dict[str, Any]:
    out = deepcopy(anns)
    n_changed = 0

    for ann in out.get("annotations", []):
        keypoints = ann.get("keypoints")
        if not keypoints:
            continue
        ann["keypoints"] = _zero_keypoints(keypoints, kp_indices, num_kpts=num_kpts)
        if "num_keypoints" in ann:
            ann["num_keypoints"] = _count_visible_keypoints(ann["keypoints"])
        n_changed += 1

    return out, n_changed


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Write a copy of a COCO annotations JSON with selected keypoints "
            "zeroed to [0, 0, 0]."
        )
    )
    parser.add_argument(
        "json_path",
        type=Path,
        help="Path to the input COCO annotation JSON file",
    )
    parser.add_argument(
        "--kp-indices",
        nargs="+",
        required=True,
        metavar="INDEX",
        help="0-based keypoint indices to zero (space- or comma-separated)",
    )
    parser.add_argument(
        "--num-keypoints",
        type=int,
        default=17,
        help="Number of keypoints per person (default: 17 for COCO)",
    )
    args = parser.parse_args()

    input_path = args.json_path.resolve()
    if not input_path.is_file():
        print(f"Error: File not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    try:
        kp_indices = _parse_kp_indices(args.kp_indices)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    output_path = _output_path(input_path)

    try:
        with input_path.open(encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        print(f"Error: Invalid JSON: {exc}", file=sys.stderr)
        sys.exit(1)

    try:
        updated, n_anns = remove_keypoints(
            data,
            kp_indices,
            num_kpts=args.num_keypoints,
        )
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(updated, f, indent=2)
        f.write("\n")

    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"Zeroed keypoint indices: {kp_indices}")
    print(f"Updated annotations: {n_anns}")


if __name__ == "__main__":
    main()
