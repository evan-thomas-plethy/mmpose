#!/usr/bin/env python3
"""Scale COCO-format bounding boxes in an annotation JSON (about bbox center).

Each annotation ``bbox`` is [x, y, width, height]. Default scale factor is 1.2.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List


def scale_bbox_xywh(bbox: List[float], scale: float) -> List[float]:
    x, y, w, h = bbox
    cx = x + w / 2.0
    cy = y + h / 2.0
    w2 = w * scale
    h2 = h * scale
    return [cx - w2 / 2.0, cy - h2 / 2.0, w2, h2]


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scale each COCO bbox in an annotations JSON by a factor (default 1.2)."
    )
    parser.add_argument(
        "ann_file",
        help="Path to COCO-style JSON (must contain an 'annotations' list with 'bbox').",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.2,
        help="Uniform scale applied to width and height about bbox center (default: 1.2).",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output JSON path. Default: <stem>_bboxes_scaled.json next to the input file.",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.ann_file):
        raise SystemExit(f"Not found: {args.ann_file}")

    data = load_json(args.ann_file)
    anns = data.get("annotations")
    if not isinstance(anns, list):
        raise SystemExit("JSON must contain an 'annotations' array.")

    for ann in anns:
        if "bbox" not in ann:
            continue
        bbox = ann["bbox"]
        if not bbox or len(bbox) != 4:
            continue
        new_bbox = scale_bbox_xywh([float(x) for x in bbox], args.scale)
        ann["bbox"] = new_bbox
        # COCO area for axis-aligned box
        ann["area"] = float(new_bbox[2] * new_bbox[3])

    out_path = args.output
    if out_path is None:
        base, ext = os.path.splitext(args.ann_file)
        out_path = f"{base}_bboxes_scaled{ext or '.json'}"

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
        f.write("\n")

    print(f"Wrote {len(anns)} annotations to {out_path} (bbox scale={args.scale}).")


if __name__ == "__main__":
    main()
