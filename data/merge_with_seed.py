#!/usr/bin/env python3
"""Merge component1 (COCO root) with a seeded sample from component2.

Component1 must be a standard COCO layout (annotations/, train2017/, val2017/).
Val in the output is exactly component1's val (train = component1 train + sample).
If component1 has mirrored val (``val2017_mirrored/`` and
``annotations/person_keypoints_val2017_mirrored.json``), those are copied too.

Component2 may be:
  - COCO layout: sample ``n`` images/anns from ``person_keypoints_train2017.json`` + train2017/
  - Flat layout: one ``*.json`` at the root + ``frames/`` (e.g. LSPe)

After merge, ``info.source2_info`` gets ``seed`` and ``n`` (after ``description``).
Empty ``val2017_2/`` from the staging placeholder is removed by default.

Usage:
    python merge_with_seed.py \\
        data/coco_ground_based_exercises_v6 \\
        data/LSPe \\
        --seed 1 --n-component2 500 \\
        --output data/merged_gbe_v6_lspe/gbe_v6_lspe_seed1_n500 \\
        --force
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from merge_datasets import (
    TRAIN_JSON,
    VAL_JSON,
    VAL_DIR_2,
    VAL_JSON_2,
    assert_layout,
    copy_tree_images,
    merge_coco,
    _dump_json,
    _load_json,
)

SOURCE2_INFO_FIELD_ORDER = (
    "year",
    "version",
    "description",
    "seed",
    "n",
    "contributor",
    "s3_uri",
    "frames_s3_uri",
)

PLACEHOLDER_VAL = {
    "info": {"description": "placeholder val for merge_datasets layout"},
    "licenses": [],
    "images": [],
    "annotations": [],
}

VAL_JSON_MIRRORED = "person_keypoints_val2017_mirrored.json"
VAL_DIR_MIRRORED = "val2017_mirrored"


def _resolve_component2(root: Path) -> tuple[Path, Path]:
    """Return (annotation_json, frames_directory) for sampling."""
    ann_dir = root / "annotations"
    train_json = ann_dir / TRAIN_JSON
    train_dir = root / "train2017"
    if train_json.is_file() and train_dir.is_dir():
        return train_json, train_dir

    frames_dir = root / "frames"
    if frames_dir.is_dir():
        json_files = sorted(
            p for p in root.glob("*.json") if p.is_file()
        )
        if len(json_files) == 1:
            return json_files[0], frames_dir
        if len(json_files) == 0:
            raise SystemExit(
                f"component2 {root}: expected one *.json beside frames/, found none"
            )
        raise SystemExit(
            f"component2 {root}: expected one *.json beside frames/, found: "
            + ", ".join(p.name for p in json_files)
        )

    raise SystemExit(
        f"component2 {root}: unrecognized layout. Expected COCO "
        "(annotations/person_keypoints_train2017.json + train2017/) or "
        "flat (*.json + frames/)."
    )


def _stage_component2_sample(
    component2: Path,
    seed: int,
    n: int,
    staging: Path,
) -> None:
    ann_path, frames_dir = _resolve_component2(component2)

    with open(ann_path, encoding="utf-8") as f:
        data = json.load(f)

    images = data.get("images", [])
    if n > len(images):
        raise SystemExit(
            f"Requested n_component2={n} but {component2} only has {len(images)} images"
        )

    rng = random.Random(seed)
    sampled_imgs = rng.sample(images, n)
    sampled_ids = {img["id"] for img in sampled_imgs}
    anns = [a for a in data.get("annotations", []) if a["image_id"] in sampled_ids]

    img_id_map: dict[int, int] = {}
    new_images: list[dict[str, Any]] = []
    for i, img in enumerate(sampled_imgs, start=1):
        img_id_map[img["id"]] = i
        row = deepcopy(img)
        row["id"] = i
        row["file_name"] = os.path.basename(row["file_name"].replace("\\", "/"))
        new_images.append(row)

    new_anns: list[dict[str, Any]] = []
    for j, ann in enumerate(anns, start=1):
        aa = deepcopy(ann)
        aa["id"] = j
        aa["image_id"] = img_id_map[ann["image_id"]]
        new_anns.append(aa)

    train_payload: dict[str, Any] = {
        "info": deepcopy(data.get("info", {})),
        "licenses": deepcopy(data.get("licenses", [])),
        "categories": deepcopy(data.get("categories", [])),
        "images": new_images,
        "annotations": new_anns,
    }
    val_payload = deepcopy(PLACEHOLDER_VAL)
    val_payload["categories"] = deepcopy(data.get("categories", []))

    if staging.exists():
        shutil.rmtree(staging)
    ann_out = staging / "annotations"
    train_out = staging / "train2017"
    val_out = staging / "val2017"
    ann_out.mkdir(parents=True)
    train_out.mkdir(parents=True)
    val_out.mkdir(parents=True)

    _dump_json(str(ann_out / TRAIN_JSON), train_payload)
    _dump_json(str(ann_out / VAL_JSON), val_payload)

    copied, missing = 0, 0
    for img in new_images:
        src = frames_dir / img["file_name"]
        dst = train_out / img["file_name"]
        if src.is_file():
            shutil.copy2(src, dst)
            copied += 1
        else:
            missing += 1
            print(f"   ⚠️ Missing frame: {src}")

    print(
        f"Staged component2 sample: {len(new_images)} images, {len(new_anns)} anns, "
        f"copied={copied}, missing={missing}"
    )


def _patch_source2_info(train_json_path: Path, seed: int, n: int) -> None:
    data = _load_json(str(train_json_path))
    info = data.get("info", {})
    s2 = dict(info.get("source2_info", {}))
    s2["seed"] = seed
    s2["n"] = n
    new_s2: dict[str, Any] = {}
    for key in SOURCE2_INFO_FIELD_ORDER:
        if key in s2:
            new_s2[key] = s2[key]
    for key, val in s2.items():
        if key not in new_s2:
            new_s2[key] = val
    info["source2_info"] = new_s2
    data["info"] = info
    _dump_json(str(train_json_path), data)


def _copy_component1_mirrored_val(component1: Path, out_root: Path) -> int:
    """Copy mirrored val images and annotations from component1 when present."""
    src_json = component1 / "annotations" / VAL_JSON_MIRRORED
    src_dir = component1 / VAL_DIR_MIRRORED
    if not src_json.is_file() and not src_dir.is_dir():
        return 0
    if not src_json.is_file() or not src_dir.is_dir():
        missing = []
        if not src_json.is_file():
            missing.append(str(src_json))
        if not src_dir.is_dir():
            missing.append(str(src_dir))
        raise SystemExit(
            f"component1 mirrored val is incomplete (expected both json and dir). "
            f"Missing: {', '.join(missing)}"
        )

    ann_out = out_root / "annotations"
    ann_out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_json, ann_out / VAL_JSON_MIRRORED)
    n = copy_tree_images(
        str(src_dir),
        str(out_root / VAL_DIR_MIRRORED),
        f"{VAL_DIR_MIRRORED} (component1)",
    )
    return n


def _remove_val2_artifacts(out_root: Path) -> None:
    val2_dir = out_root / VAL_DIR_2
    val2_json = out_root / "annotations" / VAL_JSON_2
    if val2_dir.is_dir():
        shutil.rmtree(val2_dir)
    if val2_json.is_file():
        val2_json.unlink()


def merge_with_seed(
    component1: Path,
    component2: Path,
    seed: int,
    n_component2: int,
    output: Path,
    *,
    force: bool = False,
    keep_val2: bool = False,
    staging_dir: Path | None = None,
    keep_staging: bool = False,
) -> None:
    c1 = component1.resolve()
    c2 = component2.resolve()
    out = output.resolve()

    if c1 == c2:
        raise SystemExit("component1 and component2 must be different paths.")
    if out in (c1, c2):
        raise SystemExit("--output must not be the same as either component.")

    assert_layout(str(c1), "component1")

    own_staging = staging_dir is None
    if own_staging:
        staging = Path(tempfile.mkdtemp(prefix="merge_with_seed_"))
    else:
        staging = staging_dir.resolve()
        if staging.exists():
            shutil.rmtree(staging)

    try:
        _stage_component2_sample(c2, seed, n_component2, staging)
        assert_layout(str(staging), "staged component2")

        if out.exists():
            if not force:
                raise SystemExit(f"Output exists: {out} (use --force to replace)")
            shutil.rmtree(out)

        out.mkdir(parents=True)
        ann_out = out / "annotations"
        train_out = out / "train2017"
        val_out = out / "val2017"
        val_out_2 = out / VAL_DIR_2
        ann_out.mkdir()
        train_out.mkdir()
        val_out.mkdir()
        val_out_2.mkdir()

        train_a = _load_json(str(c1 / "annotations" / TRAIN_JSON))
        train_b = _load_json(str(staging / "annotations" / TRAIN_JSON))
        merged_train = merge_coco(train_a, train_b)
        _dump_json(str(ann_out / TRAIN_JSON), merged_train)

        shutil.copy2(c1 / "annotations" / VAL_JSON, ann_out / VAL_JSON)
        shutil.copy2(
            staging / "annotations" / VAL_JSON,
            ann_out / VAL_JSON_2,
        )

        n_tr_a = copy_tree_images(
            str(c1 / "train2017"), str(train_out), "train2017 (component1)"
        )
        n_tr_b = copy_tree_images(
            str(staging / "train2017"), str(train_out), "train2017 (component2 sample)"
        )
        n_va_a = copy_tree_images(
            str(c1 / "val2017"), str(val_out), "val2017 (component1)"
        )
        n_va_b = copy_tree_images(
            str(staging / "val2017"), str(val_out_2), f"{VAL_DIR_2} (placeholder)"
        )

        n_va_mirrored = _copy_component1_mirrored_val(c1, out)

        _patch_source2_info(ann_out / TRAIN_JSON, seed, n_component2)

        if not keep_val2:
            _remove_val2_artifacts(out)

        print(f"\nWrote merged dataset to: {out}")
        print(f"  Train images copied: {n_tr_a} + {n_tr_b} = {n_tr_a + n_tr_b}")
        print(
            f"  Train annotations: {len(merged_train['images'])} images, "
            f"{len(merged_train['annotations'])} instances"
        )
        print(f"  Val (component1 only): val2017/ {n_va_a} images")
        if n_va_mirrored:
            print(
                f"  Mirrored val (component1): {VAL_DIR_MIRRORED}/ "
                f"{n_va_mirrored} images"
            )
        print(f"  source2_info: seed={seed}, n={n_component2}")
        if keep_val2:
            print(f"  (kept empty {VAL_DIR_2}/ and {VAL_JSON_2})")
    finally:
        if own_staging and staging.exists() and not keep_staging:
            shutil.rmtree(staging)
        elif not own_staging and staging.exists() and not keep_staging:
            shutil.rmtree(staging)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Merge component1 COCO dataset with a seeded random sample of n images "
            "from component2 (COCO train split or flat json+frames)."
        )
    )
    parser.add_argument(
        "component1",
        help="First dataset root (val preserved; train merged in)",
    )
    parser.add_argument(
        "component2",
        help="Second source: COCO root or flat dir with *.json + frames/",
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help="RNG seed for sampling component2",
    )
    parser.add_argument(
        "--n-component2",
        type=int,
        required=True,
        help="Number of images/annotations to sample from component2",
    )
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Output COCO dataset root",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace existing --output",
    )
    parser.add_argument(
        "--keep-val2",
        action="store_true",
        help="Keep empty val2017_2/ and person_keypoints_val2017_2.json",
    )
    parser.add_argument(
        "--staging-dir",
        type=str,
        default=None,
        help="Reuse this staging path instead of a temp directory",
    )
    parser.add_argument(
        "--keep-staging",
        action="store_true",
        help="Do not delete staging directory after merge",
    )
    args = parser.parse_args()

    staging = Path(args.staging_dir) if args.staging_dir else None
    merge_with_seed(
        Path(args.component1),
        Path(args.component2),
        args.seed,
        args.n_component2,
        Path(args.output),
        force=args.force,
        keep_val2=args.keep_val2,
        staging_dir=staging,
        keep_staging=args.keep_staging,
    )


if __name__ == "__main__":
    main()
