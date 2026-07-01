#!/usr/bin/env python3
"""Merge two COCO-format pose datasets into a new tree.

Expected per source: annotations/, train2017/, val2017/. Mirrored val is not handled.

Train is always merged (one ``person_keypoints_train2017.json`` + ``train2017/``).

Val: use ``--merge-val`` to produce a single merged ``val2017/`` and
``person_keypoints_val2017.json``. Default (no flag): keep val separate —
``val2017/`` + ``person_keypoints_val2017.json`` from dataset A, and
``val2017_2/`` + ``person_keypoints_val2017_2.json`` from dataset B (copies; no id remapping).

Source directories are only read; all output is written under --output. Existing output
is removed only when --force is passed.

Merged train JSON ``info`` contains ``source1_info`` … ``sourceN_info`` for every
component source from both inputs (already-merged datasets keep their prior sources).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import re
from copy import deepcopy
from typing import Any, Dict, List

_SOURCE_INFO_RE = re.compile(r"^source(\d+)_info$")


TRAIN_JSON = "person_keypoints_train2017.json"
VAL_JSON = "person_keypoints_val2017.json"
VAL_JSON_2 = "person_keypoints_val2017_2.json"
VAL_DIR_2 = "val2017_2"


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _dump_json(path: str, data: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
        f.write("\n")


def collect_source_infos(info: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return ordered source info dicts from a COCO ``info`` block.

    If ``info`` already has ``sourceN_info`` keys, those are collected in numeric
    order. Otherwise the whole ``info`` dict is treated as a single source.
    """
    if not info:
        return []

    numbered: List[tuple[int, Dict[str, Any]]] = []
    for key, val in info.items():
        m = _SOURCE_INFO_RE.match(key)
        if m and isinstance(val, dict):
            numbered.append((int(m.group(1)), deepcopy(val)))

    if numbered:
        numbered.sort(key=lambda x: x[0])
        return [val for _, val in numbered]

    return [deepcopy(info)]


def build_merged_info(sources: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Pack source dicts into ``source1_info``, ``source2_info``, …"""
    return {f"source{i}_info": src for i, src in enumerate(sources, start=1)}


def merge_coco(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    """Merge B into A's namespace: remap all image and annotation ids from B.

    ``info`` lists every component source from A then B as ``source1_info`` …
    ``sourceN_info``.
    """
    merged_sources = collect_source_infos(a.get("info", {})) + collect_source_infos(
        b.get("info", {})
    )
    out: Dict[str, Any] = {
        "info": build_merged_info(merged_sources),
        "licenses": deepcopy(a.get("licenses", [])),
        "categories": deepcopy(a.get("categories", b.get("categories", []))),
        "images": deepcopy(a.get("images", [])),
        "annotations": deepcopy(a.get("annotations", [])),
    }

    if not b.get("images"):
        return out

    max_img = max((img["id"] for img in out["images"]), default=0)
    max_ann = max((ann["id"] for ann in out["annotations"]), default=0)

    img_id_map: Dict[int, int] = {}
    next_img = max_img + 1
    for img in b["images"]:
        old = img["id"]
        img_id_map[old] = next_img
        im = deepcopy(img)
        im["id"] = next_img
        out["images"].append(im)
        next_img += 1

    next_ann = max_ann + 1
    for ann in b.get("annotations", []):
        oid = ann["image_id"]
        if oid not in img_id_map:
            raise ValueError(
                f"Annotation references image_id={oid} not in B images"
            )
        aa = deepcopy(ann)
        aa["id"] = next_ann
        aa["image_id"] = img_id_map[oid]
        out["annotations"].append(aa)
        next_ann += 1

    # Prefer categories from A; warn if B differs
    ca, cb = a.get("categories"), b.get("categories")
    if ca and cb and json.dumps(ca, sort_keys=True) != json.dumps(cb, sort_keys=True):
        print(
            "Warning: categories differ between datasets; keeping first dataset's categories."
        )

    return out


def copy_tree_images(src_dir: str, dst_dir: str, label: str) -> int:
    """Copy all files from src_dir into dst_dir. Error if a basename already exists in dst."""
    os.makedirs(dst_dir, exist_ok=True)
    collisions: List[str] = []
    n = 0
    if not os.path.isdir(src_dir):
        return 0
    for name in sorted(os.listdir(src_dir)):
        sp = os.path.join(src_dir, name)
        if not os.path.isfile(sp):
            continue
        dp = os.path.join(dst_dir, name)
        if os.path.exists(dp):
            collisions.append(name)
            continue
        shutil.copy2(sp, dp)
        n += 1
    if collisions:
        raise RuntimeError(
            f"Filename collision copying into {label}: {collisions[:15]}"
            + (f" ... (+{len(collisions) - 15} more)" if len(collisions) > 15 else "")
        )
    return n


def assert_layout(root: str, name: str) -> None:
    ann = os.path.join(root, "annotations")
    tr = os.path.join(root, "train2017")
    va = os.path.join(root, "val2017")
    missing = [p for p in (ann, tr, va) if not os.path.isdir(p)]
    if missing:
        raise SystemExit(
            f"{name}: expected directories annotations/, train2017/, val2017/. Missing:\n"
            + "\n".join(missing)
        )
    tj = os.path.join(ann, TRAIN_JSON)
    vj = os.path.join(ann, VAL_JSON)
    if not os.path.isfile(tj):
        raise SystemExit(f"{name}: missing {tj}")
    if not os.path.isfile(vj):
        raise SystemExit(f"{name}: missing {vj}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge two COCO pose dataset roots into a new directory (read-only on sources)."
    )
    parser.add_argument("dataset_a", help="First dataset root (annotations/, train2017/, val2017/)")
    parser.add_argument("dataset_b", help="Second dataset root (same layout)")
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="New output root (created fresh; see --force)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="If --output exists, delete it before writing",
    )
    parser.add_argument(
        "--merge-val",
        action="store_true",
        default=False,
        help="Merge val into one val2017/ and person_keypoints_val2017.json (default: "
        "dataset A → val2017/ + person_keypoints_val2017.json, B → val2017_2/ + *_2.json)",
    )
    args = parser.parse_args()

    a_root = os.path.abspath(args.dataset_a)
    b_root = os.path.abspath(args.dataset_b)
    out_root = os.path.abspath(args.output)

    if a_root == b_root:
        raise SystemExit("dataset_a and dataset_b must be different paths.")
    if out_root in (a_root, b_root):
        raise SystemExit("--output must not be the same as either source.")

    assert_layout(a_root, "dataset_a")
    assert_layout(b_root, "dataset_b")

    if os.path.exists(out_root):
        if not args.force:
            raise SystemExit(f"Output exists: {out_root} (use --force to replace)")
        shutil.rmtree(out_root)

    os.makedirs(os.path.join(out_root, "annotations"), exist_ok=True)
    train_out = os.path.join(out_root, "train2017")
    val_out = os.path.join(out_root, "val2017")
    val_out_2 = os.path.join(out_root, VAL_DIR_2)
    os.makedirs(train_out, exist_ok=True)
    merge_val = args.merge_val
    if merge_val:
        os.makedirs(val_out, exist_ok=True)
    else:
        os.makedirs(val_out, exist_ok=True)
        os.makedirs(val_out_2, exist_ok=True)

    # --- train (always merged) ---
    train_a = _load_json(os.path.join(a_root, "annotations", TRAIN_JSON))
    train_b = _load_json(os.path.join(b_root, "annotations", TRAIN_JSON))
    merged_train = merge_coco(train_a, train_b)
    _dump_json(os.path.join(out_root, "annotations", TRAIN_JSON), merged_train)

    val_a_path = os.path.join(a_root, "annotations", VAL_JSON)
    val_b_path = os.path.join(b_root, "annotations", VAL_JSON)
    merged_val = None
    if merge_val:
        val_a = _load_json(val_a_path)
        val_b = _load_json(val_b_path)
        merged_val = merge_coco(val_a, val_b)
        _dump_json(os.path.join(out_root, "annotations", VAL_JSON), merged_val)
    else:
        shutil.copy2(val_a_path, os.path.join(out_root, "annotations", VAL_JSON))
        shutil.copy2(val_b_path, os.path.join(out_root, "annotations", VAL_JSON_2))

    # --- images (copy only; sources untouched) ---
    n_tr_a = copy_tree_images(os.path.join(a_root, "train2017"), train_out, "train2017 (after A)")
    n_tr_b = copy_tree_images(os.path.join(b_root, "train2017"), train_out, "train2017 (after B)")
    n_va_a = n_va_b = 0
    if merge_val:
        n_va_a = copy_tree_images(os.path.join(a_root, "val2017"), val_out, "val2017 (after A)")
        n_va_b = copy_tree_images(os.path.join(b_root, "val2017"), val_out, "val2017 (after B)")
    else:
        n_va_a = copy_tree_images(
            os.path.join(a_root, "val2017"), val_out, "val2017 (dataset A)")
        n_va_b = copy_tree_images(
            os.path.join(b_root, "val2017"), val_out_2, f"{VAL_DIR_2} (dataset B)")

    print(f"Wrote merged dataset to: {out_root}")
    print(f"  Train images copied: {n_tr_a} + {n_tr_b} = {n_tr_a + n_tr_b}")
    if merge_val:
        print(f"  Val images copied:   {n_va_a} + {n_va_b} = {n_va_a + n_va_b}")
        print(
            f"  Train annotations: {len(merged_train['images'])} images, "
            f"{len(merged_train['annotations'])} instances"
        )
        if merged_val is not None:
            print(
                f"  Val annotations:     {len(merged_val['images'])} images, "
                f"{len(merged_val['annotations'])} instances"
            )
    else:
        print(f"  Val (separate): val2017/ {n_va_a} images, {VAL_DIR_2}/ {n_va_b} images")
        print(f"  Val JSON: {VAL_JSON}, {VAL_JSON_2} (copies)")
        print(
            f"  Train annotations: {len(merged_train['images'])} images, "
            f"{len(merged_train['annotations'])} instances"
        )


if __name__ == "__main__":
    main()
