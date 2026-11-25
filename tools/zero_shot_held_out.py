#!/usr/bin/env python
"""AP of the pretrained checkpoint and the four v7 fold models on full held-out vals.

The pretrained checkpoint is scored on every
full_held_out_<subject>_ske.json and full_held_out_<subject>_gbe.json that
exists. Each sweep run is scored only on the ske and gbe files for the two
subjects that run held out. Missing files are skipped. The CSV has one row per subject and split, with
finetuned, pretrained, and old_ground_based AP. Each run also writes a COCO
JSON of the predicted keypoints to
data/predictions/<model>_<subject>_<split>.json for overlay_dataset.py.

Example:
  python tools/zero_shot_held_out.py \\
    --sweep experiments/rtmpose-m_single_model/sweep_config_app_vids_gbe_v7.yaml
"""

import argparse
import copy
import csv
import json
import os
import os.path as osp
import re
from pathlib import Path

import numpy as np
import torch
import yaml
from mmengine.config import Config
from mmengine.evaluator import Evaluator
from mmengine.registry import init_default_scope
from mmengine.runner import Runner, load_checkpoint

import mmdet.models  # noqa: F401  CSPNeXt backbone
import mmpose.datasets  # noqa: F401
import mmpose.evaluation  # noqa: F401
import mmpose.models  # noqa: F401
from mmpose.registry import MODELS


DEFAULT_SWEEP = (
    "experiments/rtmpose-m_single_model/sweep_config_app_vids_gbe_v7.yaml"
)
DEFAULT_OUT = "data/zero_shot_full_held_out_ap.csv"
PRED_DIR = Path("data/predictions")
FOLDS_DIR = Path("data/App_Vids_v2_GBE_v4_folds")
HELD_OUT_FRAMES = "../../../App_Vids_v1_GBE_v1_lyingperson_InfiniteForm_frames/"
SUBJECT_RE = re.compile(r"^held_out_(?P<subject>[a-z]+)_")
CSV_FIELDS = ("subject", "split", "finetuned", "pretrained", "old_ground_based")
OLD_GROUND_BASED = (
    "work_dirs/rtmpose-m_merged_gbe_v9_gbe_app_vids_various_versions_sweep/"
    "merged_gbe_v9_app_vids_v4_lyingperson_full_infiniteform_seed4_n736_"
    "unfreeze_full_ld0.5_lr1e-3"
)


def resolve_checkpoint(path):
    """A .pth file, or a run directory containing best_coco_AP_*.pth."""
    if osp.isfile(path):
        return path
    if not osp.isdir(path):
        raise FileNotFoundError(path)
    best = sorted(Path(path).glob("best_coco_AP_*.pth"))
    if len(best) == 1:
        return str(best[0])
    if len(best) > 1:
        raise FileNotFoundError(
            f"{path} has {len(best)} best_coco_AP checkpoints; pass one file"
        )
    epochs = sorted(Path(path).glob("epoch_*.pth"))
    if not epochs:
        raise FileNotFoundError(f"No checkpoint in {path}")
    return str(epochs[-1])


def full_held_out_path(subject, split):
    return (
        FOLDS_DIR
        / f"{subject}_held_out"
        / "held_out"
        / "annotations"
        / f"full_held_out_{subject}_{split}.json"
    )


def existing_evals(subjects):
    """ske then gbe, only when that full held-out JSON exists."""
    found = []
    for subject in subjects:
        for split in ("ske", "gbe"):
            path = full_held_out_path(subject, split)
            if path.is_file():
                found.append((subject, split, path))
    return found


def subjects_for_experiment(overrides):
    """Short names of the two held-out people, in hook order."""
    subjects = []
    indexed = []
    for key, value in overrides.items():
        parts = key.split(".")
        if len(parts) == 3 and parts[0] == "custom_hooks" and parts[2] == "metric_prefix":
            match = SUBJECT_RE.match(str(value))
            if match:
                indexed.append((int(parts[1]), match.group("subject")))
    for _, subject in sorted(indexed):
        if subject not in subjects:
            subjects.append(subject)
    return subjects


def all_subjects(experiments):
    subjects = []
    for experiment in experiments:
        for subject in subjects_for_experiment(experiment.get("overrides", {})):
            if subject not in subjects:
                subjects.append(subject)
    return subjects


def build_loader(template, data_root, ann_file, img_prefix):
    loader_cfg = copy.deepcopy(template)
    loader_cfg.dataset.data_root = data_root
    loader_cfg.dataset.ann_file = ann_file
    loader_cfg.dataset.data_prefix = dict(img=img_prefix)
    loader_cfg.setdefault("drop_last", False)
    loader_cfg.setdefault("sampler", dict(type="DefaultSampler", shuffle=False))
    loader_cfg["persistent_workers"] = False
    return Runner.build_dataloader(loader_cfg)


def score_ap(model, loader, ann_file):
    evaluator = Evaluator(dict(type="CocoMetric", ann_file=str(ann_file), prefix="coco"))
    dataset_meta = getattr(loader.dataset, "metainfo", None)
    if dataset_meta:
        for metric in evaluator.metrics:
            metric.dataset_meta = dataset_meta
    model.eval()
    with torch.no_grad():
        for data_batch in loader:
            outputs = model.val_step(data_batch)
            evaluator.process(data_samples=outputs, data_batch=data_batch)
    predictions = list(evaluator.metrics[0].results)
    metrics = evaluator.evaluate(len(loader.dataset))
    ap = metrics.get("coco/AP")
    if ap is None:
        raise RuntimeError(f"CocoMetric did not return AP for {ann_file}")
    return float(ap), predictions


def prediction_coco(source_ann, results):
    """COCO JSON whose annotations are the predicted keypoints, for overlay_dataset.py."""
    with open(source_ann, encoding="utf-8") as handle:
        source = json.load(handle)
    annotations = []
    ann_id = 1
    for pred, _gt in results:
        keypoints = np.asarray(pred["keypoints"])
        scores = np.asarray(pred["keypoint_scores"])
        bboxes = pred.get("bbox")
        image_id = int(pred["img_id"])
        category_id = int(pred.get("category_id", 1))
        for index in range(len(keypoints)):
            stacked = np.concatenate(
                [keypoints[index], scores[index][:, None]], axis=1
            )
            flat = stacked.reshape(-1).tolist()
            ann = {
                "id": ann_id,
                "image_id": image_id,
                "category_id": category_id,
                "keypoints": flat,
                "num_keypoints": int((scores[index] > 0).sum()),
                "score": float(np.mean(scores[index])),
            }
            if bboxes is not None:
                ann["bbox"] = np.asarray(bboxes[index]).reshape(-1).tolist()
            annotations.append(ann)
            ann_id += 1
    return {
        "images": source.get("images", []),
        "annotations": annotations,
        "categories": source.get("categories", []),
    }


def write_predictions(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle)
        handle.write("\n")


def load_model(cfg, checkpoint, device):
    model = MODELS.build(cfg.model)
    load_checkpoint(model, checkpoint, map_location="cpu")
    model.to(device)
    model.eval()
    return model


def dataloader_template(cfg):
    for hook in cfg.custom_hooks:
        if hook.get("type") == "CustomDatasetHook":
            return hook.dataloader
    raise RuntimeError("Config has no CustomDatasetHook to copy a val dataloader from")


def write_csv(path, row_keys, cells):
    os.makedirs(osp.dirname(osp.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for subject, split in row_keys:
            cell = cells[(subject, split)]
            writer.writerow({
                "subject": subject,
                "split": split,
                "finetuned": cell.get("finetuned", ""),
                "pretrained": cell.get("pretrained", ""),
                "old_ground_based": cell.get("old_ground_based", ""),
            })


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", default=DEFAULT_SWEEP)
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Pretrained .pth or URL. Defaults to load_from in the sweep base config.",
    )
    parser.add_argument("--out", default=DEFAULT_OUT, help="CSV of AP scores")
    parser.add_argument(
        "--old-ground-based",
        default=OLD_GROUND_BASED,
        help="Checkpoint or run directory for the old ground-based model. "
        "Scored on every full held-out ske and gbe file.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.sweep, encoding="utf-8") as handle:
        sweep = yaml.safe_load(handle)
    cfg = Config.fromfile(sweep["base_config"])
    pretrained = args.checkpoint or cfg.load_from
    if not pretrained:
        raise SystemExit("Pass --checkpoint or set load_from in the config")

    experiments = sweep["experiments"]
    if len(experiments) != 4:
        raise SystemExit(f"Expected 4 sweep models, found {len(experiments)}")
    run_root = Path("work_dirs") / sweep["experiment_name"]
    fold_models = []
    for experiment in experiments:
        run_dir = run_root / experiment["name"]
        subjects = subjects_for_experiment(experiment.get("overrides", {}))
        if len(subjects) != 2:
            raise SystemExit(
                f"{experiment['name']} held out {subjects}; expected 2 subjects"
            )
        fold_models.append({
            "name": experiment["name"],
            "checkpoint": resolve_checkpoint(str(run_dir)),
            "subjects": subjects,
        })
    old_checkpoint = resolve_checkpoint(args.old_ground_based)

    init_default_scope("mmpose")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    template = dataloader_template(cfg)
    row_keys = []
    cells = {}

    def record(column, subject, split, ap):
        key = (subject, split)
        if key not in cells:
            row_keys.append(key)
            cells[key] = {}
        cells[key][column] = f"{ap:.4f}"
        write_csv(args.out, row_keys, cells)

    def run_one(column, pred_name, model, subject, split, ann_path):
        data_root = ann_path.parents[1]
        loader = build_loader(
            template,
            str(data_root),
            f"annotations/{ann_path.name}",
            HELD_OUT_FRAMES,
        )
        if len(loader.dataset) == 0:
            raise SystemExit(f"{ann_path} has no images")
        ap, results = score_ap(model, loader, ann_path.resolve())
        pred_path = PRED_DIR / f"{pred_name}_{subject}_{split}.json"
        write_predictions(pred_path, prediction_coco(ann_path, results))
        print(
            f"{column}  {subject}  {split}  AP {ap:.4f}  "
            f"images={len(loader.dataset)}  {pred_path}"
        )
        record(column, subject, split, ap)

    evals = existing_evals(all_subjects(experiments))

    print(f"pretrained: {pretrained}")
    model = load_model(cfg, pretrained, device)
    for subject, split, ann_path in evals:
        run_one("pretrained", "pretrained", model, subject, split, ann_path)
    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    print(f"old_ground_based: {old_checkpoint}")
    model = load_model(cfg, old_checkpoint, device)
    for subject, split, ann_path in evals:
        run_one("old_ground_based", "old_ground_based", model, subject, split, ann_path)
    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    for fold in fold_models:
        print(f"{fold['name']}: {fold['checkpoint']}")
        model = load_model(cfg, fold["checkpoint"], device)
        for subject, split, ann_path in existing_evals(fold["subjects"]):
            run_one("finetuned", fold["name"], model, subject, split, ann_path)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    print(f"csv: {args.out}")


if __name__ == "__main__":
    main()
