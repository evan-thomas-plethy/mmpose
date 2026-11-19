#!/usr/bin/env python
"""Per-subject COCO AP on an existing val set. No training.

Each --compare runs two checkpoints (the full train set and the subject25
train set) on the same val JSON, then scores NinaVani, JamieDoe, JoeyReyna,
and YordanosWoldebirhan separately. Subject ids come from the subjects CSV
with the same fallback as split_dataset.subject_for_video. The val images
are loaded the same way as tools/test.py --val-annotations and
--val-images-dir: data_root is the frames directory and data_prefix.img is
empty, so file_name is the basename.

Example:
  python tools/per_subject_ap.py \\
    --config configs/body_2d_keypoint/rtmpose/coco/rtmpose-m_8xb256-420e_coco-256x192_finetune.py \\
    --subjects-path /path/to/users.csv \\
    --images-dir data/App_Vids_v1_GBE_v1_frames \\
    --compare v2_gbe_v4 \\
      data/App_Vids_v2_GBE_v4_val10/annotations/person_keypoints_val2017.json \\
      work_dirs/rtmpose-m_single_model/app_vids_v2_gbe_v4_val10 \\
      work_dirs/rtmpose-m_single_model/app_vids_v2_gbe_v4_val10_subject25
"""

import argparse
import contextlib
import importlib.util
import io
import json
import os
import os.path as osp
import sys
import tempfile
from pathlib import Path

from mmengine.config import Config
from mmengine.runner import Runner
from xtcocotools.coco import COCO


DEFAULT_SUBJECTS = (
    "NinaVani",
    "JamieDoe",
    "JoeyReyna",
    "YordanosWoldebirhan",
)
_SKIP_HOOKS = (
    "MirroredValHook",
    "CustomDatasetHook",
    "GeneralValHook",
    "EMAHook",
    "mmdet.PipelineSwitchHook",
)


def _load_split_dataset():
    path = Path(__file__).resolve().parents[1] / "data" / "split_dataset.py"
    spec = importlib.util.spec_from_file_location("split_dataset", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def video_of_image(image):
    video_id = image.get("video_id")
    if video_id:
        return str(video_id)
    filename = os.path.basename(str(image.get("file_name", "")).replace("\\", "/"))
    if "_frame_" in filename:
        return filename.split("_frame_")[0]
    return filename


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


def subject_image_ids(val_json, video_subjects, subject_for_video):
    with open(val_json, encoding="utf-8") as handle:
        data = json.load(handle)
    by_subject = {}
    missing = []
    for image in data["images"]:
        video_id = video_of_image(image)
        if video_subjects is not None and video_id not in video_subjects:
            missing.append(video_id)
        subject = subject_for_video(video_id, video_subjects)
        by_subject.setdefault(subject, set()).add(image["id"])
    return data, by_subject, sorted(set(missing))


def write_subset(data, image_ids, dest):
    keep = set(image_ids)
    subset = {
        "info": data.get("info", {}),
        "licenses": data.get("licenses", []),
        "categories": data["categories"],
        "images": [image for image in data["images"] if image["id"] in keep],
        "annotations": [
            ann for ann in data["annotations"] if ann.get("image_id") in keep
        ],
    }
    with open(dest, "w", encoding="utf-8") as handle:
        json.dump(subset, handle)
    return len(subset["images"]), len(subset["annotations"])


def build_cfg(config_path, checkpoint, val_json, images_dir):
    cfg = Config.fromfile(config_path)
    cfg.load_from = checkpoint
    cfg.resume = False
    cfg.work_dir = tempfile.mkdtemp(prefix="per_subject_ap_")
    ann_path = osp.abspath(val_json)
    img_dir = osp.abspath(images_dir)
    cfg.val_dataloader.dataset.data_root = img_dir
    cfg.val_dataloader.dataset.ann_file = ann_path
    cfg.val_dataloader.dataset.data_prefix = dict(img="")
    cfg.test_dataloader = cfg.val_dataloader
    cfg.test_evaluator = dict(type="CocoMetric", ann_file=ann_path)
    cfg.custom_hooks = [
        hook for hook in cfg.get("custom_hooks", [])
        if hook.get("type") not in _SKIP_HOOKS
    ]
    return cfg


def load_weights(runner, checkpoint):
    if hasattr(runner, "load_or_resume"):
        runner.load_or_resume()
        return
    from mmengine.runner import load_checkpoint
    load_checkpoint(runner.model, checkpoint)


def collect_results(cfg):
    """One val pass. Returns the CocoMetric and its (pred, gt) results."""
    runner = Runner.from_cfg(cfg)
    load_weights(runner, cfg.load_from)
    metric = runner.test_evaluator.metrics[0]
    metric.dataset_meta = runner.test_dataloader.dataset.metainfo
    metric.results = []
    runner.model.eval()
    for batch in runner.test_dataloader:
        outputs = runner.model.test_step(batch)
        metric.process(batch, outputs)
    return metric, list(metric.results)


def score_subset(metric, results, image_ids, data, tmp_dir):
    kept = [item for item in results if int(item[0]["img_id"]) in image_ids]
    if not kept:
        return None
    ann_path = osp.join(tmp_dir, "subset.json")
    write_subset(data, image_ids, ann_path)
    metric.coco = COCO(ann_path)
    metric.results = kept
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        metrics = metric.compute_metrics(kept)
    ap = metrics.get("AP")
    return None if ap is None else float(ap)


def format_ap(value):
    if value is None:
        return "  n/a"
    return f"{value:6.3f}"


def print_comparison(name, rows):
    print()
    print(name)
    print(f"{'subject':<24} {'images':>7} {'full':>8} {'subject25':>10} {'delta':>8}")
    for subject, images, full_ap, cap_ap in rows:
        if full_ap is None or cap_ap is None:
            delta = "n/a"
        else:
            delta = f"{cap_ap - full_ap:+7.3f}"
        print(
            f"{subject:<24} {images:7d} {format_ap(full_ap):>8} "
            f"{format_ap(cap_ap):>10} {delta:>8}"
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Per-subject val AP for a full checkpoint and its subject25 pair"
    )
    parser.add_argument("--config", required=True, help="Test config, usually the finetune config")
    parser.add_argument("--subjects-path", required=True, help="video_id,subject CSV")
    parser.add_argument(
        "--images-dir",
        required=True,
        help="Directory of val frames. file_name in the JSON is a basename here.",
    )
    parser.add_argument(
        "--subjects",
        nargs="+",
        default=list(DEFAULT_SUBJECTS),
        help="Subject names as stored in the CSV. Defaults to NinaVani, JamieDoe, JoeyReyna, YordanosWoldebirhan.",
    )
    parser.add_argument(
        "--compare",
        action="append",
        nargs=4,
        metavar=("NAME", "VAL_JSON", "FULL_CKPT", "SUBJECT25_CKPT"),
        required=True,
        help="One pair. A checkpoint may be a .pth file or a run directory with best_coco_AP_*.pth.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not osp.isdir(args.images_dir):
        raise SystemExit(f"Images directory not found: {args.images_dir}")
    split_dataset = _load_split_dataset()
    video_subjects = split_dataset.load_video_subjects(args.subjects_path)

    for name, val_json, full_ckpt, cap_ckpt in args.compare:
        if not osp.isfile(val_json):
            raise SystemExit(f"Val JSON not found: {val_json}")
        full_ckpt = resolve_checkpoint(full_ckpt)
        cap_ckpt = resolve_checkpoint(cap_ckpt)
        data, by_subject, missing = subject_image_ids(
            val_json, video_subjects, split_dataset.subject_for_video
        )
        if missing:
            preview = ", ".join(missing[:8])
            extra = f" (+{len(missing) - 8} more)" if len(missing) > 8 else ""
            print(
                f"{name}: {len(missing)} val videos are not in the subjects CSV "
                f"({preview}{extra}). They use the parsed name, or the video id."
            )
        print(f"{name}: full {full_ckpt}")
        print(f"{name}: subject25 {cap_ckpt}")
        full_metric, full_results = collect_results(
            build_cfg(args.config, full_ckpt, val_json, args.images_dir)
        )
        cap_metric, cap_results = collect_results(
            build_cfg(args.config, cap_ckpt, val_json, args.images_dir)
        )
        rows = []
        with tempfile.TemporaryDirectory(prefix="per_subject_ap_") as tmp_dir:
            all_ids = {image["id"] for image in data["images"]}
            rows.append((
                "all",
                len(all_ids),
                score_subset(full_metric, full_results, all_ids, data, tmp_dir),
                score_subset(cap_metric, cap_results, all_ids, data, tmp_dir),
            ))
            for subject in args.subjects:
                image_ids = by_subject.get(subject, set())
                rows.append((
                    subject,
                    len(image_ids),
                    score_subset(full_metric, full_results, image_ids, data, tmp_dir),
                    score_subset(cap_metric, cap_results, image_ids, data, tmp_dir),
                ))
        print_comparison(name, rows)


if __name__ == "__main__":
    sys.exit(main())
