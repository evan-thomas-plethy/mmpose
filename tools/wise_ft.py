#!/usr/bin/env python
"""WiSE-FT weight interpolation. No training.

Blends a pretrained checkpoint and a finetuned checkpoint:

    w = (1 - alpha) * w_pretrained + alpha * w_finetuned

Alpha runs from 0 to 1 in steps of 0.1. Each blend is scored with COCO AP
on the finetune val set and on the general val set. The plot is val AP
against general AP, one point per alpha.

The finetuned argument is a .pth file or a run directory that contains
best_coco_AP_*.pth. Pass a sweep yaml and an experiment name to take the
val and general-val paths from that run. Otherwise pass the four dataset
paths directly. The pretrained checkpoint defaults to load_from in the config.

Example:
  python tools/wise_ft.py \\
    --sweep experiments/rtmpose-m_single_model/sweep_config_app_vids_gbe_v2.yaml \\
    --experiment app_vids_v2_gbe_v4_val10
"""

import argparse
import copy
import csv
import os
import os.path as osp
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import yaml
from mmengine.config import Config
from mmengine.runner import CheckpointLoader, Runner, load_checkpoint
from mmpose.registry import METRICS


FINETUNE_CONFIG = (
    "configs/body_2d_keypoint/rtmpose/coco/"
    "rtmpose-m_8xb256-420e_coco-256x192_finetune.py"
)
_SKIP_HOOKS = (
    "MirroredValHook",
    "CustomDatasetHook",
    "GeneralValHook",
    "EMAHook",
    "mmdet.PipelineSwitchHook",
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


def load_state_dict(path):
    checkpoint = CheckpointLoader.load_checkpoint(path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f"No state_dict found in {path}")
    if "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    if "model" in checkpoint:
        return checkpoint["model"]
    return checkpoint


def blend_state_dicts(pretrained, finetuned, alpha):
    """Linear blend of floating tensors. Other tensors follow the nearer endpoint."""
    blended = type(finetuned)()
    missing = []
    shape_mismatch = []
    for key, ft_value in finetuned.items():
        if key not in pretrained:
            missing.append(key)
            blended[key] = ft_value
            continue
        pre_value = pretrained[key]
        if (
            torch.is_tensor(ft_value)
            and torch.is_tensor(pre_value)
            and torch.is_floating_point(ft_value)
            and ft_value.shape == pre_value.shape
        ):
            if alpha == 0:
                blended[key] = pre_value.detach().clone()
            elif alpha == 1:
                blended[key] = ft_value.detach().clone()
            else:
                mixed = (1 - alpha) * pre_value.float() + alpha * ft_value.float()
                blended[key] = mixed.to(dtype=ft_value.dtype)
        elif torch.is_tensor(ft_value) and torch.is_tensor(pre_value) and ft_value.shape != pre_value.shape:
            shape_mismatch.append(key)
            blended[key] = ft_value
        else:
            blended[key] = pre_value if alpha == 0 else ft_value
    if missing or shape_mismatch:
        preview = (missing + shape_mismatch)[:8]
        raise RuntimeError(
            f"{len(missing)} finetuned keys are missing from the pretrained "
            f"checkpoint and {len(shape_mismatch)} shapes differ. First keys: {preview}"
        )
    return blended


def general_hook(cfg):
    for index, hook in enumerate(cfg.custom_hooks):
        if hook.get("type") == "GeneralValHook":
            return index, hook
    raise RuntimeError("Config has no GeneralValHook")


def eval_set_from_parts(data_root, ann_file, img_prefix, metric_ann):
    if not osp.isabs(metric_ann):
        metric_ann = osp.join(data_root, metric_ann)
    return {
        "data_root": data_root,
        "ann_file": ann_file,
        "img_prefix": img_prefix,
        "metric_ann": osp.abspath(metric_ann),
    }


def eval_set_from_images(annotations, images_dir):
    ann = osp.abspath(annotations)
    return {
        "data_root": osp.abspath(images_dir),
        "ann_file": ann,
        "img_prefix": "",
        "metric_ann": ann,
    }


def sets_from_sweep(cfg, overrides):
    val_root = overrides.get(
        "val_dataloader.dataset.data_root", cfg.val_dataloader.dataset.data_root
    )
    val_ann = overrides.get(
        "val_dataloader.dataset.ann_file", cfg.val_dataloader.dataset.ann_file
    )
    val_prefix = overrides.get(
        "val_dataloader.dataset.data_prefix.img",
        cfg.val_dataloader.dataset.data_prefix.img,
    )
    val_metric = overrides.get("val_evaluator.ann_file", cfg.val_evaluator.ann_file)
    val_set = eval_set_from_parts(val_root, val_ann, val_prefix, val_metric)

    index, hook = general_hook(cfg)
    dataset = hook.dataloader.dataset
    prefix = f"custom_hooks.{index}"
    general_root = overrides.get(
        f"{prefix}.dataloader.dataset.data_root", dataset.data_root
    )
    general_ann = overrides.get(
        f"{prefix}.dataloader.dataset.ann_file", dataset.ann_file
    )
    general_img = overrides.get(
        f"{prefix}.dataloader.dataset.data_prefix.img", dataset.data_prefix.img
    )
    general_metric = overrides.get(
        f"{prefix}.evaluator.ann_file", hook.evaluator.ann_file
    )
    general_set = eval_set_from_parts(
        general_root, general_ann, general_img, general_metric
    )
    return val_set, general_set


def apply_set(loader_cfg, eval_set):
    loader_cfg.dataset.data_root = eval_set["data_root"]
    loader_cfg.dataset.ann_file = eval_set["ann_file"]
    loader_cfg.dataset.data_prefix = dict(img=eval_set["img_prefix"])
    return loader_cfg


def build_eval(config_path, finetuned, val_set, general_set):
    cfg = Config.fromfile(config_path)
    cfg.load_from = finetuned
    cfg.resume = False
    cfg.work_dir = tempfile.mkdtemp(prefix="wise_ft_")
    apply_set(cfg.val_dataloader, val_set)
    cfg.val_dataloader.persistent_workers = False
    cfg.test_dataloader = cfg.val_dataloader
    cfg.val_evaluator = dict(type="CocoMetric", ann_file=val_set["metric_ann"])
    cfg.test_evaluator = cfg.val_evaluator
    cfg.custom_hooks = [
        hook for hook in cfg.get("custom_hooks", [])
        if hook.get("type") not in _SKIP_HOOKS
    ]
    runner = Runner.from_cfg(cfg)
    load_checkpoint(runner.model, finetuned)

    general_loader_cfg = apply_set(copy.deepcopy(cfg.val_dataloader), general_set)
    general_loader_cfg.persistent_workers = False
    general_loader = Runner.build_dataloader(general_loader_cfg)
    general_metric = METRICS.build(
        dict(type="CocoMetric", ann_file=general_set["metric_ann"])
    )
    general_metric.dataset_meta = general_loader.dataset.metainfo
    return runner, general_loader, general_metric


def score(model, loader, metric):
    metric.results = []
    metric.dataset_meta = loader.dataset.metainfo
    model.eval()
    with torch.no_grad():
        for batch in loader:
            outputs = model.test_step(batch)
            metric.process(batch, outputs)
    metrics = metric.compute_metrics(list(metric.results))
    ap = metrics.get("AP")
    if ap is None:
        raise RuntimeError(f"CocoMetric did not return AP. Keys: {list(metrics)}")
    return float(ap)


def apply_weights(model, state_dict, path):
    torch.save({"state_dict": state_dict}, path)
    load_checkpoint(model, path)


def plot_results(rows, out_path):
    val_ap = [row["val_ap"] for row in rows]
    general_ap = [row["general_ap"] for row in rows]
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.plot(val_ap, general_ap, color="0.4", zorder=1)
    ax.scatter(val_ap, general_ap, zorder=2)
    for row in rows:
        ax.annotate(
            f"{row['alpha']:.1f}",
            (row["val_ap"], row["general_ap"]),
            textcoords="offset points",
            xytext=(5, 4),
            fontsize=8,
        )
    ax.set_xlabel("Val AP")
    ax.set_ylabel("General AP")
    ax.set_title("WiSE-FT")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=FINETUNE_CONFIG, help="Finetune config")
    parser.add_argument(
        "--finetuned",
        default=None,
        help="Finetuned .pth, or a run directory with best_coco_AP_*.pth. "
        "With --sweep, defaults to work_dirs/<experiment_name>/<experiment>.",
    )
    parser.add_argument(
        "--pretrained",
        default=None,
        help="Pretrained .pth or URL. Defaults to load_from in the config.",
    )
    parser.add_argument("--sweep", default=None, help="Sweep yaml used for the run")
    parser.add_argument("--experiment", default=None, help="Experiment name in the sweep yaml")
    parser.add_argument("--val-annotations", default=None)
    parser.add_argument("--val-images-dir", default=None)
    parser.add_argument("--general-annotations", default=None)
    parser.add_argument("--general-images-dir", default=None)
    parser.add_argument("--out-dir", default=None, help="Checkpoints, csv, and plot. Default: <run>/wise_ft")
    return parser.parse_args()


def resolve_inputs(args):
    overrides = {}
    experiment_name = None
    if args.sweep:
        with open(args.sweep, encoding="utf-8") as handle:
            sweep = yaml.safe_load(handle)
        experiment_name = sweep["experiment_name"]
        if args.config == FINETUNE_CONFIG and sweep.get("base_config"):
            args.config = sweep["base_config"]
        if not args.experiment:
            raise SystemExit("--experiment is required with --sweep")
        matches = [item for item in sweep["experiments"] if item["name"] == args.experiment]
        if len(matches) != 1:
            names = ", ".join(item["name"] for item in sweep["experiments"])
            raise SystemExit(f"Experiment {args.experiment!r} not in the sweep. Names: {names}")
        overrides = matches[0].get("overrides", {})
        if args.finetuned is None:
            args.finetuned = osp.join("work_dirs", experiment_name, args.experiment)

    if not args.finetuned:
        raise SystemExit("Pass --finetuned, or --sweep and --experiment")

    cfg = Config.fromfile(args.config)
    pretrained = args.pretrained or cfg.load_from
    if not pretrained:
        raise SystemExit("Pass --pretrained or set load_from in the config")

    explicit_val = args.val_annotations or args.val_images_dir
    explicit_general = args.general_annotations or args.general_images_dir
    if explicit_val and not (args.val_annotations and args.val_images_dir):
        raise SystemExit("Pass both --val-annotations and --val-images-dir")
    if explicit_general and not (args.general_annotations and args.general_images_dir):
        raise SystemExit("Pass both --general-annotations and --general-images-dir")

    if args.sweep and not explicit_val and not explicit_general:
        val_set, general_set = sets_from_sweep(cfg, overrides)
    else:
        if not explicit_val or not explicit_general:
            raise SystemExit(
                "Pass --sweep and --experiment, or all four of "
                "--val-annotations, --val-images-dir, "
                "--general-annotations, --general-images-dir"
            )
        val_set = eval_set_from_images(args.val_annotations, args.val_images_dir)
        general_set = eval_set_from_images(
            args.general_annotations, args.general_images_dir
        )

    finetuned = resolve_checkpoint(args.finetuned)
    if args.out_dir:
        out_dir = args.out_dir
    elif osp.isdir(args.finetuned):
        out_dir = osp.join(args.finetuned, "wise_ft")
    else:
        out_dir = osp.join(osp.dirname(finetuned), "wise_ft")
    return pretrained, finetuned, val_set, general_set, out_dir


def main():
    args = parse_args()
    pretrained_path, finetuned_path, val_set, general_set, out_dir = resolve_inputs(args)
    os.makedirs(out_dir, exist_ok=True)
    print(f"pretrained: {pretrained_path}")
    print(f"finetuned:  {finetuned_path}")
    print(f"val:        {val_set['metric_ann']}")
    print(f"general:    {general_set['metric_ann']}")
    print(f"out:        {out_dir}")

    for eval_set in (val_set, general_set):
        if not osp.isfile(eval_set["metric_ann"]):
            raise SystemExit(f"Annotation file not found: {eval_set['metric_ann']}")

    pretrained = load_state_dict(pretrained_path)
    finetuned = load_state_dict(finetuned_path)
    runner, general_loader, general_metric = build_eval(
        args.config, finetuned_path, val_set, general_set
    )
    val_loader = runner.test_dataloader
    val_metric = runner.test_evaluator.metrics[0]
    model = runner.model

    alphas = [round(step / 10, 1) for step in range(11)]
    rows = []
    for alpha in alphas:
        blended = blend_state_dicts(pretrained, finetuned, alpha)
        ckpt_path = osp.join(out_dir, f"alpha_{alpha:.1f}.pth")
        apply_weights(model, blended, ckpt_path)
        print(f"alpha {alpha:.1f}: val")
        val_ap = score(model, val_loader, val_metric)
        print(f"alpha {alpha:.1f}: general")
        general_ap = score(model, general_loader, general_metric)
        print(f"alpha {alpha:.1f}: val AP {val_ap:.4f}  general AP {general_ap:.4f}")
        rows.append({"alpha": alpha, "val_ap": val_ap, "general_ap": general_ap, "checkpoint": ckpt_path})

    csv_path = osp.join(out_dir, "wise_ft.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["alpha", "val_ap", "general_ap", "checkpoint"])
        writer.writeheader()
        writer.writerows(rows)
    plot_path = osp.join(out_dir, "wise_ft_val_vs_general.png")
    plot_results(rows, plot_path)

    print()
    print(f"{'alpha':>6} {'val_ap':>8} {'general_ap':>11}")
    for row in rows:
        print(f"{row['alpha']:6.1f} {row['val_ap']:8.4f} {row['general_ap']:11.4f}")
    print(f"csv:  {csv_path}")
    print(f"plot: {plot_path}")


if __name__ == "__main__":
    main()
