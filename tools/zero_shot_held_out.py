#!/usr/bin/env python
"""Zero-shot pretrained baseline on the held-out subject frames.

Uses the held-out annotation files the sweep already gives to
CustomDatasetHook, so each person is scored on the same frames as training
val. The checkpoint defaults to load_from in the finetune config. Metrics are
the COCO stats that hook logs, prefixed the same way (held_out_<person>/AP,
AP .5, AP .75, AP_M, AP_L, AR, AR .5, AR .75, AR_M, AR_L).

Results are printed and written to --out (default zero_shot_held_out.json).

Example:
  python tools/zero_shot_held_out.py \\
    --sweep experiments/rtmpose-m_single_model/sweep_config_app_vids_gbe_v2.yaml
"""

import argparse
import copy
import json
import os
import os.path as osp
import random

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
    "experiments/rtmpose-m_single_model/sweep_config_app_vids_gbe_v2.yaml"
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def held_out_jobs(cfg, experiments):
    """One job per held-out person, in sweep order, using that hook's paths."""
    jobs = []
    seen = set()
    for experiment in experiments:
        overrides = experiment.get("overrides", {})
        indexes = sorted({
            int(key.split(".")[1])
            for key in overrides
            if key.startswith("custom_hooks.")
            and key.split(".")[1].isdigit()
            and key.endswith(".metric_prefix")
        })
        for index in indexes:
            prefix = f"custom_hooks.{index}"
            metric_prefix = overrides.get(f"{prefix}.metric_prefix")
            if not metric_prefix:
                continue
            hook = cfg.custom_hooks[index]
            dataset = hook.dataloader.dataset
            data_root = overrides.get(
                f"{prefix}.dataloader.dataset.data_root", dataset.data_root
            )
            ann_file = overrides.get(
                f"{prefix}.dataloader.dataset.ann_file", dataset.ann_file
            )
            img_prefix = overrides.get(
                f"{prefix}.dataloader.dataset.data_prefix.img",
                dataset.data_prefix.img,
            )
            metric_ann = overrides.get(
                f"{prefix}.evaluator.ann_file", hook.evaluator.ann_file
            )
            if not osp.isabs(metric_ann):
                metric_ann = osp.join(data_root, metric_ann)
            metric_ann = osp.abspath(metric_ann)
            if metric_ann in seen:
                continue
            seen.add(metric_ann)
            jobs.append({
                "experiment": experiment["name"],
                "metric_prefix": metric_prefix,
                "data_root": data_root,
                "ann_file": ann_file,
                "img_prefix": img_prefix,
                "metric_ann": metric_ann,
                "hook_index": index,
            })
    if not jobs:
        raise SystemExit("Sweep has no CustomDatasetHook metric_prefix overrides")
    return jobs


def build_loader(cfg, job):
    hook = cfg.custom_hooks[job["hook_index"]]
    loader_cfg = copy.deepcopy(hook.dataloader)
    loader_cfg.dataset.data_root = job["data_root"]
    loader_cfg.dataset.ann_file = job["ann_file"]
    loader_cfg.dataset.data_prefix = dict(img=job["img_prefix"])
    loader_cfg.setdefault("drop_last", False)
    loader_cfg.setdefault("sampler", dict(type="DefaultSampler", shuffle=False))
    loader_cfg["persistent_workers"] = False
    return Runner.build_dataloader(loader_cfg)


def score(model, loader, metric_prefix, metric_ann):
    evaluator = Evaluator(dict(
        type="CocoMetric",
        ann_file=metric_ann,
        prefix=metric_prefix,
    ))
    dataset_meta = getattr(loader.dataset, "metainfo", None)
    if dataset_meta:
        for metric in evaluator.metrics:
            metric.dataset_meta = dataset_meta
    model.eval()
    with torch.no_grad():
        for data_batch in loader:
            outputs = model.val_step(data_batch)
            evaluator.process(data_samples=outputs, data_batch=data_batch)
    return evaluator.evaluate(len(loader.dataset))


def json_ready(value):
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def write_results(path, payload):
    os.makedirs(osp.dirname(osp.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(json_ready(payload), handle, indent=2)
        handle.write("\n")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", default=DEFAULT_SWEEP, help="Sweep yaml whose held-out hooks define the frames")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Pretrained .pth or URL. Defaults to load_from in the sweep base config.",
    )
    parser.add_argument(
        "--persons",
        nargs="*",
        default=None,
        help="Optional metric prefixes to run, e.g. held_out_evan held_out_nina",
    )
    parser.add_argument(
        "--out",
        default="zero_shot_held_out.json",
        help="JSON file for the metrics. Rewritten after each person.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.sweep, encoding="utf-8") as handle:
        sweep = yaml.safe_load(handle)
    cfg = Config.fromfile(sweep["base_config"])
    checkpoint = args.checkpoint or cfg.load_from
    if not checkpoint:
        raise SystemExit("Pass --checkpoint or set load_from in the config")
    seed = cfg.get("randomness", {}).get("seed", None)
    if seed is not None:
        set_seed(seed)
        print(f"seed: {seed}")
    print(f"checkpoint: {checkpoint}")
    print(f"out: {args.out}")
    payload = {
        "sweep": osp.abspath(args.sweep),
        "checkpoint": checkpoint,
        "seed": seed,
        "results": [],
    }

    init_default_scope("mmpose")
    model = MODELS.build(cfg.model)
    load_checkpoint(model, checkpoint, map_location="cpu")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    jobs = held_out_jobs(cfg, sweep["experiments"])
    if args.persons:
        wanted = set(args.persons)
        jobs = [job for job in jobs if job["metric_prefix"] in wanted]
        missing = wanted.difference(job["metric_prefix"] for job in jobs)
        if missing:
            raise SystemExit(f"Unknown held-out prefix: {', '.join(sorted(missing))}")

    for job in jobs:
        if not osp.isfile(job["metric_ann"]):
            raise SystemExit(f"Annotation file not found: {job['metric_ann']}")
        loader = build_loader(cfg, job)
        print()
        print(
            f"{job['metric_prefix']}  images={len(loader.dataset)}  "
            f"{job['metric_ann']}"
        )
        metrics = score(model, loader, job["metric_prefix"], job["metric_ann"])
        print(f"CustomDatasetHook[{job['metric_prefix']}] results: {metrics}")
        payload["results"].append({
            "metric_prefix": job["metric_prefix"],
            "experiment": job["experiment"],
            "images": len(loader.dataset),
            "ann_file": job["metric_ann"],
            "metrics": metrics,
        })
        write_results(args.out, payload)


if __name__ == "__main__":
    main()
