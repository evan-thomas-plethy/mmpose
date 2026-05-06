#!/usr/bin/env python3
"""
Download all JSON artifacts for a given MLflow run (by name) within an experiment.

Saves files under data/artifacts_{run_name}/ preserving artifact relative paths.

Usage:
    python tools/mlflow_artifacts.py --experiment-name EXP --run-name RUN \\
        [--tracking-uri URI]
"""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile

from mlflow.tracking import MlflowClient


def iter_json_artifact_paths(client: MlflowClient, run_id: str, path: str = ''):
    """Yield artifact paths (relative to run root) for all .json files, recursively."""
    for fi in client.list_artifacts(run_id, path):
        if fi.is_dir:
            yield from iter_json_artifact_paths(client, run_id, fi.path)
        elif fi.path.lower().endswith('.json'):
            yield fi.path


def resolve_run_id(client: MlflowClient, experiment_id: str, run_name: str) -> str:
    """Return run_id for the given run name in the experiment."""
    runs = client.search_runs(experiment_ids=[experiment_id])
    matching = [r for r in runs if r.info.run_name == run_name]
    if not matching:
        available = sorted({r.info.run_name for r in runs})
        preview = available[:40]
        extra = f" ... (+{len(available) - len(preview)} more)" if len(available) > len(preview) else ""
        raise ValueError(
            f"No run with name {run_name!r} in this experiment. "
            f"Some available names ({len(available)} total): {preview}{extra}"
        )
    matching.sort(key=lambda r: r.info.start_time, reverse=True)
    if len(matching) > 1:
        print(
            f"Warning: {len(matching)} runs named {run_name!r}; "
            f"using the most recent (run_id={matching[0].info.run_id})."
        )
    return matching[0].info.run_id


def main():
    parser = argparse.ArgumentParser(
        description='Download all JSON artifacts for an MLflow run into data/artifacts_{run_name}/'
    )
    parser.add_argument(
        '--experiment-name',
        type=str,
        required=True,
        help='MLflow experiment name',
    )
    parser.add_argument(
        '--run-name',
        type=str,
        required=True,
        help='Run name (MLflow runName)',
    )
    parser.add_argument(
        '--tracking-uri',
        type=str,
        default='http://35.91.197.8:5000',
        help='MLflow tracking URI',
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Override output directory (default: data/artifacts_{run_name})',
    )
    args = parser.parse_args()

    client = MlflowClient(tracking_uri=args.tracking_uri)
    experiment = client.get_experiment_by_name(args.experiment_name)
    if experiment is None:
        raise SystemExit(f"Experiment not found: {args.experiment_name}")

    run_id = resolve_run_id(client, experiment.experiment_id, args.run_name)
    out_dir = args.output_dir or os.path.join(
        os.getcwd(), f"data/artifacts_{args.run_name}"
    )
    os.makedirs(out_dir, exist_ok=True)

    json_paths = list(iter_json_artifact_paths(client, run_id))
    if not json_paths:
        print(f"No .json artifacts found for run {run_id} ({args.run_name}).")
        print(f"Empty directory: {out_dir}")
        return

    print(f"Experiment: {args.experiment_name}")
    print(f"Run: {args.run_name} (run_id={run_id})")
    print(f"Downloading {len(json_paths)} JSON artifact(s) to {out_dir}\n")

    with tempfile.TemporaryDirectory() as tmp:
        for rel in sorted(json_paths):
            src = client.download_artifacts(run_id, rel, tmp)
            dest = os.path.join(out_dir, rel)
            parent = os.path.dirname(dest)
            if parent:
                os.makedirs(parent, exist_ok=True)
            shutil.copy2(src, dest)
            print(f"  {rel}")

    print(f"\nDone: {out_dir}")


if __name__ == '__main__':
    main()
