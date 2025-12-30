#!/usr/bin/env python
"""
Hyperparameter Sweep Runner for MMPose

Usage:
    python experiments/run_sweep.py experiments/sweep_config.yaml
    python experiments/run_sweep.py experiments/sweep_config.yaml --dry-run
    python experiments/run_sweep.py experiments/sweep_config.yaml --experiment backbone_lr_mult_0.1
"""

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def build_cfg_options(overrides: dict) -> str:
    """Convert override dict to --cfg-options string."""
    if not overrides:
        return ""
    
    options = []
    for key, value in overrides.items():
        # Handle different value types
        if isinstance(value, bool):
            value = str(value)
        elif isinstance(value, (list, tuple)):
            value = f'"{value}"'
        options.append(f"{key}={value}")
    
    return " ".join(options)


def run_experiment(
    base_config: str,
    experiment: dict,
    mlflow_config: dict,
    dry_run: bool = False
) -> int:
    """Run a single experiment."""
    name = experiment['name']
    overrides = experiment.get('overrides', {})
    
    # Compute work_dir (same logic as train.py)
    work_dir = f"work_dirs/{mlflow_config['experiment_name']}/{name}"
    
    # Build the command
    cmd_parts = [
        sys.executable,  # Use current Python interpreter
        "tools/train.py",
        base_config,
        f"--mlflow-tracking-uri", mlflow_config['tracking_uri'],
        f"--mlflow-experiment-name", mlflow_config['experiment_name'],
        f"--mlflow-run-name", name,
    ]
    
    # Add cfg-options if there are overrides
    if overrides:
        cfg_options = build_cfg_options(overrides)
        cmd_parts.extend(["--cfg-options", cfg_options])
    
    cmd = " ".join(cmd_parts)
    
    print("\n" + "=" * 80)
    print(f"EXPERIMENT: {name}")
    print("=" * 80)
    print(f"Work dir: {work_dir}")
    print(f"Command: {cmd}")
    print("-" * 80)
    
    if dry_run:
        print("[DRY RUN] Skipping execution")
        return 0
    
    # Run the training command
    result = subprocess.run(cmd, shell=True, cwd=Path(__file__).parent.parent)
    
    if result.returncode != 0:
        print(f"\n❌ Experiment '{name}' failed with return code {result.returncode}")
    else:
        print(f"\n✅ Experiment '{name}' completed successfully")
    
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description='Run hyperparameter sweep')
    parser.add_argument('config', type=str, help='Path to sweep config YAML')
    parser.add_argument('--dry-run', action='store_true', 
                        help='Print commands without executing')
    parser.add_argument('--experiment', type=str, default=None,
                        help='Run only a specific experiment by name')
    parser.add_argument('--skip-completed', action='store_true',
                        help='Skip experiments that have work_dir with checkpoints')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    base_config = config['base_config']
    mlflow_config = config['mlflow']
    experiments = config['experiments']
    
    print("=" * 80)
    print("HYPERPARAMETER SWEEP")
    print("=" * 80)
    print(f"Base config: {base_config}")
    print(f"MLflow tracking: {mlflow_config['tracking_uri']}")
    print(f"MLflow experiment: {mlflow_config['experiment_name']}")
    print(f"Total experiments: {len(experiments)}")
    
    if args.experiment:
        # Filter to specific experiment
        experiments = [e for e in experiments if e['name'] == args.experiment]
        if not experiments:
            print(f"\n❌ Experiment '{args.experiment}' not found in config")
            sys.exit(1)
        print(f"Running only: {args.experiment}")
    
    if args.dry_run:
        print("\n🔍 DRY RUN MODE - commands will be printed but not executed")
    
    # Run experiments
    results = {}
    for i, experiment in enumerate(experiments, 1):
        print(f"\n[{i}/{len(experiments)}] Starting experiment: {experiment['name']}")
        
        returncode = run_experiment(
            base_config=base_config,
            experiment=experiment,
            mlflow_config=mlflow_config,
            dry_run=args.dry_run
        )
        
        results[experiment['name']] = returncode
    
    # Summary
    print("\n" + "=" * 80)
    print("SWEEP SUMMARY")
    print("=" * 80)
    
    for name, code in results.items():
        status = "✅ SUCCESS" if code == 0 else f"❌ FAILED (code {code})"
        print(f"  {name}: {status}")
    
    failed = sum(1 for code in results.values() if code != 0)
    if failed > 0:
        print(f"\n⚠️  {failed}/{len(results)} experiments failed")
        sys.exit(1)
    else:
        print(f"\n🎉 All {len(results)} experiments completed successfully!")


if __name__ == '__main__':
    main()
