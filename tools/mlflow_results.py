#!/usr/bin/env python3
"""
Read MLflow results and print highest metrics per run.

Usage:
    python tools/mlflow_results.py [--tracking-uri URI] [--experiment-name NAME]
"""

import argparse
from mlflow.tracking import MlflowClient


def get_max_metric_value(client, run_id, metric_key):
    """Get the maximum value of a metric across all steps for a run.
    
    Returns:
        tuple: (max_value, step) or None if metric doesn't exist
    """
    try:
        metric_history = client.get_metric_history(run_id, metric_key)
        if not metric_history:
            return None
        max_metric = max(metric_history, key=lambda m: m.value)
        return (max_metric.value, max_metric.step)
    except Exception as e:
        # Metric might not exist for this run
        return None


def get_metric_value_at_step(client, run_id, metric_key, step):
    """Get the value of a metric at a specific step.
    
    Returns:
        float: metric value at the step, or None if not found
    """
    try:
        metric_history = client.get_metric_history(run_id, metric_key)
        if not metric_history:
            return None
        # Find the metric entry at the specified step
        for m in metric_history:
            if m.step == step:
                return m.value
        # If exact step not found, return None
        return None
    except Exception as e:
        return None


def main():
    parser = argparse.ArgumentParser(
        description='Print highest MLflow metrics per run'
    )
    parser.add_argument(
        '--tracking-uri',
        type=str,
        default='http://52.41.68.196:5000',
        help='MLflow tracking URI (default: http://52.41.68.196:5000)'
    )
    parser.add_argument(
        '--experiment-name',
        type=str,
        default='rtmpose-m_heel_slides_offline_augs_dedup_param_0.995',
        help='MLflow experiment name (default: rtmpose-m_heel_slides_offline_augs_dedup_param_0.995)'
    )
    
    args = parser.parse_args()
    tracking_uri = args.tracking_uri
    experiment_name = args.experiment_name
    
    print(f"Connecting to MLflow at: {tracking_uri}")
    print(f"Experiment: {experiment_name}\n")
    
    # Set tracking URI and get client
    client = MlflowClient(tracking_uri=tracking_uri)
    
    # Get experiment by name
    try:
        experiment = client.get_experiment_by_name(experiment_name)
        if experiment is None:
            print(f"Error: Experiment '{experiment_name}' not found")
            return
        experiment_id = experiment.experiment_id
    except Exception as e:
        print(f"Error getting experiment: {e}")
        return
    
    # Get all runs for this experiment
    runs = client.search_runs(experiment_ids=[experiment_id])
    
    if not runs:
        print(f"No runs found in experiment '{experiment_name}'")
        return
    
    print(f"Found {len(runs)} runs\n")
    
    # Collect metrics for each run
    run_metrics = []
    for run in runs:
        run_id = run.info.run_id
        run_name = run.info.run_name
        
        # Get max coco/AP (returns tuple: (value, step) or None)
        max_coco_ap_result = get_max_metric_value(client, run_id, 'coco/AP')
        
        # Get general_val/AP at the step where max coco/AP was achieved
        general_val_at_max_coco_step = None
        if max_coco_ap_result is not None:
            _, max_coco_step = max_coco_ap_result
            general_val_at_max_coco_step = get_metric_value_at_step(
                client, run_id, 'general_val/AP', max_coco_step
            )
        
        run_metrics.append({
            'run_id': run_id,
            'run_name': run_name,
            'max_coco_ap': max_coco_ap_result,
            'general_val_at_max_coco_step': general_val_at_max_coco_step
        })
    
    # Filter runs that have both max coco/AP and general_val/AP at that step
    runs_with_both = [
        r for r in run_metrics 
        if r['max_coco_ap'] is not None and r['general_val_at_max_coco_step'] is not None
    ]
    
    if not runs_with_both:
        print("No runs with both coco/AP and general_val/AP metrics found\n")
        return
    
    # Calculate composite scores for each run
    for run in runs_with_both:
        max_coco_value, _ = run['max_coco_ap']
        general_val_value = run['general_val_at_max_coco_step']
        run['score_1x'] = 1 * max_coco_value + general_val_value
        run['score_2x'] = 2 * max_coco_value + general_val_value
        run['score_3x'] = 3 * max_coco_value + general_val_value
    
    # Print top 3 runs sorted by 1*(max coco/AP) + general_val/AP
    print("=" * 80)
    print("Top 3 runs by: 1*(max coco/AP) + general_val/AP at that step")
    print("=" * 80)
    
    score_1x_sorted = sorted(
        runs_with_both,
        key=lambda x: x['score_1x'],
        reverse=True
    )[:3]  # Top 3 only
    
    for i, run in enumerate(score_1x_sorted, 1):
        max_coco_value, max_coco_step = run['max_coco_ap']
        general_val_value = run['general_val_at_max_coco_step']
        score = run['score_1x']
        print(f"{i}. Run: {run['run_name']}")
        print(f"   Run ID: {run['run_id']}")
        print(f"   Max coco/AP: {max_coco_value:.6f} (at step {max_coco_step})")
        print(f"   general_val/AP at step {max_coco_step}: {general_val_value:.6f}")
        print(f"   Score (1*coco + general_val): {score:.6f}\n")
    
    # Print top 3 runs sorted by 2*(max coco/AP) + general_val/AP
    print("=" * 80)
    print("Top 3 runs by: 2*(max coco/AP) + general_val/AP at that step")
    print("=" * 80)
    
    score_2x_sorted = sorted(
        runs_with_both,
        key=lambda x: x['score_2x'],
        reverse=True
    )[:3]  # Top 3 only
    
    for i, run in enumerate(score_2x_sorted, 1):
        max_coco_value, max_coco_step = run['max_coco_ap']
        general_val_value = run['general_val_at_max_coco_step']
        score = run['score_2x']
        print(f"{i}. Run: {run['run_name']}")
        print(f"   Run ID: {run['run_id']}")
        print(f"   Max coco/AP: {max_coco_value:.6f} (at step {max_coco_step})")
        print(f"   general_val/AP at step {max_coco_step}: {general_val_value:.6f}")
        print(f"   Score (2*coco + general_val): {score:.6f}\n")
    
    # Print top 3 runs sorted by 3*(max coco/AP) + general_val/AP
    print("=" * 80)
    print("Top 3 runs by: 3*(max coco/AP) + general_val/AP at that step")
    print("=" * 80)
    
    score_3x_sorted = sorted(
        runs_with_both,
        key=lambda x: x['score_3x'],
        reverse=True
    )[:3]  # Top 3 only
    
    for i, run in enumerate(score_3x_sorted, 1):
        max_coco_value, max_coco_step = run['max_coco_ap']
        general_val_value = run['general_val_at_max_coco_step']
        score = run['score_3x']
        print(f"{i}. Run: {run['run_name']}")
        print(f"   Run ID: {run['run_id']}")
        print(f"   Max coco/AP: {max_coco_value:.6f} (at step {max_coco_step})")
        print(f"   general_val/AP at step {max_coco_step}: {general_val_value:.6f}")
        print(f"   Score (3*coco + general_val): {score:.6f}\n")


if __name__ == '__main__':
    main()
