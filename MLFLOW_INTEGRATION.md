# MLflow Integration with MMPose Training

This document describes the MLflow integration for the MMPose training pipeline using MMEngine's built-in `MLflowVisBackend`.

## Overview

MLflow integration is achieved through MMEngine's visualization backend system. When enabled, **all training and validation metrics are automatically logged** to MLflow without any custom hooks or code modifications.

## Quick Start

### Enable MLflow via Command Line

```bash
python tools/train.py configs/body_2d_keypoint/rtmpose/coco/rtmpose-m_8xb256-420e_coco-256x192_finetune.py \
    --mlflow-tracking-uri http://52.41.68.196:5000
```

That's it! All metrics will be logged automatically.

### With Custom Experiment/Run Names

```bash
python tools/train.py configs/body_2d_keypoint/rtmpose/coco/rtmpose-m_8xb256-420e_coco-256x192_finetune.py \
    --mlflow-tracking-uri http://52.41.68.196:5000 \
    --mlflow-experiment-name "rtmpose_finetune" \
    --mlflow-run-name "baseline_v1"
```

## Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--mlflow-tracking-uri` | MLflow server URI. **Required to enable logging.** | None (disabled) |
| `--mlflow-experiment-name` | Experiment name | Config filename |
| `--mlflow-run-name` | Run name | Timestamp |

## Alternative: Configure via Config File

You can also enable MLflow directly in your config file:

```python
# Add to your config file
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(
        type='MLflowVisBackend',
        tracking_uri='http://52.41.68.196:5000',
        exp_name='my_experiment',
        run_name='my_run',
    )
]
visualizer = dict(
    type='PoseLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer'
)
```

## What Gets Logged

### Automatically Logged Metrics (Every Epoch)

**Training metrics:**
- `loss` - Total training loss
- `loss_kpt` - Keypoint loss
- `acc_pose` - Pose accuracy
- `lr` - Learning rate
- `time` - Iteration time
- `data_time` - Data loading time
- `memory` - GPU memory usage

**Validation metrics:**
- `coco/AP` - COCO Average Precision
- `coco/AP .5` - AP at IoU=0.5
- `coco/AP .75` - AP at IoU=0.75
- `coco/AR` - COCO Average Recall
- And all other COCO metrics

### Automatically Logged Artifacts (End of Training)

- Configuration file (`.py`)
- Training logs (`.log`)
- JSON files (`.json`)
- YAML files (`.yaml`)

### Automatically Logged Config Parameters

The full config is flattened and logged as parameters, including:
- Model architecture
- Training hyperparameters
- Dataset settings
- Optimizer configuration

## How It Works

1. **LoggerHook** collects all training/validation metrics
2. **Visualizer** receives the metrics via `add_scalars()`
3. **MLflowVisBackend** forwards them to MLflow via `mlflow.log_metrics()`
4. On training completion, `close()` logs all artifacts automatically

This is MMEngine's standard logging pipeline - no custom code needed!

## MLflow UI

View your experiments at your MLflow server:

```bash
# If running locally
mlflow ui

# Or access remote server
# http://52.41.68.196:5000
```

## Requirements

```bash
pip install mlflow
```

## Comparison with Custom Hook Approach

| Feature | MLflowVisBackend | Custom Hook |
|---------|------------------|-------------|
| Code changes | None (config only) | Custom hook code |
| Metric access | Automatic via LoggerHook | Manual extraction |
| Artifact logging | Automatic | Manual |
| Config logging | Automatic (flattened) | Manual |
| Maintenance | Built into MMEngine | Custom maintenance |

The `MLflowVisBackend` approach is recommended as it's:
- **Simpler** - Just add to config or use CLI flag
- **More reliable** - Uses MMEngine's logging pipeline
- **Feature-complete** - Logs metrics, config, and artifacts automatically
- **Maintained** - Part of MMEngine, not custom code

## Example Output

```
MLflow logging enabled:
  Tracking URI: http://52.41.68.196:5000
  Experiment: rtmpose-m_8xb256-420e_coco-256x192_finetune
  Run name: 20251216_235500
```

All metrics will appear in the MLflow UI with proper step numbers corresponding to epochs.
