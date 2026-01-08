# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import json
import os
import os.path as osp
from datetime import datetime

from mmengine.config import Config, DictAction
from mmengine.runner import Runner


def parse_args():
    parser = argparse.ArgumentParser(description='Train a pose model')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume',
        nargs='?',
        type=str,
        const='auto',
        help='If specify checkpint path, resume from it, while if not '
        'specify, try to auto resume from the latest checkpoint '
        'in the work directory.')
    parser.add_argument(
        '--amp',
        action='store_true',
        default=False,
        help='enable automatic-mixed-precision training')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    parser.add_argument(
        '--auto-scale-lr',
        action='store_true',
        help='whether to auto scale the learning rate according to the '
        'actual batch size and the original batch size.')
    parser.add_argument(
        '--show-dir',
        help='directory where the visualization images will be saved.')
    parser.add_argument(
        '--show',
        action='store_true',
        help='whether to display the prediction results in a window.')
    parser.add_argument(
        '--interval',
        type=int,
        default=1,
        help='visualize per interval samples.')
    parser.add_argument(
        '--wait-time',
        type=float,
        default=1,
        help='display time of every window. (second)')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    # MLflow arguments
    parser.add_argument(
        '--mlflow-tracking-uri',
        type=str,
        default='http://35.165.139.156:5000',
        help='MLflow tracking server URI. If provided, enables MLflow logging.')
    parser.add_argument(
        '--mlflow-experiment-name',
        type=str,
        default=None,
        help='MLflow experiment name (defaults to config filename)')
    parser.add_argument(
        '--mlflow-run-name',
        type=str,
        default=None,
        help='MLflow run name (defaults to timestamp)')
    parser.add_argument(
        '--mlflow-run-description',
        type=str,
        default=None,
        help='MLflow run description (e.g., list of tweaked params)')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def merge_args(cfg, args):
    """Merge CLI arguments to config."""
    if args.no_validate:
        cfg.val_cfg = None
        cfg.val_dataloader = None
        cfg.val_evaluator = None

    cfg.launcher = args.launcher

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    # enable automatic-mixed-precision training
    if args.amp is True:
        from mmengine.optim import AmpOptimWrapper, OptimWrapper
        optim_wrapper = cfg.optim_wrapper.get('type', OptimWrapper)
        assert optim_wrapper in (OptimWrapper, AmpOptimWrapper,
                                 'OptimWrapper', 'AmpOptimWrapper'), \
            '`--amp` is not supported custom optimizer wrapper type ' \
            f'`{optim_wrapper}.'
        cfg.optim_wrapper.type = 'AmpOptimWrapper'
        cfg.optim_wrapper.setdefault('loss_scale', 'dynamic')

    # resume training
    if args.resume == 'auto':
        cfg.resume = True
        cfg.load_from = None
    elif args.resume is not None:
        cfg.resume = True
        cfg.load_from = args.resume

    # enable auto scale learning rate
    if args.auto_scale_lr:
        cfg.auto_scale_lr.enable = True

    # visualization
    if args.show or (args.show_dir is not None):
        assert 'visualization' in cfg.default_hooks, \
            'PoseVisualizationHook is not set in the ' \
            '`default_hooks` field of config. Please set ' \
            '`visualization=dict(type="PoseVisualizationHook")`'

        cfg.default_hooks.visualization.enable = True
        cfg.default_hooks.visualization.show = args.show
        if args.show:
            cfg.default_hooks.visualization.wait_time = args.wait_time
        cfg.default_hooks.visualization.out_dir = args.show_dir
        cfg.default_hooks.visualization.interval = args.interval

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    return cfg


def add_mlflow_backend(cfg, args):
    """Add MLflowVisBackend to the visualizer if MLflow is enabled."""
    if args.mlflow_tracking_uri is None:
        return cfg
    
    # Set experiment name (default to config filename without extension)
    if args.mlflow_experiment_name is None:
        exp_name = osp.splitext(osp.basename(args.config))[0]
    else:
        exp_name = args.mlflow_experiment_name
    
    # Set run name (default to timestamp)
    if args.mlflow_run_name is None:
        run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    else:
        run_name = args.mlflow_run_name
    
    # Set work_dir to work_dirs/<experiment_name>/<run_name>
    cfg.work_dir = osp.join('./work_dirs', exp_name, run_name)
    
    # Create MLflowVisBackend config
    mlflow_backend = dict(
        type='MLflowVisBackend',
        tracking_uri=args.mlflow_tracking_uri,
        exp_name=exp_name,
        run_name=run_name,
    )
    
    # Add run description as a tag if provided
    if args.mlflow_run_description:
        mlflow_backend['tags'] = {'mlflow.note.content': args.mlflow_run_description}
    
    # Ensure vis_backends exists and add MLflow backend
    if not hasattr(cfg, 'vis_backends') or cfg.vis_backends is None:
        cfg.vis_backends = [dict(type='LocalVisBackend')]
    
    # Check if MLflowVisBackend is already configured
    has_mlflow = any(
        backend.get('type') == 'MLflowVisBackend' 
        for backend in cfg.vis_backends
    )
    
    if not has_mlflow:
        cfg.vis_backends.append(mlflow_backend)
        print(f"MLflow logging enabled:")
        print(f"  Tracking URI: {args.mlflow_tracking_uri}")
        print(f"  Experiment: {exp_name}")
        print(f"  Run name: {run_name}")
        print(f"  Work dir: {cfg.work_dir}")
    
    # Update visualizer to use the new vis_backends
    if hasattr(cfg, 'visualizer') and cfg.visualizer is not None:
        cfg.visualizer.vis_backends = cfg.vis_backends

    return cfg


def main():
    args = parse_args()

    # load config
    cfg = Config.fromfile(args.config)

    # merge CLI arguments to config
    cfg = merge_args(cfg, args)

    # Add MLflow backend if tracking URI is provided
    cfg = add_mlflow_backend(cfg, args)

    # set preprocess configs to model
    if 'preprocess_cfg' in cfg:
        cfg.model.setdefault('data_preprocessor',
                             cfg.get('preprocess_cfg', {}))

    # Helper to get full annotation path from dataloader config
    def get_ann_path(dataloader_cfg):
        data_root = dataloader_cfg.dataset.get('data_root', '')
        ann_file = dataloader_cfg.dataset.ann_file
        return osp.join(data_root, ann_file) if data_root else ann_file

    # Helper to load info from annotation file
    def load_ann_info(ann_path):
        if osp.exists(ann_path):
            with open(ann_path, 'r') as f:
                ann_data = json.load(f)
            return ann_data.get('info')
        return None

    # Add dataset info to config so it gets logged with other params
    if args.mlflow_tracking_uri is not None:
        try:
            # Train dataset info
            if hasattr(cfg, 'train_dataloader') and cfg.train_dataloader is not None:
                train_ann_path = get_ann_path(cfg.train_dataloader)
                train_info = load_ann_info(train_ann_path)
                if train_info:
                    cfg.dataset_train = train_info
                    print(f"  Train dataset info added from: {train_ann_path}")

            # Val dataset info
            if hasattr(cfg, 'val_dataloader') and cfg.val_dataloader is not None:
                val_ann_path = get_ann_path(cfg.val_dataloader)
                val_info = load_ann_info(val_ann_path)
                if val_info:
                    cfg.dataset_val = val_info
                    print(f"  Val dataset info added from: {val_ann_path}")

            # General val dataset info from GeneralValHook
            if hasattr(cfg, 'custom_hooks') and cfg.custom_hooks is not None:
                for hook in cfg.custom_hooks:
                    if hook.get('type') == 'GeneralValHook':
                        general_val_ann = hook.get('evaluator', {}).get('ann_file')
                        if general_val_ann:
                            general_val_info = load_ann_info(general_val_ann)
                            if general_val_info:
                                cfg.dataset_general_val = general_val_info
                                print(f"  General val dataset info added from: {general_val_ann}")
                        break
        except Exception as e:
            print(f"Warning: Could not add dataset info to config: {e}")

    # build the runner from config
    runner = Runner.from_cfg(cfg)

    # Log annotation JSON files as artifacts to MLflow
    if args.mlflow_tracking_uri is not None:
        try:
            import mlflow
            # Log train annotation file
            if hasattr(cfg, 'train_dataloader') and cfg.train_dataloader is not None:
                train_ann_path = get_ann_path(cfg.train_dataloader)
                if osp.exists(train_ann_path):
                    mlflow.log_artifact(train_ann_path)
                    print(f"  Logged train annotations: {osp.basename(train_ann_path)}")
                else:
                    print(f"  Warning: Train annotations not found: {train_ann_path}")
            # Log val annotation file
            if hasattr(cfg, 'val_dataloader') and cfg.val_dataloader is not None:
                val_ann_path = get_ann_path(cfg.val_dataloader)
                if osp.exists(val_ann_path):
                    mlflow.log_artifact(val_ann_path)
                    print(f"  Logged val annotations: {osp.basename(val_ann_path)}")
                else:
                    print(f"  Warning: Val annotations not found: {val_ann_path}")
            # Log mirrored_val annotation file from MirroredValHook if present
            if hasattr(cfg, 'custom_hooks') and cfg.custom_hooks is not None:
                for hook in cfg.custom_hooks:
                    if hook.get('type') == 'MirroredValHook':
                        mirrored_val_ann_file = hook.get('evaluator', {}).get('ann_file')
                        if mirrored_val_ann_file:
                            if osp.exists(mirrored_val_ann_file):
                                mlflow.log_artifact(mirrored_val_ann_file)
                                print(f"  Logged mirrored_val annotations: {osp.basename(mirrored_val_ann_file)}")
                            else:
                                print(f"  Warning: Mirrored val annotations not found: {mirrored_val_ann_file}")
                        break
            # Log general_val annotation file from GeneralValHook if present
            if hasattr(cfg, 'custom_hooks') and cfg.custom_hooks is not None:
                for hook in cfg.custom_hooks:
                    if hook.get('type') == 'GeneralValHook':
                        general_val_ann_file = hook.get('evaluator', {}).get('ann_file')
                        if general_val_ann_file:
                            if osp.exists(general_val_ann_file):
                                mlflow.log_artifact(general_val_ann_file)
                                print(f"  Logged general_val annotations: {osp.basename(general_val_ann_file)}")
                            else:
                                print(f"  Warning: General val annotations not found: {general_val_ann_file}")
                        break
        except Exception as e:
            print(f"Warning: Could not log annotation files to MLflow: {e}")

    # start training
    runner.train()


if __name__ == '__main__':
    main()
