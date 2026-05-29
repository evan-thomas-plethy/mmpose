# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
from datetime import datetime

import mmengine
from mmengine.config import Config, DictAction
from mmengine.hooks import Hook
from mmengine.runner import Runner


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMPose test (and eval) model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--work-dir', help='the directory to save evaluation results')
    parser.add_argument('--out', help='the file to save metric results.')
    parser.add_argument(
        '--dump',
        type=str,
        help='dump predictions to a pickle file for offline evaluation')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        default={},
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. For example, '
        "'--cfg-options model.backbone.depth=18 model.backbone.with_cp=True'")
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
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/test.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    parser.add_argument(
        '--badcase',
        action='store_true',
        help='whether analyze badcase in test')
    # MLflow arguments
    parser.add_argument(
        '--mlflow-tracking-uri',
        type=str,
        default='http://52.41.68.196:5000',
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
        help='MLflow run name (defaults to "pretrain_eval")')
    parser.add_argument(
        '--mlflow-append-to-existing-run',
        action='store_true',
        help='Attach to the latest existing run in the experiment whose '
        'mlflow.runName tag matches --mlflow-run-name. Log only metrics under '
        '--metric-prefix (no new params). Requires --mlflow-run-name and '
        '--mlflow-experiment-name.')
    # Multi-dataset evaluation
    parser.add_argument(
        '--eval-general',
        action='store_true',
        help='Also evaluate on general validation dataset (requires GeneralValHook config)')
    parser.add_argument(
        '--general-data-root',
        type=str,
        default=None,
        help='Override general validation data root')
    parser.add_argument(
        '--print-model',
        action='store_true',
        help='Print model architecture/layers and exit')
    # Override validation/test dataset (for running on arbitrary val images + anns)
    parser.add_argument(
        '--val-annotations',
        type=str,
        default=None,
        help='Absolute or cwd-relative path to the COCO JSON file. Use together with '
        '--val-images-dir (mutually exclusive with --val-data-root / --val-ann-file).')
    parser.add_argument(
        '--val-images-dir',
        type=str,
        default=None,
        help='Directory containing image files (file_name in JSON is relative to this '
        'directory). Use with --val-annotations.')
    parser.add_argument(
        '--val-data-root',
        type=str,
        default=None,
        help='Data root for validation/test (overrides config). Use with --val-ann-file '
        '(and optionally --val-img-prefix). Alternative: --val-annotations + --val-images-dir.')
    parser.add_argument(
        '--val-ann-file',
        type=str,
        default=None,
        help='Annotation path relative to --val-data-root (e.g. annotations/person_keypoints_val2017.json).')
    parser.add_argument(
        '--val-img-prefix',
        type=str,
        default=None,
        help='Image subfolder under --val-data-root (e.g. val2017/ or frames/). Default: use config.')
    parser.add_argument(
        '--metric-prefix',
        type=str,
        default='coco',
        help='Prefix for primary test metrics (MLflow groups by the segment before the '
        'first "/"). Default: coco. Use e.g. heel_slides when evaluating on a custom val set.')
    parser.add_argument(
        '--general-metric-prefix',
        type=str,
        default='general_val',
        help='Prefix for metrics from --eval-general (default: general_val).')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def merge_args(cfg, args):
    """Merge CLI arguments to config."""

    cfg.launcher = args.launcher
    cfg.load_from = args.checkpoint

    # -------------------- work directory --------------------
    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    # -------------------- visualization --------------------
    if (args.show and not args.badcase) or (args.show_dir is not None):
        assert 'visualization' in cfg.default_hooks, \
            'PoseVisualizationHook is not set in the ' \
            '`default_hooks` field of config. Please set ' \
            '`visualization=dict(type="PoseVisualizationHook")`'

        cfg.default_hooks.visualization.enable = True
        cfg.default_hooks.visualization.show = False \
            if args.badcase else args.show
        if args.show:
            cfg.default_hooks.visualization.wait_time = args.wait_time
        cfg.default_hooks.visualization.out_dir = args.show_dir
        cfg.default_hooks.visualization.interval = args.interval

    # -------------------- badcase analyze --------------------
    if args.badcase:
        assert 'badcase' in cfg.default_hooks, \
            'BadcaseAnalyzeHook is not set in the ' \
            '`default_hooks` field of config. Please set ' \
            '`badcase=dict(type="BadcaseAnalyzeHook")`'

        cfg.default_hooks.badcase.enable = True
        cfg.default_hooks.badcase.show = args.show
        if args.show:
            cfg.default_hooks.badcase.wait_time = args.wait_time
        cfg.default_hooks.badcase.interval = args.interval

        metric_type = cfg.default_hooks.badcase.get('metric_type', 'loss')
        if metric_type not in ['loss', 'accuracy']:
            raise ValueError('Only support badcase metric type'
                             "in ['loss', 'accuracy']")

        if metric_type == 'loss':
            if not cfg.default_hooks.badcase.get('metric'):
                cfg.default_hooks.badcase.metric = cfg.model.head.loss
        else:
            if not cfg.default_hooks.badcase.get('metric'):
                cfg.default_hooks.badcase.metric = cfg.test_evaluator

    # -------------------- Dump predictions --------------------
    if args.dump is not None:
        assert args.dump.endswith(('.pkl', '.pickle')), \
            'The dump file must be a pkl file.'
        dump_metric = dict(type='DumpResults', out_file_path=args.dump)
        if isinstance(cfg.test_evaluator, (list, tuple)):
            cfg.test_evaluator = [*cfg.test_evaluator, dump_metric]
        else:
            cfg.test_evaluator = [cfg.test_evaluator, dump_metric]

    # -------------------- Other arguments --------------------
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # -------------------- Validation dataset override --------------------
    use_explicit = (args.val_annotations is not None
                    or args.val_images_dir is not None)
    use_root_style = args.val_data_root is not None

    if use_explicit and use_root_style:
        raise ValueError(
            'Use either (--val-annotations and --val-images-dir) or '
            '(--val-data-root and --val-ann-file), not both.')

    if use_explicit:
        if args.val_annotations is None or args.val_images_dir is None:
            raise ValueError(
                '--val-annotations and --val-images-dir must be given together.')
        ann_path = osp.abspath(args.val_annotations)
        img_dir = osp.abspath(args.val_images_dir)
        if not osp.isfile(ann_path):
            raise ValueError(f'--val-annotations is not a file: {ann_path}')
        if not osp.isdir(img_dir):
            raise ValueError(f'--val-images-dir is not a directory: {img_dir}')
        cfg.val_dataloader.dataset.data_root = img_dir
        cfg.val_dataloader.dataset.ann_file = ann_path
        cfg.val_dataloader.dataset.data_prefix = dict(img='')
        if isinstance(cfg.val_evaluator, dict):
            cfg.val_evaluator = dict(cfg.val_evaluator, ann_file=ann_path)
        else:
            cfg.val_evaluator = [
                dict(ev, ann_file=ann_path) if ev.get('type') == 'CocoMetric' else ev
                for ev in cfg.val_evaluator]
        cfg.test_dataloader = cfg.val_dataloader
        cfg.test_evaluator = cfg.val_evaluator
        if args.dump is not None:
            dump_metric = dict(type='DumpResults', out_file_path=args.dump)
            if isinstance(cfg.test_evaluator, (list, tuple)):
                cfg.test_evaluator = [*cfg.test_evaluator, dump_metric]
            else:
                cfg.test_evaluator = [cfg.test_evaluator, dump_metric]
        skip_hooks = ('MirroredValHook', 'CustomDatasetHook', 'GeneralValHook')
        cfg.custom_hooks = [
            h for h in cfg.get('custom_hooks', [])
            if h.get('type') not in skip_hooks
        ]
    elif use_root_style:
        if args.val_ann_file is None:
            raise ValueError('--val-data-root requires --val-ann-file')
        cfg.val_dataloader.dataset.data_root = args.val_data_root
        cfg.val_dataloader.dataset.ann_file = args.val_ann_file
        if args.val_img_prefix is not None:
            cfg.val_dataloader.dataset.data_prefix = dict(img=args.val_img_prefix)
        ann_path = osp.join(args.val_data_root, args.val_ann_file) if not osp.isabs(
            args.val_ann_file) else args.val_ann_file
        if isinstance(cfg.val_evaluator, dict):
            cfg.val_evaluator = dict(cfg.val_evaluator, ann_file=ann_path)
        else:
            cfg.val_evaluator = [
                dict(ev, ann_file=ann_path) if ev.get('type') == 'CocoMetric' else ev
                for ev in cfg.val_evaluator]
        cfg.test_dataloader = cfg.val_dataloader
        cfg.test_evaluator = cfg.val_evaluator
        if args.dump is not None:
            dump_metric = dict(type='DumpResults', out_file_path=args.dump)
            if isinstance(cfg.test_evaluator, (list, tuple)):
                cfg.test_evaluator = [*cfg.test_evaluator, dump_metric]
            else:
                cfg.test_evaluator = [cfg.test_evaluator, dump_metric]
        skip_hooks = ('MirroredValHook', 'CustomDatasetHook', 'GeneralValHook')
        cfg.custom_hooks = [
            h for h in cfg.get('custom_hooks', [])
            if h.get('type') not in skip_hooks
        ]

    return cfg


def normalize_metric_prefix(prefix: str, default: str) -> str:
    """Strip whitespace; empty -> default. No slashes (single MLflow namespace segment)."""
    p = (prefix or default).strip()
    if not p:
        p = default
    p = p.rstrip('/')
    if '/' in p or '\\' in p:
        raise ValueError(
            f'--metric-prefix must be a single namespace (no slashes), got {prefix!r}')
    return p


def find_latest_run_id_by_name(tracking_uri: str, experiment_name: str,
                               run_name: str) -> str:
    """Return run_id of the most recent run with tag mlflow.runName == run_name."""
    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri)
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        raise ValueError(f'MLflow experiment not found: {experiment_name!r}')

    # MLflow filter: escape single quotes in run_name
    safe = run_name.replace("'", "''")
    filter_string = f'tags."mlflow.runName" = \'{safe}\''
    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string=filter_string,
        max_results=100,
        order_by=['attribute.start_time DESC'],
    )
    if not runs:
        raise ValueError(
            f'No MLflow run with tag mlflow.runName={run_name!r} in experiment '
            f'{experiment_name!r}')
    return runs[0].info.run_id


def evaluate_on_dataset(cfg, checkpoint, data_root, ann_file, img_prefix, 
                        metric_prefix='coco', work_dir=None):
    """Evaluate model on a specific dataset.
    
    Args:
        cfg: Base config
        checkpoint: Path to checkpoint file
        data_root: Root directory of dataset
        ann_file: Annotation file path (relative to data_root)
        img_prefix: Image directory prefix (relative to data_root)
        metric_prefix: Prefix for metric names (e.g., 'coco' or 'general_val')
        work_dir: Working directory
    
    Returns:
        dict: Evaluation metrics with prefixed names
    """
    import copy
    eval_cfg = copy.deepcopy(cfg)
    
    # Update test dataloader
    eval_cfg.test_dataloader.dataset.data_root = data_root
    eval_cfg.test_dataloader.dataset.ann_file = ann_file
    eval_cfg.test_dataloader.dataset.data_prefix = dict(img=img_prefix)
    
    # Update test evaluator
    if isinstance(eval_cfg.test_evaluator, dict):
        eval_cfg.test_evaluator['ann_file'] = osp.join(data_root, ann_file)
    elif isinstance(eval_cfg.test_evaluator, list):
        for evaluator in eval_cfg.test_evaluator:
            if evaluator.get('type') == 'CocoMetric':
                evaluator['ann_file'] = osp.join(data_root, ann_file)
    
    # Set work dir
    if work_dir:
        eval_cfg.work_dir = work_dir
    
    # Remove custom hooks that might interfere (like GeneralValHook)
    eval_cfg.custom_hooks = [
        hook for hook in eval_cfg.get('custom_hooks', [])
        if hook.get('type') not in ['GeneralValHook']
    ]
    
    # Build runner and test
    runner = Runner.from_cfg(eval_cfg)
    metrics = runner.test()
    
    # Prefix metrics
    prefixed_metrics = {}
    for key, value in metrics.items():
        # Replace 'coco/' prefix with the desired prefix
        if key.startswith('coco/'):
            new_key = f"{metric_prefix}/{key[5:]}"  # Remove 'coco/' and add new prefix
        else:
            new_key = f"{metric_prefix}/{key}"
        prefixed_metrics[new_key] = value
    
    return prefixed_metrics


def main():
    args = parse_args()

    # load config
    cfg = Config.fromfile(args.config)
    cfg = merge_args(cfg, args)

    # Determine MLflow settings
    mlflow_client = None
    run_id = None
    mlflow_append = args.mlflow_append_to_existing_run
    if args.mlflow_tracking_uri:
        try:
            import mlflow
            mlflow.set_tracking_uri(args.mlflow_tracking_uri)

            if args.mlflow_experiment_name is None:
                exp_name = osp.splitext(osp.basename(args.config))[0]
            else:
                exp_name = args.mlflow_experiment_name

            mlflow.set_experiment(exp_name)

            if args.mlflow_run_name is None:
                run_name = "pretrain_eval_" + datetime.now().strftime("%Y%m%d_%H%M%S")
            else:
                run_name = args.mlflow_run_name

            if mlflow_append:
                if args.mlflow_run_name is None or args.mlflow_experiment_name is None:
                    raise ValueError(
                        '--mlflow-append-to-existing-run requires '
                        '--mlflow-run-name and --mlflow-experiment-name')
                run_id = find_latest_run_id_by_name(
                    args.mlflow_tracking_uri, exp_name, run_name)
                mlflow.start_run(run_id=run_id)
                print(f"MLflow: appending metrics to existing run "
                      f"{run_id} (runName={run_name!r}, experiment={exp_name!r})")
            else:
                mlflow.start_run(run_name=run_name)
                run_id = mlflow.active_run().info.run_id

            print(f"MLflow logging enabled:")
            print(f"  Tracking URI: {args.mlflow_tracking_uri}")
            print(f"  Experiment: {exp_name}")
            print(f"  Run name: {run_name}")
            print(f"  Run ID: {run_id}")

            if not mlflow_append:
                mlflow.log_param("checkpoint", args.checkpoint)
                mlflow.log_param("config", args.config)
                mlflow.log_param(
                    "primary_metric_prefix",
                    normalize_metric_prefix(args.metric_prefix, 'coco'))
                mlflow.log_param(
                    "general_metric_prefix",
                    normalize_metric_prefix(args.general_metric_prefix, 'general_val'))
                if args.val_annotations is not None:
                    mlflow.log_param("val_annotations", osp.abspath(args.val_annotations))
                if args.val_images_dir is not None:
                    mlflow.log_param("val_images_dir", osp.abspath(args.val_images_dir))

            mlflow_client = mlflow
        except ImportError:
            print("Warning: mlflow not installed, skipping MLflow logging")
        except Exception as e:
            print(f"Warning: Failed to initialize MLflow: {e}")

    all_metrics = {}
    
    primary_metric_prefix = normalize_metric_prefix(args.metric_prefix, 'coco')
    general_metric_prefix = normalize_metric_prefix(args.general_metric_prefix, 'general_val')

    # ==================== Primary Evaluation (main test dataloader) ====================
    print("\n" + "="*60)
    print(f"Evaluating on primary test dataset (metrics prefix: {primary_metric_prefix}/)...")
    print("="*60)
    
    # Build runner and run test
    runner = Runner.from_cfg(cfg)
    
    # Print model layers if requested
    if args.print_model:
        print("\n" + "="*60)
        print("MODEL ARCHITECTURE")
        print("="*60)
        print(runner.model)
        print("\n" + "="*60)
        print("NAMED MODULES (layer by layer)")
        print("="*60)
        for name, module in runner.model.named_modules():
            if name:  # Skip root module (empty name)
                print(f"{name}: {module.__class__.__name__}")
        print("\n" + "="*60)
        print("PARAMETERS SUMMARY")
        print("="*60)
        total_params = 0
        trainable_params = 0
        for name, param in runner.model.named_parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
            print(f"{name}: {list(param.shape)}, requires_grad={param.requires_grad}")
        print(f"\nTotal parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        return {}
    
    if args.out:
        class SaveMetricHook(Hook):
            def after_test_epoch(self, _, metrics=None):
                if metrics is not None:
                    mmengine.dump(metrics, args.out)
        runner.register_hook(SaveMetricHook(), 'LOWEST')
    
    primary_metrics = runner.test()

    # Remap CocoMetric keys (coco/...) to --metric-prefix for MLflow / dashboards
    for key, value in primary_metrics.items():
        if key.startswith('coco/'):
            new_key = f"{primary_metric_prefix}/{key[5:]}"
        else:
            new_key = f"{primary_metric_prefix}/{key}"
        all_metrics[new_key] = value

    print(f"\nPrimary ({primary_metric_prefix}/) metrics: {primary_metrics}")
    
    # ==================== General Validation Evaluation ====================
    if args.eval_general:
        print("\n" + "="*60)
        print("Evaluating on general validation dataset...")
        print("="*60)
        
        # Find general_val_data_root from config or CLI
        general_data_root = args.general_data_root
        if general_data_root is None:
            general_data_root = cfg.get('general_val_data_root', None)
        
        if general_data_root is None:
            print("Warning: --eval-general specified but no general_val_data_root found.")
            print("Use --general-data-root to specify the path.")
        else:
            general_metrics = evaluate_on_dataset(
                cfg=cfg,
                checkpoint=args.checkpoint,
                data_root=general_data_root,
                ann_file='annotations/person_keypoints_val2017.json',
                img_prefix='val2017/',
                metric_prefix=general_metric_prefix,
                work_dir=cfg.work_dir,
            )
            all_metrics.update(general_metrics)
            print(f"\nGeneral validation metrics: {general_metrics}")
    
    # ==================== Log to MLflow ====================
    if mlflow_client:
        try:
            if mlflow_append:
                # Only log primary-eval metrics (e.g. heel_slides/AP), not general_val
                pfx = primary_metric_prefix + '/'
                to_log = {
                    k: v
                    for k, v in all_metrics.items()
                    if isinstance(v, (int, float)) and k.startswith(pfx)
                }
                if not to_log:
                    print(f"\nWarning: no metrics under {pfx!r} to log to MLflow.")
                for key, value in to_log.items():
                    mlflow_client.log_metric(key, value, step=0)
                print(f"\nLogged {len(to_log)} metrics to MLflow (prefix {primary_metric_prefix}/):")
                for key, value in sorted(to_log.items()):
                    print(f"  {key}: {value:.4f}")
            else:
                for key, value in all_metrics.items():
                    if isinstance(value, (int, float)):
                        mlflow_client.log_metric(key, value, step=0)
                print(f"\nLogged metrics to MLflow at step 0:")
                for key, value in all_metrics.items():
                    if isinstance(value, (int, float)):
                        print(f"  {key}: {value:.4f}")

            mlflow_client.end_run()
            print(f"\nMLflow run completed: {run_id}")
        except Exception as e:
            print(f"Warning: Failed to log to MLflow: {e}")
            try:
                mlflow_client.end_run()
            except:
                pass
    
    # ==================== Summary ====================
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    for key, value in sorted(all_metrics.items()):
        if isinstance(value, (int, float)):
            print(f"  {key}: {value:.4f}")
    
    return all_metrics


if __name__ == '__main__':
    main()
