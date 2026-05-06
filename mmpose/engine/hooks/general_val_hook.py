"""
Hook to evaluate on a secondary validation dataset during training.

This hook runs evaluation on a "general" dataset after each primary validation,
logging metrics with a different prefix to track catastrophic forgetting.
"""

import copy
import torch
from mmengine.hooks import Hook
from mmengine.registry import HOOKS
from mmengine.runner import Runner
from mmengine.evaluator import Evaluator


@HOOKS.register_module()
class GeneralValHook(Hook):
    """Evaluate on a secondary validation dataset to monitor forgetting.
    
    This hook runs after each validation epoch and evaluates the model
    on a general/diverse dataset, logging metrics with a 'general_val/' prefix.
    
    Args:
        dataloader (dict): Dataloader config for general validation dataset.
        evaluator (dict): Evaluator config (typically CocoMetric).
        interval (int): Run general validation every N epochs. Default: 1.
        priority (int): Hook priority. Lower runs earlier in each hook phase;
            use below ``EMAHook`` (e.g. 49) so ``after_val_epoch`` runs before EMA
            restores training weights. Default: 48.
        
    Example config:
        custom_hooks = [
            dict(
                type='GeneralValHook',
                dataloader=dict(
                    batch_size=16,
                    num_workers=4,
                    dataset=dict(
                        type='CocoDataset',
                        data_root='data/coco_general_val/',
                        ann_file='annotations/person_keypoints_val2017.json',
                        data_prefix=dict(img='val2017/'),
                        test_mode=True,
                        pipeline=val_pipeline,
                    ),
                ),
                evaluator=dict(
                    type='CocoMetric',
                    ann_file='data/coco_general_val/annotations/person_keypoints_val2017.json',
                ),
            ),
        ]
    """
    
    def __init__(
        self,
        dataloader: dict,
        evaluator: dict,
        interval: int = 1,
        priority: int = 48,
    ):
        self.priority = priority
        # Deep copy to avoid modifying original config
        self.dataloader_cfg = copy.deepcopy(dataloader)
        self.evaluator_cfg = copy.deepcopy(evaluator)
        self.interval = interval
        self._dataloader = None
        self._evaluator = None
    
    def before_run(self, runner: Runner) -> None:
        """Build dataloader and evaluator before training starts."""
        # Build dataloader using Runner's static method
        dataloader_cfg = copy.deepcopy(self.dataloader_cfg)
        dataloader_cfg.setdefault('persistent_workers', True)
        dataloader_cfg.setdefault('drop_last', False)
        dataloader_cfg.setdefault('sampler', dict(type='DefaultSampler', shuffle=False))
        
        self._dataloader = Runner.build_dataloader(dataloader_cfg)
        
        # Build evaluator with prefix
        evaluator_cfg = copy.deepcopy(self.evaluator_cfg)
        evaluator_cfg.setdefault('prefix', 'general_val')
        self._evaluator = Evaluator(evaluator_cfg)
        
        # Set dataset_meta on the evaluator's metrics (required by CocoMetric)
        dataset_meta = getattr(self._dataloader.dataset, 'metainfo', None)
        if dataset_meta:
            for metric in self._evaluator.metrics:
                metric.dataset_meta = dataset_meta
        
        runner.logger.info(f'GeneralValHook: Built dataloader with {len(self._dataloader.dataset)} samples')
    
    def after_val_epoch(self, runner: Runner, metrics: dict = None) -> None:
        """Run general validation after each primary validation."""
        epoch = runner.epoch # NOT 0-indexed
        
        if epoch % self.interval != 0:
            return
        
        if self._dataloader is None or self._evaluator is None:
            runner.logger.warning('GeneralValHook: dataloader or evaluator not initialized!')
            return
        
        runner.logger.info(f'Running general validation at epoch {epoch}...')
        
        # Run inference on general dataset
        # Metrics are automatically reset by the evaluator
        model = runner.model
        was_training = model.training
        model.eval()
        
        try:
            with torch.no_grad():
                for idx, data_batch in enumerate(self._dataloader):
                    # Forward pass
                    outputs = model.val_step(data_batch)
                    
                    # Process outputs for evaluation
                    self._evaluator.process(
                        data_samples=outputs,
                        data_batch=data_batch,
                    )
            
            # Compute metrics
            general_metrics = self._evaluator.evaluate(len(self._dataloader.dataset))
            
            # Log metrics
            runner.logger.info(f'General validation results: {general_metrics}')
            
            # Log to visualizer (which logs to MLflow)
            if runner.visualizer is not None:
                for key, value in general_metrics.items():
                    if isinstance(value, (int, float)):
                        runner.visualizer.add_scalar(key, value, step=epoch)
            
            # Also add to message hub so it appears in logs
            runner.message_hub.update_info('general_val_metrics', general_metrics)
            
        except Exception as e:
            runner.logger.error(f'GeneralValHook failed: {e}')
            import traceback
            runner.logger.error(traceback.format_exc())

        finally:
            if was_training:
                model.train()