"""
Hook to evaluate on an extra validation dataset during training.

Same behavior as :class:`MirroredValHook`, but the MLflow / log metric namespace
is configurable via ``metric_prefix`` (CocoMetric ``prefix``).
"""

import copy
import torch
from mmengine.hooks import Hook
from mmengine.registry import HOOKS
from mmengine.runner import Runner
from mmengine.evaluator import Evaluator


@HOOKS.register_module()
class CustomDatasetHook(Hook):
    """Evaluate on a secondary validation dataset with a custom metric prefix.

    Args:
        dataloader (dict): Dataloader config for the extra validation dataset.
        evaluator (dict): Evaluator config (typically CocoMetric).
        metric_prefix (str): Prefix for logged metrics (e.g. ``'heel_slides'``
            -> ``heel_slides/AP`` in MLflow). Passed to the evaluator as
            ``prefix`` unless the evaluator config already sets ``prefix``.
        interval (int): Run every N epochs. Default: 1.
        priority (int): Hook priority. Default: 48.
    """

    def __init__(
        self,
        dataloader: dict,
        evaluator: dict,
        metric_prefix: str = 'custom',
        interval: int = 1,
        priority: int = 48,
    ):
        self.priority = priority
        self.metric_prefix = metric_prefix
        self.dataloader_cfg = copy.deepcopy(dataloader)
        self.evaluator_cfg = copy.deepcopy(evaluator)
        self.interval = interval
        self._dataloader = None
        self._evaluator = None
        self._message_hub_key = f'{metric_prefix.replace("/", "_")}_metrics'

    def before_run(self, runner: Runner) -> None:
        """Build dataloader and evaluator before training starts."""
        dataloader_cfg = copy.deepcopy(self.dataloader_cfg)
        dataloader_cfg.setdefault('persistent_workers', True)
        dataloader_cfg.setdefault('drop_last', False)
        dataloader_cfg.setdefault('sampler', dict(type='DefaultSampler', shuffle=False))

        self._dataloader = Runner.build_dataloader(dataloader_cfg)

        evaluator_cfg = copy.deepcopy(self.evaluator_cfg)
        evaluator_cfg.setdefault('prefix', self.metric_prefix)
        self._evaluator = Evaluator(evaluator_cfg)

        dataset_meta = getattr(self._dataloader.dataset, 'metainfo', None)
        if dataset_meta:
            for metric in self._evaluator.metrics:
                metric.dataset_meta = dataset_meta

        runner.logger.info(
            f'CustomDatasetHook[{self.metric_prefix}]: Built dataloader with '
            f'{len(self._dataloader.dataset)} samples')

    def after_val_epoch(self, runner: Runner, metrics: dict = None) -> None:
        """Run secondary validation after each primary validation."""
        epoch = runner.epoch

        if epoch % self.interval != 0:
            return

        if self._dataloader is None or self._evaluator is None:
            runner.logger.warning(
                f'CustomDatasetHook[{self.metric_prefix}]: dataloader or '
                'evaluator not initialized!')
            return

        runner.logger.info(
            f'Running CustomDatasetHook[{self.metric_prefix}] validation at '
            f'epoch {epoch}...')

        model = runner.model
        was_training = model.training
        model.eval()

        try:
            with torch.no_grad():
                for idx, data_batch in enumerate(self._dataloader):
                    outputs = model.val_step(data_batch)
                    self._evaluator.process(
                        data_samples=outputs,
                        data_batch=data_batch,
                    )

            extra_metrics = self._evaluator.evaluate(len(self._dataloader.dataset))

            runner.logger.info(
                f'CustomDatasetHook[{self.metric_prefix}] results: {extra_metrics}')

            if runner.visualizer is not None:
                for key, value in extra_metrics.items():
                    if isinstance(value, (int, float)):
                        runner.visualizer.add_scalar(key, value, step=epoch)

            runner.message_hub.update_info(self._message_hub_key, extra_metrics)

        except Exception as e:
            runner.logger.error(
                f'CustomDatasetHook[{self.metric_prefix}] failed: {e}')
            import traceback
            runner.logger.error(traceback.format_exc())

        finally:
            if was_training:
                model.train()
