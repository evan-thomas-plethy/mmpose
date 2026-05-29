# Copyright (c) OpenMMLab. All rights reserved.
"""Extended linear LR warmup for head param groups only."""

from mmengine.hooks import Hook
from mmengine.runner import Runner

from mmpose.registry import HOOKS


def _is_head_param_group(param_group, param_id_to_name):
    """True when every param in the group belongs to the pose head."""
    names = [param_id_to_name.get(id(p), '') for p in param_group['params']]
    return bool(names) and all(n.startswith('head.') for n in names)


@HOOKS.register_module()
class HeadWarmupHook(Hook):
    """Linear LR warmup for head param groups over a longer duration.

    Backbone (and other) param groups follow ``param_scheduler`` as usual.
    Head groups are overwritten each iteration until ``head_warmup_iters`` with
    a linear ramp from ``start_factor * base_lr`` to ``base_lr``.

    Runs after ``ParamSchedulerHook`` (priority 55 > 50) so head warmup can
    extend beyond the global ``LinearLR`` window without slowing the backbone.

    Args:
        head_warmup_iters (int): Iterations for head to reach full LR. Default
            500 matches the base config global warmup; use 1000/1500 etc. to
            keep the head at a lower LR longer while the backbone finishes
            warming up and begins adapting.
        start_factor (float): Starting LR multiplier for the head. Default 0.01
            matches the base ``LinearLR`` ``start_factor``.
    """

    priority = 55

    def __init__(self, head_warmup_iters: int = 500, start_factor: float = 0.01):
        self.head_warmup_iters = head_warmup_iters
        self.start_factor = start_factor
        self._head_base_lrs = None
        self._head_group_ids = None

    def before_train(self, runner: Runner) -> None:
        model = runner.model.module if hasattr(runner.model, 'module') else runner.model
        param_id_to_name = {id(p): n for n, p in model.named_parameters()}
        optim = runner.optim_wrapper.optimizer

        self._head_base_lrs = {}
        self._head_group_ids = set()
        for idx, group in enumerate(optim.param_groups):
            if _is_head_param_group(group, param_id_to_name):
                self._head_group_ids.add(idx)
                self._head_base_lrs[idx] = group['lr']

    def before_train_iter(self, runner: Runner, batch_idx: int,
                          data_batch=None) -> None:
        if self._head_group_ids is None:
            return

        cur_iter = runner.iter
        if cur_iter >= self.head_warmup_iters:
            return

        progress = float(cur_iter + 1) / float(self.head_warmup_iters)
        factor = self.start_factor + (1.0 - self.start_factor) * progress

        optim = runner.optim_wrapper.optimizer
        for idx in self._head_group_ids:
            base_lr = self._head_base_lrs[idx]
            optim.param_groups[idx]['lr'] = base_lr * factor
