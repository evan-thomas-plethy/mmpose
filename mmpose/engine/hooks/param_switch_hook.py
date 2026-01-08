# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional

from mmengine.hooks import Hook
from mmengine.runner import Runner

from mmpose.registry import HOOKS


@HOOKS.register_module()
class ParamSwitchHook(Hook):
    """Switch optimizer parameter-wise config at a specific epoch.

    This hook allows dynamically changing the learning rate multiplier for
    specific parameter groups (e.g., backbone) during training. This is useful
    for staged unfreezing strategies where you want to freeze the backbone
    initially and then unfreeze it after a few epochs.

    Args:
        switch_epoch (int): The epoch at which to switch the paramwise config.
        new_paramwise_cfg (dict): The new paramwise config to apply. Should
            contain 'custom_keys' with the parameter group settings.
            Example: {'custom_keys': {'backbone': {'lr_mult': 0.1}}}

    Example:
        >>> # In config file:
        >>> custom_hooks = [
        >>>     dict(
        >>>         type='ParamSwitchHook',
        >>>         switch_epoch=2,
        >>>         new_paramwise_cfg=dict(
        >>>             custom_keys={'backbone': dict(lr_mult=0.1)}
        >>>         )
        >>>     )
        >>> ]
    """

    def __init__(self,
                 switch_epoch: int,
                 new_paramwise_cfg: Dict):
        self.switch_epoch = switch_epoch
        self.new_paramwise_cfg = new_paramwise_cfg
        self._switched = False

    def before_train_epoch(self, runner: Runner):
        """Check if we should switch params at this epoch."""
        epoch = runner.epoch
        
        if epoch == self.switch_epoch and not self._switched:
            self._apply_paramwise_cfg(runner)
            self._switched = True

    def _apply_paramwise_cfg(self, runner: Runner):
        """Apply the new paramwise config to the optimizer."""
        optimizer = runner.optim_wrapper.optimizer
        
        # Get custom_keys from new config
        custom_keys = self.new_paramwise_cfg.get('custom_keys', {})
        
        for param_name, param_cfg in custom_keys.items():
            new_lr_mult = param_cfg.get('lr_mult')
            if new_lr_mult is None:
                continue
                
            # Find and update matching parameter groups
            base_lr = runner.optim_wrapper.optimizer.defaults.get('lr', 1e-4)
            new_lr = base_lr * new_lr_mult
            
            updated_count = 0
            for param_group in optimizer.param_groups:
                # Check if this param group matches the key
                # mmengine typically stores the param name pattern in the group
                group_name = param_group.get('name', '')
                
                if param_name in group_name or self._matches_param_group(
                        param_group, param_name, runner):
                    old_lr = param_group['lr']
                    param_group['lr'] = new_lr
                    updated_count += 1
                    runner.logger.info(
                        f"ParamSwitchHook: Updated '{group_name}' lr: "
                        f"{old_lr:.2e} -> {new_lr:.2e} (lr_mult={new_lr_mult})")
            
            if updated_count == 0:
                # Fallback: update all param groups that might be backbone
                # by checking parameter names in the model
                runner.logger.warning(
                    f"ParamSwitchHook: Could not find param group matching "
                    f"'{param_name}'. Attempting fallback matching...")
                self._fallback_update(runner, param_name, new_lr_mult)

    def _matches_param_group(self, param_group: Dict, key: str, 
                             runner: Runner) -> bool:
        """Check if a param group matches the given key."""
        # Try to match by checking if any parameter in the group
        # belongs to the specified module (e.g., 'backbone')
        return False  # Conservative: rely on group name matching

    def _fallback_update(self, runner: Runner, param_name: str, 
                         new_lr_mult: float):
        """Fallback: update params by checking model structure."""
        from mmengine.model import is_model_wrapper
        
        model = runner.model
        if is_model_wrapper(model):
            model = model.module
        
        # Get the target module
        target_module = None
        for name, module in model.named_modules():
            if name == param_name:
                target_module = module
                break
        
        if target_module is None:
            runner.logger.warning(
                f"ParamSwitchHook: Module '{param_name}' not found in model")
            return
        
        # Get parameter IDs for the target module
        target_param_ids = {id(p) for p in target_module.parameters()}
        
        base_lr = runner.optim_wrapper.optimizer.defaults.get('lr', 1e-4)
        new_lr = base_lr * new_lr_mult
        
        optimizer = runner.optim_wrapper.optimizer
        updated_count = 0
        
        for param_group in optimizer.param_groups:
            # Check if any param in this group belongs to target module
            for param in param_group['params']:
                if id(param) in target_param_ids:
                    old_lr = param_group['lr']
                    param_group['lr'] = new_lr
                    updated_count += 1
                    runner.logger.info(
                        f"ParamSwitchHook (fallback): Updated param group lr: "
                        f"{old_lr:.2e} -> {new_lr:.2e} (lr_mult={new_lr_mult})")
                    break  # Only update once per group
        
        runner.logger.info(
            f"ParamSwitchHook: Updated {updated_count} param groups for "
            f"'{param_name}'")

