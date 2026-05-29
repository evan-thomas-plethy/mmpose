# Copyright (c) OpenMMLab. All rights reserved.
import logging

from mmengine.logging import print_log
from mmengine.optim import DefaultOptimWrapperConstructor

from mmpose.registry import OPTIM_WRAPPER_CONSTRUCTORS


def _cspnext_stage_id(name: str):
    """Stage index of a CSPNeXt backbone param (stem=0, stageN=N).

    Returns None for non-backbone params (e.g. the head), which are treated as
    the top of the network and trained at the full base lr.
    """
    if not name.startswith('backbone.'):
        return None
    token = name.split('.')[1]
    if token == 'stem':
        return 0
    if token.startswith('stage'):
        try:
            return int(token[len('stage'):])
        except ValueError:
            return None
    return None


@OPTIM_WRAPPER_CONSTRUCTORS.register_module()
class CSPNeXtLayerDecayOptimWrapperConstructor(DefaultOptimWrapperConstructor):
    """Layer-wise LR decay constructor for CSPNeXt-backbone finetuning.

    Backbone stages receive progressively smaller learning rates toward the
    input stem. This keeps general, low-level pretrained features close to their
    pretrained values (mitigating catastrophic forgetting and protecting
    general-domain accuracy) while letting the high-level stage adapt to the new
    domain. The head (and any other non-backbone param) trains at the full base
    lr.

    For a backbone stage ``s`` (stem=0, stage1=1, ..., top stage ``N``)::

        lr = base_lr * backbone_top_lr_mult * layer_decay_rate ** (N - s)

    Weight decay follows the standard finetuning convention: normalization
    affine params and biases (``ndim == 1`` or name ends with ``.bias``) are
    excluded from weight decay; all other params use the optimizer
    ``weight_decay``. Unlike a plain ``custom_keys={'backbone': ...}`` setup,
    this correctly keeps BatchNorm gamma/beta out of weight decay.

    Frozen params (``requires_grad=False``, e.g. set by
    ``model.backbone.frozen_stages``) are skipped entirely, so they receive no
    optimizer state and never change.

    ``paramwise_cfg`` fields:
        backbone_top_lr_mult (float): lr multiplier of the top backbone stage
            relative to ``base_lr``. Default 0.1, so the top stage matches the
            conventional 0.1x backbone finetuning lr.
        layer_decay_rate (float): per-stage decay applied toward the stem.
            Default 0.5.
    """

    def add_params(self, params, module, **kwargs):
        backbone_top_lr_mult = self.paramwise_cfg.get('backbone_top_lr_mult',
                                                       0.1)
        layer_decay_rate = self.paramwise_cfg.get('layer_decay_rate', 0.5)
        base_wd = self.base_wd if self.base_wd is not None else 0.0

        # Determine the top (largest) backbone stage id actually present so the
        # top trainable stage always gets exactly `backbone_top_lr_mult`.
        top_id = 0
        for name, _ in module.named_parameters():
            sid = _cspnext_stage_id(name)
            if sid is not None:
                top_id = max(top_id, sid)

        groups: dict = {}
        for name, param in module.named_parameters():
            if not param.requires_grad:
                continue  # frozen (e.g. backbone.frozen_stages)

            sid = _cspnext_stage_id(name)
            if sid is None:
                lr_mult = 1.0
                tag = 'head'
            else:
                lr_mult = backbone_top_lr_mult * (
                    layer_decay_rate**(top_id - sid))
                tag = f'backbone_stage{sid}'

            no_decay = (param.ndim == 1) or name.endswith('.bias')
            wd = 0.0 if no_decay else base_wd
            key = f'{tag}_{"no_decay" if no_decay else "decay"}'

            if key not in groups:
                groups[key] = {
                    'params': [],
                    'param_names': [],
                    'lr': self.base_lr * lr_mult,
                    'lr_mult': lr_mult,
                    'weight_decay': wd,
                }
            groups[key]['params'].append(param)
            groups[key]['param_names'].append(name)

        for key in sorted(groups):
            g = groups[key]
            print_log(
                f'paramwise group {key}: n_params={len(g["params"])} '
                f'lr={g["lr"]:.3e} (lr_mult={g["lr_mult"]:.5g}) '
                f'weight_decay={g["weight_decay"]}',
                logger='current',
                level=logging.INFO)
            params.append(g)
