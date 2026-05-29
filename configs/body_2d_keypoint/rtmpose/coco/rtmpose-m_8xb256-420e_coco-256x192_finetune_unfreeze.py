_base_ = ['./rtmpose-m_8xb256-420e_coco-256x192_finetune.py']

# =============================================================================
# Backbone Unfreeze Variant (layer-wise LR decay)
# =============================================================================
# Goal: best primary-domain (coco/AP) AND general-domain (general_val/AP)
# accuracy for each unfreeze depth, i.e. adapt the new domain while minimizing
# catastrophic forgetting.
#
# Two coupled mechanisms:
#
# 1) Unfreeze depth is controlled by the single integer
#    `model.backbone.frozen_stages`, which IS overridable via --cfg-options
#    (unlike the base config's dotted `custom_keys`, which merge_from_dict cannot
#    target). Freeze ladder (P5 = stem + 4 stages):
#       frozen_stages=4  -> whole backbone frozen (baseline, default here)
#       frozen_stages=3  -> train stage4 only
#       frozen_stages=2  -> train stage3 + stage4
#       frozen_stages=-1 -> train entire backbone
#
# 2) Layer-wise LR decay (CSPNeXtLayerDecayOptimWrapperConstructor): whatever
#    stays trainable gets a per-stage lr that decays toward the stem, so general
#    low-level features barely move (protecting general_val) while the top stage
#    adapts (improving coco/AP). The head trains at the full base lr.
#       lr(stage s) = base_lr * backbone_top_lr_mult * layer_decay_rate**(top-s)
#    With base_lr=1e-4, backbone_top_lr_mult=0.1, layer_decay_rate=0.5:
#       stage4=1e-5, stage3=5e-6, stage2=2.5e-6, stage1=1.25e-6, stem=6.25e-7
#    (stage4 == the conventional 0.1x backbone lr, so the stage4-only run is
#    unchanged vs a flat 0.1 setup; deeper stages get progressively gentler.)
#    The constructor also keeps norm (BN gamma/beta) and bias params out of
#    weight decay - which a blanket custom_keys={'backbone': ...} does not do -
#    and skips frozen params entirely.

optim_wrapper = dict(
    # _delete_ drops the inherited paramwise_cfg (custom_keys / norm_decay_mult /
    # bias_decay_mult); weight-decay handling is done inside the constructor.
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=1e-4, weight_decay=0.05),
    constructor='CSPNeXtLayerDecayOptimWrapperConstructor',
    paramwise_cfg=dict(
        backbone_top_lr_mult=0.1,
        layer_decay_rate=0.5,
    ))

# Cosine annealing must use a per-group RELATIVE minimum (eta_min_ratio) instead
# of the base config's absolute eta_min (1e-6): with layer-wise decay the deepest
# stages have lr below 1e-6, and an absolute eta_min would push their lr back up.
# eta_min_ratio=0.01 means every group decays to 1% of its own lr.
max_epochs = 50
base_lr = 1e-4
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0e-2,
        by_epoch=False,
        begin=0,
        end=500),
    dict(
        type='CosineAnnealingLR',
        eta_min_ratio=0.01,
        begin=5,
        end=max_epochs,
        T_max=max_epochs - 5,
        by_epoch=True,
        convert_to_iter_based=True),
]

# Default fully-frozen backbone (equivalent to the baseline); the sweep
# overrides this per experiment. norm_eval=True is inherited from the base so
# BatchNorm running stats stay frozen on this small data.
model = dict(backbone=dict(frozen_stages=4))
