_base_ = ['./v1.py']

# Same unfreeze as experiments/v5_unfreeze.py, on the v1 hook layout.
# v1.py has MirroredValHook at custom_hooks.2 and GeneralValHook at
# custom_hooks.3. This dataset has no seated-knee or ground-based val files.

max_epochs = 30
base_lr = 1e-4

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.05),
    constructor='CSPNeXtLayerDecayOptimWrapperConstructor',
    paramwise_cfg=dict(
        backbone_top_lr_mult=0.1,
        layer_decay_rate=0.5,
    ))

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0e-2,
        begin=0,
        end=2,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        eta_min_ratio=0.01,
        begin=2,
        end=max_epochs,
        T_max=max_epochs - 2,
        by_epoch=True,
        convert_to_iter_based=True),
]

model = dict(backbone=dict(frozen_stages=4))
