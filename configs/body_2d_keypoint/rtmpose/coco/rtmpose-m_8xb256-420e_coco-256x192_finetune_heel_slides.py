_base_ = ['../../../_base_/default_runtime.py']

# =============================================================================
# Heel Slides Exercise Fine-tuning Configuration
# =============================================================================
# Optimized for sidelying persons performing heel slide exercises.

# Load pretrained checkpoint for fine-tuning
load_from = 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-aic-coco_pt-aic-coco_420e-256x192-63eb25f7_20230126.pth'

# runtime - optimized for fine-tuning (SHORT training to prevent forgetting)
max_epochs = 15  # Reduced from 50: heel slides AP peaks early, more epochs = more forgetting
stage2_num_epochs = 5  # Last 5 epochs use refined augmentation
base_lr = 1e-4  # Conservative learning rate for fine-tuning

train_cfg = dict(max_epochs=max_epochs, val_interval=1)
randomness = dict(seed=21)

# optimizer - optimized for fine-tuning with differential learning rates
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.05),
    paramwise_cfg=dict(
        norm_decay_mult=0,
        bias_decay_mult=0,
        bypass_duplicate=True,
        custom_keys={
            # 'backbone': dict(lr_mult=0.0),
            'backbone.stem': dict(lr_mult=0.0),
            'backbone.stage1': dict(lr_mult=0.0),
            'backbone.stage2': dict(lr_mult=0.0),
            'backbone.stage3': dict(lr_mult=0.0),
            'backbone.stage4': dict(lr_mult=0.0),
        }))

# learning rate schedule - adjusted for short 15-epoch training
# With 8000 samples, batch_size=32: ~250 iters/epoch, 3750 total iters
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0e-2,  # Start at 1e-6, warmup to 1e-4
        by_epoch=False,
        begin=0,
        end=250),  # ~1 epoch warmup (reduced from 500)
    dict(
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.1,  # End at 1e-5
        begin=3,  # Start cosine annealing at epoch 3 (after peak)
        end=max_epochs,
        T_max=max_epochs - 3,
        by_epoch=True,
        convert_to_iter_based=True),
]

# automatically scaling LR based on the actual training batch size
auto_scale_lr = dict(base_batch_size=1024)

# codec settings
codec = dict(
    type='SimCCLabel',
    input_size=(192, 256),
    sigma=(4.9, 5.66),
    simcc_split_ratio=2.0,
    normalize=False,
    use_dark=False)

# model settings
model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(
        _scope_='mmdet',
        type='CSPNeXt',
        arch='P5',
        expand_ratio=0.5,
        deepen_factor=0.67,
        widen_factor=0.75,
        out_indices=(4, ),
        channel_attention=True,
        norm_cfg=dict(type='SyncBN'),
        act_cfg=dict(type='SiLU'),
        frozen_stages=-1, # 0 freezes stem, 1 freezes stage1 + stem, ...
        # Keep BatchNorm in eval mode during training to prevent running
        # statistics from shifting to heel slides domain. Critical when using
        # low backbone LR (lr_mult=0.1) or fully frozen backbone (lr_mult=0).
        norm_eval=True),
    head=dict(
        type='RTMCCHead',
        in_channels=768,
        out_channels=17,
        input_size=codec['input_size'],
        in_featuremap_size=tuple([s // 32 for s in codec['input_size']]),
        simcc_split_ratio=codec['simcc_split_ratio'],
        final_layer_kernel_size=7,
        gau_cfg=dict(
            hidden_dims=256,
            s=128,
            expansion_factor=2,
            dropout_rate=0.,
            drop_path=0.,
            act_fn='SiLU',
            use_rel_bias=False,
            pos_enc=False),
        loss=dict(
            type='KLDiscretLoss',
            use_target_weight=True,
            beta=10.,
            label_softmax=True),
        decoder=codec),
    test_cfg=dict(flip_test=False))

# base dataset settings
dataset_type = 'CocoDataset'
data_mode = 'topdown'
data_root = 'data/coco/'  # Heel slides data
general_val_data_root = 'data/coco_general_val/'  # General exercise validation

backend_args = dict(backend='local')

# pipelines - optimized for heel slides (sidelying poses)
train_pipeline = [
    dict(type='LoadImage', backend_args=backend_args),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomFlip', direction='horizontal'),
    # RandomHalfBody removed - not suitable for sidelying full-body poses
    dict(
        type='RandomBBoxTransform',
        scale_factor=[0.8, 1.2],  # Conservative scaling to maintain exercise context
        rotate_factor=20),  # Reduced rotation - sidelying poses have consistent orientation
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='mmdet.YOLOXHSVRandomAug'),
    dict(
        type='Albumentation',
        transforms=[
            dict(type='Blur', p=0.1),
            dict(type='MedianBlur', p=0.1),
            dict(
                type='CoarseDropout',
                max_holes=1,
                max_height=0.25,  # Reduced to preserve leg/heel visibility
                max_width=0.25,
                min_holes=1,
                min_height=0.1,
                min_width=0.1,
                p=0.5),  # Reduced probability - heel visibility is critical
        ]),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs')
]
val_pipeline = [
    dict(type='LoadImage', backend_args=backend_args),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='PackPoseInputs')
]

# Stage 2 pipeline - even more conservative augmentation for refinement
train_pipeline_stage2 = [
    dict(type='LoadImage', backend_args=backend_args),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomFlip', direction='horizontal'),
    # RandomHalfBody removed - not suitable for sidelying full-body poses
    dict(
        type='RandomBBoxTransform',
        shift_factor=0.,
        scale_factor=[0.9, 1.1],  # Very conservative for final refinement
        rotate_factor=10),  # Minimal rotation in stage 2
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='mmdet.YOLOXHSVRandomAug'),
    dict(
        type='Albumentation',
        transforms=[
            dict(type='Blur', p=0.1),
            dict(type='MedianBlur', p=0.1),
            dict(
                type='CoarseDropout',
                max_holes=1,
                max_height=0.2,  # Further reduced for stage 2
                max_width=0.2,
                min_holes=1,
                min_height=0.1,
                min_width=0.1,
                p=0.3),  # Lower probability in refinement stage
        ]),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs')
]

# data loaders - optimized for fine-tuning (reduced batch size for memory)
train_dataloader = dict(
    batch_size=32,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='annotations/person_keypoints_train2017.json',
        data_prefix=dict(img='train2017/'),
        pipeline=train_pipeline,
    ))

# Validation: heel slides only (primary domain)
# General exercise validation is done via general_val_hook
val_dataloader = dict(
    batch_size=16,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='annotations/person_keypoints_val2017.json',
        data_prefix=dict(img='val2017/'),
        test_mode=True,
        pipeline=val_pipeline,
    ))

test_dataloader = val_dataloader

# hooks - optimized for fine-tuning
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=3,  # Save every 3 epochs (at 3, 6, 9, 12, 15)
        save_best='coco/AP',  # Save best based on heel slides validation
        rule='greater',
        max_keep_ckpts=5))  # Keep more checkpoints for short training

custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0001,  # Lower momentum for fine-tuning
        update_buffers=True,
        priority=49),
    # # THIS CODE DOES NOT WORK - do not uncomment
    # # Backbone unfreezing hook - switches backbone lr_mult from 0.0 to 0.1
    # # This allows the head to adapt first, then backbone fine-tuning begins
    # dict(
    #     type='ParamSwitchHook',
    #     switch_epoch=0,
    #     new_paramwise_cfg=dict(
    #         custom_keys={'backbone': dict(lr_mult=0.1)}
    #     )),
    dict(
        type='mmdet.PipelineSwitchHook',
        switch_epoch=max_epochs - stage2_num_epochs,
        switch_pipeline=train_pipeline_stage2),
    # Mirrored validation hook - assesses bias for left/right pose orientation
    dict(
        type='MirroredValHook',
        interval=1,
        dataloader=dict(
            batch_size=16,
            num_workers=4,
            dataset=dict(
                type=dataset_type,
                data_root=data_root,
                data_mode=data_mode,
                ann_file='annotations/person_keypoints_val2017_mirrored.json',
                data_prefix=dict(img='val2017_mirrored/'),
                test_mode=True,
                pipeline=val_pipeline,
            ),
        ),
        evaluator=dict(
            type='CocoMetric',
            ann_file=data_root + 'annotations/person_keypoints_val2017_mirrored.json',
        ),
    ),
    # General validation hook - monitors forgetting on diverse exercises
    dict(
        type='GeneralValHook',
        interval=1,
        dataloader=dict(
            batch_size=16,
            num_workers=4,
            dataset=dict(
                type=dataset_type,
                data_root=general_val_data_root,
                data_mode=data_mode,
                ann_file='annotations/annotations.json',
                data_prefix=dict(img='val/'),
                test_mode=True,
                pipeline=val_pipeline,
            ),
        ),
        evaluator=dict(
            type='CocoMetric',
            ann_file=general_val_data_root + 'annotations/annotations.json',
        ),
    ),
]

# Evaluator for heel slides validation
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/person_keypoints_val2017.json')

test_evaluator = val_evaluator
