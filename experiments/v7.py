_base_ = ['./v4_unfreeze.py']

# Four held-out hooks. experiments/v4_unfreeze.py has two, so this replaces
# custom_hooks. Indices stay aligned with the v4 sweeps:
#   0 EMAHook
#   1 PipelineSwitchHook (stage2 length is 0, so it does not switch)
#   2 GeneralValHook
#   3 first subject's non-ske-gbe val
#   4 first subject's ske val
#   5 first subject's gbe val
#   6 second subject's non-ske-gbe val
#   7 second subject's ske val
#   8 second subject's gbe val
# The sweep points these at held_out_<subject>_non_ske_gbe.json,
# held_out_<subject>_ske.json, and held_out_<subject>_gbe.json.

dataset_type = 'CocoDataset'
data_mode = 'topdown'
general_val_data_root = 'data/coco_general_val/'
custom_data_root = 'data/custom/'
custom_data_root_2 = 'data/custom_2/'
custom_data_root_3 = 'data/custom_3/'
custom_data_root_4 = 'data/custom_4/'
custom_data_root_5 = 'data/custom_5/'
custom_data_root_6 = 'data/custom_6/'
backend_args = dict(backend='local')
val_pipeline = [
    dict(type='LoadImage', backend_args=backend_args),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=(192, 256)),
    dict(type='PackPoseInputs')
]
train_pipeline_stage2 = [
    dict(type='LoadImage', backend_args=backend_args),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomFlip', direction='horizontal'),
    dict(
        type='RandomBBoxTransform',
        shift_factor=0.,
        scale_factor=[0.9, 1.1],
        rotate_factor=10),
    dict(type='TopdownAffine', input_size=(192, 256)),
    dict(type='mmdet.YOLOXHSVRandomAug'),
    dict(
        type='Albumentation',
        transforms=[
            dict(type='Blur', p=0.1),
            dict(type='MedianBlur', p=0.1),
            dict(
                type='CoarseDropout',
                max_holes=1,
                max_height=0.2,
                max_width=0.2,
                min_holes=1,
                min_height=0.1,
                min_width=0.1,
                p=0.3),
        ]),
    dict(type='GenerateTarget', encoder=dict(
        type='SimCCLabel',
        input_size=(192, 256),
        sigma=(4.9, 5.66),
        simcc_split_ratio=2.0,
        normalize=False,
        use_dark=False)),
    dict(type='PackPoseInputs')
]


custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        update_buffers=True,
        priority=49),
    dict(
        type='mmdet.PipelineSwitchHook',
        switch_epoch=30,
        switch_pipeline=train_pipeline_stage2),
    dict(
        type='GeneralValHook',
        interval=1,
        priority=48,
        dataloader=dict(
            batch_size=16,
            num_workers=4,
            dataset=dict(
                type=dataset_type,
                data_root=general_val_data_root,
                data_mode=data_mode,
                ann_file='annotations/general_val.json',
                data_prefix=dict(img='general_val/'),
                test_mode=True,
                pipeline=val_pipeline,
            ),
        ),
        evaluator=dict(
            type='CocoMetric',
            ann_file=general_val_data_root + 'annotations/general_val.json',
        ),
    ),
    dict(
        type='CustomDatasetHook',
        metric_prefix='custom',
        interval=1,
        priority=48,
        dataloader=dict(
            batch_size=16,
            num_workers=4,
            dataset=dict(
                type=dataset_type,
                data_root=custom_data_root,
                data_mode=data_mode,
                ann_file='annotations/person_keypoints_val2017.json',
                data_prefix=dict(img='custom/'),
                test_mode=True,
                pipeline=val_pipeline,
            ),
        ),
        evaluator=dict(
            type='CocoMetric',
            ann_file=custom_data_root + 'annotations/person_keypoints_val2017.json',
        ),
    ),
    dict(
        type='CustomDatasetHook',
        metric_prefix='custom_2',
        interval=1,
        priority=48,
        dataloader=dict(
            batch_size=16,
            num_workers=4,
            dataset=dict(
                type=dataset_type,
                data_root=custom_data_root_2,
                data_mode=data_mode,
                ann_file='annotations/person_keypoints_val2017.json',
                data_prefix=dict(img='custom/'),
                test_mode=True,
                pipeline=val_pipeline,
            ),
        ),
        evaluator=dict(
            type='CocoMetric',
            ann_file=custom_data_root_2 + 'annotations/person_keypoints_val2017.json',
        ),
    ),
    dict(
        type='CustomDatasetHook',
        metric_prefix='custom_3',
        interval=1,
        priority=48,
        dataloader=dict(
            batch_size=16,
            num_workers=4,
            dataset=dict(
                type=dataset_type,
                data_root=custom_data_root_3,
                data_mode=data_mode,
                ann_file='annotations/person_keypoints_val2017.json',
                data_prefix=dict(img='custom/'),
                test_mode=True,
                pipeline=val_pipeline,
            ),
        ),
        evaluator=dict(
            type='CocoMetric',
            ann_file=custom_data_root_3 + 'annotations/person_keypoints_val2017.json',
        ),
    ),
    dict(
        type='CustomDatasetHook',
        metric_prefix='custom_4',
        interval=1,
        priority=48,
        dataloader=dict(
            batch_size=16,
            num_workers=4,
            dataset=dict(
                type=dataset_type,
                data_root=custom_data_root_4,
                data_mode=data_mode,
                ann_file='annotations/person_keypoints_val2017.json',
                data_prefix=dict(img='custom/'),
                test_mode=True,
                pipeline=val_pipeline,
            ),
        ),
        evaluator=dict(
            type='CocoMetric',
            ann_file=custom_data_root_4 + 'annotations/person_keypoints_val2017.json',
        ),
    ),
    dict(
        type='CustomDatasetHook',
        metric_prefix='custom_5',
        interval=1,
        priority=48,
        dataloader=dict(
            batch_size=16,
            num_workers=4,
            dataset=dict(
                type=dataset_type,
                data_root=custom_data_root_5,
                data_mode=data_mode,
                ann_file='annotations/person_keypoints_val2017.json',
                data_prefix=dict(img='custom/'),
                test_mode=True,
                pipeline=val_pipeline,
            ),
        ),
        evaluator=dict(
            type='CocoMetric',
            ann_file=custom_data_root_5 + 'annotations/person_keypoints_val2017.json',
        ),
    ),
    dict(
        type='CustomDatasetHook',
        metric_prefix='custom_6',
        interval=1,
        priority=48,
        dataloader=dict(
            batch_size=16,
            num_workers=4,
            dataset=dict(
                type=dataset_type,
                data_root=custom_data_root_6,
                data_mode=data_mode,
                ann_file='annotations/person_keypoints_val2017.json',
                data_prefix=dict(img='custom/'),
                test_mode=True,
                pipeline=val_pipeline,
            ),
        ),
        evaluator=dict(
            type='CocoMetric',
            ann_file=custom_data_root_6 + 'annotations/person_keypoints_val2017.json',
        ),
    ),
]
