_base_ = ['./v5_unfreeze.py']

# Train is the require_ske_gbe app-vid set once, plus the lying-person set
# repeated `times` times. DefaultSampler then shuffles that list, so each
# lying-person image is drawn `times` times per epoch and each app-vid image
# once. The sweep sets times to 1, 2, or 4. Val stays the require_ske_gbe split.
_frames = '../App_Vids_v1_GBE_v1_lyingperson_InfiniteForm_frames/'

train_dataloader = dict(
    dataset=dict(
        _delete_=True,
        type='CombinedDataset',
        metainfo=dict(from_file='configs/_base_/datasets/coco.py'),
        datasets=[
            dict(
                type='CocoDataset',
                data_root='data/App_Vids_v2_GBE_v4_val10_subject25_require_ske_gbe/',
                data_mode='topdown',
                ann_file='annotations/person_keypoints_train2017.json',
                data_prefix=dict(img=_frames),
                pipeline=[],
            ),
            dict(
                type='RepeatDataset',
                times=2,
                dataset=dict(
                    type='CocoDataset',
                    data_root='data/Lying_Person_Dataset_Single_Ann/',
                    data_mode='topdown',
                    ann_file='Lying_Person_Dataset_Single_Ann.json',
                    data_prefix=dict(img=_frames),
                    pipeline=[],
                ),
            ),
        ],
        pipeline=_base_.train_pipeline,
        test_mode=False,
    ))
