_base_ = ['./v4_unfreeze.py']

# Same train construction as the v5p2 x1 run: the fold train set once, plus
# Lying_Person_Dataset_Single_Ann.json once. The sweep points datasets.0 at
# rishi_jamie or tim_nina. Those dirs sit two levels under data/. The lying
# JSON sits one level under data/.
_fold_frames = '../../App_Vids_v1_GBE_v1_lyingperson_InfiniteForm_frames/'
_lying_frames = '../App_Vids_v1_GBE_v1_lyingperson_InfiniteForm_frames/'

train_dataloader = dict(
    dataset=dict(
        _delete_=True,
        type='CombinedDataset',
        metainfo=dict(from_file='configs/_base_/datasets/coco.py'),
        datasets=[
            dict(
                type='CocoDataset',
                data_root='data/App_Vids_v2_GBE_v4_folds/rishi_jamie/',
                data_mode='topdown',
                ann_file='annotations/person_keypoints_train2017.json',
                data_prefix=dict(img=_fold_frames),
                pipeline=[],
            ),
            dict(
                type='RepeatDataset',
                times=1,
                dataset=dict(
                    type='CocoDataset',
                    data_root='data/Lying_Person_Dataset_Single_Ann/',
                    data_mode='topdown',
                    ann_file='Lying_Person_Dataset_Single_Ann.json',
                    data_prefix=dict(img=_lying_frames),
                    pipeline=[],
                ),
            ),
        ],
        pipeline=_base_.train_pipeline,
        test_mode=False,
    ))
