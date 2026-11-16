Scripts for building, cleaning, and checking COCO-format pose datasets.

**augment.py** — Applies Albumentations from a YAML config to a COCO image directory and annotation JSON, appending new images and annotations. `make_augmented_dataset.py` calls this so the source tree is not edited directly.

**chroma_key_augment.py** — Replaces a near-white background in each video under an input directory with a still image. A frame is composited only when more than one third of its pixels pass the brightness key.

**count_images_and_annotations.py** — Prints the number of images and annotations in a COCO JSON.

**filter_images.py** — Copies into an output directory only the frames listed in a COCO annotations JSON.

**filter_val_data.py** — Splits a target dataset into COCO train and val by video name, using the video prefixes in a reference val JSON. Frames from the same video stay in one split, and only the output directory is written.

**make_augmented_dataset.py** — Copies a COCO dataset to a new directory and runs `augment.py` there. Supports train-only augmentation and a custom augmentations YAML.

**match_anns.py** — Writes a filtered copy of a base annotations JSON, keeping only image ids that appear in a reference JSON.

**merge_datasets.py** — Merges two COCO pose trees into a new output directory. Train is always combined; val is either merged or kept as `val2017` from the first dataset and `val2017_2` from the second.

**merge_with_seed.py** — Merges a full COCO dataset with a seeded sample of a second dataset (COCO layout or a flat JSON plus `frames/`). Output val is exactly the first dataset's val, including mirrored val when that dataset has it.

**mirror_keypoints.py** — Swaps left/right COCO keypoint pairs in an annotation file to correct mislabeled sides. Images are not flipped.

**mirror_val_data.py** — Writes a horizontally flipped copy of a val split: images plus keypoints, boxes, and segmentations, with left/right keypoint pairs swapped.

**overlay_dataset.py** — Draws COCO keypoints, skeleton, and boxes on dataset frames and writes a review video.

**remove_bad_data.py** — Drops frames whose video-name prefix is on a hardcoded bad-video list, and drops the annotations linked to those frames.

**remove_filename_prefix_annotations.py** — Strips directory prefixes from `file_name` fields and writes a sibling `*_normalized` JSON.

**remove_keypoints.py** — Writes a copy of an annotations JSON with selected keypoint indices set to `[0, 0, 0]`. The input file is left unchanged.

**remove_prefix_from_anns.py** — Filters a COCO annotations JSON down to images that exist in a given images directory, and drops annotations for the removed images.

**scale_bboxes.py** — Scales each COCO bounding box about its center. The default scale factor is 1.2.

**single_frame_overlay.py** — Picks one random frame per video (name taken from the text before `_frame_`) and stitches those frames into one video, with the video name drawn on each frame.

**split_dataset.py** — Splits a `frames/` plus annotation JSON directory into a COCO train/val tree. All frames from the same video stay in one split.

**train_val_leakage.py** — Reports video-id prefixes shared by two COCO annotation JSONs. A shared prefix means frames from the same video appear in both files.
