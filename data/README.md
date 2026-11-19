Scripts for building, cleaning, and checking COCO-format pose datasets.

**augment.py** — Applies Albumentations from a YAML config to a COCO image directory and annotation JSON, appending new images and annotations. `make_augmented_dataset.py` calls this so the source tree is not edited directly.

**chroma_key_augment.py** — Replaces a near-white background in each video under an input directory with a still image. A frame is composited only when more than one third of its pixels pass the brightness key.

**count_images_and_annotations.py** — Prints the number of images and annotations in a COCO JSON.

**filter_images.py** — Copies into an output directory only the frames listed in a COCO annotations JSON.

**filter_train_data.py** — Caps the subject with the most frames in a training COCO JSON by dropping one random video of that subject at a time, then counting again. Other subjects are not trimmed. It stops before a removal that would put that subject at or under `--min-fraction-train-max-subject` (default 0.25), or below the subject with the second-most frames. Subjects come from a `video_id,subject` CSV. `--require-videos` keeps those ids by exact match. If the cap cannot be met without dropping a required video, it prints that and still writes the output. Removed video ids and the `info.config` block are written to `videos_removed.json` beside the output JSON. `--removed-json` drops every frame of the videos in that file, does not run the subject cap, and copies that config onto the output. `--frame-limit` skips the share cap and drops videos from every subject until the next video would put that subject under the limit. `input`, `annotation`, `output`, `subjects-path`, `max-subject-train-fraction`, `min-fraction-train-max-subject`, `frame-limit`, and `require-videos` follow this call; the other fields stay as saved. `max-subject-train-fraction` records the second-place stop, and `min-fraction-train-max-subject` records that floor. A frame-limit run records `frame-limit` instead of those two fields. Both modes require the subjects CSV and print each subject's share of the remaining frames. A sibling `person_keypoints_val2017.json` is copied next to the output training JSON. The input is not modified.

**filter_val_data.py** — Splits a target dataset into COCO train and val by video name, using the video prefixes in a reference val JSON. Every frame of those videos is removed from train. `--exact-val` copies that val JSON unchanged and still removes every frame of those videos from the target train JSON.

**make_augmented_dataset.py** — Copies a COCO dataset to a new directory and runs `augment.py` there. Supports train-only augmentation and a custom augmentations YAML.

**match_anns.py** — Writes a filtered copy of a base annotations JSON, keeping only image ids that appear in a reference JSON.

**merge_datasets.py** — Merges two COCO pose trees into a new output directory. Train is always combined; val is either merged or kept as `val2017` from the first dataset and `val2017_2` from the second.

**merge_jsons.py** — Merges two COCO JSON files into one new JSON. Ids from the first file are kept, and image and annotation ids from the second file are remapped. Categories come from the first file. `info` is stored as `source1_info`, `source2_info`, and so on. The script stops if the same frame filename appears in both files. Neither input is modified.

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

**held_out_subjects.py** — Moves every frame of the named subjects into `held_out/annotations/person_keypoints_val2017.json` and writes the other frames to `train_pool/` for `split_dataset.py`. The inner split then validates only on the remaining subjects. The finetune config scores that held-out JSON with `CustomDatasetHook`. The input JSON is not modified.

**split_dataset.py** — Splits a `frames/` plus annotation JSON directory into a COCO train/val tree. All frames from the same video stay in one split. `--no-frames` writes only the train and val JSON files. With a subjects CSV and `--max-subject-val-fraction`, videos whose id is `{name}_{exercise}_{date}_{id}` and that share the name and date stay in one split; any other id is its own group. Named train-only subjects stay in train, and no subject can exceed that fraction of val. `--max-subject-train-fraction` drops individual videos of the over-limit subject at random. `--require-videos` keeps those ids by exact match. The flags used for the run are stored at `info.config` in both output JSON files.

**train_val_leakage.py** — Reports video-id prefixes shared by two COCO annotation JSONs. A shared prefix means frames from the same video appear in both files.
