import csv
import json
import os
import random
import shutil
from collections import defaultdict

"""
python split_dataset.py --input Heel_Slides_v1/ --output coco/ --val-ratio 0.19 --seed 46
python split_dataset.py --input Heel_Slides_v1/ --output coco/ \
    --subjects-path subjects.csv --subjects-val AnthonyGeorge \
    --subjects-exclude JaneDoe

# App vids v4, 10% val. YordanosWoldebirhan stays in train. No person is
# more than 25% of val. Same user and date stay in one split.
python split_dataset.py \
    --input /home/ubuntu/visionAI/mmpose/data/App_Vids_11-06-2026/v4 \
    --output /home/ubuntu/visionAI/mmpose/data/App_Vids_11-06-2026/v4_val10 \
    --val-ratio 0.10 \
    --subjects-path /home/ubuntu/visionAI/mmpose/data/App_Vids_11-06-2026/users.csv \
    --train-only-subjects YordanosWoldebirhan \
    --max-subject-val-fraction 0.25 \
    --no-frames \
    --seed 42

# Same rules, plus no person is more than 25% of train. Videos may be dropped.
# Ids in already_2b_annotated_gbe.txt are kept.
python split_dataset.py \
    --input /home/ubuntu/visionAI/mmpose/data/App_Vids_11-06-2026/v4 \
    --output /home/ubuntu/visionAI/mmpose/data/App_Vids_11-06-2026/v4_val10_train25 \
    --val-ratio 0.10 \
    --subjects-path /home/ubuntu/visionAI/mmpose/data/App_Vids_11-06-2026/users.csv \
    --train-only-subjects YordanosWoldebirhan \
    --max-subject-val-fraction 0.25 \
    --max-subject-train-fraction 0.25 \
    --require-videos /home/ubuntu/visionAI/Vision-ML/app_vids_dataset/training_dataset/outputs_txt/already_2b_annotated_gbe.txt \
    --no-frames \
    --seed 42
"""

_MONTHS = {
    "jan", "feb", "mar", "apr", "may", "jun",
    "jul", "aug", "sep", "oct", "nov", "dec",
}


def load_video_subjects(subjects_path):
    """Load a video_id,subject CSV. A header row is skipped when present."""
    video_subjects = {}
    with open(subjects_path, newline="") as f:
        for row_number, row in enumerate(csv.reader(f), start=1):
            if not row or not row[0].strip():
                continue
            if len(row) < 2 or not row[1].strip():
                raise ValueError(
                    f"{subjects_path}:{row_number} must be video_id,subject"
                )
            video_id = row[0].strip()
            subject = row[1].strip()
            if row_number == 1 and video_id.lower() in {"video_id", "video"}:
                continue
            if video_id in video_subjects:
                raise ValueError(
                    f"Duplicate video_id in {subjects_path}: {video_id}"
                )
            video_subjects[video_id] = subject
    if not video_subjects:
        raise ValueError(f"No video_id,subject rows found in {subjects_path}")
    return video_subjects


def is_date_token(token):
    """Six digits (080326), or a 3-letter month then digits (Jun0226)."""
    if len(token) == 6 and token.isdigit():
        return True
    return (
        len(token) > 3
        and token[:3].lower() in _MONTHS
        and token[3:].isdigit()
    )


def parsed_name_date(video_id):
    """Return (name, date) for {name}_{exercise}_{date}_{id}, else None.

    Same four-part stem rule as exercise_name_from_video_id in
    Vision-ML/app_vids_dataset/training_dataset/video_id_utils.py.
    """
    parts = video_id.split("_")
    if len(parts) != 4 or not parts[0] or not parts[1]:
        return None
    if not is_date_token(parts[2]) or len(parts[3]) != 6:
        return None
    return parts[0], parts[2]


def load_id_list(path):
    """Load video ids. A line may be video_id or video_id: date."""
    video_ids = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            video_id = line.split(":", 1)[0].strip()
            if video_id:
                video_ids.append(video_id)
    if not video_ids:
        raise ValueError(f"No video ids found in {path}")
    return video_ids


def required_video_set(video_ids):
    """Ids that must stay in the dataset. Matching is exact."""
    return set(video_ids)


def subject_for_video(video, video_subjects):
    if video_subjects and video in video_subjects:
        return video_subjects[video]
    parsed = parsed_name_date(video)
    if parsed:
        return parsed[0]
    return video


def _over_fraction(frames, total, fraction):
    if total <= 0:
        return frames > 0
    return frames > fraction * total


def build_user_day_groups(video_to_images, video_subjects, train_only_subjects, required_ids):
    """Group videos that share a parsed name and date. Anything else is alone."""
    buckets = defaultdict(list)
    missing_subject = []
    for video in video_to_images:
        parsed = parsed_name_date(video)
        if parsed:
            buckets[("day", parsed[0], parsed[1])].append(video)
        else:
            buckets[("video", video)].append(video)
        if video_subjects is not None and video not in video_subjects:
            missing_subject.append(video)
    if missing_subject:
        preview = ", ".join(missing_subject[:10])
        extra = f" (+{len(missing_subject) - 10} more)" if len(missing_subject) > 10 else ""
        print(
            f"⚠️ {len(missing_subject)} videos are not in the subjects CSV "
            f"({preview}{extra}). Each uses the name in its id, or the id "
            "itself when the id has no name and date."
        )

    train_only = set(train_only_subjects or [])
    groups = []
    for key, videos in buckets.items():
        subject_frames = defaultdict(int)
        for video in videos:
            subject = subject_for_video(video, video_subjects)
            subject_frames[subject] += len(video_to_images[video])
        groups.append({
            "key": key,
            "videos": videos,
            "frames": sum(len(video_to_images[video]) for video in videos),
            "subject_frames": dict(subject_frames),
            "protected": any(video in required_ids for video in videos),
            "train_only": any(
                subject_for_video(video, video_subjects) in train_only
                for video in videos
            ),
        })

    present = set()
    for group in groups:
        present.update(group["subject_frames"])
    unused = sorted(train_only - present)
    if unused:
        print(f"⚠️ Train-only subjects with no videos: {', '.join(unused)}")
    day_groups = sum(1 for group in groups if group["key"][0] == "day")
    print(
        f"👤 User-day groups: {day_groups} shared-day groups, "
        f"{len(groups) - day_groups} videos on their own day"
    )
    return groups


def assign_val_groups(groups, val_ratio, max_val_fraction):
    """Pick val groups at random until the frame target, within the subject cap.

    Train-only groups stay out. A group is added only when every subject in it
    stays at or under max_val_fraction of the val target (or of the val set,
    once the target is reached). After that, random groups are removed until
    the cap holds on the actual val size.
    """
    total_frames = sum(group["frames"] for group in groups)
    if total_frames <= 0:
        raise ValueError("No frames left to split")
    target = val_ratio * total_frames
    eligible = [group for group in groups if not group["train_only"]]

    def can_add(val_frames, subj_val, group):
        if max_val_fraction is None:
            return True
        new_val = val_frames + group["frames"]
        denom = max(new_val, target)
        for subject, frames in group["subject_frames"].items():
            if _over_fraction(subj_val[subject] + frames, denom, max_val_fraction):
                return False
        return True

    def add_group(val, val_frames, subj_val, group):
        val.append(group)
        val_frames += group["frames"]
        for subject, frames in group["subject_frames"].items():
            subj_val[subject] += frames
        return val_frames

    def repair(val, val_frames, subj_val):
        if max_val_fraction is None:
            return val_frames
        while val_frames > 0:
            overweight = [
                subject
                for subject, frames in subj_val.items()
                if _over_fraction(frames, val_frames, max_val_fraction)
            ]
            if not overweight:
                break
            subject = random.choice(overweight)
            owned = [group for group in val if subject in group["subject_frames"]]
            if not owned:
                break
            drop = random.choice(owned)
            val.remove(drop)
            val_frames -= drop["frames"]
            for subject_name, frames in drop["subject_frames"].items():
                subj_val[subject_name] -= frames
                if subj_val[subject_name] <= 0:
                    del subj_val[subject_name]
        return val_frames

    val = []
    val_frames = 0
    subj_val = defaultdict(int)
    order = list(eligible)
    random.shuffle(order)
    for group in order:
        if val_frames >= target:
            break
        if can_add(val_frames, subj_val, group):
            val_frames = add_group(val, val_frames, subj_val, group)
    val_frames = repair(val, val_frames, subj_val)

    for _ in range(len(eligible) + 1):
        if val_frames >= target:
            break
        before = val_frames
        rest = [group for group in eligible if group not in val]
        random.shuffle(rest)
        for group in rest:
            if val_frames >= target:
                break
            if can_add(val_frames, subj_val, group):
                val_frames = add_group(val, val_frames, subj_val, group)
        val_frames = repair(val, val_frames, subj_val)
        if val_frames <= before:
            break

    return val


def _subject_frame_totals(groups):
    totals = defaultdict(int)
    total = 0
    for group in groups:
        total += group["frames"]
        for subject, frames in group["subject_frames"].items():
            totals[subject] += frames
    return totals, total


def _print_subject_shares(title, groups, max_fraction):
    totals, total = _subject_frame_totals(groups)
    if total <= 0:
        print(f"   {title}: empty")
        return
    ranked = sorted(totals.items(), key=lambda item: (-item[1], item[0]))
    top = ranked[:8]
    shares = ", ".join(f"{subject} {frames / total:.1%}" for subject, frames in top)
    print(f"   {title} frames {total}: {shares}")
    if max_fraction is None:
        return
    over = [
        f"{subject} {frames / total:.1%}"
        for subject, frames in ranked
        if _over_fraction(frames, total, max_fraction)
    ]
    if over:
        raise RuntimeError(f"{title} exceeds the subject cap: {', '.join(over)}")


def select_capped_videos(
    video_to_images,
    val_ratio,
    subjects_path,
    train_only_subjects,
    max_val_fraction,
    max_train_fraction,
    require_videos,
):
    """Split by user-day group with optional per-subject frame caps.

    Returns (val video ids, excluded video ids). Validation is assigned at
    random by user-day group. The train cap drops individual videos of the
    over-limit subject at random. Required videos stay in the dataset.
    """
    if not subjects_path:
        raise ValueError(
            "--subjects-path is required with subject caps, "
            "--train-only-subjects, or --require-videos"
        )
    if max_val_fraction is not None and not 0 < max_val_fraction <= 1:
        raise ValueError("--max-subject-val-fraction must be in (0, 1]")
    if max_train_fraction is not None and not 0 < max_train_fraction <= 1:
        raise ValueError("--max-subject-train-fraction must be in (0, 1]")

    video_subjects = load_video_subjects(subjects_path)
    required_ids = set()
    listed_required = []
    if require_videos:
        listed_required = load_id_list(require_videos)
        required_ids = required_video_set(listed_required)

    groups = build_user_day_groups(
        video_to_images,
        video_subjects,
        train_only_subjects,
        required_ids,
    )
    if required_ids:
        present = {
            video
            for video in video_to_images
            if video in required_ids
        }
        print(
            f"📌 Required videos present: {len(present)} "
            f"(list has {len(listed_required)} ids)"
        )

    kept = list(groups)
    dropped_videos = set()
    stalls = 0
    drops = 0
    max_stalls = 100
    train_only = set(train_only_subjects or [])
    max_steps = sum(
        1
        for group in groups
        for video in group["videos"]
        if video not in required_ids
    ) + max_stalls

    def detach_video(group, video):
        frames = len(video_to_images[video])
        subject_name = subject_for_video(video, video_subjects)
        group["videos"].remove(video)
        group["frames"] -= frames
        group["subject_frames"][subject_name] -= frames
        if group["subject_frames"][subject_name] <= 0:
            del group["subject_frames"][subject_name]
        group["protected"] = any(item in required_ids for item in group["videos"])
        group["train_only"] = any(
            subject_for_video(item, video_subjects) in train_only
            for item in group["videos"]
        )
        dropped_videos.add(video)

    for _ in range(max_steps + 1):
        val_groups = assign_val_groups(kept, val_ratio, max_val_fraction)
        if max_train_fraction is None:
            break
        val_ids = {id(group) for group in val_groups}
        train_groups = [group for group in kept if id(group) not in val_ids]
        train_totals, train_frames = _subject_frame_totals(train_groups)
        offenders = [
            subject
            for subject, frames in train_totals.items()
            if _over_fraction(frames, train_frames, max_train_fraction)
        ]
        if not offenders:
            break
        random.shuffle(offenders)
        dropped = False
        for subject in offenders:
            removable = [
                (group, video)
                for group in train_groups
                for video in group["videos"]
                if subject_for_video(video, video_subjects) == subject
                and video not in required_ids
            ]
            if not removable:
                continue
            group, video = random.choice(removable)
            detach_video(group, video)
            if not group["videos"]:
                kept.remove(group)
            drops += 1
            dropped = True
            stalls = 0
            if drops % 100 == 0:
                print(f"   Dropped {drops} videos to meet the train cap...")
            break
        if dropped:
            continue
        stalls += 1
        if stalls > max_stalls:
            details = ", ".join(
                f"{subject} {train_totals[subject] / train_frames:.1%}"
                for subject in offenders
            )
            raise ValueError(
                "Train subject cap cannot be met without dropping required "
                f"videos: {details}"
            )
    else:
        raise ValueError("Train subject cap did not converge")

    if not val_groups:
        raise ValueError("Val split is empty under the subject caps")

    kept_ids = {id(group) for group in kept}
    excluded_groups = [group for group in groups if id(group) not in kept_ids]
    val_ids = {id(group) for group in val_groups}
    train_groups = [group for group in kept if id(group) not in val_ids]
    val_videos = {video for group in val_groups for video in group["videos"]}
    excluded_videos = {video for group in excluded_groups for video in group["videos"]}
    excluded_videos.update(dropped_videos)

    for group in groups:
        if not group["videos"]:
            continue
        places = set()
        for video in group["videos"]:
            if video in excluded_videos:
                places.add("excluded")
            elif video in val_videos:
                places.add("val")
            else:
                places.add("train")
        if len(places) != 1:
            raise RuntimeError(f"User-day group was split across {places}: {group['key']}")

    kept_frames = sum(group["frames"] for group in kept)
    val_frames = sum(group["frames"] for group in val_groups)
    if val_frames + 1e-6 < val_ratio * kept_frames:
        print(
            f"⚠️ Val is {val_frames / kept_frames:.2%} of kept frames, "
            f"under the {val_ratio:.2%} target, so the subject cap can hold"
        )

    if required_ids:
        leaked = [
            video
            for video in excluded_videos
            if video in required_ids
        ]
        if leaked:
            raise RuntimeError(
                f"Required videos were excluded: {', '.join(leaked[:10])}"
            )

    print(f"   Dropped {drops} train videos")
    _print_subject_shares("Val", val_groups, max_val_fraction)
    _print_subject_shares("Train", train_groups, max_train_fraction)
    return val_videos, excluded_videos


def select_val_videos(
    video_to_images,
    val_ratio,
    subjects_path,
    subjects_val,
    subjects_exclude=None,
):
    """Return val video prefixes and video prefixes left out of both splits.

    Subject mode puts every remaining video whose subject is listed in
    subjects_val into val and every other remaining video into train. The frame
    ratio is not used. Excluded subjects are omitted from both splits.
    """
    if subjects_val and not subjects_path:
        raise ValueError("--subjects-val requires --subjects-path")
    if subjects_exclude and not subjects_path:
        raise ValueError("--subjects-exclude requires --subjects-path")
    if subjects_path and not subjects_val and not subjects_exclude:
        raise ValueError(
            "--subjects-path requires --subjects-val or --subjects-exclude"
        )

    videos = list(video_to_images.keys())
    video_subjects = load_video_subjects(subjects_path) if subjects_path else None
    if video_subjects is not None:
        missing = [video for video in videos if video not in video_subjects]
        if missing:
            preview = ", ".join(missing[:10])
            extra = f" (+{len(missing) - 10} more)" if len(missing) > 10 else ""
            raise ValueError(
                f"{len(missing)} videos have no subject in {subjects_path}: "
                f"{preview}{extra}"
            )

    excluded_videos = set()
    if subjects_exclude:
        exclude_subjects = set(subjects_exclude)
        overlap = sorted(exclude_subjects & set(subjects_val or []))
        if overlap:
            print(
                "⚠️ Subjects in both --subjects-val and --subjects-exclude "
                f"are excluded: {', '.join(overlap)}"
            )
        present_subjects = {video_subjects[video] for video in videos}
        unused = sorted(exclude_subjects - present_subjects)
        if unused:
            print(
                "⚠️ Excluded subjects with no videos in this dataset: "
                f"{', '.join(unused)}"
            )
        excluded_videos = {
            video
            for video in videos
            if video_subjects[video] in exclude_subjects
        }
        if len(excluded_videos) == len(videos):
            raise ValueError("Subject exclude removed every video")
        if excluded_videos:
            excluded_names = sorted(
                {video_subjects[video] for video in excluded_videos}
            )
            excluded_frames = sum(
                len(video_to_images[video]) for video in excluded_videos
            )
            print(
                f"🚫 Excluding {', '.join(excluded_names)}: "
                f"{len(excluded_videos)} videos, {excluded_frames} frames"
            )

    pool = [video for video in videos if video not in excluded_videos]
    if subjects_val is not None:
        val_subjects = set(subjects_val)
        val_videos = {
            video
            for video in pool
            if video_subjects[video] in val_subjects
        }
        present_subjects = {video_subjects[video] for video in pool}
        unused = sorted(val_subjects - present_subjects)
        if unused:
            print(f"⚠️ Val subjects with no videos left in this dataset: {', '.join(unused)}")
        if not val_videos:
            raise ValueError("Subject split produced an empty val set")
        if len(val_videos) == len(pool):
            raise ValueError("Subject split produced an empty train set")

        assigned = sorted(present_subjects & val_subjects)
        print(f"👤 Val subjects: {', '.join(assigned)}")
        print("   --val-ratio ignored")
        return val_videos, excluded_videos

    random.shuffle(pool)
    pool_frames = sum(len(video_to_images[video]) for video in pool)
    target_val_frames = int(val_ratio * pool_frames)
    val_videos = set()
    val_frame_count = 0
    for video in pool:
        if val_frame_count >= target_val_frames:
            break
        val_videos.add(video)
        val_frame_count += len(video_to_images[video])
    return val_videos, excluded_videos


def split_dataset(
    input_dir,
    output_dir,
    annotation_name=None,
    val_ratio=0.15,
    seed=42,
    subjects_path=None,
    subjects_val=None,
    subjects_exclude=None,
    train_only_subjects=None,
    max_val_fraction=None,
    max_train_fraction=None,
    require_videos=None,
    no_frames=False,
    run_config=None,
):
    """
    Splits a combined dataset into train and val sets by video.
    
    All frames from the same video (based on filename prefix before '_frame_')
    are kept together in either train or val to prevent data leakage.
    
    Input structure:
        input_dir/
        ├── frames/
        └── <annotation_name>.json
    
    Output structure (COCO format):
        output_dir/
        ├── annotations/
        │   ├── person_keypoints_train2017.json
        │   └── person_keypoints_val2017.json
        ├── train2017/
        └── val2017/
    
    Args:
        input_dir: Input directory containing frames/ and annotation JSON
        output_dir: Output directory for COCO-structured dataset
        annotation_name: Name of annotation JSON file (auto-detected if None)
        val_ratio: Fraction of frames to use for validation (default 0.15).
            Ignored when subjects_path and subjects_val are set.
        seed: Random seed for reproducible ratio splits (default 42)
        subjects_path: CSV of video_id,subject. Required for subject val/exclude
            and for the capped user-day split.
        subjects_val: Subject names assigned entirely to val. Every other
            subject is assigned entirely to train.
        subjects_exclude: Subject names omitted from both train and val.
            Requires subjects_path.
        train_only_subjects: Subject names that stay in train. Requires
            subjects_path. Cannot be combined with subjects_val.
        max_val_fraction: Maximum share of val frames from one subject.
            Groups videos by the name and date in the id. Requires subjects_path.
        max_train_fraction: Maximum share of train frames from one subject.
            User-day groups may be dropped at random to meet it. Required videos
            are not dropped.
        require_videos: Text file of video ids that must remain in the dataset.
        no_frames: Write only the train and val annotation JSON files. Image
            directories are not created or copied.
        run_config: Dict of script flag name to the value used for this run.
            Stored at info.config in both output JSON files.
    """
    random.seed(seed)
    
    # Define input paths
    input_frames_dir = os.path.join(input_dir, "frames")
    
    if annotation_name:
        json_path = os.path.join(input_dir, annotation_name)
    else:
        # Auto-detect JSON file in input_dir
        json_files = [f for f in os.listdir(input_dir) if f.endswith('.json')]
        if len(json_files) == 0:
            raise FileNotFoundError(f"No JSON file found in {input_dir}")
        elif len(json_files) > 1:
            raise ValueError(f"Multiple JSON files found in {input_dir}: {json_files}. Specify --annotation")
        json_path = os.path.join(input_dir, json_files[0])
    
    print(f"📂 Input: {input_dir}")
    if no_frames:
        print("   Frames: skipped")
    else:
        print(f"   Frames: {input_frames_dir}")
    print(f"   Annotations: {json_path}")

    # Load combined dataset
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Group images by video prefix
    video_to_images = defaultdict(list)
    for img in data["images"]:
        filename = os.path.basename(img["file_name"])
        # Extract video prefix (everything before '_frame_')
        if "_frame_" in filename:
            video_prefix = filename.split("_frame_")[0]
        else:
            video_prefix = filename  # fallback: treat each image as its own group
        video_to_images[video_prefix].append(img)

    videos = list(video_to_images.keys())
    total_frames = sum(len(imgs) for imgs in video_to_images.values())
    use_caps = any((
        train_only_subjects,
        max_val_fraction is not None,
        max_train_fraction is not None,
        require_videos,
    ))
    if use_caps and (subjects_val or subjects_exclude):
        raise ValueError(
            "--subjects-val and --subjects-exclude cannot be combined with "
            "--train-only-subjects, subject fraction caps, or --require-videos"
        )
    if use_caps:
        val_videos, excluded_videos = select_capped_videos(
            video_to_images,
            val_ratio,
            subjects_path,
            train_only_subjects,
            max_val_fraction,
            max_train_fraction,
            require_videos,
        )
    else:
        val_videos, excluded_videos = select_val_videos(
            video_to_images,
            val_ratio,
            subjects_path,
            subjects_val,
            subjects_exclude,
        )

    # Define COCO output structure
    annotations_dir = os.path.join(output_dir, "annotations")
    train_dir = os.path.join(output_dir, "train2017")
    val_dir = os.path.join(output_dir, "val2017")
    train_json_out = os.path.join(annotations_dir, "person_keypoints_train2017.json")
    val_json_out = os.path.join(annotations_dir, "person_keypoints_val2017.json")

    # Create output directories
    os.makedirs(annotations_dir, exist_ok=True)

    if not no_frames:
        if os.path.exists(train_dir):
            shutil.rmtree(train_dir)
        os.makedirs(train_dir, exist_ok=True)

        if os.path.exists(val_dir):
            shutil.rmtree(val_dir)
        os.makedirs(val_dir, exist_ok=True)

    # Split images into train and val
    train_images = []
    val_images = []
    
    for video, images in video_to_images.items():
        if video in excluded_videos:
            continue
        if video in val_videos:
            val_images.extend(images)
        else:
            train_images.extend(images)

    # Get corresponding annotations
    train_ids = {img["id"] for img in train_images}
    val_ids = {img["id"] for img in val_images}
    
    train_annotations = [ann for ann in data["annotations"] if ann["image_id"] in train_ids]
    val_annotations = [ann for ann in data["annotations"] if ann["image_id"] in val_ids]

    # Copy images to respective directories
    if not no_frames:
        print(f"📁 Copying {len(train_images)} training images...")
        for img in train_images:
            src = os.path.join(input_frames_dir, os.path.basename(img["file_name"]))
            dst = os.path.join(train_dir, os.path.basename(img["file_name"]))
            if os.path.exists(src):
                shutil.copy(src, dst)
            else:
                print(f"⚠️ Missing: {src}")

        print(f"📁 Copying {len(val_images)} validation images...")
        for img in val_images:
            src = os.path.join(input_frames_dir, os.path.basename(img["file_name"]))
            dst = os.path.join(val_dir, os.path.basename(img["file_name"]))
            if os.path.exists(src):
                shutil.copy(src, dst)
            else:
                print(f"⚠️ Missing: {src}")

    # Prepare output JSON structure
    info = dict(data.get("info") or {})
    if run_config is not None:
        info["config"] = run_config
    categories = data.get("categories", [])

    # Normalize file_name to just basename (remove any path prefixes like 'frames/')
    def normalize_images(images):
        normalized = []
        for img in images:
            normalized_img = dict(img)
            normalized_img["file_name"] = os.path.basename(img["file_name"])
            normalized.append(normalized_img)
        return normalized

    # Save train JSON
    train_data = {
        "info": info,
        "images": normalize_images(train_images),
        "annotations": train_annotations,
        "categories": categories
    }
    with open(train_json_out, "w") as f:
        json.dump(train_data, f, indent=4)

    # Save val JSON
    val_data = {
        "info": info,
        "images": normalize_images(val_images),
        "annotations": val_annotations,
        "categories": categories
    }
    with open(val_json_out, "w") as f:
        json.dump(val_data, f, indent=4)

    # Print summary
    kept_frames = len(train_images) + len(val_images)
    train_videos = len(videos) - len(val_videos) - len(excluded_videos)
    print(f"\n✅ Dataset split complete!")
    print(f"   Output: {output_dir}")
    print(
        f"   Videos: {len(videos)} total → {train_videos} train, "
        f"{len(val_videos)} val, {len(excluded_videos)} excluded"
    )
    print(
        f"   Images: {total_frames} total → {len(train_images)} train, "
        f"{len(val_images)} val, {total_frames - kept_frames} excluded"
    )
    print(f"   Annotations: {len(train_annotations)} train, {len(val_annotations)} val")
    print(f"   Actual val ratio: {len(val_images) / kept_frames:.2%}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Split dataset into train/val by video (COCO format)")
    parser.add_argument("--input", required=True, help="Input directory (contains frames/ and .json)")
    parser.add_argument("--annotation", default=None, help="Annotation JSON filename (auto-detected if omitted)")
    parser.add_argument(
        "--no-frames",
        action="store_true",
        help="Write only the train and val JSON files. Do not copy images.",
    )
    parser.add_argument("--output", required=True, help="Output directory for COCO-structured dataset")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation ratio (default: 0.15). Ignored with --subjects-path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument(
        "--subjects-path",
        default=None,
        help="CSV of video_id,subject (e.g. AnthonyGeorge_SeatedStraightLegRaise_072426_oN2pDT,AnthonyGeorge)",
    )
    parser.add_argument(
        "--subjects-val",
        nargs="+",
        default=None,
        help="Subjects to put entirely in val. Remaining subjects go to train. Requires --subjects-path",
    )
    parser.add_argument(
        "--subjects-exclude",
        nargs="+",
        default=None,
        help="Subjects to omit from both train and val. Requires --subjects-path",
    )
    parser.add_argument(
        "--train-only-subjects",
        nargs="+",
        default=None,
        help="Subjects whose videos stay in train. Requires --subjects-path",
    )
    parser.add_argument(
        "--max-subject-val-fraction",
        type=float,
        default=None,
        help=(
            "Max fraction of val frames from one subject. "
            "Same name and date in the video id stay in one split"
        ),
    )
    parser.add_argument(
        "--max-subject-train-fraction",
        type=float,
        default=None,
        help=(
            "Max fraction of train frames from one subject. "
            "Individual videos of that subject may be dropped at random to meet it"
        ),
    )
    parser.add_argument(
        "--require-videos",
        default=None,
        help=(
            "Text file of video ids that must stay in the dataset. "
            "A line may be video_id or video_id: date"
        ),
    )

    args = parser.parse_args()

    def script_flag_config(parser, args):
        """Flag name to the value used for this run."""
        config = {}
        for action in parser._actions:
            if not action.option_strings or action.dest in {"help", "no_frames"}:
                continue
            flag = max(action.option_strings, key=len).lstrip("-")
            config[flag] = getattr(args, action.dest)
        return config

    def parse_subjects(values):
        if not values:
            return None
        subjects = [
            subject.strip()
            for value in values
            for subject in value.split(",")
            if subject.strip()
        ]
        return subjects or None

    split_dataset(
        input_dir=args.input,
        output_dir=args.output,
        annotation_name=args.annotation,
        val_ratio=args.val_ratio,
        seed=args.seed,
        subjects_path=args.subjects_path,
        subjects_val=parse_subjects(args.subjects_val),
        subjects_exclude=parse_subjects(args.subjects_exclude),
        train_only_subjects=parse_subjects(args.train_only_subjects),
        max_val_fraction=args.max_subject_val_fraction,
        max_train_fraction=args.max_subject_train_fraction,
        require_videos=args.require_videos,
        no_frames=args.no_frames,
        run_config=script_flag_config(parser, args),
    )
