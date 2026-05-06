#!/usr/bin/env python3
"""One random frame per video, assembled into a single video.

Frame files are expected to look like ``{videoname}_frame_{...}.jpg``; the video
name is the text before ``_frame_`` (same rule as filter_val_images). If
``_frame_`` is missing, the whole basename (without extension) is treated as one
video.

Video encoding matches ``overlay_dataset.create_video_from_images``: max
width/height across selected frames, dimensions rounded up to a multiple of 16,
``mp4v`` codec, each frame resized to that size. The video name is drawn in blue
at the top center of each frame.
"""

from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path

import cv2


def video_name_for_frame_file(path: Path) -> str:
    """Derive video name from a frame filename."""
    name = path.name
    if "_frame_" in name:
        return name.split("_frame_")[0]
    return path.stem


def _draw_video_title_top_center(
    image,
    text: str,
    *,
    margin_y: int = 12,
    color_bgr: tuple[int, int, int] = (255, 0, 0),
) -> None:
    """In-place: blue title centered at top (OpenCV BGR: blue = 255,0,0)."""
    h, w = image.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.45, min(1.4, w / 900.0))
    pad = 20
    while font_scale >= 0.3:
        thickness = max(1, int(round(font_scale * 2)))
        (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
        if text_w <= w - pad:
            break
        font_scale -= 0.05
    thickness = max(1, int(round(font_scale * 2)))
    (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
    x = max(0, (w - text_w) // 2)
    y = min(h - 1, text_h + margin_y)
    cv2.putText(image, text, (x, y), font, font_scale, color_bgr, thickness, cv2.LINE_AA)


def create_video_from_image_paths(
    frame_entries: list[tuple[Path, str]], output_path: Path, fps: float = 1.0
) -> None:
    """Create a video from (image path, video name) pairs; scaling matches overlay_dataset."""
    if not frame_entries:
        print("No images for video")
        return

    print(f"Encoding {len(frame_entries)} frames at {fps} fps")

    max_width, max_height = 0, 0
    for image_file, _ in frame_entries:
        img = cv2.imread(str(image_file))
        if img is not None:
            h, w = img.shape[:2]
            max_width = max(max_width, w)
            max_height = max(max_height, h)

    if max_width == 0 or max_height == 0:
        print("Could not read any input images")
        return

    target_width = ((max_width + 15) // 16) * 16
    target_height = ((max_height + 15) // 16) * 16
    print(f"Video dimensions: {target_width}x{target_height}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(
        str(output_path), fourcc, fps, (target_width, target_height)
    )

    if not video_writer.isOpened():
        print("Could not open video writer")
        return

    for i, (image_file, video_label) in enumerate(frame_entries):
        if (i + 1) % 50 == 0:
            print(f"Adding frame {i + 1}/{len(frame_entries)}")

        image = cv2.imread(str(image_file))
        if image is not None:
            resized = cv2.resize(image, (target_width, target_height))
            _draw_video_title_top_center(resized, video_label)
            video_writer.write(resized)

    video_writer.release()
    print(f"Video saved to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pick one random frame per video and write a single MP4."
    )
    parser.add_argument(
        "frames_dir",
        type=str,
        help="Directory containing frame images (e.g. Heel_Slides/v1/frames)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output video path (default: <frames_dir>/random_frame_per_video.mp4)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=1.0,
        help="Video FPS (default: 1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional RNG seed for reproducible frame choices",
    )
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    script_dir = Path(__file__).parent
    frames_dir = Path(args.frames_dir)
    if not frames_dir.is_absolute():
        frames_dir = script_dir / frames_dir

    if not frames_dir.is_dir():
        print(f"Error: not a directory: {frames_dir}")
        return

    paths = list(frames_dir.glob("*.jpg")) + list(frames_dir.glob("*.png"))
    if not paths:
        print(f"No .jpg or .png files in {frames_dir}")
        return

    by_video: dict[str, list[Path]] = defaultdict(list)
    for p in paths:
        by_video[video_name_for_frame_file(p)].append(p)

    for v in by_video:
        by_video[v].sort(key=lambda x: x.name)

    video_names = sorted(by_video.keys())
    chosen: list[tuple[Path, str]] = [
        (random.choice(by_video[name]), name) for name in video_names
    ]

    print(f"frames_dir: {frames_dir}")
    print(f"frame_files: {len(paths)}")
    print(f"distinct_videos: {len(video_names)}")

    out = args.output
    if out is None:
        output_path = frames_dir / "random_frame_per_video.mp4"
    else:
        output_path = Path(out)
        if not output_path.is_absolute():
            output_path = script_dir / output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    create_video_from_image_paths(chosen, output_path, fps=args.fps)


if __name__ == "__main__":
    main()
