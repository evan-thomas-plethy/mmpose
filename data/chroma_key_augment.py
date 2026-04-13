"""
Replace a near-white background in each video under an input directory with a still image.
Writes `{stem}_output{ext}` into the output directory for every supported video file.
Background may be JPEG, PNG, WebP, etc. (WebP and some formats load via Pillow if OpenCV fails).

Per frame, background replacement runs only if strictly more than one third of pixels
have all BGR channels >= threshold (same rule as the key); otherwise the frame is unchanged.
"""

import argparse
import os

import cv2
import numpy as np

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm"}

# Apply compositing only when more than this fraction of pixels exceed the BGR threshold (raw, no morphology).
MIN_BRIGHT_PIXEL_FRACTION = 1.0 / 3.0
MAX_COLORFULNESS = 24

# COCO person keypoints: 17 joints × (x, y, v).
COCO_NUM_KEYPOINTS = 17
COCO_KP_STRIDE = 3

# Same edges as Ground_Based_Exercises JSON → ``categories[0].skeleton`` (COCO-17, 0-index).
# Torso / arms / legs use the same thickness multiplier; head uses a smaller one.
# Draw order: largest multiplier first (head last).
GROUND_BASED_SKELETON_SEGMENTS: list[tuple[int, int, float]] = [
    (0, 2, 0.6),
    (0, 1, 0.6),
    (1, 3, 0.6),
    (2, 4, 0.6),
    (5, 6, 1.5),
    (11, 12, 1.5),
    (5, 11, 1.5),
    (6, 12, 1.5),
    (5, 7, 1.5),
    (6, 8, 1.5),
    (11, 13, 1.5),
    (12, 14, 1.5),
    (7, 9, 1.5),
    (8, 10, 1.5),
    (13, 15, 1.5),
    (14, 16, 1.5),
]


def _coco_v_visible(v: float) -> bool:
    return int(round(float(v))) > 0


def _bbox_xywh_from_keypoints(kps: list[float]) -> tuple[float, float, float, float]:
    """Tight axis-aligned box from visible keypoints (source image space)."""
    xs: list[float] = []
    ys: list[float] = []
    for i in range(COCO_NUM_KEYPOINTS):
        v = kps[i * COCO_KP_STRIDE + 2]
        if not _coco_v_visible(v):
            continue
        xs.append(float(kps[i * COCO_KP_STRIDE]))
        ys.append(float(kps[i * COCO_KP_STRIDE + 1]))
    if len(xs) < 2:
        return (0.0, 0.0, 1.0, 1.0)
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    return (x0, y0, max(1e-3, x1 - x0), max(1e-3, y1 - y0))


def thickness_px_coco_segment(
    bbox_xywh: tuple[float, float, float, float],
    scale_x: float,
    scale_y: float,
    segment_multiplier: float,
) -> int:
    _x, _y, bw, bh = bbox_xywh
    bbox_width = float(bw) * float(scale_x)
    bbox_height = float(bh) * float(scale_y)
    body_scale = 0.7 * bbox_height + 0.3 * bbox_width
    t = 0.015 * body_scale * float(segment_multiplier)
    return int(max(3, min(50, round(t))))


def _each_visible_skeleton_tube_segment(
    keypoints_list: list[list[float]],
    bboxes_xywh: list[tuple[float, float, float, float]],
    scale_x: float,
    scale_y: float,
):
    """Yield ``(person_index, p0, p1, thickness)`` for each skeleton edge with both endpoints visible."""
    need = COCO_NUM_KEYPOINTS * COCO_KP_STRIDE
    segments_draw_order = sorted(
        GROUND_BASED_SKELETON_SEGMENTS, key=lambda s: s[2], reverse=True
    )
    for pi, kps in enumerate(keypoints_list):
        if len(kps) < need:
            continue
        if pi < len(bboxes_xywh):
            bbox = bboxes_xywh[pi]
        else:
            bbox = _bbox_xywh_from_keypoints(kps)
        _bx, _by, bw, bh = bbox
        if bw <= 1.0 or bh <= 1.0:
            bbox = _bbox_xywh_from_keypoints(kps)

        for i, j, mult in segments_draw_order:
            vi = kps[i * COCO_KP_STRIDE + 2]
            vj = kps[j * COCO_KP_STRIDE + 2]
            if not (_coco_v_visible(vi) and _coco_v_visible(vj)):
                continue
            x0 = float(kps[i * COCO_KP_STRIDE])
            y0 = float(kps[i * COCO_KP_STRIDE + 1])
            x1 = float(kps[j * COCO_KP_STRIDE])
            y1 = float(kps[j * COCO_KP_STRIDE + 1])
            p0 = (int(round(x0 * scale_x)), int(round(y0 * scale_y)))
            p1 = (int(round(x1 * scale_x)), int(round(y1 * scale_y)))
            thick = thickness_px_coco_segment(bbox, scale_x, scale_y, mult)
            yield pi, p0, p1, thick


def overlay_coco17_pose(
    img_bgr: np.ndarray,
    keypoints_list: list[list[float]],
    bboxes_xywh: list[tuple[float, float, float, float]],
    scale_x: float,
    scale_y: float,
    *,
    draw_skeleton: bool = False,
    draw_keypoints: bool = False,
) -> None:
    """
    Optional **visual** overlay only: colored skeleton tubes and/or joint dots on ``img_bgr``.

    Tubes: both endpoints ``v > 0``. Joints: ``v > 0``.
    ``bboxes_xywh``: COCO ``[x, y, width, height]`` per person (for tube thickness).
    """
    if not draw_skeleton and not draw_keypoints:
        return
    palette = (
        (0, 255, 0),
        (255, 128, 0),
        (0, 165, 255),
        (255, 0, 255),
        (200, 200, 0),
        (0, 255, 255),
        (128, 0, 255),
        (255, 255, 0),
    )
    need = COCO_NUM_KEYPOINTS * COCO_KP_STRIDE

    if draw_skeleton:
        for pi, p0, p1, thick in _each_visible_skeleton_tube_segment(
            keypoints_list, bboxes_xywh, scale_x, scale_y
        ):
            color = palette[pi % len(palette)]
            cv2.line(
                img_bgr,
                p0,
                p1,
                color,
                thick,
                lineType=cv2.LINE_AA,
            )

    for pi, kps in enumerate(keypoints_list):
        if len(kps) < need:
            continue
        color = palette[pi % len(palette)]

        if draw_keypoints:
            for ki in range(COCO_NUM_KEYPOINTS):
                v = kps[ki * COCO_KP_STRIDE + 2]
                if not _coco_v_visible(v):
                    continue
                x = float(kps[ki * COCO_KP_STRIDE])
                y = float(kps[ki * COCO_KP_STRIDE + 1])
                c = (int(round(x * scale_x)), int(round(y * scale_y)))
                cv2.circle(img_bgr, c, 4, color, -1, lineType=cv2.LINE_AA)


def _guided_filter(guide_gray: np.ndarray, src: np.ndarray, radius: int = 6, eps: float = 1e-3) -> np.ndarray:
    """Guided filter for edge-aware smoothing of a single-channel float image."""
    r = max(1, int(radius))
    I = np.clip(guide_gray.astype(np.float32), 0.0, 1.0)
    p = np.clip(src.astype(np.float32), 0.0, 1.0)

    mean_I = cv2.boxFilter(I, ddepth=-1, ksize=(2 * r + 1, 2 * r + 1), normalize=True)
    mean_p = cv2.boxFilter(p, ddepth=-1, ksize=(2 * r + 1, 2 * r + 1), normalize=True)
    corr_I = cv2.boxFilter(I * I, ddepth=-1, ksize=(2 * r + 1, 2 * r + 1), normalize=True)
    corr_Ip = cv2.boxFilter(I * p, ddepth=-1, ksize=(2 * r + 1, 2 * r + 1), normalize=True)

    var_I = corr_I - mean_I * mean_I
    cov_Ip = corr_Ip - mean_I * mean_p

    a = cov_Ip / (var_I + float(eps))
    b = mean_p - a * mean_I

    mean_a = cv2.boxFilter(a, ddepth=-1, ksize=(2 * r + 1, 2 * r + 1), normalize=True)
    mean_b = cv2.boxFilter(b, ddepth=-1, ksize=(2 * r + 1, 2 * r + 1), normalize=True)
    q = mean_a * I + mean_b
    return np.clip(q, 0.0, 1.0)


def _load_background(path: str, width: int, height: int) -> np.ndarray:
    bg = cv2.imread(path, cv2.IMREAD_COLOR)
    if bg is None:
        from PIL import Image

        try:
            im = Image.open(path)
        except OSError as e:
            raise FileNotFoundError(f"Could not read background image: {path}") from e
        if im.mode == "RGBA":
            im = im.convert("RGB")
        elif im.mode != "RGB":
            im = im.convert("RGB")
        arr = np.asarray(im)
        bg = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    if bg.ndim == 2:
        bg = cv2.cvtColor(bg, cv2.COLOR_GRAY2BGR)
    return cv2.resize(bg, (width, height), interpolation=cv2.INTER_AREA)


def _fraction_pixels_above_threshold_bgr(
    frame: np.ndarray, threshold: int, colorfulness_max: int
) -> float:
    """Share of pixels that are both bright and low-colorfulness (white-like)."""
    t = int(np.clip(threshold, 0, 255))
    cmax = int(np.clip(colorfulness_max, 0, 255))
    frame_u16 = frame.astype(np.uint16)
    min_chan = np.min(frame_u16, axis=2)
    max_chan = np.max(frame_u16, axis=2)
    spread = max_chan - min_chan
    white_like = (min_chan >= t) & (spread <= cmax)
    return float(np.count_nonzero(white_like)) / float(white_like.size)


def _white_key_mask_bgr(
    frame: np.ndarray,
    threshold: int,
    colorfulness_max: int,
    softness: int = 15,
    colorfulness_softness: int = 15,
) -> np.ndarray:
    """Soft mask (0-1): high for bright, low-colorfulness (white-like) pixels."""
    t = int(np.clip(threshold, 0, 255))
    cmax = int(np.clip(colorfulness_max, 0, 255))
    soft = max(1, int(softness))
    csoft = max(1, int(colorfulness_softness))
    frame_f = frame.astype(np.float32)

    # Use the darkest channel as whiteness confidence; true white should be high in all channels.
    min_chan = np.min(frame_f, axis=2)
    max_chan = np.max(frame_f, axis=2)
    spread = max_chan - min_chan

    bright_score = np.clip((min_chan - float(t)) / float(soft), 0.0, 1.0)
    # Penalize saturated/colored pixels even when bright (e.g., skin highlights).
    color_score = np.clip((float(cmax) - spread) / float(csoft), 0.0, 1.0)
    m = bright_score * color_score
    m = m ** 1.1

    kernel = np.ones((3, 3), np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel, iterations=1)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Edge-aware smoothing keeps mask transitions aligned to subject/background boundaries.
    guide = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    m = _guided_filter(guide, m, radius=5, eps=5e-4)
    return m


def _compose(frame: np.ndarray, bg: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Alpha-blend using soft white-key mask (float 0-1 per pixel)."""
    m = mask.astype(np.float32)
    m = cv2.GaussianBlur(m, (0, 0), sigmaX=1.0, sigmaY=1.0)
    m = np.clip(m[..., np.newaxis], 0.0, 1.0)
    f32 = frame.astype(np.float32)
    bg32 = bg.astype(np.float32)
    out = f32 * (1.0 - m) + bg32 * m
    return np.clip(out, 0, 255).astype(np.uint8)


def process_frame_white_key(
    frame: np.ndarray,
    bg_bgr: np.ndarray,
    frame_threshold: int,
    colorfulness_max: int = MAX_COLORFULNESS,
    min_bright_fraction: float = MIN_BRIGHT_PIXEL_FRACTION,
) -> np.ndarray:
    """Apply near-white key compositing to a single BGR frame (same rules as ``run``)."""
    fh, fw = frame.shape[:2]
    if (
        _fraction_pixels_above_threshold_bgr(frame, frame_threshold, colorfulness_max)
        < min_bright_fraction
    ):
        return frame
    if fh != bg_bgr.shape[0] or fw != bg_bgr.shape[1]:
        bg_use = cv2.resize(bg_bgr, (fw, fh), interpolation=cv2.INTER_AREA)
    else:
        bg_use = bg_bgr
    mask = _white_key_mask_bgr(frame, frame_threshold, colorfulness_max)
    return _compose(frame, bg_use, mask)


def run(
    video_path: str,
    background_path: str,
    output_path: str,
    threshold: int = 230,
    threshold_window: int = 10,
    colorfulness_max: int = MAX_COLORFULNESS,
    min_bright_fraction: float = MIN_BRIGHT_PIXEL_FRACTION,
) -> None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise OSError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps > 120:
        fps = 30.0

    ret, first = cap.read()
    if not ret:
        cap.release()
        raise OSError(f"Could not read frames from: {video_path}")

    h, w = first.shape[:2]
    bg = _load_background(background_path, w, h)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    if not writer.isOpened():
        cap.release()
        raise OSError(f"Could not open video writer for: {output_path}")

    window = max(0, int(threshold_window))
    half = window / 2.0
    thresh_low = int(np.clip(np.floor(float(threshold) - half), 0, 255))
    thresh_high = int(np.clip(np.ceil(float(threshold) + half), 0, 255))

    def process(frame: np.ndarray, bg_bgr: np.ndarray) -> np.ndarray:
        frame_threshold = int(np.random.randint(thresh_low, thresh_high + 1))
        return process_frame_white_key(
            frame, bg_bgr, frame_threshold, colorfulness_max, min_bright_fraction
        )

    out_f = process(first, bg)
    writer.write(out_f)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        writer.write(process(frame, bg))

    writer.release()
    cap.release()


def _list_videos(input_dir: str) -> list[tuple[str, str]]:
    """Return sorted list of (full_path, filename) for each video in ``input_dir``."""
    if not os.path.isdir(input_dir):
        return []
    out = []
    for name in sorted(os.listdir(input_dir)):
        ext = os.path.splitext(name)[1].lower()
        if ext not in VIDEO_EXTS:
            continue
        path = os.path.join(input_dir, name)
        if os.path.isfile(path):
            out.append((path, name))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Chroma-key near-white background for each video in an input directory; "
            "writes `<name>_output<ext>` files into the output directory."
        ),
    )
    parser.add_argument(
        "input_dir",
        help="Directory containing input videos.",
    )
    parser.add_argument(
        "output_dir",
        help="Directory to write processed videos (created if missing).",
    )
    parser.add_argument(
        "background",
        help="Background image (e.g. background.jpg, background.webp)",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=230,
        metavar="T",
        help="BGR lower bound per channel for white (0–255). Higher = stricter white.",
    )
    parser.add_argument(
        "--threshold-window",
        type=int,
        default=10,
        metavar="W",
        help=(
            "Random per-frame threshold window size around --threshold. "
            "Effective range is [T - W/2, T + W/2], clamped to [0, 255]."
        ),
    )
    parser.add_argument(
        "--min-bright-fraction",
        type=float,
        default=MIN_BRIGHT_PIXEL_FRACTION,
        metavar="F",
        help=(
            "Apply background replacement only if more than this fraction of pixels have "
            f"all channels >= threshold (default: {MIN_BRIGHT_PIXEL_FRACTION:.4f}, i.e. >33%%)."
        ),
    )
    parser.add_argument(
        "--colorfulness-max",
        type=int,
        default=MAX_COLORFULNESS,
        metavar="C",
        help=(
            "Maximum allowed per-pixel channel spread (max(B,G,R)-min(B,G,R)) for white-like "
            f"masking; lower values are stricter (default: {MAX_COLORFULNESS})."
        ),
    )
    args = parser.parse_args()

    input_dir = os.path.abspath(args.input_dir)
    output_dir = os.path.abspath(args.output_dir)

    if not os.path.isdir(input_dir):
        raise SystemExit(f"Not a directory: {input_dir}")

    os.makedirs(output_dir, exist_ok=True)

    videos = _list_videos(input_dir)
    if not videos:
        print(f"No video files ({', '.join(sorted(VIDEO_EXTS))}) found in {input_dir}")
        return

    for video_path, filename in videos:
        stem, ext = os.path.splitext(filename)
        out_name = f"{stem}_output{ext}"
        out_path = os.path.join(output_dir, out_name)
        print(f"Processing {filename!r} -> {out_name!r}")
        run(
            video_path,
            args.background,
            out_path,
            threshold=args.threshold,
            threshold_window=args.threshold_window,
            colorfulness_max=args.colorfulness_max,
            min_bright_fraction=args.min_bright_fraction,
        )
        print(f"  Wrote {out_path}")

    print(f"Done: {len(videos)} file(s) in {output_dir}")


if __name__ == "__main__":
    main()
