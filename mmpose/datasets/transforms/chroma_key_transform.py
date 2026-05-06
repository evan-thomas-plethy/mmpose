# Copyright (c) OpenMMLab. All rights reserved.
"""Online chroma-key (near-white) background replacement for training.

Uses :mod:`data.chroma_key_augment` (repo ``data/chroma_key_augment.py``) for
pixel compositing. One random background image per sample (per ``__getitem__``),
which matches dataloader workers and maximizes diversity vs batch-level.
"""

from __future__ import annotations

import importlib.util
import warnings
from pathlib import Path
from typing import List, Optional

import numpy as np
from mmcv.transforms import BaseTransform

from mmpose.registry import TRANSFORMS

# Image extensions for background pool
_BG_EXTS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tif', '.tiff'}

# ---------------------------------------------------------------------------
# Hardcoded defaults (override via config dict when you randomize ranges)
# ---------------------------------------------------------------------------
# bg_dir: resolved under mmpose repo root (parent of the ``mmpose`` package).
_DEFAULT_BG_DIR = 'data/House_Room_Dataset/Livingroom'
# Probability of applying this augmentation on each sample.
_DEFAULT_PROB = 0.5
# Match ``chroma_key_augment.py`` CLI / ``run()`` defaults.
_DEFAULT_THRESHOLD = 230
_DEFAULT_THRESHOLD_WINDOW = 10
_DEFAULT_COLORFULNESS_MAX = 24
_DEFAULT_MIN_BRIGHT_FRACTION = 1.0 / 3.0

_CHROMA_MODULE = None


def _repo_root() -> Path:
    """``mmpose`` repo root (contains ``data/`` and ``mmpose/`` package)."""
    return Path(__file__).resolve().parents[3]


def _load_chroma_module():
    import importlib.util

    global _CHROMA_MODULE
    if _CHROMA_MODULE is not None:
        return _CHROMA_MODULE
    path = _repo_root() / 'data' / 'chroma_key_augment.py'
    if not path.is_file():
        raise FileNotFoundError(
            f'Expected chroma_key_augment at {path}. '
            'Ensure data/chroma_key_augment.py exists.')
    spec = importlib.util.spec_from_file_location(
        'mmpose_chroma_key_augment', path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    _CHROMA_MODULE = mod
    return _CHROMA_MODULE


def _list_background_images(bg_dir: Path) -> List[str]:
    if not bg_dir.is_dir():
        return []
    out: List[str] = []
    for name in sorted(bg_dir.iterdir()):
        if not name.is_file():
            continue
        if name.suffix.lower() in _BG_EXTS:
            out.append(str(name.resolve()))
    return out


@TRANSFORMS.register_module()
class ChromaKeyAug(BaseTransform):
    """Replace near-white background with a random room image (online).

    Runs **after** :class:`LoadImage` and **before** geometric crops/warps
    (e.g. :class:`TopdownAffine`), so keying uses full-frame BGR.

    Required Keys:

        - img

    Modified Keys:

        - img

    Args:
        bg_dir (str, optional): Directory of background images. If relative,
            resolved from ``mmpose`` repo root (contains ``data/`` and the
            ``mmpose`` package). Default: ``data/House_Room_Dataset/Livingroom``.
        prob (float): Probability of applying augmentation per sample.
            Default: 0.5.
        threshold (int): Center white-key threshold (0--255). Default: 230.
        threshold_window (int): Per-sample random threshold is uniform in
            ``[threshold - W/2, threshold + W/2]`` (clamped to [0, 255]),
            same as ``run()`` in ``chroma_key_augment``. Default: 10.
        colorfulness_max (int): Max channel spread for white-like pixels.
            Default: 24.
        min_bright_fraction (float): Skip compositing if bright fraction
            below this (see ``process_frame_white_key``). Default: 1/3.
    """

    def __init__(
        self,
        bg_dir: Optional[str] = None,
        prob: float = _DEFAULT_PROB,
        threshold: int = _DEFAULT_THRESHOLD,
        threshold_window: int = _DEFAULT_THRESHOLD_WINDOW,
        colorfulness_max: int = _DEFAULT_COLORFULNESS_MAX,
        min_bright_fraction: float = _DEFAULT_MIN_BRIGHT_FRACTION,
    ) -> None:
        super().__init__()
        self.prob = float(prob)
        self.threshold = int(threshold)
        self.threshold_window = int(threshold_window)
        self.colorfulness_max = int(colorfulness_max)
        self.min_bright_fraction = float(min_bright_fraction)

        root = _repo_root()
        raw = _DEFAULT_BG_DIR if bg_dir is None else bg_dir
        p = Path(raw)
        if not p.is_absolute():
            p = (root / p).resolve()
        self.bg_dir = p
        self._bg_paths: List[str] = _list_background_images(self.bg_dir)
        self._warned_empty = False

        _load_chroma_module()  # fail fast if script missing

    def _sample_frame_threshold(self) -> int:
        w = max(0, int(self.threshold_window))
        half = w / 2.0
        low = int(np.clip(np.floor(float(self.threshold) - half), 0, 255))
        high = int(np.clip(np.ceil(float(self.threshold) + half), 0, 255))
        return int(np.random.randint(low, high + 1))

    def _apply_one(self, img: np.ndarray) -> np.ndarray:
        mod = _load_chroma_module()
        if not self._bg_paths:
            if not self._warned_empty:
                warnings.warn(
                    f'ChromaKeyAug: no background images in {self.bg_dir}. '
                    'Augmentation skipped until images are added.',
                    UserWarning,
                    stacklevel=2)
                self._warned_empty = True
            return img

        path = np.random.choice(self._bg_paths)
        h, w = img.shape[:2]
        bg = mod._load_background(path, w, h)
        t = self._sample_frame_threshold()
        return mod.process_frame_white_key(
            img,
            bg,
            t,
            self.colorfulness_max,
            self.min_bright_fraction,
        )

    def transform(self, results: dict) -> dict:
        if np.random.rand() >= self.prob:
            return results

        img = results.get('img')
        if img is None:
            return results

        if isinstance(img, list):
            results['img'] = [self._apply_one(x) for x in img]
        else:
            results['img'] = self._apply_one(img)
        return results
