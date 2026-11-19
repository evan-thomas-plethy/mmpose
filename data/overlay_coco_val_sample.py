#!/usr/bin/env python3
"""Download COCO val2017 images and keypoint annotations if missing, then overlay keypoints on a random sample.

Writes images to data/coco/overlays/.

Usage:
    python data/overlay_coco_val_sample.py
    python data/overlay_coco_val_sample.py --n 5 --seed 0
"""

import argparse
import json
import random
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from overlay_dataset import draw_keypoints, draw_skeleton

COCO_ROOT = Path(__file__).resolve().parent / "coco"
ANN_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
IMAGES_URL = "http://images.cocodataset.org/zips/val2017.zip"
ANN_FILE = COCO_ROOT / "annotations" / "person_keypoints_val2017.json"
IMAGES_DIR = COCO_ROOT / "val2017"
OUT_DIR = COCO_ROOT / "overlays"


def _download(url: str, dest: Path, retries: int = 8) -> None:
    """Stream ``url`` to ``dest``, resuming a ``.part`` file and retrying dropped connections."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    last_error: Exception | None = None

    for attempt in range(1, retries + 1):
        have = tmp.stat().st_size if tmp.is_file() else 0
        headers = {"Range": f"bytes={have}-"} if have else {}
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                status = getattr(response, "status", 200)
                if status == 200:
                    have = 0
                    mode = "wb"
                elif status == 206:
                    mode = "ab"
                else:
                    raise urllib.error.HTTPError(url, status, response.reason, response.headers, None)

                length = response.headers.get("Content-Length")
                total = (have + int(length)) if length else None
                print(f"Downloading {url} (resuming at {have / 1e6:.1f} MB)" if have else f"Downloading {url}")
                done = have
                with tmp.open(mode) as out:
                    while True:
                        chunk = response.read(1 << 20)
                        if not chunk:
                            break
                        out.write(chunk)
                        done += len(chunk)
                        if total:
                            print(
                                f"\r  {dest.name}: {100.0 * done / total:5.1f}% ({done / 1e6:.0f}/{total / 1e6:.0f} MB)",
                                end="",
                                flush=True,
                            )
                print()
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code == 416 and tmp.is_file():
                pass
            else:
                print(f"  attempt {attempt}/{retries} failed: {exc}")
                time.sleep(min(2 ** attempt, 30))
                continue
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
            last_error = exc
            print(f"\n  attempt {attempt}/{retries} failed: {exc}")
            time.sleep(min(2 ** attempt, 30))
            continue

        if zipfile.is_zipfile(tmp):
            tmp.replace(dest)
            return
        print(f"  attempt {attempt}/{retries}: file is not a complete zip yet")
        time.sleep(min(2 ** attempt, 30))

    raise RuntimeError(f"Failed to download {url}") from last_error


def _extract(zip_path: Path, dest_dir: Path) -> None:
    print(f"Extracting {zip_path.name} -> {dest_dir}")
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest_dir)


def ensure_coco() -> None:
    """Download and extract val images and keypoint annotations when they are not already on disk."""
    if not ANN_FILE.is_file():
        zip_path = COCO_ROOT / "annotations_trainval2017.zip"
        if not zip_path.is_file():
            _download(ANN_URL, zip_path)
        _extract(zip_path, COCO_ROOT)
        zip_path.unlink(missing_ok=True)
        if not ANN_FILE.is_file():
            raise FileNotFoundError(f"Expected {ANN_FILE} after extracting annotations")

    has_images = IMAGES_DIR.is_dir() and any(IMAGES_DIR.glob("*.jpg"))
    if not has_images:
        zip_path = COCO_ROOT / "val2017.zip"
        if not zip_path.is_file():
            _download(IMAGES_URL, zip_path)
        _extract(zip_path, COCO_ROOT)
        zip_path.unlink(missing_ok=True)
        if not any(IMAGES_DIR.glob("*.jpg")):
            raise FileNotFoundError(f"Expected JPEGs in {IMAGES_DIR} after extracting val2017")


def sample_images(coco: dict, n: int, rng: random.Random) -> list[tuple[dict, list]]:
    """Pick n images that have at least one person with visible keypoints."""
    anns_by_image: dict[int, list] = {}
    for ann in coco["annotations"]:
        if ann.get("num_keypoints", 0) > 0:
            anns_by_image.setdefault(ann["image_id"], []).append(ann)

    candidates = [img for img in coco["images"] if img["id"] in anns_by_image]
    if len(candidates) < n:
        raise RuntimeError(f"Only {len(candidates)} images have keypoints; asked for {n}")
    chosen = rng.sample(candidates, n)
    return [(img, anns_by_image[img["id"]]) for img in chosen]


def overlay_one(image_info: dict, anns: list, out_dir: Path) -> Path:
    image_path = IMAGES_DIR / image_info["file_name"]
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(image_path)
    for ann in anns:
        image = draw_skeleton(image, ann["keypoints"])
        image = draw_keypoints(image, ann["keypoints"])
    out_path = out_dir / image_info["file_name"]
    cv2.imwrite(str(out_path), image)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=5, help="Number of images to overlay")
    parser.add_argument("--seed", type=int, default=None, help="RNG seed")
    parser.add_argument("--out", type=Path, default=OUT_DIR, help="Directory for overlay images")
    args = parser.parse_args()

    ensure_coco()
    with ANN_FILE.open() as f:
        coco = json.load(f)

    rng = random.Random(args.seed)
    chosen = sample_images(coco, args.n, rng)
    args.out.mkdir(parents=True, exist_ok=True)

    for image_info, anns in chosen:
        out_path = overlay_one(image_info, anns, args.out)
        print(f"{image_info['file_name']}  persons={len(anns)}  -> {out_path}")


if __name__ == "__main__":
    main()
