"""
Turn class videos into a train/test image dataset.

Video names decide the label and the split:
    shoe_take1.mp4, shoe_take2.mp4, sandal_take1.mp4, sandal_take2.mp4 ...
    (name must start with "shoe" or "sandal"; .mp4 .mov .mkv .avi all work)

Whole videos go to train OR test, never both, so the test score is honest.
By default the highest-numbered take of each class becomes the test set.
Override with --test shoe_take3 sandal_take3

Usage:
    python extract_frames.py --videos videos --out dataset
Needs: pip install opencv-python numpy   (already installed in Colab)
"""
import argparse
import re
import shutil
from pathlib import Path

import cv2
import numpy as np

VIDEO_EXT = {".mp4", ".mov", ".mkv", ".avi", ".m4v", ".3gp"}


def label_of(stem):
    s = stem.lower()
    if s.startswith("shoe"):
        return "shoes"
    if s.startswith("sandal"):
        return "sandals"
    return None


def take_number(stem):
    m = re.search(r"(\d+)", stem)
    return int(m.group(1)) if m else 0


def extract(video, every, min_sharp, dup_thresh, max_side):
    """Return a list of kept frames (BGR arrays) from one video."""
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        print(f"  cannot open {video.name}")
        return []
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(fps * every)))
    kept, last_small, idx = [], None, 0
    skipped_blur = skipped_dup = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            sharp = cv2.Laplacian(gray, cv2.CV_64F).var()
            small = cv2.resize(gray, (32, 32)).astype(np.float32)
            if sharp < min_sharp:
                skipped_blur += 1
            elif last_small is not None and np.abs(small - last_small).mean() < dup_thresh:
                skipped_dup += 1
            else:
                h, w = frame.shape[:2]
                scale = max_side / max(h, w)
                if scale < 1:
                    frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
                kept.append(frame)
                last_small = small
        idx += 1
    cap.release()
    print(f"  {video.name}: kept {len(kept)}, blurry {skipped_blur}, near-duplicate {skipped_dup}")
    return kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True, help="folder with the videos")
    ap.add_argument("--out", default="dataset", help="output folder")
    ap.add_argument("--every", type=float, default=0.5, help="seconds between sampled frames")
    ap.add_argument("--min-sharp", type=float, default=40.0, help="blur cutoff, higher drops more")
    ap.add_argument("--dup", type=float, default=6.0, help="near-duplicate cutoff, higher drops more")
    ap.add_argument("--max-side", type=int, default=384, help="longest image side in pixels")
    ap.add_argument("--test", nargs="*", default=None, help="video names (no extension) to use as test")
    args = ap.parse_args()

    vids = sorted(p for p in Path(args.videos).iterdir() if p.suffix.lower() in VIDEO_EXT)
    if not vids:
        raise SystemExit("No videos found in " + args.videos)

    bad = [v.name for v in vids if label_of(v.stem) is None]
    if bad:
        raise SystemExit("Rename these so they start with 'shoe' or 'sandal': " + ", ".join(bad))

    by_class = {"shoes": [], "sandals": []}
    for v in vids:
        by_class[label_of(v.stem)].append(v)

    if args.test is not None:
        test_set = {t.lower() for t in args.test}
    else:
        test_set = set()
        for cls, vs in by_class.items():
            if len(vs) >= 2:
                test_set.add(max(vs, key=lambda p: take_number(p.stem)).stem.lower())
            else:
                print(f"WARNING: only {len(vs)} video for {cls}. Record more takes, otherwise there is no honest test set.")

    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    counts = {}
    for v in vids:
        cls = label_of(v.stem)
        split = "test" if v.stem.lower() in test_set else "train"
        folder = out / split / cls
        folder.mkdir(parents=True, exist_ok=True)
        print(f"{v.name} -> {split}/{cls}")
        for i, frame in enumerate(extract(v, args.every, args.min_sharp, args.dup, args.max_side)):
            cv2.imwrite(str(folder / f"{v.stem}_{i:04d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        counts[(split, cls)] = len(list(folder.glob("*.jpg")))

    print("\nFinal counts")
    for split in ("train", "test"):
        for cls in ("shoes", "sandals"):
            print(f"  {split:5s} {cls:8s} {counts.get((split, cls), 0)}")
    tr = [counts.get(("train", c), 0) for c in ("shoes", "sandals")]
    if min(tr) and max(tr) / min(tr) > 1.5:
        print("\nWARNING: classes are unbalanced in train. Lower --every for the smaller class or record another take.")


if __name__ == "__main__":
    main()
