"""
Classify footwear photos as shoes or sandals with the trained models.

Usage:
    python predict.py PHOTO.jpg                      # transfer-learning model (default)
    python predict.py PHOTO.jpg --model cnn          # self-built CNN
    python predict.py test_samples --model transfer  # a whole folder; prints accuracy
                                                     # when sub-folders are named shoes/ and sandals/
"""
import argparse
import glob
import os

import numpy as np
from PIL import Image, ImageOps

MODEL_FILES = {
    "transfer": "models/transfer_mobilenetv2.keras",
    "cnn": "models/self_cnn.keras",
}
CLASSES = ["sandals", "shoes"]  # model output = probability of "shoes"
IMG = 224


def load_image(path):
    """Same preprocessing as training: fix EXIF rotation, RGB, resize to 224x224, scale to -1..1."""
    import tensorflow as tf
    im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    x = tf.image.resize(np.asarray(im, dtype=np.float32), (IMG, IMG)).numpy()
    return x / 127.5 - 1.0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", help="an image file or a folder of images")
    ap.add_argument("--model", choices=MODEL_FILES, default="transfer")
    args = ap.parse_args()

    import keras
    here = os.path.dirname(os.path.abspath(__file__))
    model = keras.models.load_model(os.path.join(here, MODEL_FILES[args.model]))

    if os.path.isdir(args.path):
        paths = sorted(p for p in glob.glob(os.path.join(args.path, "**", "*"), recursive=True)
                       if p.lower().endswith((".jpg", ".jpeg", ".png", ".webp")))
    else:
        paths = [args.path]
    if not paths:
        raise SystemExit("No images found at " + args.path)

    correct = labelled = 0
    for p in paths:
        p_shoe = float(model.predict(load_image(p)[None], verbose=0)[0, 0])
        label = CLASSES[int(p_shoe >= 0.5)]
        print(f"{label:8s}  P(shoe)={p_shoe:.3f}  {p}")
        parent = os.path.basename(os.path.dirname(p)).lower()
        if parent in CLASSES:
            labelled += 1
            correct += int(parent == label)
    if labelled:
        print(f"\nAccuracy on labelled images: {correct}/{labelled} = {100 * correct / labelled:.1f}%")


if __name__ == "__main__":
    main()
