"""
Classify footwear photos as shoes or sandals with the trained weights.

The architectures are defined below (identical to the training notebook) and the trained
weights are loaded from models/*.weights.h5.

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

HERE = os.path.dirname(os.path.abspath(__file__))
WEIGHT_FILES = {
    "transfer": "models/transfer_mobilenetv2.weights.h5",
    "cnn": "models/self_cnn.weights.h5",
}
CLASSES = ["sandals", "shoes"]  # model output = probability of "shoes"
IMG = 224


def build_cnn():
    """Self-built CNN, trained from scratch."""
    from keras import Input, Model, layers
    inp = Input((IMG, IMG, 3))
    x = inp
    for f in [32, 64, 128, 128]:
        x = layers.Conv2D(f, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization(momentum=0.9)(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling2D()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    return Model(inp, out, name="self_cnn")


def build_transfer():
    """MobileNetV2 base + new classification head (transfer learning)."""
    import keras
    from keras import Input, Model, layers
    base = keras.applications.MobileNetV2(input_shape=(IMG, IMG, 3), include_top=False, weights=None)
    inp = Input((IMG, IMG, 3))
    x = base(inp, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    return Model(inp, out, name="transfer_mobilenetv2")


def weights_file(path):
    """The transfer weights are stored as path.part00, .part01, ... because GitHub's upload
    page limits file size. Join them into one file the first time they are needed."""
    if os.path.exists(path):
        return path
    parts = sorted(glob.glob(path + ".part*"))
    if not parts:
        raise SystemExit("Weights not found: " + path)
    with open(path, "wb") as out:
        for part in parts:
            with open(part, "rb") as f:
                out.write(f.read())
    print(f"Joined {len(parts)} parts into {path}")
    return path


def load_model(name):
    model = build_transfer() if name == "transfer" else build_cnn()
    model.load_weights(weights_file(os.path.join(HERE, WEIGHT_FILES[name])))
    return model


def load_image(path):
    """Same preprocessing as training: fix EXIF rotation, RGB, resize to 224x224, scale to -1..1."""
    import tensorflow as tf
    im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    x = tf.image.resize(np.asarray(im, dtype=np.float32), (IMG, IMG)).numpy()
    return x / 127.5 - 1.0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", help="an image file or a folder of images")
    ap.add_argument("--model", choices=WEIGHT_FILES, default="transfer")
    args = ap.parse_args()

    model = load_model(args.model)

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
