"""
Train landmark_model.h5 on WFLW (98-point facial landmarks).

WFLW: https://wywu.github.io/projects/LAB/WFLW.html
  - WFLW_images/                         (raw images)
  - WFLW_annotations/list_98pt_rect_attr_train_test/
        list_98pt_rect_attr_train.txt
        list_98pt_rect_attr_test.txt

Each annotation line is space-separated:
  x0 y0 ... x97 y97   (196 landmark coords, pixel space)
  x_min y_min x_max y_max   (face rect)
  pose expr illum makeup occl blur   (6 attribute flags)
  image_name

The model regresses 98 landmarks in the normalised [0,1] crop space.
Metric: NME (Normalised Mean Error), normalised by inter-ocular distance
(WFLW outer eye corners = points 60 and 72) — the standard WFLW protocol.

Usage:
  python train_landmark.py --wflw_root /path/to/WFLW --epochs 120 --out landmark_model.h5
"""

import argparse
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers, callbacks

from build_landmark_model import build_landmark_model

NUM_LANDMARKS = 98
CROP = 112
# WFLW inter-ocular normalisation points (outer corners of the two eyes).
EYE_L, EYE_R = 60, 72


# ----------------------------- data pipeline ------------------------------ #
def parse_annotation_line(line):
    """Return (image_name, landmarks[98,2] in pixels, rect[4])."""
    p = line.strip().split(" ")
    coords = np.array(p[:196], dtype=np.float32).reshape(NUM_LANDMARKS, 2)
    rect = np.array(p[196:200], dtype=np.float32)      # x_min y_min x_max y_max
    name = p[-1]
    return name, coords, rect


def load_samples(ann_file):
    with open(ann_file) as f:
        return [parse_annotation_line(l) for l in f if l.strip()]


def crop_and_normalise(img, landmarks, rect, margin=0.25):
    """Crop the face box (with margin), resize to CROP, and map landmarks to
    normalised [0,1] coords relative to the crop."""
    x0, y0, x1, y1 = rect
    w, h = x1 - x0, y1 - y0
    x0 -= margin * w; y0 -= margin * h
    x1 += margin * w; y1 += margin * h
    cw, ch = max(x1 - x0, 1.0), max(y1 - y0, 1.0)

    lm = landmarks.copy()
    lm[:, 0] = (lm[:, 0] - x0) / cw
    lm[:, 1] = (lm[:, 1] - y0) / ch

    crop = tf.image.crop_to_bounding_box  # not used directly; we resize via crop-resize
    box = [[y0 / img.shape[0], x0 / img.shape[1],
            y1 / img.shape[0], x1 / img.shape[1]]]
    out = tf.image.crop_and_resize(img[None], box, [0], (CROP, CROP))[0]
    return out, lm.reshape(-1)


def make_dataset(samples, images_root, batch_size, training):
    def gen():
        idx = np.arange(len(samples))
        if training:
            np.random.shuffle(idx)
        for i in idx:
            name, coords, rect = samples[i]
            path = os.path.join(images_root, name)
            raw = tf.io.read_file(path)
            img = tf.image.decode_jpeg(raw, channels=3)
            img = tf.cast(img, tf.float32) / 255.0
            crop, target = crop_and_normalise(img.numpy(), coords, rect)
            yield crop, target

    ds = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec((CROP, CROP, 3), tf.float32),
            tf.TensorSpec((NUM_LANDMARKS * 2,), tf.float32),
        ),
    )
    if training:
        ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def augment(img, target):
    """Light photometric augmentation (geometry is fixed by the crop)."""
    img = tf.image.random_brightness(img, 0.1)
    img = tf.image.random_contrast(img, 0.8, 1.2)
    return tf.clip_by_value(img, 0.0, 1.0), target


# --------------------------- loss & metric -------------------------------- #
def wing_loss(w=10.0, eps=2.0):
    """Wing loss (Feng et al. 2018) — better than L2 for small landmark errors."""
    C = w - w * np.log(1.0 + w / eps)

    def loss(y_true, y_pred):
        d = tf.abs(y_true - y_pred)
        return tf.reduce_mean(
            tf.where(d < w, w * tf.math.log(1.0 + d / eps), d - C)
        )
    return loss


def nme_metric(y_true, y_pred):
    """Normalised Mean Error, normalised by inter-ocular distance. Reported in %."""
    t = tf.reshape(y_true, (-1, NUM_LANDMARKS, 2))
    p = tf.reshape(y_pred, (-1, NUM_LANDMARKS, 2))
    per_pt = tf.norm(t - p, axis=-1)                    # (B, 98)
    iod = tf.norm(t[:, EYE_L] - t[:, EYE_R], axis=-1)   # (B,)
    return 100.0 * tf.reduce_mean(per_pt, axis=-1) / (iod + 1e-6)


# ------------------------------- main ------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wflw_root", required=True,
                    help="Folder containing WFLW_images/ and WFLW_annotations/")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default="landmark_model.h5")
    args = ap.parse_args()

    ann_dir = os.path.join(args.wflw_root, "WFLW_annotations",
                           "list_98pt_rect_attr_train_test")
    images_root = os.path.join(args.wflw_root, "WFLW_images")
    train = load_samples(os.path.join(ann_dir, "list_98pt_rect_attr_train.txt"))
    test = load_samples(os.path.join(ann_dir, "list_98pt_rect_attr_test.txt"))
    print(f"train={len(train)}  test={len(test)}")

    train_ds = make_dataset(train, images_root, args.batch_size, training=True)
    val_ds = make_dataset(test, images_root, args.batch_size, training=False)

    model = build_landmark_model(input_shape=(CROP, CROP, 3),
                                 num_landmarks=NUM_LANDMARKS)
    model.compile(
        optimizer=optimizers.Adam(args.lr),
        loss=wing_loss(),
        metrics=[nme_metric],
    )
    model.summary()

    cbs = [
        callbacks.ReduceLROnPlateau(monitor="val_nme_metric", factor=0.5,
                                    patience=8, min_lr=1e-5, mode="min"),
        callbacks.ModelCheckpoint(args.out, monitor="val_nme_metric",
                                  save_best_only=True, mode="min"),
        callbacks.EarlyStopping(monitor="val_nme_metric", patience=20,
                                mode="min", restore_best_weights=True),
    ]
    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=cbs)

    val = model.evaluate(val_ds, return_dict=True)
    print(f"\nBest WFLW test NME: {val['nme_metric']:.2f}%")
    model.save(args.out)
    print(f"Saved trained model -> {args.out}")


if __name__ == "__main__":
    main()
