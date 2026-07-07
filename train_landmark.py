"""
Train landmark_model.h5 on WFLW (98-point facial landmarks).

Dataset layout (PFLD-style, pre-cropped 112x112 faces):
    <root>/train_data/labels.csv  + train_data/imgs/*.png
    <root>/test_data/labels.csv   + test_data/imgs/*.png

Each labels.csv row (206 columns, comma-separated):
    0..195   : 98 landmarks x 2, normalised to [0,1] in the 112x112 crop
    196..201 : 6 attribute flags (pose/expr/illum/makeup/occl/blur)
    202..204 : euler angles (pitch/yaw/roll)  -- unused here
    205      : original image path (we use only its basename)

The model regresses the 196 normalised coords directly.
Metric: NME (Normalised Mean Error) normalised by inter-ocular distance
(WFLW outer eye corners = points 60 and 72) — the standard WFLW protocol.

A HF mirror in this exact format: duongnguy/WFLW_augmented
(train_data.tar.gz / test_data.tar.gz).

Usage:
    python train_landmark.py --wflw_root /path/to/wflw --epochs 40 --out landmark_model.h5
"""

import argparse
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers, callbacks

from build_landmark_model import build_landmark_model

NUM_LANDMARKS = 98
CROP = 112
EYE_L, EYE_R = 60, 72   # WFLW inter-ocular normalisation points


# ----------------------------- data pipeline ------------------------------ #
def read_labels(csv_path, imgs_dir):
    """Return (image_paths[list], targets[N,196] float32)."""
    paths, targets = [], []
    with open(csv_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            p = line.split(",")
            coords = np.asarray(p[:NUM_LANDMARKS * 2], dtype=np.float32)
            name = os.path.basename(p[-1])
            paths.append(os.path.join(imgs_dir, name))
            targets.append(coords)
    return paths, np.stack(targets)


def make_dataset(paths, targets, batch_size, training):
    ds = tf.data.Dataset.from_tensor_slices((paths, targets))
    if training:
        ds = ds.shuffle(min(len(paths), 8192), reshuffle_each_iteration=True)

    def load(path, target):
        img = tf.io.decode_png(tf.io.read_file(path), channels=3)
        img = tf.image.resize(img, (CROP, CROP))
        img = tf.cast(img, tf.float32) / 255.0
        if training:
            img = tf.image.random_brightness(img, 0.1)
            img = tf.image.random_contrast(img, 0.8, 1.2)
            img = tf.clip_by_value(img, 0.0, 1.0)
        return img, target

    ds = ds.map(load, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


# --------------------------- loss & metric -------------------------------- #
def wing_loss(w=10.0, eps=2.0):
    """Wing loss (Feng et al. 2018) — better than L2 for small landmark errors.
    Operates on landmark coords scaled to pixels (x112) so w/eps are in pixels."""
    C = w - w * np.log(1.0 + w / eps)

    def loss(y_true, y_pred):
        d = tf.abs(y_true - y_pred) * CROP
        return tf.reduce_mean(
            tf.where(d < w, w * tf.math.log(1.0 + d / eps), d - C)
        )
    return loss


def nme_metric(y_true, y_pred):
    """Normalised Mean Error (%), inter-ocular normalisation."""
    t = tf.reshape(y_true, (-1, NUM_LANDMARKS, 2))
    p = tf.reshape(y_pred, (-1, NUM_LANDMARKS, 2))
    per_pt = tf.norm(t - p, axis=-1)                    # (B, 98)
    iod = tf.norm(t[:, EYE_L] - t[:, EYE_R], axis=-1)   # (B,)
    return 100.0 * tf.reduce_mean(per_pt, axis=-1) / (iod + 1e-6)


# ------------------------------- main ------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wflw_root", required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max_train", type=int, default=0,
                    help="Optional cap on #train samples (0 = use all).")
    ap.add_argument("--out", default="landmark_model.h5")
    args = ap.parse_args()

    tr_paths, tr_y = read_labels(
        os.path.join(args.wflw_root, "train_data", "labels.csv"),
        os.path.join(args.wflw_root, "train_data", "imgs"))
    te_paths, te_y = read_labels(
        os.path.join(args.wflw_root, "test_data", "labels.csv"),
        os.path.join(args.wflw_root, "test_data", "imgs"))
    if args.max_train and args.max_train < len(tr_paths):
        idx = np.random.RandomState(0).permutation(len(tr_paths))[:args.max_train]
        tr_paths = [tr_paths[i] for i in idx]; tr_y = tr_y[idx]
    print(f"train={len(tr_paths)}  test={len(te_paths)}")

    train_ds = make_dataset(tr_paths, tr_y, args.batch_size, training=True)
    val_ds = make_dataset(te_paths, te_y, args.batch_size, training=False)

    model = build_landmark_model((CROP, CROP, 3), NUM_LANDMARKS)
    model.compile(optimizer=optimizers.Adam(args.lr),
                  loss=wing_loss(), metrics=[nme_metric])
    model.summary()

    cbs = [
        callbacks.ReduceLROnPlateau(monitor="val_nme_metric", factor=0.5,
                                    patience=5, min_lr=1e-5, mode="min"),
        callbacks.ModelCheckpoint(args.out, monitor="val_nme_metric",
                                  save_best_only=True, mode="min", verbose=1),
        callbacks.EarlyStopping(monitor="val_nme_metric", patience=12,
                                mode="min", restore_best_weights=True),
    ]
    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=cbs)

    val = model.evaluate(val_ds, return_dict=True, verbose=0)
    print(f"\nBest WFLW test NME: {val['nme_metric']:.2f}%")
    model.save(args.out)
    print(f"Saved trained model -> {args.out}")


if __name__ == "__main__":
    main()
