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


def geo_augment(img, target, max_deg=25.0, scale_lo=0.85, scale_hi=1.20, trans=0.10):
    """Random rotation + scale + translation applied CONSISTENTLY to the image and
    the landmarks. Landmarks are the forward map; the image uses the matching
    inverse warp (ImageProjectiveTransformV3 samples output->input)."""
    lm = tf.reshape(target, (NUM_LANDMARKS, 2)) * CROP     # pixel coords
    ang = tf.random.uniform([], -max_deg, max_deg) * np.pi / 180.0
    s = tf.random.uniform([], scale_lo, scale_hi)
    tx = tf.random.uniform([], -trans, trans) * CROP
    ty = tf.random.uniform([], -trans, trans) * CROP
    c = CROP / 2.0
    cos, sin = tf.cos(ang), tf.sin(ang)

    # forward: p' = s * R * (p - c) + c + t          (moves landmarks)
    R = tf.stack([[cos, -sin], [sin, cos]])
    lm_new = tf.matmul(lm - c, R, transpose_b=True) * s + c + tf.stack([tx, ty])

    # inverse for the image op: p = B p' + d,  B = (1/s) R^-1
    inv = 1.0 / s
    b00, b01 = inv * cos, inv * sin
    b10, b11 = -inv * sin, inv * cos
    ctx, cty = c + tx, c + ty
    d0 = c - (b00 * ctx + b01 * cty)
    d1 = c - (b10 * ctx + b11 * cty)
    xf = tf.stack([b00, b01, d0, b10, b11, d1, 0.0, 0.0])
    img = tf.raw_ops.ImageProjectiveTransformV3(
        images=img[None], transforms=xf[None], output_shape=[CROP, CROP],
        interpolation="BILINEAR", fill_mode="REFLECT", fill_value=0.0)[0]
    return img, tf.reshape(lm_new / CROP, (-1,))


def compute_flip_map(targets):
    """Derive the horizontal-flip landmark permutation empirically: mirror the
    mean shape (x -> 1-x) and match each point to its nearest neighbour. Robust,
    and self-checked to be an involution permutation (no hardcoded 98-pt table)."""
    mean = targets.mean(0).reshape(NUM_LANDMARKS, 2)
    mirror = mean.copy(); mirror[:, 0] = 1.0 - mirror[:, 0]
    # cost[i, j] = distance between point i and the mirror of point j
    d = np.linalg.norm(mean[:, None, :] - mirror[None, :, :], axis=-1)
    # greedy mutual assignment -> guaranteed involution permutation
    pairs = sorted(((d[i, j], i, j) for i in range(NUM_LANDMARKS)
                    for j in range(NUM_LANDMARKS)), key=lambda x: x[0])
    fmap = -np.ones(NUM_LANDMARKS, dtype=np.int32)
    for _, i, j in pairs:
        if fmap[i] == -1 and fmap[j] == -1:
            fmap[i] = j; fmap[j] = i
    assert len(set(fmap.tolist())) == NUM_LANDMARKS, "flip map not a permutation"
    assert np.all(fmap[fmap] == np.arange(NUM_LANDMARKS)), "flip map not an involution"
    return fmap


def maybe_flip(img, target, flip_map):
    def do():
        im = tf.image.flip_left_right(img)
        lm = tf.reshape(target, (NUM_LANDMARKS, 2))
        lm = tf.stack([1.0 - lm[:, 0], lm[:, 1]], axis=1)
        lm = tf.gather(lm, flip_map)                 # re-map left<->right semantics
        return im, tf.reshape(lm, (-1,))
    return tf.cond(tf.random.uniform([]) < 0.5, do, lambda: (img, target))


def make_dataset(paths, targets, batch_size, training, flip_map=None):
    ds = tf.data.Dataset.from_tensor_slices((paths, targets))
    if training:
        ds = ds.shuffle(min(len(paths), 8192), reshuffle_each_iteration=True)
    fm = tf.constant(flip_map) if flip_map is not None else None

    def load(path, target):
        img = tf.io.decode_png(tf.io.read_file(path), channels=3)
        img = tf.image.resize(img, (CROP, CROP))
        img = tf.cast(img, tf.float32) / 255.0
        if training:
            if fm is not None:
                img, target = maybe_flip(img, target, fm)
            img, target = geo_augment(img, target)
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
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
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

    flip_map = compute_flip_map(tr_y)
    train_ds = make_dataset(tr_paths, tr_y, args.batch_size, training=True,
                            flip_map=flip_map)
    val_ds = make_dataset(te_paths, te_y, args.batch_size, training=False)

    # Cosine-decayed LR with a short warmup gives a cleaner descent than
    # reduce-on-plateau for this regression.
    steps_per_epoch = int(np.ceil(len(tr_paths) / args.batch_size))
    lr_sched = optimizers.schedules.CosineDecay(
        initial_learning_rate=args.lr, decay_steps=args.epochs * steps_per_epoch,
        warmup_target=args.lr, warmup_steps=2 * steps_per_epoch, alpha=0.02)

    model = build_landmark_model((CROP, CROP, 3), NUM_LANDMARKS)
    model.compile(optimizer=optimizers.Adam(lr_sched),
                  loss=wing_loss(), metrics=[nme_metric])
    model.summary()

    cbs = [
        callbacks.ModelCheckpoint(args.out, monitor="val_nme_metric",
                                  save_best_only=True, mode="min", verbose=1),
    ]
    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=cbs)

    # ModelCheckpoint already saved the best-NME weights to args.out; reload and
    # report them (the in-memory model holds last-epoch weights, not the best).
    best = tf.keras.models.load_model(args.out, compile=False)
    best.compile(loss=wing_loss(), metrics=[nme_metric])
    val = best.evaluate(val_ds, return_dict=True, verbose=0)
    print(f"\nBest WFLW test NME: {val['nme_metric']:.2f}%")
    print(f"Saved trained model -> {args.out}")


if __name__ == "__main__":
    main()
