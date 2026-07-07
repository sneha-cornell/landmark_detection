"""
Evaluate a trained landmark_model.h5 on the WFLW test set and report the
standard facial-landmark accuracy metrics:

  - NME   : Normalised Mean Error (inter-ocular), mean over the test set (lower is better)
  - PCK@t : Percentage of Correct Keypoints — fraction of the 98 landmarks whose
            error is below t x inter-ocular distance (higher is better = "accuracy")
  - FR@0.10 : Failure Rate — % of faces with NME > 10% (lower is better)
  - AUC@0.10 : area under the cumulative-error-distribution curve up to 10% (higher is better)

Usage:
    python evaluate.py --wflw_root /tmp/wflw --model landmark_model.h5
"""

import argparse
import numpy as np
import tensorflow as tf

from train_landmark import read_labels, make_dataset, NUM_LANDMARKS, EYE_L, EYE_R


def per_image_nme(gt, pred):
    """gt, pred: (N,196). Returns per-image NME and per-point normalised errors."""
    g = gt.reshape(-1, NUM_LANDMARKS, 2)
    p = pred.reshape(-1, NUM_LANDMARKS, 2)
    dist = np.linalg.norm(g - p, axis=-1)                    # (N,98)
    iod = np.linalg.norm(g[:, EYE_L] - g[:, EYE_R], axis=-1) # (N,)
    norm_err = dist / (iod[:, None] + 1e-9)                  # (N,98)
    nme = norm_err.mean(axis=1)                              # (N,)
    return nme, norm_err


def auc_and_fr(nme, thresh=0.10, step=0.0001):
    """AUC of the cumulative error distribution up to `thresh`, and failure rate."""
    xs = np.arange(0, thresh + step, step)
    ced = np.array([(nme <= x).mean() for x in xs])
    auc = np.trapz(ced, xs) / thresh
    fr = float((nme > thresh).mean())
    return auc, fr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wflw_root", default="/tmp/wflw")
    ap.add_argument("--model", default="landmark_model.h5")
    ap.add_argument("--batch_size", type=int, default=64)
    args = ap.parse_args()

    te_paths, te_y = read_labels(
        f"{args.wflw_root}/test_data/labels.csv",
        f"{args.wflw_root}/test_data/imgs")
    ds = make_dataset(te_paths, te_y, args.batch_size, training=False)

    model = tf.keras.models.load_model(args.model, compile=False)
    pred = model.predict(ds, verbose=0)

    nme, norm_err = per_image_nme(te_y, pred)
    auc, fr = auc_and_fr(nme, 0.10)

    print(f"Test images        : {len(te_y)}")
    print(f"NME (inter-ocular) : {100*nme.mean():.2f}%")
    print(f"AUC@0.10           : {auc:.4f}")
    print(f"Failure Rate @0.10 : {100*fr:.2f}%")
    print("Accuracy (PCK — % of landmarks within t x inter-ocular):")
    for t in (0.05, 0.08, 0.10, 0.15, 0.20):
        pck = 100.0 * (norm_err <= t).mean()
        print(f"   PCK@{t:.2f} : {pck:5.2f}%")


if __name__ == "__main__":
    main()
