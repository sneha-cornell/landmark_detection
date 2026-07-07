"""
Build a GPX-10-compatible single-face detector in Keras and save as .h5.

This replaces RetinaFace's role in the alignment front-end for the common
single-dominant-face case. RetinaFace itself is NOT GPX-10-deployable (it uses
FPN Resize + Add + depthwise conv, and a multi-head Functional graph). GPX-10
allows only a Sequential model over a restricted op set, which rules out
multi-scale anchor detectors — so this is a direct-regression design instead:

    full image (112x112x3)  ->  Conv/BN/ReLU trunk  ->  Flatten  ->  Dense
    ->  14 outputs: [x1, y1, x2, y2, (lx, ly) x 5]   (box + 5 landmarks, all in
        normalised [0,1] image coords)

Input is 112x112 to match the rest of the pipeline's resolution. Note this is
the DETECTOR input (the raw photo downscaled to 112x112); the aligned crop it
ultimately produces for the landmark/recognition stage is a separate 112x112.

Assumes exactly one face present (no confidence head, no NMS). The 5 landmarks
are all the downstream similarity-transform needs; the box is a useful bonus.

Only GPX-10-supported layers: Conv2D (groups=1) / BatchNormalization / ReLU /
Flatten / Dense. No depthwise, no Add/Concat, no pooling, no upsampling.

Usage:
    python build_detector_model.py --out detector_model.h5
"""

import argparse
from tensorflow.keras import layers, Sequential


def add_conv_block(model, filters, kernel_size=3, strides=1, name=None):
    model.add(layers.Conv2D(filters, kernel_size, strides=strides, padding="same",
                            use_bias=False, name=f"{name}_conv"))
    model.add(layers.BatchNormalization(name=f"{name}_bn"))
    model.add(layers.ReLU(name=f"{name}_relu"))


def build_detector_model(input_shape=(112, 112, 3), num_landmarks=5):
    # box (4) + num_landmarks * 2. ~130k params at this width.
    out_dim = 4 + num_landmarks * 2
    model = Sequential(name="face_detector_sequential")
    model.add(layers.Input(shape=input_shape, name="image"))

    add_conv_block(model, 16, strides=2, name="stem")     # 56x56
    add_conv_block(model, 24, strides=2, name="block1")    # 28x28
    add_conv_block(model, 24, strides=1, name="block2")
    add_conv_block(model, 32, strides=2, name="block3")    # 14x14
    add_conv_block(model, 32, strides=1, name="block4")
    add_conv_block(model, 48, strides=2, name="block5")    # 7x7
    add_conv_block(model, 64, strides=2, name="block6")    # 4x4

    model.add(layers.Flatten(name="flatten"))              # 4*4*64 = 1024
    model.add(layers.Dense(64, name="fc1"))
    model.add(layers.ReLU(name="fc1_relu"))
    model.add(layers.Dense(out_dim, name="detections"))    # box + landmarks
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="detector_model.h5")
    ap.add_argument("--input_size", type=int, default=112)
    args = ap.parse_args()
    model = build_detector_model((args.input_size, args.input_size, 3))
    model.summary()
    model.save(args.out)
    print(f"\nSaved: {args.out}")
    print(f"Parameters: {model.count_params():,}")


if __name__ == "__main__":
    main()
