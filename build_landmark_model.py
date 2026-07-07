"""
Build a PFLD-style facial landmark detector in Keras and save as .h5.

Design choices driven by EdgeSphere GPX-10 compatibility (Keras-level rules):
  - **Sequential** model API only (Functional/subclassed models are rejected)
  - Pure sequential conv-bn-relu blocks (no residual Adds, no merge layers)
  - No depthwise / separable / grouped convolutions (groups must be 1)
  - AveragePooling2D + Flatten head instead of GlobalAveragePooling2D
    (global pooling is not in the GPX-10 supported set)
  - All ops on the supported list: Conv2D, BatchNormalization, ReLU,
    AveragePooling2D, Flatten, Dense
  - Input 112x112x3 (matches the alignment pipeline's output)
  - Output: 196 = 98 landmarks x 2 coords (PFLD convention)
  - Exported with TensorFlow / Keras 2.15 as a full .h5 (architecture + weights)

Usage:
    python build_landmark_model.py --out landmark_model.h5
"""

import argparse
from tensorflow.keras import layers, Sequential


def add_conv_block(model, filters, kernel_size=3, strides=1, name=None):
    """Standard Conv -> BN -> ReLU. GPX-10-friendly."""
    model.add(layers.Conv2D(
        filters, kernel_size, strides=strides, padding="same",
        use_bias=False, name=f"{name}_conv",
    ))
    model.add(layers.BatchNormalization(name=f"{name}_bn"))
    model.add(layers.ReLU(name=f"{name}_relu"))


def build_landmark_model(input_shape=(112, 112, 3), num_landmarks=98):
    # Narrow width schedule to keep total params < 200k while preserving the
    # GPX-10-safe layer vocabulary (Conv2D / BN / ReLU / Flatten / Dense).
    #
    # IMPORTANT: landmark *localisation* needs spatial detail. Global/large
    # average pooling collapses the H x W map to 1x1 and throws that away, which
    # caps accuracy badly (~19% NME). Instead we keep the 7x7 map and Flatten it
    # straight into the FC head, so the regressor sees where features are.
    # ~150k params at this width.
    model = Sequential(name="pfld_sequential")
    model.add(layers.Input(shape=input_shape, name="image"))

    # Stem
    add_conv_block(model, 16, kernel_size=3, strides=2, name="stem")    # 56x56
    add_conv_block(model, 16, kernel_size=3, strides=1, name="block1")

    # Stage 1
    add_conv_block(model, 24, kernel_size=3, strides=2, name="block2")   # 28x28
    add_conv_block(model, 24, kernel_size=3, strides=1, name="block3")

    # Stage 2
    add_conv_block(model, 32, kernel_size=3, strides=2, name="block4")   # 14x14
    add_conv_block(model, 32, kernel_size=3, strides=1, name="block5")

    # Stage 3
    add_conv_block(model, 32, kernel_size=3, strides=2, name="block6")   # 7x7

    # Head: flatten the 7x7 spatial map (preserves localisation) -> dense -> dense
    model.add(layers.Flatten(name="flatten"))                            # (7*7*32=1568,)
    model.add(layers.Dense(64, name="fc1"))
    model.add(layers.ReLU(name="fc1_relu"))
    model.add(layers.Dense(num_landmarks * 2, name="landmarks"))         # (196,)

    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="landmark_model.h5",
                        help="Path to save the .h5 model")
    parser.add_argument("--num_landmarks", type=int, default=98,
                        help="Number of landmark points (98 for PFLD, 68 for dlib)")
    parser.add_argument("--input_size", type=int, default=112,
                        help="Input image size (square)")
    args = parser.parse_args()

    model = build_landmark_model(
        input_shape=(args.input_size, args.input_size, 3),
        num_landmarks=args.num_landmarks,
    )
    # No compile() — keep inference-only graph for clean export to the compiler.
    model.summary()

    model.save(args.out)
    print(f"\nSaved: {args.out}")
    print(f"Parameters: {model.count_params():,}")


if __name__ == "__main__":
    main()
