"""
Build a PFLD-style facial landmark detector in Keras and save as .h5.

Design choices driven by gpx-compiler compatibility:
  - Pure sequential conv-bn-relu blocks (no residual Adds — the unknown op)
  - No depthwise convolutions (also unknown to the compiler)
  - One AveragePool head (no pool→pool chains)
  - All ops on the supported list: Conv2D, BatchNorm, ReLU, AveragePool, Dense, Reshape
  - Input 112x112x3 (matches the alignment pipeline's output)
  - Output: 196 = 98 landmarks × 2 coords (PFLD convention)

Usage:
    python build_landmark_model.py --out landmark_model.h5
"""

import argparse
import tensorflow as tf
from tensorflow.keras import layers, Model


def conv_block(x, filters, kernel_size=3, strides=1, name=None):
    """Standard Conv -> BN -> ReLU. Compiler-friendly."""
    x = layers.Conv2D(
        filters, kernel_size, strides=strides, padding="same",
        use_bias=False, name=f"{name}_conv",
    )(x)
    x = layers.BatchNormalization(name=f"{name}_bn")(x)
    x = layers.ReLU(name=f"{name}_relu")(x)
    return x


def build_landmark_model(input_shape=(112, 112, 3), num_landmarks=98):
    inputs = layers.Input(shape=input_shape, name="image")

    # Stem
    x = conv_block(inputs, 32, kernel_size=3, strides=2, name="stem")     # 56x56
    x = conv_block(x, 32, kernel_size=3, strides=1, name="block1")

    # Stage 1
    x = conv_block(x, 64, kernel_size=3, strides=2, name="block2")        # 28x28
    x = conv_block(x, 64, kernel_size=3, strides=1, name="block3")
    x = conv_block(x, 64, kernel_size=3, strides=1, name="block4")

    # Stage 2
    x = conv_block(x, 128, kernel_size=3, strides=2, name="block5")       # 14x14
    x = conv_block(x, 128, kernel_size=3, strides=1, name="block6")
    x = conv_block(x, 128, kernel_size=3, strides=1, name="block7")

    # Stage 3
    x = conv_block(x, 256, kernel_size=3, strides=2, name="block8")       # 7x7
    x = conv_block(x, 256, kernel_size=3, strides=1, name="block9")

    # Head: global average pool -> dense -> dense
    x = layers.GlobalAveragePooling2D(name="gap")(x)                       # (256,)
    x = layers.Dense(128, name="fc1")(x)
    x = layers.ReLU(name="fc1_relu")(x)
    landmarks = layers.Dense(num_landmarks * 2, name="landmarks")(x)       # (196,)

    return Model(inputs, landmarks, name="pfld_simplified")


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
