# Models

Pretrained face detection / landmark models for compiler-compatibility testing.

## Files

| File | Format | Model | Size | Source |
|------|--------|-------|------|--------|
| `retinaface_scrfd500m.onnx` | ONNX | SCRFD-500MF (InsightFace's "RetinaFace") | 2.4 MB | InsightFace buffalo_sc pack |
| `pfld_backbone.onnx`        | ONNX | PFLD landmark backbone | 4.8 MB | Exported from PyTorch checkpoint |
| `pfld_backbone.h5`          | Keras h5 | Same PFLD, converted via onnx2tf | 27 MB | onnx2tf conversion |
| `pfld_backbone.tflite`      | TFLite fp32 | Same PFLD | 4.8 MB | onnx2tf conversion |
| `pfld_pretrained.pth.tar`   | PyTorch | Original PFLD checkpoint | 6.8 MB | polarisZhao/PFLD-pytorch |

## Format availability — why not all in h5?

Neither model ships as `.h5` natively:

- **RetinaFace (SCRFD)** — originally MXNet, distributed as ONNX. No official .h5.
- **PFLD** — originally PyTorch, distributed as `.pth`. No official .h5.

The `.h5` here is **converted** via [onnx2tf](https://github.com/PINTO0309/onnx2tf) from the ONNX file.

## Conversion results

**PFLD: ONNX → h5/tflite** — **succeeded**. All ops are convertible:
```
onnx2tf -i pfld_backbone.onnx -o pfld_tf -oh5
```
Output includes float32 .h5, .tflite, saved_model, and Keras format.

**RetinaFace: ONNX → h5** — **failed** at `Resize_108`:
```
ERROR: onnx_op_name: Resize_108
UnboundLocalError: local variable 'new_size' referenced before assignment
```
This is the exact op flagged as the incompatibility killer in [COMPILER_COMPATIBILITY.md](../COMPILER_COMPATIBILITY.md).
RetinaFace's FPN upsample uses dynamic-shape Resize, which neither onnx2tf nor the gpx-compiler can handle without re-architecting.

## Why PFLD's h5 is much bigger than its ONNX (27 MB vs 4.8 MB)

- ONNX stores weights as raw protobuf bytes with quantization-friendly tensor packing
- Converted .h5 includes the full Keras graph metadata, layer configs as JSON, plus weights as separate HDF5 datasets
- ONNX is closer to a packed binary; h5 is a hierarchical filesystem-like container

For deployment to the gpx-compiler, prefer the **.h5 → re-export to ONNX** path (round-trip via tf2onnx) for clean ONNX with named ops, OR use the original ONNX directly if your compiler frontend accepts it.

## Input/output shapes

| Model | Input | Output |
|-------|-------|--------|
| RetinaFace | (1, 3, dynamic, dynamic) | 9 tensors: cls/bbox/kps × 3 FPN scales |
| PFLD | (1, 3, 112, 112) | features (1, 64, 28, 28) + landmarks (1, 196) |

PFLD output is 196 = 98 landmarks × 2 coordinates.
