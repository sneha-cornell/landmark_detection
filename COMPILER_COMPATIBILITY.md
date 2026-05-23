# RetinaFace & PFLD Compatibility Analysis for gpx-compiler

Target: embedded NPU (gpx-compiler).
Supported ONNX ops: `Conv` (2D), `MaxPool`, `AveragePool`, `Gemm`, `Relu`, `Sigmoid`, `Softmax`, `Tanh`, `Reshape`, `Transpose`.
Quantization: int8 / power-of-2 scales. Layout: NCHW. No pool→pool. AON path: input must be 1×1, 2×2, 3×3, 4×4, or 10×10 (or multiple of 20). Main CNN path: arbitrary shapes.

Models actually inspected:
- **RetinaFace** — `det_500m.onnx` from InsightFace `buffalo_sc` bundle (this is the SCRFD-500MF variant — the production "RetinaFace" model that ships in `insightface`). Already on disk at `/Users/snehamanimurugan/.insightface/models/buffalo_sc/det_500m.onnx`.
- **PFLD** — exported fresh from `polarisZhao/PFLD-pytorch` (`/tmp/pfld_repo/models/pfld.py`) into `/tmp/pfld_backbone.onnx` and `/tmp/pfld_aux.onnx`. The original `guoqiangqi/PFLD-pytorch` repo is no longer reachable; `polarisZhao/PFLD-pytorch` is the canonical mirror and matches the architecture in the PFLD paper.

---

## 1. RetinaFace (SCRFD-500MF / `det_500m.onnx`)

### Basic facts
| Field | Value |
|---|---|
| Producer | PyTorch 1.6 → ONNX opset 11 |
| File size | 2.4 MB (`.onnx`) |
| Parameters | **626,330 (~0.63 M)** |
| Input | `input.1` shape `[1, 3, H, W]` — H/W dynamic; runtime default 640×640 (or 320×320 in `buffalo_sc`) |
| Outputs | **9 tensors**: 3 scales (stride 8 / 16 / 32) × 3 heads (cls / bbox / kps) |
| Layout | NCHW |
| Backbone | MobileNet-style depthwise-separable + FPN neck + SCRFD detection heads |

Output heads (per scale, 2 anchors / cell):
- cls: `[N, 1]` (objectness)
- bbox: `[N, 4]` (distance-based)
- kps: `[N, 10]` (5 facial landmarks)

Element counts (12800 / 3200 / 800) correspond to stride 8 / 16 / 32 over a 640×640 input × 2 anchors.

### ONNX op inventory (146 nodes total)
| Op | Count | Compiler support |
|---|---|---|
| Conv | 60 (40 standard + **20 depthwise**, group ≠ 1) | Conv 2D supported; **depthwise = grouped Conv listed as "unknown support"** |
| Relu | 41 | Supported |
| Transpose | 9 | Supported |
| Reshape | 9 | Supported |
| Shape | 6 | **NOT in supported list** (dynamic-shape op) |
| Gather | 4 | **NOT in supported list** |
| Unsqueeze | 4 | **NOT in supported list** |
| Add | 4 | **"Unknown support" (residual element-wise add)** |
| Sigmoid | 3 | Supported |
| Slice | 2 | **NOT in supported list** |
| Concat | 2 | **"Unknown support"** (axis=0, used in dynamic Resize-target wiring) |
| Resize | 2 | **NOT supported** (nearest upsample for FPN top-down) |

(No `Gemm`, no `MaxPool`/`AveragePool`, no `BatchNorm` — BN is folded into Conv. There is no Softmax: classification uses Sigmoid.)

### Architectural features
- **Depthwise-separable MobileNet backbone** (stages 16→16→40→72→152→288 channels).
- **FPN neck**: three 1×1 lateral convs (`Conv_87/88/89`) + two `Resize` (nearest, scale 2) upsamples + two `Add` merges = top-down feature pyramid.
- **Shared detection head** at each FPN level: 1× depthwise 3×3 + 1×1 + depthwise 3×3 + 1×1 followed by three 3×3 head convs producing cls (2 ch), bbox (8 ch), kps (20 ch).
- Per-head `Reshape` + `Transpose` to flatten predictions; `Shape`/`Gather`/`Unsqueeze`/`Concat`/`Slice` exist purely to compute the dynamic reshape targets (because input H/W are dynamic).

### Compatibility verdict — RetinaFace: **INCOMPATIBLE as-is (partial after surgery)**
Hard blockers against the supported-op list:
1. **`Resize` (nearest upsample)** — used twice in the FPN. Compiler list flags upsample/interpolation as unknown. This is a structural op the model **cannot run without**.
2. **`Add` (residual)** — 4 instances at FPN merge points; element-wise add is flagged unknown.
3. **`Concat`** — only used internally for dynamic shape arithmetic; would disappear if input shape is fixed.
4. **Depthwise Conv (20 nodes)** — grouped Conv, flagged unknown. Required by the entire backbone.
5. **`Shape`/`Gather`/`Unsqueeze`/`Slice`** — 16 nodes of dynamic-shape glue; all become trivially foldable if input H/W is frozen at export time.

If the export is re-done with a **fixed input shape** (e.g. 320×320), the Shape/Gather/Unsqueeze/Slice/Concat path disappears via constant-folding, leaving Conv / Relu / Sigmoid / Reshape / Transpose / Resize / Add / depthwise-Conv. The two real architectural blockers that remain are **Resize and Add (FPN)** and **depthwise Conv**.

---

## 2. PFLD (`PFLDInference` + `AuxiliaryNet`)

### Basic facts
| Field | Value |
|---|---|
| Source | `polarisZhao/PFLD-pytorch` → exported with `torch.onnx`, opset 11 |
| File size | 5.0 MB backbone + 1.8 MB aux (FP32 ONNX) |
| Parameters | **1,264,804** backbone (1.27 M) + **463,811** aux (0.46 M) = 1.73 M combined |
| Input | `[1, 3, 112, 112]` (fixed) |
| Outputs | backbone: `features [1,64,28,28]` + `landmarks [1,196]` (98 landmarks × 2); aux: `euler [1,3]` (pitch/yaw/roll, training-only) |
| Layout | NCHW |
| Backbone | MobileNetV2-style inverted-residual blocks |

### ONNX op inventory — backbone (125 nodes, ignoring Constants)
| Op | Count | Compiler support |
|---|---|---|
| Conv | 43 (30 standard + **13 depthwise** inside inverted residuals) | Conv supported; depthwise = unknown |
| Relu | 30 | Supported |
| Add | 9 | **Unknown (residuals inside inverted-residual blocks)** |
| AveragePool | 2 (14×14 and 7×7 global pools) | Supported |
| Concat | 1 (axis=1, multi-scale feature fusion) | **Unknown** |
| Reshape | 3 | Supported |
| Gemm | 1 (FC 176 → 196) | Supported |
| Identity | 36 | Trivial (drop in pre-pass) |
| Constant | 3 | Trivial |

### ONNX op inventory — aux net (15 nodes)
| Op | Count | Compiler support |
|---|---|---|
| Conv | 4 (all standard) | Supported |
| Relu | 4 | Supported |
| MaxPool | 1 (3×3 stride 3) | Supported |
| Gemm | 2 | Supported |
| Reshape | 1 | Supported |
| Identity, Constant | 2, 1 | Trivial |

### Architectural features
- **Backbone**: stem (Conv 3×3 s2 + Conv 3×3 s1) → 5 inverted-residual blocks at 64ch → 1 reduction IR-block to 128ch → 6 inverted-residual blocks at 128ch → 1 IR to 16ch → Conv 3×3 s2 → Conv 7×7. Aux features tap out after the first IR-stack (`out1` = `[1,64,28,28]`).
- **Multi-scale head**: three pooled feature maps are flattened and **concatenated on the channel axis**, then a single FC produces 196 landmark coords (98 × (x,y)).
- **Pooling: no pool→pool sequence** in backbone. Aux net has only one MaxPool → FC. Compiler restriction satisfied.
- **Residual connections (9 Add ops)** inside every inverted-residual block where `use_res_connect=True`.
- All activations are plain ReLU. No ReLU6 / LeakyReLU / Sigmoid / Softmax.
- BatchNorm is folded into Conv weights at export time (the PFLD source has BN after every conv; you must export in `eval()` mode + constant-folding to get this clean graph, which we did).

### Compatibility verdict — PFLD: **PARTIALLY COMPATIBLE**
Blockers against the supported-op list:
1. **`Add` (9 instances)** — residual connections inside every `InvertedResidual` block with `use_res_connect=True`. Cannot be removed without retraining (drops accuracy materially).
2. **`Concat` (1 instance, axis=1)** — multi-scale feature fusion before the final FC. Could be replaced by three parallel FCs whose outputs are added/concatenated externally, but that changes the parameter shape of `fc` and requires retraining.
3. **Depthwise Conv (13 nodes)** — inside every inverted residual. Required.

The aux net (Euler angle head) is **fully compatible** with the supported-op list — it's pure Conv + Relu + MaxPool + Gemm + Reshape. But it's a training-only auxiliary and is normally not deployed.

Everything else (Conv, Relu, AveragePool, Reshape, Gemm) is on the supported list.

### Pooling-after-pooling check
Backbone has two `AveragePool`s, but they consume **different** feature maps in parallel branches — they are not pool→pool serially. **OK.**

### AON sensor-fusion path
Input is 112×112 — not in `{1, 2, 3, 4, 10}` and not a multiple of 20. **PFLD cannot run on the AON path**; must run on the main CNN path. Same is true for RetinaFace at 320/640.

---

## 3. Op-by-op compatibility matrix

| Op | RetinaFace | PFLD backbone | PFLD aux | Compiler list |
|---|---|---|---|---|
| Conv (standard) | 40 | 30 | 4 | **Yes** |
| Conv (depthwise / grouped) | 20 | 13 | 0 | Unknown |
| Relu | 41 | 30 | 4 | **Yes** |
| Sigmoid | 3 | 0 | 0 | **Yes** |
| Gemm | 0 | 1 | 2 | **Yes** |
| Reshape | 9 | 3 | 1 | **Yes** |
| Transpose | 9 | 0 | 0 | **Yes** |
| MaxPool | 0 | 0 | 1 | **Yes** |
| AveragePool | 0 | 2 | 0 | **Yes** |
| Softmax / Tanh | 0 | 0 | 0 | **Yes** |
| Add (residual) | 4 | 9 | 0 | Unknown |
| Concat | 2 | 1 | 0 | Unknown |
| Resize (upsample) | 2 | 0 | 0 | Unknown |
| Shape / Gather / Unsqueeze / Slice | 16 | 0 | 0 | **No** (dynamic-shape glue) |

---

## 4. Verdicts and recommendations

### RetinaFace — *Incompatible* (heavy modification required)
The fundamental architecture relies on:
- **FPN with Resize + Add** — these are the load-bearing feature-pyramid ops and there is no way to remove them while preserving multi-scale detection.
- **Depthwise convolutions** throughout the backbone.
- **Dynamic input shape** causes 16 unsupported Shape/Gather/Unsqueeze/Slice nodes; these *can* be eliminated by re-exporting with a fixed input shape (use `torch.onnx.export(..., dynamic_axes=None, opset_version=11)` and a frozen 320×320 input).

Even after fixing the input shape, **Resize + Add + depthwise Conv** would all need compiler-side support added before this model is deployable. If `Add`/`Concat`/depthwise Conv are added but Resize is not, RetinaFace is still dead.

### PFLD — *Partially compatible* (depends on three "unknown" ops)
If the compiler can support **element-wise Add** (residual) and **depthwise Conv**, PFLD is essentially deployable. The one channel-axis `Concat` is the smallest blocker — it could be replaced by **three independent FCs** whose results are summed, eliminating Concat entirely (small refactor, requires re-training).

Recommended modifications, in order of cost:
1. **Drop the aux net** at deploy time (it's training-only anyway) — already standard practice.
2. **Replace channel-Concat + single FC** with three parallel FCs + Add — removes the only Concat. Requires retraining final layer.
3. **Confirm depthwise-Conv support** with the compiler team — depthwise is just Conv with `group == channels`; many embedded NPUs handle it natively even if it's listed as "unknown."
4. **Confirm element-wise Add support** — same story; very common op, often supported as a free skip-connection primitive even if undocumented.
5. If Add cannot be supported, **retrain PFLD with `use_res_connect=False` everywhere** (`InvertedResidual` already supports this flag). Expect a few-percent landmark-NME regression but produces a fully-supported graph: only `Conv / Relu / AvgPool / Reshape / Gemm`.

### Bottom line
- **PFLD is clearly the more compatible model.** With residual-Add support (very likely) and one tiny Concat refactor, the whole 1.27 M-param backbone fits into the supported op set.
- **RetinaFace is not viable** without first adding `Resize` (nearest upsample) and `Add` support to the compiler. The dependency on FPN is structural.
- If the goal is "face landmark detection on the NPU," PFLD alone (with an external bounding box prior, e.g. from the AON path or a hand-rolled face detector) is the right path. If the goal needs detection *and* landmarks, consider replacing RetinaFace with a simpler single-scale face detector (no FPN, no upsample) — e.g. a YuNet-tiny or BlazeFace variant pruned to remove upsampling — and keep PFLD for landmarks.

---

## 5. Files referenced
- `/Users/snehamanimurugan/.insightface/models/buffalo_sc/det_500m.onnx` — RetinaFace (SCRFD-500MF) ONNX, 2.4 MB.
- `/Users/snehamanimurugan/.insightface/models/buffalo_sc/w600k_mbf.onnx` — face-recognition model (MobileFaceNet), 13 MB, not part of this analysis.
- `/tmp/pfld_repo/models/pfld.py` — PFLD PyTorch source (cloned from `polarisZhao/PFLD-pytorch`).
- `/tmp/pfld_backbone.onnx` — freshly exported PFLD backbone ONNX, 5.0 MB.
- `/tmp/pfld_aux.onnx` — freshly exported PFLD aux net ONNX, 1.8 MB.
