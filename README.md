# Landmark Detection & Face Alignment Pipeline

Standalone pipeline that takes raw photos, detects faces using RetinaFace, localises 5 facial landmarks, and produces geometrically aligned 112×112 crops ready for a face recognition model.

---

## Pipeline

```
Raw photo
    ↓
RetinaFace (MobileNet) — face bounding box + 5 landmarks
    ↓
Similarity transform (cv2.warpAffine) — rotate + scale + translate
    ↓
112×112 aligned crop — ready for ArcFace recognition
```

---

## The 5 Landmarks

| Point | Colour in visualisation |
|---|---|
| Left eye | Green |
| Right eye | Green |
| Nose tip | Orange |
| Left mouth corner | Blue |
| Right mouth corner | Blue |

---

## Results

### Detection + Landmark Localisation

![Detection](example_detection.jpg)

RetinaFace detects the face bounding box (yellow) and places all 5 landmarks in the correct positions. Confidence score shown top-left of the bounding box.

### Aligned 112×112 Crop

![Aligned crop](example_crop.jpg)

Output of the similarity transform — face centred, upright, eyes at fixed pixel positions. This is the format expected by the ArcFace recognition backbone.

### Alignment: before vs after

| | Before | After |
|---|---|---|
| Resolution | Any | 112 × 112 px |
| Face position | Anywhere in frame | Centred |
| Rotation | Any | Upright |
| Scale | Any | Fixed — eyes always ~35px apart |
| Input to recognition | ❌ | ✅ |

### Target landmark positions (112×112 space)

| Point | x | y |
|---|---|---|
| Left eye | 38.3 | 51.7 |
| Right eye | 73.5 | 51.5 |
| Nose tip | 56.0 | 71.7 |
| Left mouth | 41.5 | 92.4 |
| Right mouth | 70.7 | 92.2 |

These fixed targets are the InsightFace standard — the same coordinates used when creating CASIA-WebFace, so alignment is consistent end-to-end with the recognition training data.

---

## Model

**RetinaFace** with MobileNet backbone (`buffalo_sc` from InsightFace).

| Property | Value |
|---|---|
| Model | RetinaFace MobileNet |
| Input size | 640 × 640 (detector) |
| Output | Bounding box + 5 landmarks per face |
| Model size | ~1.6 MB |
| Faces per image | Multiple supported |
| Framework | ONNX via insightface |

RetinaFace was chosen because it performs face detection and landmark localisation in a single forward pass — no separate detector + landmark model needed. It is also the same model used to produce the alignment in CASIA-WebFace, ensuring consistency with the downstream recognition pipeline.

---

## Geometric Alignment

The alignment is a **similarity transform** — rotation, scale, and translation only. No warping or perspective distortion.

Given detected landmarks `src` and fixed target positions `dst`, OpenCV solves for the 4-parameter transform (angle, scale, tx, ty) using least squares:

```python
transform, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC)
aligned = cv2.warpAffine(image, transform, (112, 112))
```

This ensures every face crop has:
- Eyes at the same height and horizontal distance
- Face at a consistent scale
- No tilt

---

## Usage

```bash
pip install insightface onnxruntime opencv-python-headless
```

**Single image:**
```bash
python detect_and_align.py --input photo.jpg --output_dir ./crops
```

**Directory of images:**
```bash
python detect_and_align.py --input ./photos/ --output_dir ./crops
```

**Visualise detections (bounding box + landmarks):**
```bash
python detect_and_align.py --input photo.jpg --output_dir ./crops --visualise
```

Output crops are named: `<original_stem>_face<N>_score<confidence>.jpg`

---

## Output Format

Each detected face is saved as a 112×112 RGB JPEG — the exact format expected by the [face recognition pipeline](https://github.com/sneha-cornell/face_recognition).

```python
# Feed directly into ArcFace backbone
import tensorflow as tf
img = tf.image.decode_jpeg(tf.io.read_file("crop.jpg"), channels=3)
img = (tf.cast(img, tf.float32) - 127.5) / 128.0   # normalise to [-1, 1]
embedding = backbone(img[None], training=False)      # (1, 128)
```

---

## Training Dataset: WIDER FACE

RetinaFace was trained on **WIDER FACE**, a large-scale face detection benchmark scraped from the web across 61 event categories (parade, protest, concert, sports, etc.).

### Scale

| Stat | Value |
|---|---|
| Total images | 32,203 |
| Total annotated faces | 393,703 |
| Training split | 12,880 images |
| Validation split | 3,226 images |
| Test split | 16,097 images |

### Annotations

Each face in the training set has:
- **Bounding box** — (x, y, width, height) in pixel coordinates
- **5-point landmarks** — added by the RetinaFace authors on top of WIDER FACE for the ~13,000 training images (left eye, right eye, nose tip, left mouth corner, right mouth corner)
- **Difficulty flag** — Easy / Medium / Hard based on scale, occlusion, and pose

The landmark annotations are not part of the original WIDER FACE release — they were added specifically for RetinaFace training and are distributed separately by the InsightFace team.

### Difficulty split

WIDER FACE deliberately includes hard conditions to build a robust detector:

| Difficulty | Criteria | % of faces |
|---|---|---|
| Easy | Large faces, frontal, minimal occlusion | ~22% |
| Medium | Moderate scale, some occlusion or pose | ~33% |
| Hard | Small faces (<10px), heavy occlusion, extreme pose | ~45% |

45% of faces fall in the Hard category — this is why RetinaFace generalises well to real-world conditions like the partially occluded, slightly off-angle face in the example above.

### Image conditions

| Condition | Range in dataset |
|---|---|
| Faces per image | 1 – 160+ |
| Face scale | 10px to full-image |
| Pose | Frontal, profile, upside-down |
| Occlusion | None to heavy (sunglasses, masks, hands) |
| Lighting | Indoor, outdoor, night, flash |
| Image source | Web photos across 61 event categories |

### Why this matters for alignment quality

The wide distribution of poses and scales in WIDER FACE means RetinaFace is robust to the kinds of faces you encounter in real deployments — not just clean frontal studio shots. The landmark predictions remain accurate even at moderate pose angles, which directly affects the quality of the similarity transform and the resulting 112×112 crop fed to the recognition model.

---

## References

- Deng et al., [RetinaFace: Single-Shot Multi-Level Face Localisation in the Wild](https://arxiv.org/abs/1905.00641), CVPR 2020
- InsightFace: [github.com/deepinsight/insightface](https://github.com/deepinsight/insightface)
- Related: [face_recognition](https://github.com/sneha-cornell/face_recognition) — ArcFace backbone trained on CASIA-WebFace
