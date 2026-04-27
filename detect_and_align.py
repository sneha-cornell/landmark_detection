"""
Standalone face detection + landmark + alignment pipeline.

Takes raw photos → detects faces with RetinaFace → aligns to 112×112 crops
ready for the ArcFace recognition pipeline.

Usage:
    # Single image — save all detected faces
    python detect_and_align.py --input photo.jpg --output_dir ./crops

    # Directory of images
    python detect_and_align.py --input ./photos/ --output_dir ./crops

    # Visualise detections (draws boxes + landmarks, doesn't save crops)
    python detect_and_align.py --input photo.jpg --visualise
"""

import argparse
import os
import cv2
import numpy as np


# ── Alignment constants ───────────────────────────────────────────────────────

# Standard 5-point target positions in 112×112 space (InsightFace convention)
# Order: left eye, right eye, nose tip, left mouth, right mouth
LANDMARK_TARGETS = np.array([
    [38.29459953, 51.69630051],
    [73.53179932, 51.50139999],
    [56.02519989, 71.73660278],
    [41.54930115, 92.3655014 ],
    [70.72990036, 92.20410156],
], dtype=np.float32)

OUTPUT_SIZE = (112, 112)


# ── Alignment ─────────────────────────────────────────────────────────────────

def align_face(image_bgr: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    """
    Align a detected face to the standard 112×112 crop.

    landmarks: (5, 2) float array — (x, y) for each of the 5 keypoints
                in the same coordinate space as image_bgr.

    Steps:
      1. Estimate the similarity transform (rotation + scale + translation)
         that maps detected landmarks → fixed target positions.
      2. Apply it with warpAffine — resamples the image onto 112×112 grid.
    """
    transform, _ = cv2.estimateAffinePartial2D(
        landmarks.astype(np.float32),
        LANDMARK_TARGETS,
        method=cv2.RANSAC,
    )
    if transform is None:
        # Fallback: just resize the face bounding box crop
        return cv2.resize(image_bgr, OUTPUT_SIZE)

    aligned = cv2.warpAffine(
        image_bgr, transform, OUTPUT_SIZE,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )
    return aligned


# ── Detection ─────────────────────────────────────────────────────────────────

def load_detector():
    """Load RetinaFace (MobileNet backbone) via insightface."""
    import insightface
    from insightface.app import FaceAnalysis

    # det_name='retinaface_mnet025_v2' is the lightweight MobileNet variant
    # det_size controls input resolution — larger = more accurate on small faces
    app = FaceAnalysis(
        name="buffalo_sc",          # small model pack: detector + no recogniser
        allowed_modules=["detection"],
        providers=["CPUExecutionProvider"],
    )
    app.prepare(ctx_id=0, det_size=(640, 640))
    print("RetinaFace loaded.")
    return app


def detect_and_align(detector, image_bgr: np.ndarray) -> list[dict]:
    """
    Run RetinaFace on one image.

    Returns list of dicts, one per detected face:
        {
          "crop":      np.ndarray (112, 112, 3) BGR uint8 — aligned crop
          "bbox":      [x1, y1, x2, y2] in original image coords
          "landmarks": (5, 2) float array in original image coords
          "score":     float detection confidence
        }
    """
    faces = detector.get(image_bgr)
    results = []
    for face in faces:
        kps = face.kps           # (5, 2) landmarks
        bbox = face.bbox         # [x1, y1, x2, y2]
        score = float(face.det_score)

        crop = align_face(image_bgr, kps)
        results.append({
            "crop":      crop,
            "bbox":      bbox.tolist(),
            "landmarks": kps.tolist(),
            "score":     score,
        })
    return results


# ── Visualisation ─────────────────────────────────────────────────────────────

LANDMARK_COLOURS = [
    (0, 255, 0),    # left eye  — green
    (0, 255, 0),    # right eye — green
    (0, 128, 255),  # nose      — orange
    (255, 0, 0),    # left mouth — blue
    (255, 0, 0),    # right mouth — blue
]

def visualise(image_bgr: np.ndarray, detections: list[dict]) -> np.ndarray:
    vis = image_bgr.copy()
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(vis, f"{det['score']:.2f}", (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        for (x, y), colour in zip(det["landmarks"], LANDMARK_COLOURS):
            cv2.circle(vis, (int(x), int(y)), 3, colour, -1)
    return vis


# ── I/O helpers ───────────────────────────────────────────────────────────────

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def collect_images(path: str) -> list[str]:
    if os.path.isfile(path):
        return [path]
    return [
        os.path.join(root, f)
        for root, _, files in os.walk(path)
        for f in files
        if os.path.splitext(f)[1].lower() in IMAGE_EXTS
    ]


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    detector = load_detector()
    image_paths = collect_images(args.input)
    print(f"Processing {len(image_paths)} image(s)...")

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    total_faces = 0
    for img_path in image_paths:
        image = cv2.imread(img_path)
        if image is None:
            print(f"  [skip] could not read {img_path}")
            continue

        detections = detect_and_align(detector, image)
        stem = os.path.splitext(os.path.basename(img_path))[0]

        if not detections:
            print(f"  {stem}: no faces detected")
            continue

        print(f"  {stem}: {len(detections)} face(s)")

        if args.visualise:
            vis = visualise(image, detections)
            vis_path = os.path.join(
                args.output_dir or ".", f"{stem}_vis.jpg"
            )
            cv2.imwrite(vis_path, vis)
            print(f"    saved visualisation: {vis_path}")

        if args.output_dir:
            for i, det in enumerate(detections):
                out_name = f"{stem}_face{i:02d}_score{det['score']:.2f}.jpg"
                out_path = os.path.join(args.output_dir, out_name)
                cv2.imwrite(out_path, det["crop"])

        total_faces += len(detections)

    print(f"\nDone. {total_faces} face crop(s) saved to {args.output_dir or '(none)'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Face detection + alignment pipeline")
    parser.add_argument("--input",      required=True, help="Image file or directory")
    parser.add_argument("--output_dir", default="./aligned_crops", help="Where to save 112×112 crops")
    parser.add_argument("--visualise",  action="store_true", help="Also save annotated detection images")
    args = parser.parse_args()
    main(args)
