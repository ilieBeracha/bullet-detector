import argparse
import os

import csv
import math
from datetime import datetime
from typing import List, Tuple, Dict

import cv2
import numpy as np

# ---------- Utilities ----------


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def get_timestamp():
    """Get current timestamp in YYYYMMDD_HHMMSS format."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def get_image_basename(image_path: str) -> str:
    """Extract image basename without extension."""
    return os.path.splitext(os.path.basename(image_path))[0]


def imread_gray(path: str):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img, gray


def save_overlay(image, circles_kept, circles_rejected, out_path):
    vis = image.copy()
    for (x, y, r, *_rest) in circles_rejected:
        cv2.circle(vis, (int(x), int(y)), int(r), (0, 0, 255), 1)
    for (x, y, r, *_rest) in circles_kept:
        cv2.circle(vis, (int(x), int(y)), int(r), (0, 255, 0), 2)
        cv2.circle(vis, (int(x), int(y)), 2, (0, 255, 0), 2)
    cv2.imwrite(out_path, vis)


def radial_profile(gray_crop: np.ndarray) -> np.ndarray:
    """Mean intensity as a function of radius from center."""
    h, w = gray_crop.shape[:2]
    cx, cy = w // 2, h // 2
    y, x = np.indices((h, w))
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(np.int32)
    rmax = r.max()
    tbin = np.bincount(r.ravel(), weights=gray_crop.ravel(), minlength=rmax + 1)
    nr = np.bincount(r.ravel(), minlength=rmax + 1)
    prof = tbin / np.maximum(nr, 1)
    return prof


def bullet_heuristics(crop_bgr: np.ndarray) -> Dict[str, float]:
    """Physics-driven checks: dark core + circularity + ring peak."""
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    cx, cy = w // 2, h // 2

    # 1) Dark core (mean in 5x5 around center)
    sz = max(2, min(h, w) // 16)
    core = gray[max(0, cy - sz) : min(h, cy + sz), max(0, cx - sz) : min(w, cx + sz)]
    core_mean = float(np.mean(core))

    # 2) Circularity (from Canny edges)
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    circ = 0.0
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        per = cv2.arcLength(cnt, True)
        area = cv2.contourArea(cnt)
        if per > 1e-6:
            circ = float(4 * math.pi * area / (per**2))

    # 3) Bright ring peak (radial intensity profile)
    prof = radial_profile(gray)
    # Normalize for stability
    p = (prof - prof.min()) / (prof.ptp() + 1e-6)
    # look for a local max between radii ~5..20% of crop radius
    rmax = max(1, len(p) - 1)
    r1 = int(0.08 * rmax)
    r2 = int(0.35 * rmax)
    ring_strength = 0.0
    if r2 > r1 + 2:
        segment = p[r1:r2]
        peak = float(segment.max())
        ring_strength = peak

    # Scoring: lower core_mean is better (darker); higher circ and ring_strength are better
    score = 0.0
    if core_mean < 90:
        score += 1.0
    if 0.45 <= circ <= 1.25:
        score += 1.0
    if ring_strength > 0.45:
        score += 1.0

    return {
        "core_mean": core_mean,
        "circularity": circ,
        "ring_strength": ring_strength,
        "heur_score": score,
    }


def crop_from_circle(
    img: np.ndarray, x: int, y: int, r: int, margin_mult: float = 2.0
) -> np.ndarray:
    m = int(max(6, margin_mult * r))
    h, w = img.shape[:2]
    x1, y1 = max(0, x - m), max(0, y - m)
    x2, y2 = min(w, x + m), min(h, y + m)
    return img[y1:y2, x1:x2]


# ---------- Stage 1: Over-detect candidates ----------


def detect_candidates(
    image_path: str,
    out_dir: str,
    dp=1.2,
    minDist=20,
    param1=100,
    param2=25,
    minRadius=4,
    maxRadius=25,
) -> List[Tuple[int, int, int]]:
    # Create output directory with image name and timestamp
    image_basename = get_image_basename(image_path)
    timestamp = get_timestamp()
    final_out_dir = os.path.join(out_dir, f"{image_basename}_{timestamp}")
    ensure_dir(final_out_dir)

    img, gray = imread_gray(image_path)
    gray_blur = cv2.medianBlur(gray, 5)

    circles = cv2.HoughCircles(
        gray_blur,
        cv2.HOUGH_GRADIENT,
        dp=dp,
        minDist=minDist,
        param1=param1,
        param2=param2,
        minRadius=minRadius,
        maxRadius=maxRadius,
    )

    results = []
    meta_path = os.path.join(final_out_dir, f"{image_basename}_candidates.csv")
    with open(meta_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["crop_path", "x", "y", "r", "image_path"])
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i, (x, y, r) in enumerate(circles[0, :]):
                crop = crop_from_circle(img, int(x), int(y), int(r))
                # New naming: imageName_counter_x_y.jpg
                crop_name = f"{image_basename}_{i:03d}_{int(x)}_{int(y)}.jpg"
                crop_path = os.path.join(final_out_dir, crop_name)
                cv2.imwrite(crop_path, crop)
                writer.writerow([crop_name, int(x), int(y), int(r), image_path])
                results.append((int(x), int(y), int(r)))
    print(f"[detect] Saved {len(results)} crops to {final_out_dir}")
    # also save an overlay so you can sanity-check recall
    overlay_path = os.path.join(
        final_out_dir, f"{image_basename}_candidates_overlay.jpg"
    )
    save_overlay(img, [(x, y, r) for (x, y, r) in results], [], overlay_path)
    return results


# ---------- Stage 2: Refine detections with heuristics ----------


def refine_image(
    image_path: str,
    overlay_path: str = None,
    csv_path: str = None,
    dp: float = 1.2,
    minDist: float = 20,
    param1: float = 100,
    param2: float = 25,
    minRadius: int = 4,
    maxRadius: int = 25,
    use_heuristics: bool = True,
) -> List[Tuple[int, int, int]]:
    """
    Detect circles and filter them using physics-based heuristics.
    Returns list of (x, y, radius) tuples for valid bullet holes.
    """
    # Generate output paths with image name and timestamp
    image_basename = get_image_basename(image_path)
    timestamp = get_timestamp()

    if overlay_path is None:
        overlay_path = f"{image_basename}_{timestamp}_refined_overlay.jpg"
    if csv_path is None:
        csv_path = f"{image_basename}_{timestamp}_refined_detections.csv"

    # First, get all candidate circles
    candidates = detect_candidates(
        image_path, "temp_candidates", dp, minDist, param1, param2, minRadius, maxRadius
    )

    if not candidates:
        print("[refine] No candidates found")
        return []

    # Load original image for cropping
    img, gray = imread_gray(image_path)

    # Filter candidates using heuristics
    valid_circles = []
    rejected_circles = []

    for x, y, r in candidates:
        # Crop around the circle
        crop = crop_from_circle(img, x, y, r)

        # Apply heuristics if enabled
        if use_heuristics:
            heuristics = bullet_heuristics(crop)
            # Accept if heuristics score is good enough
            if heuristics["heur_score"] >= 2.0:  # At least 2 out of 3 criteria
                valid_circles.append((x, y, r))
            else:
                rejected_circles.append((x, y, r))
        else:
            # If no heuristics, accept all candidates
            valid_circles.append((x, y, r))

    # Save overlay visualization
    save_overlay(img, valid_circles, rejected_circles, overlay_path)

    # Save CSV with results
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y", "radius", "image_path"])
        for x, y, r in valid_circles:
            writer.writerow([x, y, r, image_path])

    # Clean up temp directory
    import shutil

    if os.path.exists("temp_candidates"):
        shutil.rmtree("temp_candidates")

    print(
        f"[refine] Found {len(valid_circles)} valid bullet holes out of {len(candidates)} candidates"
    )
    print(f"[refine] Results saved to {overlay_path} and {csv_path}")
    return valid_circles


# ---------- CLI ----------


def main():
    ap = argparse.ArgumentParser(
        description="Bullet-hole detection pipeline using classical computer vision."
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    # detect
    ap_det = sub.add_parser(
        "detect", help="Over-detect circle candidates and save crops."
    )
    ap_det.add_argument("--image", required=True)
    ap_det.add_argument("--out", default="candidates")
    ap_det.add_argument("--dp", type=float, default=1.2)
    ap_det.add_argument("--minDist", type=float, default=20)
    ap_det.add_argument("--param1", type=float, default=100)
    ap_det.add_argument("--param2", type=float, default=25)
    ap_det.add_argument("--minRadius", type=int, default=4)
    ap_det.add_argument("--maxRadius", type=int, default=25)

    # refine
    ap_ref = sub.add_parser(
        "refine", help="Filter detections with physics heuristics, save overlay + CSV."
    )
    ap_ref.add_argument("--image", required=True)
    ap_ref.add_argument("--overlay", default="refined_overlay.jpg")
    ap_ref.add_argument("--csv", default="refined_detections.csv")
    ap_ref.add_argument("--dp", type=float, default=1.2)
    ap_ref.add_argument("--minDist", type=float, default=20)
    ap_ref.add_argument("--param1", type=float, default=100)
    ap_ref.add_argument("--param2", type=float, default=25)
    ap_ref.add_argument("--minRadius", type=int, default=4)
    ap_ref.add_argument("--maxRadius", type=int, default=25)
    ap_ref.add_argument(
        "--no-heur", action="store_true", help="Disable physics heuristics."
    )

    args = ap.parse_args()

    if args.cmd == "detect":
        detect_candidates(
            args.image,
            args.out,
            args.dp,
            args.minDist,
            args.param1,
            args.param2,
            args.minRadius,
            args.maxRadius,
        )

    elif args.cmd == "refine":
        refine_image(
            args.image,
            args.overlay,
            args.csv,
            dp=args.dp,
            minDist=args.minDist,
            param1=args.param1,
            param2=args.param2,
            minRadius=args.minRadius,
            maxRadius=args.maxRadius,
            use_heuristics=not args.no_heur,
        )


if __name__ == "__main__":
    main()
