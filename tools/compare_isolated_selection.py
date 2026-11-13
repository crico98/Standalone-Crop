from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

from crop_pipeline import (
    BOX_FUSION_CANDIDATE_IOU,
    CROP_TARGET_COVERAGE,
    DETECTION_CONFIDENCE_TOLERANCE,
    OUTPUT_SIZE,
    centered_detection_image,
    crop_square,
    detection_region,
    fuse_detection_candidates,
    isometric_square_crop_geometry,
    map_box_to_full_image,
    square_crop_geometry,
)
from image_orientation import normalize_uploaded_image


DEFAULT_MODELS = {
    "top": PROJECT_ROOT / "models" / "top_view_field_v1" / "weights" / "best.pt",
    "isometric": PROJECT_ROOT / "models" / "isometric_view" / "weights" / "best.pt",
}


def choose_confidence_first(candidates, image_size):
    """Reproduce the established confidence-band selection on fused candidates."""
    if not candidates:
        return None
    highest = max(float(item["confidence"]) for item in candidates)
    tied = [
        item for item in candidates
        if highest - float(item["confidence"]) <= DETECTION_CONFIDENCE_TOLERANCE
    ]
    if len(tied) == 1:
        return tied[0]

    width, height = image_size
    center_x, center_y = width / 2.0, height / 2.0

    def key(item):
        x1, y1, x2, y2 = item["box"]
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        return (cx - center_x) ** 2 + (cy - center_y) ** 2, -float(item["confidence"])

    return min(tied, key=key)


def choose_center_nearest(candidates):
    """Experimental selection: nearest fused-box center, confidence only breaks ties."""
    if not candidates:
        return None

    def key(item):
        x1, y1, x2, y2 = item["box"]
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        return (cx, cy)

    # Distance is measured from the center of the detector input image.
    # The caller attaches detector_center_distance for deterministic ordering.
    return min(candidates, key=lambda item: (
        item["detector_center_distance"], -float(item["confidence"])
    ))


def load_font(size):
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def save_overlay(image, offset, region_size, candidates, selected, method, path):
    canvas = image.copy().convert("RGB")
    draw = ImageDraw.Draw(canvas)
    font = load_font(max(24, image.width // 170))
    small = load_font(max(20, image.width // 210))
    left, top = offset
    rw, rh = region_size
    line_width = max(8, image.width // 550)

    draw.rectangle(
        (left, top, left + rw, top + rh),
        outline=(255, 190, 0), width=line_width,
    )
    draw.text(
        (left + 16, top + 16),
        f"{method} | selected {selected['candidate'] if selected else 'NONE'}",
        fill=(255, 255, 255), font=font, stroke_width=2, stroke_fill=(0, 0, 0),
    )

    for item in candidates:
        x1, y1, x2, y2 = item["box"]
        coords = (x1 + left, y1 + top, x2 + left, y2 + top)
        active = selected is item
        color = (30, 220, 100) if active else (235, 75, 75)
        draw.rectangle(coords, outline=color, width=line_width)
        label = (
            f"{item['candidate']} conf={item['confidence']:.3f} "
            f"n={item['member_count']} "
            f"center={item['detector_center_distance']:.1f}px"
        )
        label_y = max(0, int(coords[1] - small.size - 8))
        label_width = draw.textlength(label, font=small) + 12
        draw.rectangle(
            (coords[0], label_y, coords[0] + label_width, label_y + small.size + 8),
            fill=(20, 20, 20),
        )
        draw.text((coords[0] + 6, label_y + 3), label, fill=color, font=small)

    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, format="PNG")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run one original image through one isolated view model and compare "
            "confidence-first versus center-nearest selection after identical WBF."
        )
    )
    parser.add_argument("image", type=Path, help="Path to one original image; no manifest required.")
    parser.add_argument("--view", choices=("top", "isometric"), default="top")
    parser.add_argument("--model", type=Path, default=None, help="Optional model checkpoint override.")
    parser.add_argument(
        "--output-dir", type=Path,
        default=PROJECT_ROOT / "work" / "diagnostics" / "isolated_selection",
    )
    args = parser.parse_args()

    image_path = args.image.expanduser().resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"Input image not found: {image_path}")

    model_path = (args.model or DEFAULT_MODELS[args.view]).expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"{args.view.title()} model checkpoint not found: {model_path}")

    with Image.open(image_path) as source:
        image = normalize_uploaded_image(source)

    region_width, region_height = detection_region(args.view, image)
    model_image, offset = centered_detection_image(image, region_width, region_height)

    print(f"Image: {image_path}")
    print(f"View: {args.view}")
    print(f"Model: {model_path}")
    print(f"Normalized image: {image.width}x{image.height}")
    print(f"Detector region: {region_width}% x {region_height}% ({model_image.width}x{model_image.height})")
    print(f"Inference IoU: {BOX_FUSION_CANDIDATE_IOU}")

    detector = YOLO(str(model_path))
    result = detector.predict(
        source=model_image,
        verbose=False,
        iou=BOX_FUSION_CANDIDATE_IOU,
    )[0]

    boxes = getattr(result, "boxes", None)
    raw_candidates = []
    if boxes is not None:
        for index, raw_box in enumerate(boxes.xyxy.tolist()):
            x1, y1, x2, y2 = map(float, raw_box)
            if x2 <= x1 or y2 <= y1:
                continue
            raw_candidates.append({
                "candidate": f"C{len(raw_candidates) + 1}",
                "confidence": float(boxes.conf[index].item()),
                "box": (x1, y1, x2, y2),
            })

    fused = fuse_detection_candidates(raw_candidates)
    detector_cx, detector_cy = model_image.width / 2.0, model_image.height / 2.0
    for item in fused:
        x1, y1, x2, y2 = item["box"]
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        item["detector_center_distance"] = (
            (cx - detector_cx) ** 2 + (cy - detector_cy) ** 2
        ) ** 0.5

    selected = {
        "confidence_first": choose_confidence_first(fused, model_image.size),
        "center_nearest": choose_center_nearest(fused),
    }

    if not fused:
        print("NO DETECTIONS: no crop was generated.")
    else:
        for method, candidate in selected.items():
            print(
                f"{method}: {candidate['candidate']} | "
                f"confidence={candidate['confidence']:.4f} | "
                f"members={','.join(candidate['members'])} | "
                f"box={tuple(round(v, 1) for v in candidate['box'])}"
            )

        coverage = CROP_TARGET_COVERAGE[args.view]
        for method, candidate in selected.items():
            full_box = map_box_to_full_image(candidate["box"], offset)
            if args.view == "isometric":
                crop_box = isometric_square_crop_geometry(image.size, full_box, coverage)
            else:
                crop_box = square_crop_geometry(image.size, full_box, coverage)
            crop = crop_square(image, crop_box, OUTPUT_SIZE)
            crop_path = args.output_dir / method / f"{image_path.stem}_{method}_crop.png"
            crop_path.parent.mkdir(parents=True, exist_ok=True)
            crop.save(crop_path, format="PNG")
            overlay_path = args.output_dir / method / f"{image_path.stem}_{method}_overlay.png"
            save_overlay(
                image, offset, model_image.size, fused, candidate, method,
                overlay_path,
            )
            print(f"{method} crop: {crop_path}")
            print(f"{method} overlay: {overlay_path}")

    metrics_path = args.output_dir / f"{image_path.stem}_candidate_metrics.csv"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "candidate", "confidence", "member_count", "members",
        "x1", "y1", "x2", "y2", "center_distance_px",
        "selected_confidence_first", "selected_center_nearest",
    ]
    with metrics_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in fused:
            writer.writerow({
                "candidate": item["candidate"],
                "confidence": f"{item['confidence']:.6f}",
                "member_count": item["member_count"],
                "members": ",".join(item["members"]),
                "x1": f"{item['box'][0]:.2f}",
                "y1": f"{item['box'][1]:.2f}",
                "x2": f"{item['box'][2]:.2f}",
                "y2": f"{item['box'][3]:.2f}",
                "center_distance_px": f"{item['detector_center_distance']:.3f}",
                "selected_confidence_first": item is selected["confidence_first"],
                "selected_center_nearest": item is selected["center_nearest"],
            })

    print(f"Raw detections: {len(raw_candidates)}")
    print(f"Fused candidates: {len(fused)}")
    print(f"Metrics: {metrics_path}")


if __name__ == "__main__":
    main()
