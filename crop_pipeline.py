from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

from PIL import Image


# Explicit angle IDs are supplied by the JSON manifest, not inferred from image content.
ANGLE_VIEWS = {
    "0": ("isometric", 0),
    "1": ("top", 1),
    "2": ("face", 2),
}
VIEW_LABELS = {
    "isometric": "Isometric",
    "top": "Top",
    "face": "Face",
}
# Width and height are percentages of the full EXIF-normalized working image.
DETECTION_REGIONS = {
    ("top", "landscape"): (50, 68),
    ("top", "portrait"): (63, 62),
    ("isometric", "landscape"): (55, 80),
    ("isometric", "portrait"): (65, 55),
    ("face", "landscape"): (55, 80),
    ("face", "portrait"): (60, 45),
}
# Per-view square-crop target coverage. Top framing has been visually accepted;
# Isometric remains provisional pending representative crop review.
CROP_TARGET_COVERAGE = {
    "top": 0.70,
    "isometric": 0.80,
}
OUTPUT_SIZE = 1200


@dataclass(frozen=True)
class BatchItem:
    file_path: str
    angle_id: str
    view: str
    order: int


def resolve_angle_index(angle_index) -> tuple[str, str, int]:
    """Resolve the explicit JSON angle_index into (ID, view, processing order)."""
    if isinstance(angle_index, bool):
        raise ValueError("angle_index must be 0, 1, or 2.")
    if isinstance(angle_index, int):
        angle_id = str(angle_index)
    elif isinstance(angle_index, str) and angle_index.strip().isdigit():
        angle_id = str(int(angle_index.strip()))
    else:
        raise ValueError(f"Invalid angle_index: {angle_index!r}. Expected 0, 1, or 2.")
    if angle_id not in ANGLE_VIEWS:
        raise ValueError(f"Unsupported angle_index {angle_id}. Expected 0, 1, or 2.")
    view, order = ANGLE_VIEWS[angle_id]
    return angle_id, view, order


def normalize_manifest_records(records: Iterable[Mapping]) -> list[BatchItem]:
    """Validate explicit {file_path, angle_index} records and group stably by view."""
    items = []
    seen_paths = set()
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError(f"Manifest record {index + 1} must be an object.")
        raw_path = record.get("file_path")
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise ValueError(f"Manifest record {index + 1} is missing file_path.")
        normalized_path = Path(raw_path.replace("\\", "/")).name
        if not normalized_path:
            raise ValueError(f"Manifest record {index + 1} has an invalid file_path.")
        key = normalized_path.casefold()
        if key in seen_paths:
            raise ValueError(f"Manifest contains duplicate photo filename: {normalized_path}")
        seen_paths.add(key)
        if "angle_index" not in record:
            raise ValueError(f"Manifest record {index + 1} is missing angle_index.")
        angle_id, view, order = resolve_angle_index(record["angle_index"])
        items.append(BatchItem(raw_path, angle_id, view, order))
    return sorted(items, key=lambda item: item.order)


def load_manifest_records(path: str | Path) -> list[BatchItem]:
    """Load JSON records containing file_path and angle_index."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("The StandAlone manifest must be a JSON array of photo records.")
    return normalize_manifest_records(payload)


def orientation_name(image: Image.Image) -> str:
    return "landscape" if image.width >= image.height else "portrait"


def detection_region(view: str, image: Image.Image) -> tuple[int, int]:
    try:
        return DETECTION_REGIONS[(view, orientation_name(image))]
    except KeyError as exc:
        raise ValueError(f"No detection-region configuration for view: {view}") from exc


def centered_detection_image(image: Image.Image, width_percent: int, height_percent: int):
    """Return the centered model input and its full-image (x, y) offset."""
    if not (1 <= width_percent <= 100 and 1 <= height_percent <= 100):
        raise ValueError("Detection-region percentages must be between 1 and 100.")
    crop_width = max(1, int(round(image.width * width_percent / 100.0)))
    crop_height = max(1, int(round(image.height * height_percent / 100.0)))
    left = (image.width - crop_width) // 2
    top = (image.height - crop_height) // 2
    return image.crop((left, top, left + crop_width, top + crop_height)), (left, top)


def map_box_to_full_image(box, offset: tuple[int, int]) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = map(float, box)
    dx, dy = offset
    return x1 + dx, y1 + dy, x2 + dx, y2 + dy


def square_crop_geometry(
    image_size: tuple[int, int],
    box: tuple[float, float, float, float],
    target_coverage: float,
) -> tuple[int, int, int, int]:
    """Build a square crop around a detected box; pad is applied by crop_square."""
    if not 0.0 < target_coverage <= 1.0:
        raise ValueError("target_coverage must be greater than 0 and at most 1.")
    image_width, image_height = image_size
    x1, y1, x2, y2 = map(float, box)
    if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
        raise ValueError("Detection box coordinates must be finite.")
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Detection box must have positive width and height.")
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    half = max(1, int(math.ceil(max(x2 - x1, y2 - y1) / (2.0 * target_coverage))))
    left, top = int(round(cx)) - half, int(round(cy)) - half
    side = 2 * half
    return left, top, left + side, top + side


def isometric_square_crop_geometry(
    image_size: tuple[int, int],
    box: tuple[float, float, float, float],
    target_coverage: float,
    target_y: float = 0.70,
) -> tuple[int, int, int, int]:
    """Zoom the Isometric target while placing its center slightly below crop center.

    The complete detected diamond-table box is kept inside the square crop.
    target_y=0.70 places the box center 70% down the output, below the image midpoint.
    """
    if not 0.0 < target_coverage <= 1.0:
        raise ValueError("target_coverage must be greater than 0 and at most 1.")
    if not 0.0 < target_y < 1.0:
        raise ValueError("target_y must be greater than 0 and less than 1.")
    x1, y1, x2, y2 = map(float, box)
    if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
        raise ValueError("Detection box coordinates must be finite.")
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Detection box must have positive width and height.")
    width, height = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    side = int(math.ceil(max(
        max(width, height) / target_coverage,
        height / (2.0 * min(target_y, 1.0 - target_y)),
    )))
    left = int(math.floor(cx - side / 2.0))
    top = int(math.floor(cy - target_y * side))
    return left, top, left + side, top + side


def crop_square(image: Image.Image, box: tuple[int, int, int, int], output_size: int = OUTPUT_SIZE):
    """Square crop with dark padding outside the source; never stretch."""
    left, top, right, bottom = box
    side = max(right - left, bottom - top)
    if side <= 0:
        raise ValueError("Crop geometry must have positive dimensions.")
    right, bottom = left + side, top + side
    canvas = Image.new("RGB", (side, side), (24, 24, 24))
    src_left, src_top = max(0, left), max(0, top)
    src_right, src_bottom = min(image.width, right), min(image.height, bottom)
    if src_right > src_left and src_bottom > src_top:
        canvas.paste(
            image.crop((src_left, src_top, src_right, src_bottom)),
            (src_left - left, src_top - top),
        )
    if output_size and canvas.size != (output_size, output_size):
        canvas = canvas.resize((output_size, output_size), Image.Resampling.LANCZOS)
    return canvas


DETECTION_CONFIDENCE_TOLERANCE = 0.03


def choose_detection(result):
    """Select by confidence, using image-center proximity within 3 points."""
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return None

    candidates = []
    for index, raw_box in enumerate(boxes.xyxy.tolist()):
        x1, y1, x2, y2 = map(float, raw_box)
        if x2 <= x1 or y2 <= y1:
            continue
        confidence = float(boxes.conf[index].item())
        candidates.append({
            "confidence": confidence,
            "box": (x1, y1, x2, y2),
        })

    if not candidates:
        return None

    highest_confidence = max(item["confidence"] for item in candidates)
    tied_candidates = [
        item for item in candidates
        if highest_confidence - item["confidence"] <= DETECTION_CONFIDENCE_TOLERANCE
    ]

    # Ultralytics Results exposes the dimensions of the detector input image.
    # If unavailable, preserve the previous confidence-only behavior.
    original_shape = getattr(result, "orig_shape", None)
    if original_shape and len(original_shape) >= 2 and len(tied_candidates) > 1:
        image_height, image_width = map(float, original_shape[:2])
        center_x, center_y = image_width / 2.0, image_height / 2.0

        def selection_key(item):
            x1, y1, x2, y2 = item["box"]
            box_center_x = (x1 + x2) / 2.0
            box_center_y = (y1 + y2) / 2.0
            distance_squared = (
                (box_center_x - center_x) ** 2
                + (box_center_y - center_y) ** 2
            )
            return distance_squared, -item["confidence"]

        selected = min(tied_candidates, key=selection_key)
    else:
        selected = max(tied_candidates, key=lambda item: item["confidence"])

    return {
        "box": selected["box"],
        "confidence": selected["confidence"],
    }


# The permissive inference pass retains overlapping candidates that the normal
# detector NMS pass would otherwise suppress before crop localization.
BOX_FUSION_CANDIDATE_IOU = 0.99
BOX_FUSION_CLUSTER_IOU = 0.60


def _box_iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    intersection_width = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    intersection_height = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = intersection_width * intersection_height
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def _confidence_weighted_box(members: list[dict]) -> tuple[float, float, float, float]:
    weights = [max(0.0, float(item["confidence"])) for item in members]
    total_weight = sum(weights)
    if total_weight <= 0:
        weights = [1.0] * len(members)
        total_weight = float(len(members))
    coordinates = [
        sum(float(item["box"][axis]) * weight for item, weight in zip(members, weights))
        / total_weight
        for axis in range(4)
    ]
    return (
        coordinates[0],
        coordinates[1],
        coordinates[2],
        coordinates[3],
    )


def fuse_detection_candidates(
    candidates: list[dict],
    iou_threshold: float = BOX_FUSION_CLUSTER_IOU,
) -> list[dict]:
    """Greedily cluster overlapping detections and confidence-weight their boxes.

    Single-member clusters are retained unchanged. This means unrelated
    detections are not averaged together and a one-candidate input needs no
    coordinate fusion.
    """
    if not 0.0 < iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be greater than 0 and at most 1.")
    ordered = sorted(candidates, key=lambda item: float(item["confidence"]), reverse=True)
    clusters = []
    for candidate in ordered:
        best_cluster = None
        best_overlap = 0.0
        for cluster in clusters:
            overlap = _box_iou(candidate["box"], cluster["box"])
            if overlap >= iou_threshold and overlap > best_overlap:
                best_cluster, best_overlap = cluster, overlap
        if best_cluster is None:
            clusters.append({"members": [candidate], "box": candidate["box"]})
        else:
            best_cluster["members"].append(candidate)
            best_cluster["box"] = _confidence_weighted_box(best_cluster["members"])

    fused = []
    for index, cluster in enumerate(clusters, start=1):
        members = cluster["members"]
        fused.append({
            "candidate": f"F{index}",
            "box": _confidence_weighted_box(members),
            "confidence": sum(float(item["confidence"]) for item in members) / len(members),
            "member_count": len(members),
            "members": tuple(item["candidate"] for item in members),
        })
    return fused


def _choose_candidate(candidates: list[dict], image_size: tuple[int, int] | None):
    """Select the candidate whose box center is nearest the detector-image center.

    Confidence is used only as a tie-breaker when candidates have equal center
    distance. This is an experimental alternative to confidence-first selection.
    """
    if not candidates:
        return None
    if image_size:
        image_width, image_height = map(float, image_size)
        center_x, center_y = image_width / 2.0, image_height / 2.0

        def selection_key(item):
            x1, y1, x2, y2 = item["box"]
            box_center_x = (float(x1) + float(x2)) / 2.0
            box_center_y = (float(y1) + float(y2)) / 2.0
            distance_squared = (
                (box_center_x - center_x) ** 2
                + (box_center_y - center_y) ** 2
            )
            return distance_squared, -float(item["confidence"])

        return min(candidates, key=selection_key)
    return max(candidates, key=lambda item: float(item["confidence"]))


def choose_fused_detection(result):
    """Select a crop target from a permissive candidate pass using WBF-style fusion.

    The model must be run with BOX_FUSION_CANDIDATE_IOU so overlapping candidates
    remain available. A single candidate is returned unchanged. Multiple
    candidates are clustered at BOX_FUSION_CLUSTER_IOU; only members of the
    same overlapping cluster contribute to a fused box. No detections returns
    None, preserving the app's existing no-crop behavior.
    """
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return None

    candidates = []
    for index, raw_box in enumerate(boxes.xyxy.tolist()):
        x1, y1, x2, y2 = map(float, raw_box)
        if x2 <= x1 or y2 <= y1:
            continue
        candidates.append({
            "candidate": f"C{len(candidates) + 1}",
            "confidence": float(boxes.conf[index].item()),
            "box": (x1, y1, x2, y2),
        })
    if not candidates:
        return None

    if len(candidates) == 1:
        selected = candidates[0]
        return {
            "box": selected["box"],
            "confidence": selected["confidence"],
            "fusion_applied": False,
            "fusion_source_count": 1,
            "fusion_members": (selected["candidate"],),
        }

    fused_candidates = fuse_detection_candidates(candidates)
    original_shape = getattr(result, "orig_shape", None)
    image_size = None
    if original_shape and len(original_shape) >= 2:
        image_height, image_width = original_shape[:2]
        image_size = (int(image_width), int(image_height))
    selected = _choose_candidate(fused_candidates, image_size)
    if selected is None:
        return None
    return {
        "box": selected["box"],
        "confidence": selected["confidence"],
        "fusion_applied": selected["member_count"] > 1,
        "fusion_source_count": selected["member_count"],
        "fusion_members": selected["members"],
    }
