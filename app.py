from __future__ import annotations

import io
import json
import os
import shutil
import uuid
import zipfile
from pathlib import Path

from flask import Flask, abort, redirect, render_template, request, send_file, url_for
from PIL import Image
from werkzeug.utils import secure_filename

from crop_pipeline import (
    OUTPUT_SIZE,
    CROP_TARGET_COVERAGE,
    VIEW_LABELS,
    BOX_FUSION_CANDIDATE_IOU,
    choose_fused_detection,
    centered_detection_image,
    crop_square,
    detection_region,
    map_box_to_full_image,
    isometric_square_crop_geometry,
    normalize_manifest_records,
    square_crop_geometry,
)
from image_orientation import normalize_uploaded_image, read_exif_orientation
from face_model import FaceFinder
from top_isometric_web import top_isometric
from view_uploads import VIEW_UPLOAD_FIELDS, unique_view_filename

ROOT = Path(__file__).resolve().parent
WORK = ROOT / "work"
WORK.mkdir(exist_ok=True)
ALLOWED = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

MAX_UPLOAD_MB = max(1, int(os.environ.get("CROPPER_MAX_UPLOAD_MB", "50000")))
MAX_FILES_PER_BATCH = max(1, int(os.environ.get("CROPPER_MAX_FILES", "500")))

MODEL_PATHS = {
    "top": ROOT / "models" / "top_view_field_v1" / "weights" / "best.pt",
    "isometric": ROOT / "models" / "isometric_view" / "weights" / "best.pt",
}
EXPECTED_CLASS_NAMES = {"top": "full_cutter", "isometric": "diamond_table"}
# The Top field checkpoint only returns usable detections at 640 on the
# validated diagnostic image; keep Isometric on its checkpoint default pending validation.
OBJECT_INFERENCE_IMAGE_SIZES = {"top": 640}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024
finder = FaceFinder()
object_detectors = {}
app.register_blueprint(top_isometric)

# Local prototype state only; restart clears active batch metadata.
batches: dict[str, dict] = {}
jobs: dict[str, dict] = {}


def batch_dir(batch_id: str) -> Path:
    return WORK / batch_id


def item_dir(batch_id: str, job_id: str) -> Path:
    return batch_dir(batch_id) / job_id


def load_rgb(path: Path) -> Image.Image:
    """Load pixels normalized exactly once during upload."""
    with Image.open(path) as source:
        return source.convert("RGB")


def save_rgb(image: Image.Image, path: Path) -> None:
    image.save(path, format="PNG")


def load_object_detector(view: str):
    """Load only the detector associated with the resolved Top/Isometric view."""
    if view not in MODEL_PATHS:
        raise ValueError(f"No object detector is configured for view: {view}")
    if view not in object_detectors:
        model_path = MODEL_PATHS[view]
        if not model_path.is_file():
            raise FileNotFoundError(
                f"{view.title()} detector checkpoint not found: {model_path}"
            )
        from ultralytics import YOLO
        detector = YOLO(str(model_path))
        if getattr(detector, "task", "detect") != "detect":
            raise ValueError(f"{view.title()} checkpoint must be a YOLO object-detection model.")
        names = getattr(detector, "names", {})
        class_zero = names.get(0) if isinstance(names, dict) else (names[0] if names else None)
        if class_zero != EXPECTED_CLASS_NAMES[view]:
            raise ValueError(
                f"{view.title()} checkpoint class 0 must be "
                f"{EXPECTED_CLASS_NAMES[view]!r}; found {class_zero!r}."
            )
        object_detectors[view] = detector
    return object_detectors[view]


def padded_square_crop(image: Image.Image, face: dict, max_size: int = 1200):
    """Existing Face V2 crop geometry; intentionally unchanged."""
    cx, cy, radius = float(face["cx"]), float(face["cy"]), float(face["r"])
    half = max(1, int(round(radius / 0.875)))
    left, top = int(round(cx)) - half, int(round(cy)) - half
    right, bottom = int(round(cx)) + half + 1, int(round(cy)) + half + 1
    side = max(2, right - left, bottom - top)
    right, bottom = left + side, top + side

    canvas = Image.new("RGB", (side, side), (24, 24, 24))
    src_left, src_top = max(0, left), max(0, top)
    src_right, src_bottom = min(image.width, right), min(image.height, bottom)
    if src_right > src_left and src_bottom > src_top:
        canvas.paste(
            image.crop((src_left, src_top, src_right, src_bottom)),
            (src_left - left, src_top - top),
        )
    if side > max_size:
        canvas = canvas.resize((max_size, max_size), Image.Resampling.LANCZOS)
    return canvas, (left, top, right, bottom)


def adjusted_detection_region(width: int, height: int, manually_rotated: bool) -> tuple[int, int]:
    """Existing Face-only interactive region expansion; preserved for regression."""
    if manually_rotated:
        return width, min(100, int(round(height * 1.20)))
    return (
        min(100, int(round(width * 1.05))),
        min(100, int(round(height * 1.05))),
    )


def require_batch(batch_id: str) -> dict:
    batch = batches.get(batch_id)
    if batch is None:
        abort(404)
    return batch


@app.errorhandler(413)
def too_large(_error):
    return render_template(
        "index.html",
        model_available=finder.available,
        top_model_available=MODEL_PATHS['top'].is_file(),
        isometric_model_available=MODEL_PATHS['isometric'].is_file(),
        max_upload_mb=MAX_UPLOAD_MB,
        max_files=MAX_FILES_PER_BATCH,
        error=f"The combined upload exceeds the current {MAX_UPLOAD_MB} MB limit.",
    ), 413


@app.get("/")
def index():
    return render_template(
        "index.html",
        model_available=finder.available,
        top_model_available=MODEL_PATHS["top"].is_file(),
        isometric_model_available=MODEL_PATHS["isometric"].is_file(),
        max_upload_mb=MAX_UPLOAD_MB,
        max_files=MAX_FILES_PER_BATCH,
        error=None,
    )


def _collect_view_uploads():
    """Build the explicit manifest from the three view-specific upload areas."""
    files = []
    manifest_payload = []
    used_names = set()
    for view, field_name, angle_index in VIEW_UPLOAD_FIELDS:
        for uploaded_file in request.files.getlist(field_name):
            if not uploaded_file or not uploaded_file.filename:
                continue

            original_name = secure_filename(uploaded_file.filename)
            if not original_name:
                abort(400, "An uploaded image has an invalid filename.")

            output_name = unique_view_filename(view, original_name, used_names)
            uploaded_file.filename = output_name
            files.append(uploaded_file)
            manifest_payload.append({
                "file_path": output_name,
                "angle_index": angle_index,
            })

    return files, manifest_payload


@app.post("/upload")
def upload():
    files, manifest_payload = _collect_view_uploads()

    # The prototype UI assigns views by the dropzone used. Keep the existing
    # JSON-manifest upload path available for older clients and batch scripts.
    if not manifest_payload:
        files = [f for f in request.files.getlist("images") if f and f.filename]
        if not files:
            abort(400, "Add images to the Isometric, Top, or Face upload area.")
        manifest_file = request.files.get("manifest")
        if manifest_file is None or not manifest_file.filename:
            abort(400, "Upload images into the view-specific areas.")
        try:
            manifest_payload = json.loads(manifest_file.read().decode("utf-8-sig"))
        except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
            abort(400, f"Invalid JSON manifest: {exc}")

    if not files:
        abort(400, "Choose at least one image.")
    if len(files) > MAX_FILES_PER_BATCH:
        abort(400, f"Too many images. Maximum per batch: {MAX_FILES_PER_BATCH}.")

    try:
        manifest_items = normalize_manifest_records(manifest_payload)
    except (TypeError, ValueError) as exc:
        abort(400, f"Invalid JSON manifest: {exc}")

    # Match by photo basename. The manifest's angle_index is authoritative;
    # filenames, EXIF metadata, and upload position never determine the view.
    uploaded_by_name = {}
    for file in files:
        original_name = secure_filename(file.filename) or "cutter_image"
        key = original_name.casefold()
        if key in uploaded_by_name:
            abort(400, f"Duplicate uploaded photo filename: {original_name}")
        suffix = Path(original_name).suffix.lower()
        if suffix not in ALLOWED:
            abort(400, f"Unsupported image type: {original_name}")
        uploaded_by_name[key] = (file, original_name)

    prepared = []
    matched_names = set()
    for manifest_item in manifest_items:
        photo_name = Path(manifest_item.file_path.replace("\\", "/")).name
        key = secure_filename(photo_name).casefold()
        if key not in uploaded_by_name:
            abort(400, f"Manifest photo was not included in this upload: {photo_name}")
        if key in matched_names:
            abort(400, f"Manifest maps the same uploaded photo more than once: {photo_name}")
        matched_names.add(key)
        file, original_name = uploaded_by_name[key]
        prepared.append((
            manifest_item.order, len(prepared), file, original_name,
            manifest_item.angle_id, manifest_item.view,
        ))

    unmatched = sorted(set(uploaded_by_name) - matched_names)
    if unmatched:
        abort(400, "Uploaded photos missing from JSON manifest: " + ", ".join(unmatched))

    batch_id = uuid.uuid4().hex
    folder = batch_dir(batch_id)
    folder.mkdir(parents=True, exist_ok=True)
    item_ids = []
    used_names: set[str] = set()

    try:
        for _order, _index, file, original_name, angle_id, view in prepared:
            stem = Path(original_name).stem or "cutter_image"
            output_stem = stem
            counter = 2
            while output_stem.lower() in used_names:
                output_stem = f"{stem}_{counter}"
                counter += 1
            used_names.add(output_stem.lower())

            job_id = uuid.uuid4().hex
            one_dir = item_dir(batch_id, job_id)
            one_dir.mkdir()
            try:
                with Image.open(file.stream) as source:
                    source_exif_orientation = read_exif_orientation(source)
                    normalized = normalize_uploaded_image(source)
                    save_rgb(normalized, one_dir / "working.png")
            except Exception as exc:
                raise ValueError(f"Could not read {original_name} as an image.") from exc

            jobs[job_id] = {
                "id": job_id,
                "filename": original_name,
                "output_name": output_stem,
                "source_path": original_name,
                "angle_id": angle_id,
                "view": view,
                "source_exif_orientation": source_exif_orientation,
                "rotation": 0,
                "crop": False,
                "face": None,
                "detection": None,
                "region_width": None,
                "region_height": None,
                "error": None,
            }
            item_ids.append(job_id)

        batches[batch_id] = {"items": item_ids}
    except Exception as exc:
        for job_id in item_ids:
            jobs.pop(job_id, None)
        shutil.rmtree(folder, ignore_errors=True)
        abort(400, str(exc))

    return redirect(url_for("batch_view", batch_id=batch_id))


@app.get("/batch/<batch_id>")
def batch_view(batch_id):
    batch = require_batch(batch_id)
    items = [jobs[job_id] for job_id in batch["items"]]
    landscape_width = batch.get("landscape_width", 55)
    landscape_height = batch.get("landscape_height", 80)
    portrait_width = batch.get("portrait_width", 60)
    portrait_height = batch.get("portrait_height", 45)
    for item in items:
        image = load_rgb(item_dir(batch_id, item["id"]) / "working.png")
        is_landscape = image.width >= image.height
        if item["view"] == "face":
            base_width = landscape_width if is_landscape else portrait_width
            base_height = landscape_height if is_landscape else portrait_height
            item["preview_region_width"], item["preview_region_height"] = adjusted_detection_region(
                base_width, base_height, bool(item["rotation"] % 360)
            )
        else:
            item["preview_region_width"], item["preview_region_height"] = detection_region(
                item["view"], image
            )
    completed = sum(bool(item["crop"]) for item in items)
    failed = sum(bool(item["error"]) for item in items)
    return render_template(
        "batch.html", batch_id=batch_id, items=items,
        completed=completed, failed=failed, total=len(items),
        landscape_width=landscape_width, landscape_height=landscape_height,
        portrait_width=portrait_width, portrait_height=portrait_height,
    )


@app.post("/rotate/<batch_id>/<job_id>/<direction>")
def rotate(batch_id, job_id, direction):
    batch = require_batch(batch_id)
    if job_id not in batch["items"] or direction not in {"left", "right"}:
        abort(404)
    item = jobs[job_id]
    path = item_dir(batch_id, job_id) / "working.png"
    image = load_rgb(path)
    angle = 90 if direction == "left" else -90
    save_rgb(image.rotate(angle, expand=True), path)
    item["rotation"] = (item["rotation"] + angle) % 360
    item["crop"] = False
    item["face"] = None
    item["detection"] = None
    item["error"] = None
    (item_dir(batch_id, job_id) / "crop.png").unlink(missing_ok=True)
    return redirect(url_for("batch_view", batch_id=batch_id))


def process_face_item(batch_id: str, item: dict, image: Image.Image, settings: dict) -> None:
    # Keep the established Face V2 model, segmentation geometry, confidence
    # flags, square framing, and region adjustment behavior intact.
    is_landscape = image.width >= image.height
    base_width = settings["landscape_width" if is_landscape else "portrait_width"]
    base_height = settings["landscape_height" if is_landscape else "portrait_height"]
    manually_rotated = bool(item["rotation"] % 360)
    region_width, region_height = adjusted_detection_region(
        base_width, base_height, manually_rotated
    )
    orientation_name = "landscape" if is_landscape else "portrait"
    model_image, (offset_x, offset_y) = centered_detection_image(
        image, region_width, region_height
    )
    print(
        f"\n[Face detector] {item['filename']} | {image.width}x{image.height} "
        f"| region={region_width}%W x {region_height}%H",
        flush=True,
    )
    face = finder.find(model_image)
    if face is None:
        item["error"] = (
            f"No cutter face detected in the centered {region_width}% width × "
            f"{region_height}% height region ({orientation_name})."
        )
        return
    face["cx"] = float(face["cx"]) + offset_x
    face["cy"] = float(face["cy"]) + offset_y
    crop, box = padded_square_crop(image, face)
    save_rgb(crop, item_dir(batch_id, item["id"]) / "crop.png")
    item["crop"] = True
    item["region_width"], item["region_height"] = region_width, region_height
    item["face"] = {
        "confidence": round(float(face.get("conf", 0)), 3),
        "center": [round(float(face["cx"]), 1), round(float(face["cy"]), 1)],
        "radius": round(float(face["r"]), 1),
        "reconstructed": bool(face.get("geometry_reconstructed")),
        "review": bool(
            face.get("geometry_needs_review")
            or face.get("low_conf")
            or face.get("not_very_circular")
        ),
        "box": box,
        "region_width": region_width,
        "region_height": region_height,
    }


def process_object_item(batch_id: str, item: dict, image: Image.Image) -> None:
    view = item["view"]
    region_width, region_height = detection_region(view, image)
    model_image, offset = centered_detection_image(image, region_width, region_height)
    detector = load_object_detector(view)
    # Preserve overlapping candidates before standard NMS can suppress them.
    # WBF then combines only overlapping boxes for the same crop target.
    predict_options = {
        "source": model_image,
        "verbose": False,
        "iou": BOX_FUSION_CANDIDATE_IOU,
    }
    inference_size = OBJECT_INFERENCE_IMAGE_SIZES.get(view)
    if inference_size is not None:
        predict_options["imgsz"] = inference_size
    results = detector.predict(**predict_options)
    selected = choose_fused_detection(results[0])
    if selected is None:
        item["error"] = (
            f"No {VIEW_LABELS[view]} target detected in the centered "
            f"{region_width}% width × {region_height}% height region."
        )
        return

    full_box = map_box_to_full_image(selected["box"], offset)
    coverage = CROP_TARGET_COVERAGE[view]
    if view == "isometric":
        crop_box = isometric_square_crop_geometry(image.size, full_box, coverage)
    else:
        crop_box = square_crop_geometry(image.size, full_box, coverage)
    crop = crop_square(image, crop_box, OUTPUT_SIZE)
    save_rgb(crop, item_dir(batch_id, item["id"]) / "crop.png")
    x1, y1, x2, y2 = full_box
    item["crop"] = True
    item["region_width"], item["region_height"] = region_width, region_height
    item["detection"] = {
        "confidence": round(float(selected["confidence"]), 3),
        "box": [round(value, 1) for value in full_box],
        "center": [round((x1 + x2) / 2.0, 1), round((y1 + y2) / 2.0, 1)],
        "crop_box": crop_box,
        "target_coverage": coverage,
        "fusion_applied": bool(selected.get("fusion_applied", False)),
        "fusion_source_count": int(selected.get("fusion_source_count", 1)),
        "fusion_members": list(selected.get("fusion_members", ())),
        "provisional": False,
    }


@app.post("/autocrop/<batch_id>")
def autocrop_batch(batch_id):
    batch = require_batch(batch_id)
    try:
        settings = {
            "landscape_width": int(request.form.get("landscape_width", "55")),
            "landscape_height": int(request.form.get("landscape_height", "80")),
            "portrait_width": int(request.form.get("portrait_width", "60")),
            "portrait_height": int(request.form.get("portrait_height", "45")),
        }
    except (TypeError, ValueError):
        abort(400, "All Face detection-region settings must be whole-number percentages.")
    if any(value < 25 or value > 100 for value in settings.values()):
        abort(400, "Face detection-region settings must each be between 25% and 100%.")
    batch.update(settings)

    # batch['items'] was sorted on upload; process all Isometric, then Top,
    # then Face, retaining original upload order inside each group.
    for job_id in batch["items"]:
        item = jobs[job_id]
        item["crop"], item["face"], item["detection"], item["error"] = False, None, None, None
        try:
            image = load_rgb(item_dir(batch_id, job_id) / "working.png")
            if item["view"] == "face":
                process_face_item(batch_id, item, image, settings)
            else:
                process_object_item(batch_id, item, image)
            if item["crop"]:
                print(
                    f"[{item['view'].title()} detector] CROPPED {item['filename']} "
                    f"| confidence={item['face']['confidence'] if item['face'] else item['detection']['confidence']}"
                    + (
                        f" | WBF sources={item['detection']['fusion_source_count']}"
                        if item['detection'] and item['detection'].get('fusion_applied')
                        else ""
                    ),
                    flush=True,
                )
            else:
                print(f"[{item['view'].title()} detector] NO CROP {item['filename']}: {item['error']}", flush=True)
        except Exception as exc:
            import traceback
            item["error"] = f"Processing error: {type(exc).__name__}: {exc}"
            print(f"[{item['view'].title()} detector] ERROR on {item['filename']}: {exc}", flush=True)
            traceback.print_exc()
    return redirect(url_for("batch_view", batch_id=batch_id))


@app.get("/image/<batch_id>/<job_id>/<kind>")
def image_file(batch_id, job_id, kind):
    batch = require_batch(batch_id)
    if job_id not in batch["items"]:
        abort(404)
    name = {"working": "working.png", "crop": "crop.png"}.get(kind)
    if not name:
        abort(404)
    path = item_dir(batch_id, job_id) / name
    if not path.exists():
        abort(404)
    return send_file(path, mimetype="image/png", max_age=0)


@app.get("/download/<batch_id>/<job_id>")
def download_one(batch_id, job_id):
    batch = require_batch(batch_id)
    if job_id not in batch["items"]:
        abort(404)
    item = jobs[job_id]
    path = item_dir(batch_id, job_id) / "crop.png"
    if not item["crop"] or not path.exists():
        abort(404, "This image has not been cropped successfully.")
    return send_file(path, as_attachment=True, download_name=f"{item['output_name']}_centered.png")


@app.get("/download-batch/<batch_id>")
def download_batch(batch_id):
    batch = require_batch(batch_id)
    memory_file = io.BytesIO()
    added = 0
    with zipfile.ZipFile(memory_file, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for job_id in batch["items"]:
            item = jobs[job_id]
            path = item_dir(batch_id, job_id) / "crop.png"
            if item["crop"] and path.exists():
                archive.write(path, arcname=f"{item['output_name']}_centered.png")
                added += 1
    if not added:
        abort(404, "There are no successfully cropped images to download.")
    memory_file.seek(0)
    return send_file(
        memory_file, mimetype="application/zip", as_attachment=True,
        download_name="centered_cutter_views.zip",
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
