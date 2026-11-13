# Architecture

## Purpose and scope

Cropper StandAlone is a local validation harness for the crop-processing workflow intended for eventual integration into BitHub. It exercises the production-oriented view assignment and crop behavior without being a production multi-user service.

The app has three independent view paths. View assignment comes from the JSON manifest; it is never inferred from filename tokens, EXIF, image orientation, pixels, or upload position.

## End-to-end processing flow

1. **Upload:** on this prototype branch, the browser submits photos in three independent multi-file fields to Flask POST /upload: Isometric, Top, and Face. The server creates manifest records from the selected field (0 = Isometric, 1 = Top, 2 = Face). The legacy combined-photo plus JSON-manifest request remains supported.
2. **Manifest validation:** crop_pipeline.normalize_manifest_records validates each generated or uploaded file_path and angle_index. Values 0, 1, and 2 map to Isometric, Top, and Face.
3. **Matching:** the app matches each manifest record to exactly one uploaded image by basename. Missing, duplicate, unmatched, or unsupported records are rejected.
4. **Stable grouping:** records are ordered Isometric first, Top second, Face third. Original order within each view is preserved.
5. **Image normalization:** EXIF orientation is applied once. The normalized RGB working image is saved as working.png; subsequent model and crop coordinates refer to these normalized pixels.
6. **Review and correction:** the batch page shows each image and its assigned view. The user may apply an additional manual 90-degree rotation before processing.
7. **Inference and crop:** the app dispatches each item to its view-specific path and saves a crop when a target is found.
8. **Review and export:** the batch page presents results, and successful crops can be downloaded individually or as a ZIP.

## Module responsibilities

| File | Responsibility |
|---|---|
| app.py | Flask routes, uploads, in-memory batch/job state, view dispatch, Face processing, and result delivery |
| crop_pipeline.py | Manifest validation, angle-to-view mapping, stable ordering, detection-region geometry, coordinate mapping, Top/Isometric crop geometry, and box fusion |
| image_orientation.py | Reads EXIF orientation and normalizes uploaded images exactly once |
| face_model.py | Face V2 segmentation inference and face-geometry reconstruction |
| top_isometric_web.py | Flask blueprint for the Top/Isometric annotation editor and its API |
| templates/ | Flask HTML pages for upload, batch review, and annotation |
| static/ | Browser-side styles and annotation-editor behavior |
| tools/ | Dataset preparation, annotation export, training, batch preparation, and detector comparison utilities |
| tests/ | Automated unit tests for manifest routing, geometry, orientation, and training preparation |

## View assignment contract

The StandAlone manifest is a JSON array. Each record has:

- file_path: source photo path or filename; the app matches the uploaded image by basename.
- angle_index: explicit numeric view identifier.

| Value | View | Model path |
|---:|---|---|
| 0 | Isometric | models/isometric_view/weights/best.pt |
| 1 | Top | models/top_view/weights/best.pt |
| 2 | Face | Existing Face V2 model |

The app does not infer a missing value. Invalid or incomplete records fail validation before inference.

## View-specific inference

### Face

Face uses the existing FaceFinder V2 segmentation model through face_model.py. The app fits/reconstructs circular face geometry and produces a padded square crop. The established crop framing, confidence/review flags, and interactive region behavior are retained. Face remains independent of the Top and Isometric YOLO object detectors.

Base Face detection regions:

| Working image orientation | Width | Height |
|---|---:|---:|
| Landscape | 55% | 80% |
| Portrait | 60% | 45% |

The Face-only region adjustment expands both dimensions by 5% for the normal path; after manual rotation it leaves width unchanged and increases height by 20%.

### Top and Isometric

Top and Isometric use independent Ultralytics YOLO object-detection checkpoints. Each detector has its own expected class name and is loaded lazily only when that view is processed.

| View | Detection target | Landscape region | Portrait region | Crop target coverage |
|---|---|---:|---:|---:|
| Top | Complete visible cutter (full_cutter) | 50% × 68% | 63% × 62% | 0.70 |
| Isometric | Visible diamond table (diamond_table) | 55% × 80% | 65% × 55% | 0.80 |

The detection regions are centered portions of the EXIF-normalized source image. Predicted boxes are translated back into full-image coordinates before crop geometry is calculated.

Top and Isometric use confidence-weighted box fusion for overlapping candidates. Single-member clusters remain unchanged; non-overlapping detections are not averaged together. If no candidate is detected, the item retains the no-crop/error result rather than receiving a fabricated crop.

Both views produce square 1200 × 1200 crops, preserve source aspect ratio, and pad beyond image edges rather than stretching the subject. Isometric crop geometry places the diamond-table center at 70% of output height while retaining the full detected table box and surrounding carbide context.

## Runtime state and storage

- Flask batch and job metadata is stored in process memory (batches and jobs dictionaries).
- Uploaded and intermediate files are stored under work/<batch_id>/<job_id>/.
- The normalized source is saved as working.png; successful output is saved as crop.png.
- Restarting the Flask process clears active batch/job metadata. Files under work/ may remain and require manual cleanup.
- The current implementation processes the batch sequentially. It is a local single-user prototype, not a persistent job service.

## Model and data boundaries

- Face V2 checkpoint: models/face_combined_v2_finetune/weights/best.pt is currently present in the repository.
- Top checkpoint: models/top_view/weights/best.pt is excluded from Git.
- Isometric checkpoint: models/isometric_view/weights/best.pt is excluded from Git.
- Source photos, annotation manifests, and generated training datasets are local/untracked by default.
- work/, .venv/, training outputs, and Top/Isometric model directories are ignored by Git.

Repository access grants access to committed files. Collaborators who need Top and Isometric inference must receive the corresponding checkpoints through an approved separate channel and place them at the documented paths.
