# Cropper StandAlone

Cropper StandAlone is a local development and validation harness for the view-specific cutter crop pipeline intended for eventual integration into BitHub. It is not a production multi-user service.

The validated crop-pipeline baseline is **feature/view-specific-crop-pipeline**. This separate branch, **prototype/isolated-view-upload-dropzones-do-not-integrate**, adds a temporary three-dropzone upload interface for evaluation only. Do not merge this prototype into the main application; keep it isolated until upload UX integration is separately approved. The repository default branch remains **main**.

## What the application does

On this prototype branch, the Flask upload page has three separate multi-file dropzones. The server generates the JSON manifest from the dropzone used for each image. The manifest explicitly assigns each photo to a view:

| angle_index | View | Processing order |
|---:|---|---:|
| 0 | Isometric | First |
| 1 | Top | Second |
| 2 | Face | Third |

The filename is used only to match a manifest record to an uploaded photo. **The filename, EXIF metadata, image orientation, image pixels, and upload order do not determine the view.** Images are grouped Isometric → Top → Face while preserving their order within each group.

A starter manifest is available at manifest_template.json. Replace the example filenames with the exact photo filenames being uploaded.

## Current crop paths

- **Face:** existing Face V2 segmentation path and square-crop geometry. The Face model and crop behavior are intentionally kept independent from the Top and Isometric detectors.
- **Top:** independent YOLO object detector for the complete visible cutter. Accepted target coverage: 0.70.
- **Isometric:** independent YOLO object detector for the visible diamond table. Accepted target coverage: 0.80; the diamond-table center is positioned at 70% of output height.

Top and Isometric outputs are square 1200 × 1200 presentation crops. See docs/ARCHITECTURE.md and CROP_PIPELINE_PLAN.md for the detailed processing contract and validation history.

## Quick start

The current development workstation baseline uses Python 3.12 and a CUDA-enabled PyTorch installation. See docs/DEVELOPER_SETUP.md before installing dependencies, especially the separate PyTorch/CUDA step.

From the repository root in PowerShell:

~~~powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
~~~

Install the CUDA-compatible PyTorch and torchvision build appropriate for the machine, then install the project dependencies:

~~~powershell
python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python app.py
~~~

The PyTorch command is for a CUDA 12.8-capable NVIDIA setup. For CPU-only machines or a different CUDA environment, use the official PyTorch install selector and install a matching build. Do not replace a working CUDA-enabled PyTorch installation with a CPU-only build.

Open http://127.0.0.1:5000.

## Run a batch

On this prototype branch, the upload page has three independent multi-file dropzones:

- Drop all Isometric photos into **Isometric**.
- Drop all Top photos into **Top**.
- Drop all Face photos into **Face**.

The server creates the manifest from the selected dropzone. The assignments remain explicit: Isometric = 0, Top = 1, Face = 2. Processing remains Isometric → Top → Face, preserving upload order within each view. Duplicate filenames across or within dropzones are given a unique view-prefixed name so every uploaded photo can be matched unambiguously.

The original JSON-manifest upload contract remains supported for older clients and scripts. The three-dropzone interface is intentionally isolated to this prototype branch and is not a main-branch integration.

## Documentation

- Architecture and processing flow: docs/ARCHITECTURE.md
- Developer setup and model requirements: docs/DEVELOPER_SETUP.md
- Testing and validation: docs/TESTING.md
- Project status and handoff checklist: docs/PROJECT_STATUS.md
- Contributor workflow and protected assets: CONTRIBUTING.md
- Detailed implementation plan and validation record: CROP_PIPELINE_PLAN.md
- Top / Isometric model training plan: TRAINING_PLAN.md
- Preparing the Top + Isometric crop-test batch: CROP_TEST_BATCH.md

## Important project boundaries

- This is a local, single-user prototype. Batch/job metadata is held in memory; restarting Flask clears active batch metadata.
- Uploaded working files and generated crops are stored under work/, which is excluded from Git. They are not automatically removed when a batch finishes.
- All three inference checkpoints are shared with authorized collaborators. The Face checkpoint is committed directly; Top and Isometric checkpoints are stored with Git LFS. Install Git LFS and run `git lfs pull` if model files are missing or appear as pointer text.
- The Face V2 checkpoint is currently present in the repository under models/face_combined_v2_finetune/weights/best.pt. Anyone granted repository access can access committed files, including this checkpoint.
- Proprietary source photos, annotation manifests, and generated training datasets should remain local/untracked unless there is explicit approval to share them.
- Automated unit tests do not replace inference tests with the actual trained checkpoints or visual review of full-batch results.

See docs/PROJECT_STATUS.md for what has and has not yet been validated.
