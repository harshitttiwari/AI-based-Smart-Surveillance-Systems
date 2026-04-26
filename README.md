# AI-based Smart Surveillance Systems

This project combines accident detection, violence classification, and automated incident reporting for CCTV-style video streams.

The current workflow uses:

- YOLOv9 for accident vs non-accident object/event detection.
- SlowFast (video model) for temporal violence/weaponized classification.
- Gemini (vision-language) for natural-language incident report generation.

## Repository Layout

### Core notebooks

- `pipeline.ipynb`
	End-to-end inference pipeline: reads a stream/video, runs YOLO + SlowFast, triggers alerts, saves evidence, and generates reports.

- `Acidental/main.ipynb`
	YOLOv9 accident-detection training and evaluation notebook.

- `Voilence/main2.ipynb`
	SlowFast training notebook for classes: `Normal`, `Violence`, `Weaponized`.

### Data and artifacts in this repo

- `Acidental/data.yaml`
	Dataset config (`accident`, `non-accident`) for YOLO training/eval.

- `Acidental/runs/`
	Training/validation artifacts for accident detection (curves, confusion matrices, predictions, CSV metrics, sample visualizations).

- `Voilence/Train/Weaponized` and `Voilence/Test/Weaponized`
	Curated relevant subset for weaponized class content.

### Generated at runtime (kept local by default)

- `incident_frames/`
- `incident_clips/`
- `incident_reports/` (JSON + TXT)

## System Pipeline

1. Read frame stream from webcam/video source.
2. Run YOLO detection on each frame.
3. Keep a temporal frame buffer and run SlowFast periodically.
4. Trigger incident if:
	 - YOLO detects target with confidence above threshold, or
	 - SlowFast predicts non-normal class above trigger confidence.
5. Save evidence:
	 - snapshot frame,
	 - short clip,
	 - structured report (`json` + `txt`).
6. Use Gemini vision model to convert frame evidence into analyst-style incident report text.

## Models and Key Configuration

### Accident detection (YOLO)

- Model base used in training notebook: `yolov9e.pt`
- Dataset config: `Acidental/data.yaml`
- Classes: `accident`, `non-accident`
- Typical training parameters seen in notebook:
	- `epochs=25`
	- `batch=6`
	- `imgsz=250` (auto-adjusted to 256 by Ultralytics due to stride)
	- `workers=8`
	- `device=0`

In pipeline inference:

- YOLO model path:
	`X:\Epics\Acidental\runs\detect\train1\weights\best.pt`
- Confidence threshold:
	`YOLO_CONF_THRESHOLD = 0.6`

### Violence/weaponized classification (SlowFast)

- Backbone source: `facebookresearch/pytorchvideo`, `slowfast_r50`
- Fine-tuned output classes:
	- `Normal`
	- `Violence`
	- `Weaponized`
- Input settings:
	- `NUM_FRAMES = 32`
	- `ALPHA = 4` (slow pathway sampling stride)
	- resize: `224x224`
	- normalization mean/std: `[0.45, 0.45, 0.45]` / `[0.225, 0.225, 0.225]`

In pipeline inference:

- Model weights loaded from: `slowfast_best.pth`
- Incident trigger threshold default in `run_pipeline`:
	`slowfast_trigger_conf=0.75`

### Reporting model (Gemini)

- Model in notebook: `gemini-2.5-flash`
- API key is user-supplied in notebook cell (`GEMINI_API_KEY`).
- Prompt requests structured sections:
	summary, activity, severity, observations, recommended action, confidence notes.

## Runtime Output Format

When an incident is triggered:

- Frame saved to `incident_frames/<incident>_<timestamp>.jpg`
- Clip saved to `incident_clips/<incident>_<timestamp>.avi`
- Structured JSON saved to `incident_reports/<incident>_<timestamp>.json`
- Human-readable TXT saved to `incident_reports/<incident>_<timestamp>.txt`

JSON report fields include:

- `timestamp`
- `incident`
- `confidence`
- `frame_path`
- `clip_path`
- `report`

## Training and Evaluation Notes

### YOLO metrics observed in notebook

Example `model2.val(split="test")` results shown:

- overall mAP50 around `0.927`
- class-wise metrics printed for `accident` and `non-accident`

Artifacts are visible under:

- `Acidental/runs/detect/train1/`
- `Acidental/runs/detect/val3/`

### SlowFast training behavior observed

Notebook uses class balancing through weighted sampling on training set:

- Train distribution shown:
	- `Normal: 200`
	- `Violence: 99`
	- `Weaponized: 100`

- Test distribution shown:
	- `Normal: 46`
	- `Violence: 12`
	- `Weaponized: 24`

Model checkpoint is saved as `slowfast_best.pth` when best validation accuracy improves.

## Quick Start

### 1) Install dependencies

The notebooks use these primary packages:

- `ultralytics`
- `opencv-python`
- `torch`
- `torchvision`
- `numpy`
- `Pillow`
- `google-generativeai`
- `tqdm`
- `pandas`

### 2) Prepare local files

- Keep local model files available:
	- `slowfast_best.pth`
	- YOLO best checkpoint used by pipeline
- Ensure dataset folders expected by notebooks are present.

### 3) Configure API key

Set a valid Gemini API key in the notebook cell before running report generation.

### 4) Run end-to-end pipeline

Open `pipeline.ipynb` and run cells in order, then call/run:

- `run_pipeline(source=0)` for webcam, or
- `run_pipeline(source="path/to/video.mp4")` for file input.

## Important Constraints

- This repo intentionally avoids uploading all raw datasets and large weight binaries.
- `.gitignore` is tuned to keep relevant outputs and notebooks while preventing heavy/unnecessary uploads.
- Some notebook paths are absolute (for local machine setup), so adjust paths if you run in a different environment.

## Disclaimer

This project is for research/prototyping and monitoring assistance. It should not be treated as a fully autonomous safety-critical system without additional validation, bias checks, and human oversight.