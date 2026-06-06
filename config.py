"""
config.py — Central configuration for the YOLOv8x + ByteTrack pipeline.

All paths, hyperparameters, and sequence settings live here.
Edit this file to change detector thresholds, ByteTrack parameters,
or which MOT17 sequences to evaluate.
"""

from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────
MOT17_ROOT    = Path("/content/MOT17/train")
DET_OUT_ROOT  = Path("/content/detections")
TRACK_ROOT    = Path("/content/tracks")
RESULTS_DIR   = Path("/content/trackeval_results")
TRACKEVAL_DIR = Path("/content/TrackEval")

# ── Sequences ─────────────────────────────────────────────────────────────
# 5-sequence subset covering diverse scenarios on T4 (~12 min total).
# Static camera: MOT17-02 (low density), MOT17-04 (high density), MOT17-09 (medium)
# Moving camera: MOT17-05 (low fps, 14Hz), MOT17-13 (indoor)
SEQUENCES = [
    "MOT17-02-SDP",
    "MOT17-04-SDP",
    "MOT17-05-SDP",
    "MOT17-09-SDP",
    "MOT17-13-SDP",
]

# Ground truth sequence lengths (frames) — used by TrackEval
SEQ_LENGTHS = {
    "MOT17-02-SDP": 600,
    "MOT17-04-SDP": 1050,
    "MOT17-05-SDP": 837,
    "MOT17-09-SDP": 525,
    "MOT17-13-SDP": 750,
}

# ── Detector (YOLOv8x) ────────────────────────────────────────────────────
DETECTOR_MODEL  = "yolov8x.pt"   # downloads ~130 MB on first run
CONF_THRESH     = 0.35           # kept low — ByteTrack uses low-conf dets in step 2
PERSON_CLASS_ID = 0              # COCO class 0 = person
BATCH_SIZE      = 8              # increase to 16 if >15 GB VRAM available

# ── ByteTrack ─────────────────────────────────────────────────────────────
TRACK_HIGH_THRESH = 0.6    # high-confidence threshold for first association pass
TRACK_LOW_THRESH  = 0.1    # low-confidence threshold for second association pass
NEW_TRACK_THRESH  = 0.7    # minimum score to initialise a new tracklet
TRACK_BUFFER      = 30     # frames a lost track is kept alive (scaled by fps)
MATCH_THRESH      = 0.8    # IoU threshold for Hungarian matching

# ── TrackEval ─────────────────────────────────────────────────────────────
TRACKER_NAME = "YOLOv8x-ByteTrack"

# ── Kaggle credentials ────────────────────────────────────────────────────
# Fill these in before running data.py.
# Get your key: kaggle.com/settings → API → Create New Token
KAGGLE_USERNAME = "your_username_here"
KAGGLE_KEY      = "your_key_here"
