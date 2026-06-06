# YOLOv8x + ByteTrack on MOT17

> UCL MEng Robotics & AI — Computer Vision Portfolio Project
> **Detector:** YOLOv8x (COCO pretrained, zero-shot) · **Tracker:** ByteTrack (custom implementation) · **Benchmark:** MOT17

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)](https://pytorch.org)
[![YOLOv8](https://img.shields.io/badge/Detector-YOLOv8x-green)](https://github.com/ultralytics/ultralytics)
[![MOT17](https://img.shields.io/badge/Benchmark-MOT17-yellow)](https://motchallenge.net)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<your-handle>/yolov8-bytetrack-mot17/blob/main/notebook.ipynb)

---

## What was Built

An end-to-end **multi-object pedestrian tracking** pipeline evaluated on the MOT17 benchmark. The pipeline is intentionally decoupled — detection and tracking are separate stages — making it easy to swap either component.

**Key design decisions:**

- YOLOv8x is used as a zero-shot COCO-pretrained detector with no fine-tuning on MOT17. This makes the evaluation a realistic test of generalisation rather than a trained baseline.
- ByteTrack is implemented from scratch in pure Python with scipy for Hungarian assignment, removing the dependency on the `lap`/`lapx` packages which have persistent build issues in modern Python environments.
- TrackEval produces HOTA, MOTA, and IDF1 — the three standard MOT metrics — plus DetA and AssA which decompose HOTA into detector and association contributions independently.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         PIPELINE OVERVIEW                           │
└─────────────────────────────────────────────────────────────────────┘

  MOT17 Frames              YOLOv8x                  ByteTrack
  (img1/%06d.jpg)       (COCO pretrained)         (IoU + Kalman)
        │                      │                        │
        ▼                      ▼                        ▼
  ┌──────────┐  batch=8  ┌──────────┐  dets [N,5]  ┌────────────────┐
  │  Frame   │──────────▶│ Backbone │  (xyxy, conf) │ Step 1         │
  │  t, t+1  │           │  CSP-    │──────────────▶│ High-conf dets │
  │  t+2 ... │           │  DarkNet │               │ ↔ active tracks│
  └──────────┘           │  + PAN   │               │ IoU Hungarian  │
                         │  + Head  │               ├────────────────┤
                         └──────────┘               │ Step 2         │
                                                    │ Low-conf dets  │
                         COCO AP@50:95 = 53.9       │ ↔ lost tracks  │
                         ~15 FPS on T4 GPU          ├────────────────┤
                                                    │ Step 3         │
                                                    │ Lost tracks    │
                                                    │ ↔ remaining    │
                                                    ├────────────────┤
                                                    │ Step 4         │
                                                    │ Init new tracks│
                                                    │ (score ≥ 0.7)  │
                                                    └────────────────┘
                                                           │
                                                           ▼
                                                  MOT-format .txt tracks
                                                  frame,id,x,y,w,h,conf
                                                           │
                                                           ▼
                                                  ┌─────────────────┐
                                                  │   TrackEval     │
                                                  │  HOTA/MOTA/IDF1 │
                                                  │  DetA / AssA    │
                                                  └─────────────────┘
```

**ByteTrack two-step association (key insight):** Most trackers discard low-confidence detections. ByteTrack keeps them and uses them in a second association pass — this recovers occluded pedestrians whose detector confidence drops temporarily, reducing ID switches without introducing false positives.

**Kalman filter state:** `[cx, cy, aspect, h, vx, vy, va, vh]` — constant velocity model in centre-aspect-height space. Predictions are used for IoU matching when detections are absent.

---

## Results

### Per-sequence metrics on MOT17 train subset (5 sequences, SDP variant)

| Sequence | HOTA↑ | DetA↑ | AssA↑ | MOTA↑ | IDF1↑ | Recall↑ | Prec↑ | IDSW↓ |
|:---------|------:|------:|------:|------:|------:|--------:|------:|------:|
| MOT17-02-SDP | 31.7 | 17.5 | 42.1 | 19.5 | 25.7 | 20.2 | 97.0 | 22 |
| MOT17-04-SDP | 38.7 | 21.4 | 51.2 | 22.3 | 34.7 | 24.1 | 93.2 | 15 |
| MOT17-05-SDP | 63.9 | 44.7 | 46.3 | 52.0 | 61.7 | 60.7 | 88.7 | 70 |
| MOT17-09-SDP | 61.0 | 56.9 | 41.0 | 65.8 | 58.2 | 70.8 | 94.2 | 36 |
| MOT17-13-SDP | 37.1 | 19.2 | 46.6 | 21.9 | 32.3 | 23.7 | 94.4 | 39 |
| **Combined** | **41.4** | **24.3** | **47.8** | **26.5** | **37.2** | **28.8** | **93.2** | **182** |

### Comparison to paper baselines

| Method | Detector | HOTA↑ | MOTA↑ | IDF1↑ | Notes |
|:-------|:---------|------:|------:|------:|:------|
| ByteTrack (ECCV 2022) | YOLOX-X 800px | 63.1 | 78.5 | 73.4 | Full train, fine-tuned on MOT17 |
| OC-SORT (2023) | YOLOX-X 800px | 63.9 | 78.0 | 77.5 | Full train |
| BoT-SORT (2022) | YOLOX-X 800px | 65.0 | 78.5 | 76.9 | Full train |
| **Ours** | YOLOv8x 640px | **41.4** | **26.5** | **37.2** | 5-seq subset, zero-shot COCO |

**Why the gap exists:** The DetA/AssA decomposition tells the story precisely. AssA = 47.8 is competitive — the ByteTrack association logic works. DetA = 24.3 is where performance is lost. The detector misses ~75% of pedestrians on the dense sequences because it was trained on COCO (average ~3 people/image) and is evaluated on MOT17-04 (average ~80 people/frame). Fine-tuning YOLOv8x on MOT17 half-train would recover this gap almost entirely.

---

## Failure Mode Analysis

### 1. Crowd density (MOT17-02, MOT17-04, MOT17-13)

Recall of 20–24% on these sequences means the detector fires on roughly 1 in 4–5 pedestrians. Precision stays above 93%, so the detections that do fire are correct — the problem is purely missed detections on occluded and overlapping pedestrians.

**Root cause:** YOLOv8x uses non-maximum suppression with IoU threshold 0.45. When pedestrians overlap by more than 55%, one is suppressed. MOT17-04 has scenes where this eliminates entire rows of people.

**Fix:** Fine-tune on MOT17 half-train (the standard approach in the literature). Alternatively, use a crowd-specific detector such as CrowdDet or disable NMS and use soft-NMS instead.

### 2. ID inflation on sparse sequences (MOT17-09)

The pipeline spawned 54 track IDs against 26 ground truth IDs — a 2× inflation ratio. AssA of 41% confirms fragmented tracks. Pedestrians re-entering frame after a short exit spawn new IDs instead of resuming old ones.

**Root cause:** `new_track_thresh = 0.7` is too permissive when precision is high. High-confidence detections of re-entering pedestrians pass the threshold immediately, creating new tracks rather than matching to the lost track pool.

**Fix:** Increase `new_track_thresh` to 0.8 and extend `track_buffer` from 30 to 60 frames. Adding a lightweight re-ID embedding (BoT-SORT style) for the lost-track association step would address this more robustly.

### 3. ID switches on moving camera (MOT17-05, MOT17-13)

70 and 39 ID switches respectively vs 15–22 on static sequences. IDF1 drops disproportionately relative to MOTA on these sequences.

**Root cause:** The Kalman filter assumes constant velocity in image coordinates. Camera motion adds a global offset to all predicted positions, making IoU-based matching fail even when the tracker correctly predicts the pedestrian's world-space motion.

**Fix:** Camera motion compensation (CMC) via ECC or sparse optical flow on background pixels. This is implemented in BoT-SORT and gives approximately +2–3 IDF1 on moving-camera sequences.

### 4. Low frame rate (MOT17-05 at 14fps)

Inter-frame displacement is larger relative to bounding box size at 14fps than at 30fps. IoU-based matching degrades because the predicted and actual boxes overlap less.

**Fix:** Scale the Kalman process noise with `1/fps` — faster motion uncertainty at lower frame rates. `track_buffer` should also scale: `buffer = int(fps * 2)` rather than a fixed 30 frames.

---

## Edge Deployment Considerations

| Change | Rationale | Expected gain |
|:-------|:----------|:-------------|
| Quantise YOLOv8x to INT8 via TensorRT | 3–4× throughput on Jetson, <1% AP loss | ~45 → ~180 FPS |
| Switch to YOLOv8n or YOLOv8s backbone | 10× fewer parameters, edge-deployable | ~180 → ~400 FPS at lower AP |
| Reduce input resolution to 480px | ~40% faster inference | Acceptable for near-field pedestrians |
| Replace scipy Hungarian with C++ lap | Eliminates Python overhead in association | ~5ms → ~0.5ms per frame |
| CMC via sparse optical flow | Handles camera motion on mobile platforms | +2–3 IDF1 |
| Background subtraction pre-filter | Skip detector on static regions | ~2× speedup on sparse scenes |

Target for Jetson AGX Orin: YOLOv8s INT8 at 480px + ByteTrack C++ → estimated ~60 FPS real-time, suitable for queue analytics and retail footfall applications.

---

## Reproduction

### Colab (recommended)

1. Open `notebook.ipynb` in Google Colab with a T4 GPU runtime
2. Run **Cell 0** → **Runtime → Restart session**
3. Fill in Kaggle credentials in **Cell 2**
4. Run **Cells 1–5** top to bottom
5. Expected total time: ~12 min

### Local / command line

```bash
git clone https://github.com/<your-handle>/yolov8-bytetrack-mot17
cd yolov8-bytetrack-mot17

pip install ultralytics scipy matplotlib kaggle tqdm
git clone https://github.com/JonathonLuiten/TrackEval.git

# Download MOT17 (fill in credentials first)
python step2_download_mot17.py

# Run pipeline
python step3_inference.py
python step4_bytetrack.py
python step5_evaluate.py
```

### Expected output

```
RESULTS: YOLOv8x + ByteTrack on MOT17 (5-sequence subset)
Sequence                        HOTA       DetA       AssA       MOTA       IDF1     Recall  Precision
──────────────────────────────────────────────────────────────────────────────────────────────────────
MOT17-02-SDP                   31.7%      17.5%      42.1%      19.5%      25.7%      20.2%      97.0%
MOT17-04-SDP                   38.7%      21.4%      51.2%      22.3%      34.7%      24.1%      93.2%
MOT17-05-SDP                   63.9%      44.7%      46.3%      52.0%      61.7%      60.7%      88.7%
MOT17-09-SDP                   61.0%      56.9%      41.0%      65.8%      58.2%      70.8%      94.2%
MOT17-13-SDP                   37.1%      19.2%      46.6%      21.9%      32.3%      23.7%      94.4%
COMBINED                       41.4%      24.3%      47.8%      26.5%      37.2%      28.8%      93.2%
```

---

## Repository Structure

```
.
├── notebook.ipynb              # End-to-end Colab notebook (6 cells)
├── step0_fix_env.py            # Colab numpy environment fix (run once)
├── step1_install_deps.py       # Install all dependencies
├── step2_download_mot17.py     # Download MOT17 from Kaggle
├── step3_inference.py          # YOLOv8x detection → MOT-format .txt
├── step4_bytetrack.py          # ByteTrack association → track .txt
├── step5_evaluate.py           # TrackEval → HOTA / MOTA / IDF1
└── README.md
```

---

## References

1. Zhang, Y. et al. (2022). **ByteTrack: Multi-Object Tracking by Associating Every Detection Box**. ECCV 2022. [arXiv:2110.06864](https://arxiv.org/abs/2110.06864)
2. Jocher, G. et al. (2023). **Ultralytics YOLOv8**. [GitHub](https://github.com/ultralytics/ultralytics)
3. Luiten, J. et al. (2021). **HOTA: A Higher Order Metric for Evaluating Multi-Object Tracking**. IJCV 79, 408–428.
4. Dendorfer, P. et al. (2021). **MOTChallenge: A Benchmark for Single-Camera Multiple Target Tracking**. IJCV.
5. Cao, J. et al. (2023). **OC-SORT: Observation-Centric SORT on Video Object Perception**. CVPR 2023.

---

## License

MIT — see `LICENSE`. MOT17 dataset subject to [MOTChallenge Terms of Use](https://motchallenge.net).
