"""
detect.py — YOLOv8x inference on MOT17 sequences.

Runs YOLOv8x (COCO pretrained) on each configured sequence and writes
per-sequence detection files in MOT challenge format:

    <frame>, -1, <x>, <y>, <w>, <h>, <conf>, -1, -1, -1

where x, y, w, h are in pixel coordinates (top-left origin).

Usage:
    python detect.py

Expected runtime on T4 GPU: ~8 min for 5-sequence subset.
"""

import time
from pathlib import Path

from tqdm import tqdm
from ultralytics import YOLO
import torch

import config
from data import read_seqinfo, get_frame_paths, write_detections


# ── Model loading ─────────────────────────────────────────────────────────

def load_model(device: str) -> YOLO:
    """Load YOLOv8x, downloading weights if not cached."""
    print(f"Loading {config.DETECTOR_MODEL}...")
    model = YOLO(config.DETECTOR_MODEL)
    print(f"YOLOv8x loaded on {device} ✓")
    return model


# ── Per-sequence inference ────────────────────────────────────────────────

def run_inference(model: YOLO, seq_path: Path, device: str) -> list:
    """
    Run YOLOv8x on all frames of a single sequence.

    Args:
        model:    loaded YOLO model
        seq_path: path to the sequence directory
        device:   'cuda' or 'cpu'

    Returns:
        List of MOT-format detection strings.
    """
    info   = read_seqinfo(seq_path)
    frames = get_frame_paths(seq_path, info)
    lines  = []

    for bs in tqdm(range(0, len(frames), config.BATCH_SIZE),
                   desc=seq_path.name, unit="batch"):
        batch_frames = frames[bs:bs + config.BATCH_SIZE]
        results = model(
            [str(f) for f in batch_frames],
            classes=[config.PERSON_CLASS_ID],
            conf=config.CONF_THRESH,
            verbose=False,
            device=device,
        )

        for frame_path, result in zip(batch_frames, results):
            fid = int(frame_path.stem)

            if result.boxes is None or len(result.boxes) == 0:
                continue

            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = box.conf.item()
                w, h = x2 - x1, y2 - y1
                # MOT detection format: frame,-1,x,y,w,h,conf,-1,-1,-1
                lines.append(
                    f"{fid},-1,{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},"
                    f"{conf:.4f},-1,-1,-1"
                )

    return lines


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = load_model(device)

    config.DET_OUT_ROOT.mkdir(parents=True, exist_ok=True)
    total_start = time.time()

    for seq_name in config.SEQUENCES:
        seq_path = config.MOT17_ROOT / seq_name
        print(f"\n{seq_name}")

        t0    = time.time()
        lines = run_inference(model, seq_path, device)
        fps   = read_seqinfo(seq_path)["seq_len"] / (time.time() - t0)

        out_path = config.DET_OUT_ROOT / f"{seq_name}.txt"
        write_detections(lines, out_path)

        n_frames = read_seqinfo(seq_path)["seq_len"]
        print(f"  {len(lines)} detections | "
              f"{fps:.1f} FPS | "
              f"{len(lines)/n_frames:.1f} dets/frame | "
              f"written {out_path.stat().st_size // 1024} KB ✓")

    elapsed = time.time() - total_start
    print(f"\nInference complete in {elapsed/60:.1f} min")
    print(f"Detection files: {config.DET_OUT_ROOT}")


if __name__ == "__main__":
    main()
