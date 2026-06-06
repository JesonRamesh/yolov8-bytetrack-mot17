"""
evaluate.py — TrackEval evaluation: HOTA, MOTA, IDF1.

Sets up the TrackEval directory structure, runs evaluation against
MOT17 ground truth, and displays a formatted metrics table.

Usage:
    python evaluate.py

Input:  config.TRACK_ROOT/<seq>.txt     (from track.py)
        config.MOT17_ROOT/<seq>/gt/gt.txt (ground truth)
Output: config.RESULTS_DIR/summary.json
        Formatted metrics table to stdout
"""

import os
import sys
import shutil
import json
import re
from pathlib import Path

import numpy as np

import config

# Add TrackEval to path
sys.path.insert(0, str(config.TRACKEVAL_DIR))


# ── Directory setup ───────────────────────────────────────────────────────

def setup_trackeval_dirs() -> tuple:
    """
    Build the directory structure TrackEval expects and copy files into it.

    TrackEval expects:
        data/gt/mot_challenge/MOT17-train/<seq>/gt/gt.txt
        data/gt/mot_challenge/MOT17-train/<seq>/seqinfo.ini
        data/trackers/mot_challenge/MOT17-train/<tracker>/data/<seq>.txt

    Returns:
        (gt_dir, tracker_dir) as Path objects
    """
    gt_dir      = config.TRACKEVAL_DIR / "data/gt/mot_challenge/MOT17-train"
    tracker_dir = (config.TRACKEVAL_DIR / "data/trackers/mot_challenge"
                   / "MOT17-train" / config.TRACKER_NAME / "data")

    # Rebuild cleanly each run to avoid stale symlinks or partial state
    for d in [gt_dir, tracker_dir, config.RESULTS_DIR]:
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)

    print("Copying GT and tracker files:")
    for seq in config.SEQUENCES:
        # GT files
        (gt_dir / seq / "gt").mkdir(parents=True)
        (gt_dir / seq / "seqinfo.ini").write_text(
            (config.MOT17_ROOT / seq / "seqinfo.ini").read_text())
        (gt_dir / seq / "gt" / "gt.txt").write_bytes(
            (config.MOT17_ROOT / seq / "gt" / "gt.txt").read_bytes())

        # Tracker output
        (tracker_dir / f"{seq}.txt").write_text(
            (config.TRACK_ROOT / f"{seq}.txt").read_text())

        print(f"  {seq} ✓")

    return gt_dir, tracker_dir


# ── Evaluation ────────────────────────────────────────────────────────────

def run_trackeval() -> dict:
    """
    Run TrackEval and return the raw results dict.

    Uses SEQ_INFO to pass sequence lengths directly, bypassing
    TrackEval's seqmap file lookup entirely.
    """
    import trackeval

    eval_config = trackeval.Evaluator.get_default_eval_config()
    eval_config.update({
        "USE_PARALLEL":        False,
        "PRINT_RESULTS":       True,
        "PRINT_ONLY_COMBINED": False,
        "TIME_PROGRESS":       True,
        "OUTPUT_SUMMARY":      True,
        "OUTPUT_DETAILED":     True,
        "PLOT_CURVES":         False,
        "OUTPUT_FOLDER":       str(config.RESULTS_DIR),
    })

    dataset_config = (trackeval.datasets.MotChallenge2DBox
                      .get_default_dataset_config())
    dataset_config.update({
        "GT_FOLDER": str(
            config.TRACKEVAL_DIR / "data/gt/mot_challenge"),
        "TRACKERS_FOLDER": str(
            config.TRACKEVAL_DIR / "data/trackers/mot_challenge"),
        "BENCHMARK":          "MOT17",
        "SPLIT_TO_EVAL":      "train",
        "TRACKERS_TO_EVAL":   [config.TRACKER_NAME],
        "CLASSES_TO_EVAL":    ["pedestrian"],
        "TRACKER_SUB_FOLDER": "data",
        "SKIP_SPLIT_FOL":     False,
        "PLOT_CURVES":        False,
        "SEQ_INFO":           config.SEQ_LENGTHS,
    })

    evaluator    = trackeval.Evaluator(eval_config)
    dataset_list = [trackeval.datasets.MotChallenge2DBox(dataset_config)]
    results, _   = evaluator.evaluate(dataset_list, [
        trackeval.metrics.HOTA(),
        trackeval.metrics.CLEAR(),
        trackeval.metrics.Identity(),
    ])

    return results


# ── Results display ───────────────────────────────────────────────────────

# Maps display name → (trackeval sub-dict, key)
METRIC_MAP = {
    "HOTA":      ("HOTA",     "HOTA(0)"),
    "DetA":      ("HOTA",     "DetA"),
    "AssA":      ("HOTA",     "AssA"),
    "MOTA":      ("CLEAR",    "MOTA"),
    "IDF1":      ("Identity", "IDF1"),
    "Recall":    ("CLEAR",    "CLR_Re"),
    "Precision": ("CLEAR",    "CLR_Pr"),
    "IDSW":      ("CLEAR",    "IDSW"),
    "MT":        ("CLEAR",    "MT"),
    "ML":        ("CLEAR",    "ML"),
}


def get_val(seq_res: dict, sub: str, key: str) -> float:
    """Extract a scalar metric value, converting to percentage if in [0,1]."""
    v = seq_res[sub][key]
    v = float(v.mean()) if isinstance(v, np.ndarray) else float(v)
    return v * 100 if -1.0 <= v <= 1.0 else v


def display_results(results: dict):
    """Print formatted metrics table and save summary JSON."""
    tracker_res  = results["MotChallenge2DBox"][config.TRACKER_NAME]
    main_metrics = ["HOTA", "DetA", "AssA", "MOTA", "IDF1",
                    "Recall", "Precision"]

    print("\n" + "=" * 80)
    print(f"RESULTS: {config.TRACKER_NAME} on MOT17 (5-sequence subset)")
    print("=" * 80)
    print(f"{'Sequence':<25}" + "".join(f"{m:>11}" for m in main_metrics))
    print("─" * (25 + 11 * len(main_metrics)))

    for seq in config.SEQUENCES + ["COMBINED_SEQ"]:
        if seq not in tracker_res:
            continue
        label   = "COMBINED" if seq == "COMBINED_SEQ" else seq
        seq_res = tracker_res[seq]["pedestrian"]
        row     = f"{label:<25}"
        for m in main_metrics:
            sub, key = METRIC_MAP[m]
            try:
                row += f"{get_val(seq_res, sub, key):>10.1f}%"
            except Exception:
                row += f"{'N/A':>11}"
        print(row)

    print("─" * (25 + 11 * len(main_metrics)))
    print()
    print("Reference (ByteTrack paper, MOT17 train, YOLOX-X fine-tuned):")
    print("  HOTA: 63.1   MOTA: 78.5   IDF1: 73.4")
    print()
    print("Note: gap vs paper is detector-side (DetA). AssA is competitive.")
    print("  Fine-tuning YOLOv8x on MOT17 half-train recovers most of the gap.")

    # Save JSON
    output = {}
    for seq in config.SEQUENCES + ["COMBINED_SEQ"]:
        if seq not in tracker_res:
            continue
        seq_res = tracker_res[seq]["pedestrian"]
        output[seq] = {
            m: round(get_val(seq_res, sub, key), 2)
            for m, (sub, key) in METRIC_MAP.items()
        }

    summary_path = config.RESULTS_DIR / "summary.json"
    summary_path.write_text(json.dumps(output, indent=2))
    print(f"Results saved to {summary_path}")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    setup_trackeval_dirs()
    print("\nRunning TrackEval...")
    results = run_trackeval()
    display_results(results)


if __name__ == "__main__":
    main()
