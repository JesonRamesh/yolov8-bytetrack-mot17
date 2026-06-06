"""
data.py — MOT17 dataset download and loading utilities.

Handles:
  - Kaggle authentication and dataset download
  - Sequence metadata parsing (seqinfo.ini)
  - Frame path enumeration
  - Detection file loading (MOT format → numpy arrays)

Usage:
  python data.py                   # download MOT17
  from data import load_detections # use as library
"""

import os
import json
import subprocess
import configparser
from pathlib import Path

import numpy as np

import config


# ── Download ───────────────────────────────────────────────────────────────

def setup_kaggle_credentials(username: str, key: str):
    """Write kaggle.json credentials file."""
    os.makedirs("/root/.kaggle", exist_ok=True)
    with open("/root/.kaggle/kaggle.json", "w") as f:
        json.dump({"username": username, "key": key}, f)
    os.chmod("/root/.kaggle/kaggle.json", 0o600)


def download_mot17():
    """
    Download MOT17 from Kaggle and verify the expected sequences are present.

    Dataset: https://www.kaggle.com/datasets/wenhoujinjust/mot-17
    Requires KAGGLE_USERNAME and KAGGLE_KEY set in config.py.
    """
    if (config.MOT17_ROOT.exists()
            and any(config.MOT17_ROOT.iterdir())):
        print(f"MOT17 already present at {config.MOT17_ROOT}, skipping download.")
        _verify_sequences()
        return

    print("Setting up Kaggle credentials...")
    setup_kaggle_credentials(config.KAGGLE_USERNAME, config.KAGGLE_KEY)

    print("Downloading MOT17 from Kaggle (~5.5 GB)...")
    subprocess.run([
        "kaggle", "datasets", "download",
        "-d", "wenhoujinjust/mot-17",
        "-p", "/content/",
        "--unzip"
    ], check=True)

    print("Download complete.")
    _verify_sequences()


def _verify_sequences():
    """Print frame counts for all configured sequences."""
    print(f"\nSequence verification ({config.MOT17_ROOT}):")
    for seq in config.SEQUENCES:
        info   = read_seqinfo(config.MOT17_ROOT / seq)
        frames = get_frame_paths(config.MOT17_ROOT / seq, info)
        status = "✓" if len(frames) == info["seq_len"] else "✗ MISMATCH"
        print(f"  {seq}: {len(frames)}/{info['seq_len']} frames {status}")


# ── Sequence metadata ──────────────────────────────────────────────────────

def read_seqinfo(seq_path: Path) -> dict:
    """
    Parse seqinfo.ini and return a metadata dict.

    Returns:
        {
            name:    sequence name string
            seq_len: total frame count
            fps:     frame rate (float)
            im_dir:  subdirectory containing frames (usually 'img1')
            im_ext:  image extension (usually '.jpg')
            im_w:    image width in pixels
            im_h:    image height in pixels
        }
    """
    cfg = configparser.ConfigParser()
    cfg.read(seq_path / "seqinfo.ini")
    s = cfg["Sequence"]
    return {
        "name":    s["name"],
        "seq_len": int(s["seqLength"]),
        "fps":     float(s.get("frameRate", 30)),
        "im_dir":  s.get("imDir", "img1"),
        "im_ext":  s.get("imExt", ".jpg"),
        "im_w":    int(s.get("imWidth", 0)),
        "im_h":    int(s.get("imHeight", 0)),
    }


def get_frame_paths(seq_path: Path, info: dict) -> list:
    """Return sorted list of frame Paths for a sequence."""
    img_dir = seq_path / info["im_dir"]
    frames  = sorted(img_dir.glob(f"*{info['im_ext']}"))
    return frames


# ── Detection file I/O ────────────────────────────────────────────────────

def load_detections(det_file: Path) -> dict:
    """
    Load a MOT-format detection file into a per-frame dict.

    MOT detection format (per line):
        frame, -1, x, y, w, h, conf, -1, -1, -1

    Returns:
        dict mapping frame_id (int) → np.ndarray of shape (N, 5)
        where columns are [x1, y1, x2, y2, conf] in xyxy format.
    """
    d = {}
    for line in det_file.read_text().strip().splitlines():
        if not line.strip():
            continue
        p    = line.split(",")
        fid  = int(p[0])
        x1   = float(p[2])
        y1   = float(p[3])
        w    = float(p[4])
        h    = float(p[5])
        conf = float(p[6])
        d.setdefault(fid, []).append([x1, y1, x1 + w, y1 + h, conf])
    return {k: np.array(v, dtype=np.float32) for k, v in d.items()}


def write_detections(lines: list, out_path: Path):
    """Write detection lines to a MOT-format .txt file."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))


def write_tracks(lines: list, out_path: Path):
    """Write track lines to a MOT-format .txt file."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))


# ── CLI entry point ───────────────────────────────────────────────────────

if __name__ == "__main__":
    download_mot17()
