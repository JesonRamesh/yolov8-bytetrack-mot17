"""
track.py — ByteTrack multi-object tracking.

Self-contained implementation of ByteTrack (Zhang et al., ECCV 2022)
with no external tracker dependency. Uses scipy for Hungarian
assignment, which is compatible with all numpy versions.

Algorithm overview:
  Step 1 — high-conf detections ↔ active tracks    (IoU, match_thresh)
  Step 2 — low-conf detections  ↔ unmatched active  (IoU, 0.5)
  Step 3 — unmatched high-conf  ↔ lost tracks       (IoU, 0.5)
  Step 4 — remaining unmatched high-conf → new track (score ≥ new_thresh)

Usage:
    python track.py

Input:  config.DET_OUT_ROOT/<seq>.txt  (from detect.py)
Output: config.TRACK_ROOT/<seq>.txt    (MOT challenge track format)
"""

import numpy as np
import configparser
from pathlib import Path
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment

import config
from data import load_detections, read_seqinfo, write_tracks


# ═══════════════════════════════════════════════════════════════════════════
# Hungarian assignment
# ═══════════════════════════════════════════════════════════════════════════

def linear_assignment(cost: np.ndarray, thresh: float):
    """
    Solve linear assignment problem, ignoring pairs above cost threshold.

    Args:
        cost:   (M, N) cost matrix
        thresh: maximum allowable cost for a valid match

    Returns:
        matches:  list of [row, col] index pairs
        u_rows:   unmatched row indices
        u_cols:   unmatched column indices
    """
    if cost.size == 0:
        return [], list(range(cost.shape[0])), list(range(cost.shape[1]))

    cost_w = cost.copy()
    cost_w[cost_w > thresh] = thresh + 1e-4
    row_ind, col_ind = linear_sum_assignment(cost_w)

    matches, matched_r, matched_c = [], set(), set()
    for r, c in zip(row_ind, col_ind):
        if cost[r, c] <= thresh:
            matches.append([r, c])
            matched_r.add(r)
            matched_c.add(c)

    u_rows = [r for r in range(cost.shape[0]) if r not in matched_r]
    u_cols = [c for c in range(cost.shape[1]) if c not in matched_c]
    return matches, u_rows, u_cols


# ═══════════════════════════════════════════════════════════════════════════
# IoU distance
# ═══════════════════════════════════════════════════════════════════════════

def iou_distance(atracks: list, btracks: list) -> np.ndarray:
    """
    Compute pairwise 1-IoU cost matrix between two lists of STrack.

    Returns:
        (M, N) cost matrix where cost[i,j] = 1 - IoU(atracks[i], btracks[j])
    """
    if not atracks or not btracks:
        return np.zeros((len(atracks), len(btracks)))

    def to_xyxy(t):
        x, y, w, h = t.tlwh
        return [x, y, x + w, y + h]

    ab   = np.array([to_xyxy(t) for t in atracks])
    bb   = np.array([to_xyxy(t) for t in btracks])
    ious = np.zeros((len(ab), len(bb)))

    for i, a in enumerate(ab):
        xi1   = np.maximum(a[0], bb[:, 0])
        yi1   = np.maximum(a[1], bb[:, 1])
        xi2   = np.minimum(a[2], bb[:, 2])
        yi2   = np.minimum(a[3], bb[:, 3])
        inter = np.maximum(xi2 - xi1, 0) * np.maximum(yi2 - yi1, 0)
        aa    = (a[2] - a[0]) * (a[3] - a[1])
        ba    = (bb[:, 2] - bb[:, 0]) * (bb[:, 3] - bb[:, 1])
        ious[i] = inter / (aa + ba - inter + 1e-6)

    return 1 - ious


# ═══════════════════════════════════════════════════════════════════════════
# Kalman Filter  (constant velocity model in centre-aspect-height space)
# ═══════════════════════════════════════════════════════════════════════════

class KalmanFilter:
    """
    8-dimensional state: [cx, cy, aspect, h, vx, vy, va, vh]

    Measurement: [cx, cy, aspect, h]  (centre, aspect ratio, height)
    The aspect ratio is width/height, kept separate from height so
    the filter handles non-square boxes correctly.
    """

    def __init__(self):
        ndim, dt = 4, 1.
        self._F = np.eye(2 * ndim, 2 * ndim)          # state transition
        for i in range(ndim):
            self._F[i, ndim + i] = dt
        self._H = np.eye(ndim, 2 * ndim)               # measurement matrix
        self._std_pos = 1. / 20
        self._std_vel = 1. / 160

    def initiate(self, measurement: np.ndarray):
        mean = np.r_[measurement, np.zeros_like(measurement)]
        std  = [2 * self._std_pos * measurement[3],
                2 * self._std_pos * measurement[3],
                1e-2,
                2 * self._std_pos * measurement[3],
                10 * self._std_vel * measurement[3],
                10 * self._std_vel * measurement[3],
                1e-5,
                10 * self._std_vel * measurement[3]]
        return mean, np.diag(np.square(std))

    def predict(self, mean: np.ndarray, cov: np.ndarray):
        std = [self._std_pos * mean[3], self._std_pos * mean[3], 1e-2,
               self._std_pos * mean[3], self._std_vel * mean[3],
               self._std_vel * mean[3], 1e-5, self._std_vel * mean[3]]
        Q    = np.diag(np.square(std))
        mean = self._F @ mean
        cov  = self._F @ cov @ self._F.T + Q
        return mean, cov

    def update(self, mean: np.ndarray, cov: np.ndarray,
               measurement: np.ndarray):
        std = [self._std_pos * mean[3], self._std_pos * mean[3],
               1e-1, self._std_pos * mean[3]]
        R = np.diag(np.square(std))
        S = self._H @ cov @ self._H.T + R
        K = cov @ self._H.T @ np.linalg.inv(S)
        mean = mean + K @ (measurement - self._H @ mean)
        cov  = cov - K @ S @ K.T
        return mean, cov


# ═══════════════════════════════════════════════════════════════════════════
# Track state and STrack
# ═══════════════════════════════════════════════════════════════════════════

class TrackState:
    New = 0; Tracked = 1; Lost = 2; Removed = 3


class STrack:
    """Single object track with Kalman filter state."""

    _count = 0  # global ID counter, reset per sequence

    def __init__(self, tlwh: list, score: float):
        self._tlwh         = np.array(tlwh, dtype=float)
        self.score         = score
        self.is_activated  = False
        self.state         = TrackState.New
        self.mean          = None
        self.covariance    = None
        self.tracklet_len  = 0
        self.frame_id      = 0
        self.start_frame   = 0
        self.kalman_filter = None
        STrack._count     += 1
        self.track_id      = STrack._count

    @staticmethod
    def tlwh_to_xyah(tlwh: np.ndarray) -> np.ndarray:
        """Convert [x, y, w, h] → [cx, cy, aspect, h]."""
        ret = np.array(tlwh, dtype=float)
        ret[:2] += ret[2:] / 2
        ret[2]  /= ret[3]
        return ret

    @property
    def tlwh(self) -> np.ndarray:
        """Current bounding box as [x, y, w, h] (top-left, width, height)."""
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    def activate(self, kf: KalmanFilter, frame_id: int):
        self.kalman_filter = kf
        self.mean, self.covariance = kf.initiate(
            self.tlwh_to_xyah(self._tlwh))
        self.state        = TrackState.Tracked
        self.is_activated = True
        self.frame_id     = frame_id
        self.start_frame  = frame_id
        self.tracklet_len = 0

    def re_activate(self, new_track: "STrack", frame_id: int):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance,
            self.tlwh_to_xyah(new_track._tlwh))
        self.state        = TrackState.Tracked
        self.is_activated = True
        self.frame_id     = frame_id
        self.tracklet_len = 0
        self.score        = new_track.score

    def update(self, new_track: "STrack", frame_id: int):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance,
            self.tlwh_to_xyah(new_track._tlwh))
        self.state        = TrackState.Tracked
        self.is_activated = True
        self.frame_id     = frame_id
        self.tracklet_len += 1
        self.score        = new_track.score


# ═══════════════════════════════════════════════════════════════════════════
# ByteTracker
# ═══════════════════════════════════════════════════════════════════════════

class ByteTracker:
    """
    ByteTrack multi-object tracker.

    Implements the two-step association from Zhang et al. (ECCV 2022):
    low-confidence detections are used in a second association pass to
    recover partially occluded objects that would otherwise be lost.
    """

    def __init__(self,
                 track_high_thresh: float = config.TRACK_HIGH_THRESH,
                 track_low_thresh:  float = config.TRACK_LOW_THRESH,
                 new_track_thresh:  float = config.NEW_TRACK_THRESH,
                 track_buffer:      int   = config.TRACK_BUFFER,
                 match_thresh:      float = config.MATCH_THRESH):
        self.high_thresh  = track_high_thresh
        self.low_thresh   = track_low_thresh
        self.new_thresh   = new_track_thresh
        self.match_thresh = match_thresh
        self.buffer       = track_buffer
        self.kf           = KalmanFilter()
        self.tracked: list[STrack] = []
        self.lost:    list[STrack] = []
        self.frame_id = 0
        STrack._count = 0   # reset IDs at start of each sequence

    def update(self, dets: np.ndarray) -> list:
        """
        Process one frame of detections.

        Args:
            dets: (N, 5) array of [x1, y1, x2, y2, score]

        Returns:
            List of active STrack objects with valid track_id and tlwh.
        """
        self.frame_id += 1
        fid = self.frame_id

        scores = dets[:, 4] if len(dets) else np.array([])

        def make_stracks(d):
            return [STrack([r[0], r[1], r[2]-r[0], r[3]-r[1]], r[4])
                    for r in d]

        high = make_stracks(dets[scores >= self.high_thresh]) \
               if len(dets) else []
        low  = make_stracks(
               dets[(scores >= self.low_thresh) &
                    (scores <  self.high_thresh)]) if len(dets) else []

        # Predict all existing tracks forward one step
        for t in self.tracked + self.lost:
            t.mean, t.covariance = self.kf.predict(t.mean, t.covariance)

        active   = [t for t in self.tracked if t.is_activated]
        inactive = [t for t in self.tracked if not t.is_activated]
        pool     = active + inactive

        # Step 1: high-conf dets ↔ all tracked tracks
        m1, u_trk1, u_det1 = linear_assignment(
            iou_distance(pool, high), self.match_thresh)
        for ti, di in m1:
            pool[ti].update(high[di], fid)

        # Step 2: low-conf dets ↔ unmatched active tracks
        u_active = [active[i] for i in u_trk1 if i < len(active)]
        m2, u_trk2, _ = linear_assignment(iou_distance(u_active, low), 0.5)
        for ti, di in m2:
            u_active[ti].update(low[di], fid)

        newly_lost = [u_active[i] for i in u_trk2]
        for t in newly_lost:
            t.state = TrackState.Lost

        # Step 3: lost tracks ↔ remaining unmatched high-conf dets
        unmatched_high = [high[i] for i in u_det1]
        m3, _, u_det2  = linear_assignment(
            iou_distance(self.lost, unmatched_high), 0.5)
        for ti, di in m3:
            self.lost[ti].re_activate(unmatched_high[di], fid)

        # Step 4: initialise new tracks from remaining unmatched high-conf dets
        for i in u_det2:
            d = unmatched_high[i]
            if d.score >= self.new_thresh:
                d.activate(self.kf, fid)
                self.tracked.append(d)

        # Bookkeeping
        reactivated  = [self.lost[ti] for ti, _ in m3]
        self.lost    = ([t for t in self.lost if t not in reactivated]
                        + newly_lost)
        self.lost    = [t for t in self.lost
                        if fid - t.frame_id <= self.buffer]
        self.tracked = [t for t in self.tracked + reactivated
                        if t.state == TrackState.Tracked]

        return [t for t in self.tracked if t.is_activated]


# ═══════════════════════════════════════════════════════════════════════════
# Tracking loop
# ═══════════════════════════════════════════════════════════════════════════

def track_sequence(seq_name: str) -> list:
    """
    Run ByteTrack on one sequence.

    Args:
        seq_name: e.g. 'MOT17-02-SDP'

    Returns:
        List of MOT-format track strings.
    """
    seq_path = config.MOT17_ROOT / seq_name
    info     = read_seqinfo(seq_path)
    dets     = load_detections(config.DET_OUT_ROOT / f"{seq_name}.txt")

    tracker = ByteTracker(track_buffer=int(info["fps"]))
    lines   = []

    for fid in tqdm(range(1, info["seq_len"] + 1), desc=seq_name):
        fd = dets.get(fid, np.empty((0, 5), dtype=np.float32))
        for t in tracker.update(fd):
            x, y, w, h = t.tlwh
            lines.append(
                f"{fid},{t.track_id},{x:.2f},{y:.2f},{w:.2f},{h:.2f},"
                f"{t.score:.4f},-1,-1,-1"
            )

    return lines


def main():
    config.TRACK_ROOT.mkdir(parents=True, exist_ok=True)

    for seq_name in config.SEQUENCES:
        lines = track_sequence(seq_name)
        write_tracks(lines, config.TRACK_ROOT / f"{seq_name}.txt")
        ids = len({int(l.split(",")[1]) for l in lines})
        print(f"  {seq_name}: {ids} unique IDs, {len(lines)} track rows")

    print(f"\nTracking complete. Track files: {config.TRACK_ROOT}")


if __name__ == "__main__":
    main()
