from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


def _xyxy_to_cxcywh(box: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = box.astype(np.float32)
    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)
    cx = x1 + 0.5 * w
    cy = y1 + 0.5 * h
    return np.array([cx, cy, w, h], dtype=np.float32)


def _cxcywh_to_xyxy(state: np.ndarray) -> np.ndarray:
    cx, cy, w, h = state.astype(np.float32)
    w = max(1.0, w)
    h = max(1.0, h)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    if denom <= 0.0:
        return 0.0
    return float(inter / denom)


@dataclass
class TrackState:
    track_id: int
    kf: cv2.KalmanFilter
    bbox_xyxy: np.ndarray
    conf: float
    age: int = 0
    time_since_update: int = 0


class KalmanPersonTracker:
    """Tiny multi-person tracker (1-3 people) using Kalman + IoU matching.

    This is intentionally simpler than BoT-SORT/ByteTrack: no ReID, no
    appearance model, just temporal smoothing + short coasting.
    """

    def __init__(
        self,
        iou_threshold: float = 0.3,
        max_age: int = 10,
    ):
        self.iou_threshold = float(iou_threshold)
        self.max_age = int(max_age)
        self._next_id = 1
        self._tracks: list[TrackState] = []

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 1

    @staticmethod
    def _make_kf(initial_cxcywh: np.ndarray) -> cv2.KalmanFilter:
        kf = cv2.KalmanFilter(8, 4)

        # State: [cx, cy, w, h, vcx, vcy, vw, vh]
        # Meas:  [cx, cy, w, h]
        kf.transitionMatrix = np.array(
            [
                [1, 0, 0, 0, 1, 0, 0, 0],
                [0, 1, 0, 0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0, 0, 1, 0],
                [0, 0, 0, 1, 0, 0, 0, 1],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 1],
            ],
            dtype=np.float32,
        )
        kf.measurementMatrix = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
            ],
            dtype=np.float32,
        )

        # Noise tuning: conservative defaults for webcam-ish motion.
        kf.processNoiseCov = np.eye(8, dtype=np.float32) * 1e-2
        kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 1e-1
        kf.errorCovPost = np.eye(8, dtype=np.float32)

        kf.statePost = np.zeros((8, 1), dtype=np.float32)
        kf.statePost[0:4, 0] = initial_cxcywh
        return kf

    def _spawn_track(self, bbox_xyxy: np.ndarray, conf: float) -> None:
        meas = _xyxy_to_cxcywh(bbox_xyxy)
        kf = self._make_kf(meas)
        self._tracks.append(
            TrackState(
                track_id=self._next_id,
                kf=kf,
                bbox_xyxy=bbox_xyxy.astype(np.float32),
                conf=float(conf),
            )
        )
        self._next_id += 1

    def update(self, detections_xyxy_conf: list[tuple[np.ndarray, float]]) -> list[TrackState]:
        """Update tracker with detections for the current frame.

        Args:
            detections_xyxy_conf: list of (bbox_xyxy, conf).

        Returns:
            Active tracks after update.
        """
        # 1) Predict all tracks forward.
        for trk in self._tracks:
            trk.age += 1
            trk.time_since_update += 1
            pred = trk.kf.predict()
            trk.bbox_xyxy = _cxcywh_to_xyxy(pred[0:4, 0])

        # 2) Match detections -> tracks by IoU (greedy is fine for 1-3 people).
        unmatched_dets = set(range(len(detections_xyxy_conf)))
        unmatched_trks = set(range(len(self._tracks)))

        if detections_xyxy_conf and self._tracks:
            iou_mat = np.zeros((len(self._tracks), len(detections_xyxy_conf)), dtype=np.float32)
            for ti, trk in enumerate(self._tracks):
                for di, (det_box, _det_conf) in enumerate(detections_xyxy_conf):
                    iou_mat[ti, di] = iou_xyxy(trk.bbox_xyxy, det_box)

            while True:
                ti, di = np.unravel_index(int(iou_mat.argmax()), iou_mat.shape)
                best = float(iou_mat[ti, di])
                if best < self.iou_threshold:
                    break
                if ti in unmatched_trks and di in unmatched_dets:
                    unmatched_trks.remove(ti)
                    unmatched_dets.remove(di)

                    det_box, det_conf = detections_xyxy_conf[di]
                    meas = _xyxy_to_cxcywh(det_box)
                    self._tracks[ti].kf.correct(meas.reshape(4, 1))
                    post = self._tracks[ti].kf.statePost
                    self._tracks[ti].bbox_xyxy = _cxcywh_to_xyxy(post[0:4, 0])
                    self._tracks[ti].conf = float(det_conf)
                    self._tracks[ti].time_since_update = 0

                iou_mat[ti, :] = -1.0
                iou_mat[:, di] = -1.0

        # 3) Spawn new tracks for unmatched detections.
        for di in sorted(unmatched_dets):
            det_box, det_conf = detections_xyxy_conf[di]
            self._spawn_track(det_box, det_conf)

        # 4) Prune dead tracks.
        self._tracks = [t for t in self._tracks if t.time_since_update <= self.max_age]

        return list(self._tracks)
