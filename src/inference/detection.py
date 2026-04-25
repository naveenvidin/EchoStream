"""YOLO-World open-vocabulary detector wrapped for the edge server.

Differences from the original wrapper:

- `warmup(h, w)` runs a dummy forward pass so the first real inference
  doesn't pay cuDNN autotune + CUDA graph build latency.
- `infer(frame_bgr)` does ONE bulk `.cpu().numpy()` extraction per
  tensor instead of a Python `.item()` per box — on scenes with 20+
  detections this is measurably cheaper and keeps the loop off the GIL
  for longer at a time.
- The heatmap is filled at a **configurable low resolution** (default
  80×60) instead of full frame size. Rectangles in frame coords are
  rescaled once via a single multiply. Bandwidth win: 640×480 heatmap =
  307 KB/reply; 80×60 = 4.7 KB/reply (64×).
- `infer(...)` returns `(metric, heatmap, detections, infer_us)` so the
  server can pass the timing through the protocol without calling the
  clock twice.
"""
from __future__ import annotations

import time
from typing import List, Sequence, Tuple

import numpy as np


Detection = Tuple[float, float, float, float, float, int]


def select_device(preference: str = "auto") -> str:
    """Resolve 'auto' into a concrete torch device string.

    Order: CUDA > MPS > CPU. Callers can force a specific device
    ("cuda", "mps", "cpu") to bypass auto-selection.
    """
    pref = (preference or "auto").lower()
    try:
        import torch
    except Exception:
        return "cpu"
    if pref != "auto":
        return pref
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class YoloWorldDetector:
    """Open-vocab YOLO-World wrapper.

    Typical usage:

        det = YoloWorldDetector(model_path="yolov8s-world.pt", device="auto")
        det.set_classes(["person", "forklift", "pallet"])
        det.set_heatmap_size(80, 60)
        det.warmup(height=480, width=640)
        metric, heatmap, detections, infer_us = det.infer(frame_bgr)
    """

    def __init__(self, model_path: str = "yolov8s-world.pt",
                 device: str = "auto", conf_threshold: float = 0.05,
                 heatmap_wh: Tuple[int, int] = (80, 60)):
        self.model_path = model_path
        self.device = select_device(device)
        self.conf_threshold = float(conf_threshold)
        self._class_names: List[str] = []
        self._heat_w = int(heatmap_wh[0]) or 80
        self._heat_h = int(heatmap_wh[1]) or 60

        from ultralytics import YOLOWorld  # noqa: WPS433
        self._model = YOLOWorld(model_path)
        try:
            self._model.to(self.device)
        except Exception:
            pass

    @property
    def class_names(self) -> List[str]:
        return list(self._class_names)

    @property
    def heatmap_size(self) -> Tuple[int, int]:
        return (self._heat_w, self._heat_h)

    def set_heatmap_size(self, width: int, height: int) -> None:
        self._heat_w = max(1, int(width))
        self._heat_h = max(1, int(height))

    def set_classes(self, names: Sequence[str]) -> None:
        cleaned = [n.strip() for n in names if n and n.strip()]
        if not cleaned:
            cleaned = ["object"]
        self._class_names = cleaned
        self._model.set_classes(cleaned)

    def warmup(self, height: int = 480, width: int = 640, runs: int = 2) -> None:
        """Flush first-run overhead — cuDNN autotune, graph build, etc."""
        if not self._class_names:
            self.set_classes(["object"])
        dummy = np.zeros((int(height), int(width), 3), dtype=np.uint8)
        for _ in range(max(1, runs)):
            try:
                self._model.predict(
                    dummy, verbose=False, device=self.device,
                    conf=self.conf_threshold,
                )
            except Exception:
                break

    def infer(self, frame_bgr: np.ndarray
              ) -> Tuple[float, np.ndarray, List[Detection], int]:
        """Run one forward pass and flatten output to the wire shape.

        Returns (metric, low_res_heatmap_uint8, detections, infer_us).
        """
        t0 = time.perf_counter()
        h, w = frame_bgr.shape[:2]
        heatmap = np.zeros((self._heat_h, self._heat_w), dtype=np.uint8)
        detections: List[Detection] = []

        results = self._model.predict(
            frame_bgr, verbose=False, device=self.device,
            conf=self.conf_threshold,
        )

        metric = 0.5
        if results and results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            # Single host-side transfer per tensor — avoids N Python .item() syncs.
            xyxy = boxes.xyxy.detach().cpu().numpy()       # (N, 4)
            confs = boxes.conf.detach().cpu().numpy()      # (N,)
            clses = boxes.cls.detach().cpu().numpy().astype(np.int32)  # (N,)

            # Map frame-space coords to heatmap-space once, vectorized.
            sx = self._heat_w / float(max(w, 1))
            sy = self._heat_h / float(max(h, 1))
            hx1 = np.clip(np.floor(xyxy[:, 0] * sx).astype(np.int32),
                          0, self._heat_w - 1)
            hy1 = np.clip(np.floor(xyxy[:, 1] * sy).astype(np.int32),
                          0, self._heat_h - 1)
            hx2 = np.clip(np.ceil(xyxy[:, 2] * sx).astype(np.int32),
                          1, self._heat_w)
            hy2 = np.clip(np.ceil(xyxy[:, 3] * sy).astype(np.int32),
                          1, self._heat_h)

            kept_confs: List[float] = []
            for i in range(xyxy.shape[0]):
                x1f, y1f, x2f, y2f = xyxy[i]
                # Frame-space clipping for the outgoing detection box.
                x1i = int(max(0, min(w - 1, int(x1f))))
                y1i = int(max(0, min(h - 1, int(y1f))))
                x2i = int(max(0, min(w, int(x2f))))
                y2i = int(max(0, min(h, int(y2f))))
                if x2i <= x1i or y2i <= y1i:
                    continue
                # Heatmap fill at low-res (already clamped above).
                if hx2[i] > hx1[i] and hy2[i] > hy1[i]:
                    heatmap[hy1[i]:hy2[i], hx1[i]:hx2[i]] = 255
                c = float(confs[i])
                kept_confs.append(c)
                detections.append(
                    (float(x1i), float(y1i), float(x2i), float(y2i),
                     c, int(clses[i]))
                )
            if kept_confs:
                # Use the minimum confidence so the camera's quality loop
                # is driven by the weakest detection in the frame.
                metric = float(min(kept_confs))

        infer_us = int((time.perf_counter() - t0) * 1_000_000)
        return float(metric), heatmap, detections, infer_us


def parse_classes(arg: str) -> List[str]:
    """Comma-separated prompt string → cleaned list of class names."""
    if not arg:
        return []
    return [c.strip() for c in arg.split(",") if c.strip()]
