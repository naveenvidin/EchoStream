"""Session artifact writer.

One `SessionArtifacts` instance owns:
- three MP4 writers (`original`, `masked`, `decoded_adaptive`),
- a per-frame metrics CSV,
- a session config JSON.

The previous implementation did all disk work on the camera thread.
Three `cv2.VideoWriter.write()` calls per frame meant the adaptive-
pipeline control loop was gated by disk throughput whenever
`--save-artifacts` was on — measured as a +6–9 ms frame budget hit on
a mid-range NVMe, more on spinning storage.

Now writes are enqueued to a background worker thread that owns the
writers. The queue is bounded; if the camera outruns the disk, we
*drop* the oldest mp4 frame and bump a counter (benchmark fidelity
prefers a dropped frame in the artifact over throttled encode
timing). CSV rows are never dropped — they're cheap.

Zero-overhead when `enabled=False` — all methods short-circuit.
"""
from __future__ import annotations

import csv
import json
import logging
import queue
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


log = logging.getLogger("echostream.artifacts")


METRIC_COLUMNS: List[str] = [
    # Identifiers
    "frame_index",
    "sequence_id",
    "timestamp_sec",          # monotonic, seconds since first logged frame
    "source_timestamp_sec",   # cap.get(CAP_PROP_POS_MSEC) scaled to seconds
    "loop_timestamp_sec",     # time.perf_counter at loop tick
    "input_source",
    "width",
    "height",
    "fps",
    # Health (running totals — rate = running / frame_index)
    "capture_ok",
    "capture_drop_count_running",
    "processing_skip_count_running",
    "encode_zero_packet_count_running",
    "response_timeout_count_running",
    "stale_response_count_running",
    "artifact_drop_count_running",
    # Observed FPS (moving window, see PipelineCounters.OBS_WINDOW_SEC)
    "expected_fps",
    "observed_input_fps",
    "observed_encoded_fps",
    "observed_response_fps",
    # Scene / codec state
    "roi_ratio",
    "conf_metric",
    "crf",
    "encoded_bytes",
    "cumulative_encoded_bytes",
    "instantaneous_bitrate_bps",
    "average_bitrate_bps",
    "num_detections",
    "detected_classes",
    "restart_count",
    "crf_transition_count",
    "proactive_mode",
    "processing_time_ms",
    # Per-phase camera-side timings
    "flow_ms",
    "encode_ms",
    "send_ms",
    "recv_ms",
    "parse_ms",
    "end_to_end_loop_ms",
    # Server-reported timings (extracted from detection response)
    "decode_us_server",
    "infer_us_server",
]


@dataclass
class SessionConfig:
    run_dir: str
    input_source: str
    width: int
    height: int
    fps: float
    gop_size: int
    classes: List[str]
    model: str
    device: str
    proactive_mode: bool
    max_frames: Optional[int]
    loop_video: bool
    seed: int
    heatmap_width: int = 0
    heatmap_height: int = 0
    controller: Dict[str, Any] = field(default_factory=dict)
    reproducibility: Dict[str, object] = field(default_factory=dict)
    # Environment + reproducibility trail
    environment: Dict[str, Any] = field(default_factory=dict)
    recorded_input: Optional[Dict[str, Any]] = None
    response_timeout_sec: float = 0.0
    # Final pipeline health counters (populated at close time; absent during
    # the initial write). summarize_run() reads this back from disk.
    pipeline_health: Dict[str, Any] = field(default_factory=dict)
    start_wall_time: float = field(default_factory=time.time)


class SessionArtifacts:
    """Write original/masked/adaptive mp4s + metrics.csv + session_config.json."""

    # Sentinel message types on the writer queue.
    _MSG_FRAME = 0
    _MSG_STOP = 1

    def __init__(self, run_dir: str, width: int, height: int, fps: float,
                 enabled: bool = True, queue_size: int = 128):
        self.enabled = bool(enabled)
        self.run_dir = Path(run_dir)
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps) if fps and fps > 0 else 30.0

        self._cum_bytes = 0
        self._start_t: Optional[float] = None
        self._prev_t: Optional[float] = None

        self._csv_file = None
        self._csv_writer: Optional[csv.DictWriter] = None
        self._writers: Dict[str, cv2.VideoWriter] = {}

        # Background writer plumbing.
        self._queue: "queue.Queue[Tuple[int, Any]]" = queue.Queue(maxsize=queue_size)
        self._worker: Optional[threading.Thread] = None
        self._dropped_frames = 0
        self._enqueue_count = 0
        self._dropped_lock = threading.Lock()

        if not self.enabled:
            return

        self.run_dir.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        for name in ("original", "masked", "decoded_adaptive"):
            path = str(self.run_dir / f"{name}.mp4")
            w = cv2.VideoWriter(path, fourcc, self.fps, (self.width, self.height))
            if not w.isOpened():
                log.warning("VideoWriter failed for %s (fourcc mp4v)", path)
            self._writers[name] = w

        csv_path = self.run_dir / "metrics.csv"
        self._csv_file = open(csv_path, "w", newline="")
        self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=METRIC_COLUMNS)
        self._csv_writer.writeheader()

        self._worker = threading.Thread(
            target=self._run_worker, name="echostream-artifacts", daemon=True,
        )
        self._worker.start()

        log.info("session artifacts → %s (async writer active)", self.run_dir)

    # ── Background worker ────────────────────────────────────────────────────

    def _run_worker(self) -> None:
        while True:
            try:
                msg_type, payload = self._queue.get()
            except Exception:
                break
            if msg_type == self._MSG_STOP:
                break
            if msg_type == self._MSG_FRAME:
                key, frame = payload
                w = self._writers.get(key)
                if w is None or frame is None:
                    continue
                try:
                    w.write(frame)
                except Exception as e:
                    log.debug("writer %s failed: %s", key, e)

    def _enqueue_frame(self, key: str, frame: np.ndarray) -> None:
        # Own the frame so the main thread can keep drawing HUD overlays
        # on the preview copy without racing the async writer worker.
        if frame.shape[0] != self.height or frame.shape[1] != self.width:
            frame = cv2.resize(frame, (self.width, self.height))
        else:
            frame = frame.copy()
        self._enqueue_count += 1
        try:
            self._queue.put_nowait((self._MSG_FRAME, (key, frame)))
        except queue.Full:
            # Drop the oldest mp4-frame message to keep the control loop
            # from stalling on disk. CSV rows are written directly on the
            # producer thread so they're never part of this queue.
            try:
                dropped = self._queue.get_nowait()
                with self._dropped_lock:
                    self._dropped_frames += 1
                # Re-enqueue the newer frame.
                self._queue.put_nowait((self._MSG_FRAME, (key, frame)))
            except queue.Empty:
                pass
            except queue.Full:
                with self._dropped_lock:
                    self._dropped_frames += 1

    # ── Videos ───────────────────────────────────────────────────────────────

    def write_original(self, frame: np.ndarray) -> None:
        if not self.enabled or frame is None:
            return
        self._enqueue_frame("original", frame)

    def write_masked(self, frame: np.ndarray) -> None:
        if not self.enabled or frame is None:
            return
        self._enqueue_frame("masked", frame)

    def write_decoded(self, frame: Optional[np.ndarray]) -> None:
        if not self.enabled or frame is None:
            return
        self._enqueue_frame("decoded_adaptive", frame)

    # ── Per-frame metrics row ────────────────────────────────────────────────

    def log_frame(self, *,
                  frame_index: int,
                  input_source: str,
                  roi_ratio: float,
                  conf_metric: float,
                  crf: int,
                  encoded_bytes: int,
                  num_detections: int,
                  detected_classes: Sequence[str],
                  restart_count: int,
                  sequence_id: Optional[int] = None,
                  source_timestamp_sec: Optional[float] = None,
                  loop_timestamp_sec: Optional[float] = None,
                  capture_ok: bool = True,
                  crf_transition_count: int = 0,
                  proactive_mode: bool = False,
                  processing_time_ms: Optional[float] = None,
                  flow_ms: Optional[float] = None,
                  encode_ms: Optional[float] = None,
                  send_ms: Optional[float] = None,
                  recv_ms: Optional[float] = None,
                  parse_ms: Optional[float] = None,
                  end_to_end_loop_ms: Optional[float] = None,
                  decode_us_server: Optional[int] = None,
                  infer_us_server: Optional[int] = None,
                  counter_snapshot: Optional[Dict[str, Any]] = None) -> None:
        if not self.enabled or self._csv_writer is None:
            return
        now = time.time()
        if self._start_t is None:
            self._start_t = now
            self._prev_t = now
        dt = max(now - (self._prev_t or now), 1e-6)
        ts = now - self._start_t
        self._cum_bytes += int(encoded_bytes or 0)
        inst_bps = (int(encoded_bytes or 0) * 8) / dt
        avg_bps = (self._cum_bytes * 8) / max(ts, 1e-6)
        self._prev_t = now

        def _opt_f(v: Optional[float]) -> str:
            return "" if v is None else f"{float(v):.3f}"

        def _opt_i(v: Optional[int]) -> str:
            return "" if v is None else str(int(v))

        snap = counter_snapshot or {}

        row = {
            "frame_index": int(frame_index),
            "sequence_id": _opt_i(sequence_id),
            "timestamp_sec": round(ts, 6),
            "source_timestamp_sec": _opt_f(source_timestamp_sec),
            "loop_timestamp_sec": _opt_f(loop_timestamp_sec),
            "input_source": input_source,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "capture_ok": int(bool(capture_ok)),
            "capture_drop_count_running":
                int(snap.get("capture_drop_count_running", 0)),
            "processing_skip_count_running":
                int(snap.get("processing_skip_count_running", 0)),
            "encode_zero_packet_count_running":
                int(snap.get("encode_zero_packet_count_running", 0)),
            "response_timeout_count_running":
                int(snap.get("response_timeout_count_running", 0)),
            "stale_response_count_running":
                int(snap.get("stale_response_count_running", 0)),
            "artifact_drop_count_running":
                int(snap.get("artifact_drop_count_running", 0)),
            "expected_fps": snap.get("expected_fps", self.fps),
            "observed_input_fps": snap.get("observed_input_fps", 0.0),
            "observed_encoded_fps": snap.get("observed_encoded_fps", 0.0),
            "observed_response_fps": snap.get("observed_response_fps", 0.0),
            "roi_ratio": round(float(roi_ratio), 6),
            "conf_metric": round(float(conf_metric), 6),
            "crf": int(crf),
            "encoded_bytes": int(encoded_bytes or 0),
            "cumulative_encoded_bytes": int(self._cum_bytes),
            "instantaneous_bitrate_bps": round(inst_bps, 3),
            "average_bitrate_bps": round(avg_bps, 3),
            "num_detections": int(num_detections),
            "detected_classes": ";".join(detected_classes),
            "restart_count": int(restart_count),
            "crf_transition_count": int(crf_transition_count),
            "proactive_mode": bool(proactive_mode),
            "processing_time_ms": _opt_f(processing_time_ms),
            "flow_ms": _opt_f(flow_ms),
            "encode_ms": _opt_f(encode_ms),
            "send_ms": _opt_f(send_ms),
            "recv_ms": _opt_f(recv_ms),
            "parse_ms": _opt_f(parse_ms),
            "end_to_end_loop_ms": _opt_f(end_to_end_loop_ms),
            "decode_us_server": _opt_i(decode_us_server),
            "infer_us_server": _opt_i(infer_us_server),
        }
        self._csv_writer.writerow(row)

    # ── Session config ───────────────────────────────────────────────────────

    def write_session_config(self, cfg: SessionConfig) -> None:
        if not self.enabled:
            return
        path = self.run_dir / "session_config.json"
        path.write_text(json.dumps(asdict(cfg), indent=2, default=str))

    @property
    def dropped_artifact_frames(self) -> int:
        with self._dropped_lock:
            return int(self._dropped_frames)

    @property
    def enqueued_artifact_frames(self) -> int:
        return int(self._enqueue_count)

    # ── Close ────────────────────────────────────────────────────────────────

    def close(self) -> None:
        if not self.enabled:
            return
        # Flush the queue, then signal the worker to exit.
        try:
            self._queue.put((self._MSG_STOP, None), timeout=5.0)
        except Exception:
            pass
        if self._worker is not None and self._worker.is_alive():
            self._worker.join(timeout=10.0)

        for w in self._writers.values():
            try:
                w.release()
            except Exception:
                pass
        self._writers.clear()

        if self._csv_file is not None:
            try:
                self._csv_file.flush()
                self._csv_file.close()
            except Exception:
                pass
            self._csv_file = None
            self._csv_writer = None

        dropped = self.dropped_artifact_frames
        if dropped:
            log.warning("dropped %d artifact mp4 frames under disk backpressure",
                        dropped)
