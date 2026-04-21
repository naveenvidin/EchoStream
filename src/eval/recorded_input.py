"""Fixed-video recorder — capture-side tap for reproducible replay.

When the camera is driven by a live webcam, the real scene is
non-deterministic: lighting, motion, auto-exposure, USB jitter, and
the controller's own feedback all conspire to make two runs impossible
to compare directly. The recorder sits BEFORE masking/encoding, captures
the resized BGR frame, and writes it to an MP4 plus a sidecar JSON with
provenance data.

Replaying that MP4 later with ``--input <path>`` gives a deterministic
stream for benchmark comparisons. The SHA-256 in the sidecar lets an
auditor prove they replayed the exact same bytes.

Design notes
------------
- The recorder is only engaged when the source is a webcam AND
  ``--record-input`` is supplied. File inputs are already reproducible.
- An async writer thread owns the mp4 container so encode latency never
  gets charged for mp4 disk I/O.
- Pacing is implicit: webcam cap.read() already blocks at the sensor's
  native rate. We just write what we get.
"""
from __future__ import annotations

import hashlib
import json
import logging
import queue
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


log = logging.getLogger("echostream.recorder")


@dataclass
class RecordedInputMetadata:
    file_path: str
    sidecar_path: str
    capture_fps: float
    width: int
    height: int
    frame_count: int = 0
    start_wall_time: float = field(default_factory=time.time)
    end_wall_time: float = 0.0
    duration_sec: float = 0.0
    input_source: str = ""
    sha256: str = ""
    size_bytes: int = 0


def _compute_sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            buf = f.read(chunk)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


class RawInputRecorder:
    """Async capture-side recorder."""

    _MSG_FRAME = 0
    _MSG_STOP = 1

    def __init__(self, output_path: str, width: int, height: int,
                 capture_fps: float, input_source: str = "",
                 max_frames: Optional[int] = None,
                 queue_size: int = 128):
        self.enabled = True
        self.output_path = Path(output_path)
        self.sidecar_path = self.output_path.with_suffix(".json")
        self.width = int(width)
        self.height = int(height)
        self.capture_fps = float(capture_fps) if capture_fps and capture_fps > 0 else 30.0
        self.input_source = input_source
        self.max_frames = int(max_frames) if max_frames else None

        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(
            str(self.output_path), fourcc, self.capture_fps,
            (self.width, self.height),
        )
        if not self._writer.isOpened():
            self.enabled = False
            log.warning("raw recorder failed to open %s — disabling",
                        self.output_path)
            return

        self._queue: "queue.Queue" = queue.Queue(maxsize=queue_size)
        self._stop_event = threading.Event()
        self._frames_written = 0
        self._frames_enqueued = 0
        self._start_ts = time.time()
        self._end_ts = 0.0
        self._worker = threading.Thread(
            target=self._run_worker, name="echostream-raw-recorder", daemon=True,
        )
        self._worker.start()
        log.info("raw input recorder → %s (%dx%d @ %.2ffps, cap=%s)",
                 self.output_path, self.width, self.height, self.capture_fps,
                 self.max_frames if self.max_frames else "unbounded")

    def _run_worker(self) -> None:
        while True:
            msg_type, payload = self._queue.get()
            if msg_type == self._MSG_STOP:
                break
            if msg_type == self._MSG_FRAME and payload is not None:
                try:
                    self._writer.write(payload)
                    self._frames_written += 1
                except Exception as e:
                    log.debug("raw recorder write failed: %s", e)

    def write(self, frame: np.ndarray) -> None:
        """Enqueue one resized BGR frame. Non-blocking.

        The caller's buffer is not borrowed — we always hand the worker
        its own copy so the main thread can continue to draw HUD
        overlays on the same frame for the preview window without
        racing the mp4 writer.
        """
        if not self.enabled or self._stop_event.is_set():
            return
        if (self.max_frames is not None
                and self._frames_enqueued >= self.max_frames):
            # Reached cap — stop silently. close() flushes remaining queue.
            self._stop_event.set()
            return
        if frame.shape[0] != self.height or frame.shape[1] != self.width:
            # cv2.resize allocates a new buffer, so no extra copy needed.
            owned = cv2.resize(frame, (self.width, self.height))
        else:
            owned = frame.copy()
        try:
            self._queue.put_nowait((self._MSG_FRAME, owned))
            self._frames_enqueued += 1
        except queue.Full:
            # Recorder should never throttle capture. Drop silently;
            # the sidecar JSON's frame_count reveals the gap.
            log.debug("raw recorder queue full; dropping frame")

    def close(self) -> RecordedInputMetadata:
        """Flush, release, hash. Returns metadata — also written to sidecar."""
        if not self.enabled:
            return RecordedInputMetadata(
                file_path=str(self.output_path),
                sidecar_path=str(self.sidecar_path),
                capture_fps=self.capture_fps,
                width=self.width, height=self.height,
                input_source=self.input_source,
            )

        self._stop_event.set()
        try:
            self._queue.put((self._MSG_STOP, None), timeout=5.0)
        except Exception:
            pass
        if self._worker.is_alive():
            self._worker.join(timeout=10.0)

        try:
            self._writer.release()
        except Exception:
            pass

        self._end_ts = time.time()
        duration = max(0.0, self._end_ts - self._start_ts)

        size_bytes = 0
        sha = ""
        try:
            if self.output_path.is_file():
                size_bytes = self.output_path.stat().st_size
                sha = _compute_sha256(self.output_path)
        except Exception as e:
            log.warning("sha256 for recorded input failed: %s", e)

        meta = RecordedInputMetadata(
            file_path=str(self.output_path),
            sidecar_path=str(self.sidecar_path),
            capture_fps=self.capture_fps,
            width=self.width,
            height=self.height,
            frame_count=self._frames_written,
            start_wall_time=self._start_ts,
            end_wall_time=self._end_ts,
            duration_sec=duration,
            input_source=self.input_source,
            sha256=sha,
            size_bytes=size_bytes,
        )

        try:
            self.sidecar_path.write_text(
                json.dumps(asdict(meta), indent=2, default=str)
            )
        except Exception as e:
            log.warning("failed to write recorder sidecar: %s", e)

        log.info(
            "raw input recorded: %d frames, %.2fs, %.1f KB, sha256=%s",
            self._frames_written, duration, size_bytes / 1024,
            sha[:16] + "…" if sha else "—",
        )
        return meta
