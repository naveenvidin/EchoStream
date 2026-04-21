"""Live-stream health counters.

Tracks the six classes of event we need to answer the question
"is the pipeline keeping up?":

    capture          — did we get a frame from the source?
    processing       — did we successfully prepare it?
    encode           — did ffmpeg emit NALs?
    response         — did the server reply in time? with a matching seq?
    artifact         — did the mp4 writer keep up with disk?
    loop             — how long was the outer loop's tick (pause detector)?

Each group has an "attempt/success/drop" triple so the derived *rate*
can be computed later without knowing the run length. Counters are
cheap ints — the overhead is a handful of ns per frame.

Observed FPS is derived from a rolling timestamp list (capped at
``OBS_WINDOW_SEC``) so it is an *actual* moving rate, not the
start-to-now average. That matches the way a user asks "is it
dropping frames *right now*".
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Optional


# Only keep per-event timestamps for the last ~10 seconds — observed
# FPS computed from this window is responsive to stalls without
# being noisy.
OBS_WINDOW_SEC = 10.0

# Loop-gap threshold for the "long gap" counter. 100 ms ≈ 3 frames
# skipped at 30 fps, which a viewer notices as a pause.
LONG_LOOP_GAP_MS = 100.0


@dataclass
class PipelineCounters:
    """Aggregate health counters for one camera run.

    Call the ``record_*`` methods from the main loop. Call
    ``summary_dict()`` at shutdown to get a flat dict suitable for
    merging into ``summary.json``. Call ``snapshot_running()`` any time
    you want the current running totals for a metrics.csv row.
    """

    expected_fps: float = 30.0

    # ── Attempt / success / drop triples ────────────────────────────
    capture_attempt: int = 0
    capture_success: int = 0
    capture_drop: int = 0

    processing_attempt: int = 0
    processing_skip: int = 0

    encode_attempt: int = 0
    encode_zero_packet: int = 0

    response_expected: int = 0
    response_received: int = 0
    response_timeout: int = 0
    stale_response: int = 0
    invalid_response: int = 0

    artifact_enqueue: int = 0
    # Populated at close time from SessionArtifacts.dropped_artifact_frames
    artifact_drop: int = 0

    # ── Loop-gap (pause) detection ──────────────────────────────────
    long_loop_gap_count: int = 0
    max_loop_gap_ms: float = 0.0
    _loop_gap_sum_ms: float = 0.0
    _loop_gap_count: int = 0
    _last_loop_ts: Optional[float] = None

    # ── Observed-FPS rolling windows ────────────────────────────────
    _capture_ts: Deque[float] = field(default_factory=deque)
    _encode_ts: Deque[float] = field(default_factory=deque)
    _response_ts: Deque[float] = field(default_factory=deque)

    _start_ts: float = field(default_factory=time.perf_counter)

    # ── Capture ─────────────────────────────────────────────────────
    def record_capture_attempt(self, ok: bool) -> None:
        self.capture_attempt += 1
        if ok:
            self.capture_success += 1
            self._push_ts(self._capture_ts)
        else:
            self.capture_drop += 1

    def record_processing_skip(self) -> None:
        self.processing_attempt += 1
        self.processing_skip += 1

    def record_processing_ok(self) -> None:
        self.processing_attempt += 1

    # ── Encode ──────────────────────────────────────────────────────
    def record_encode(self, produced_packet: bool) -> None:
        self.encode_attempt += 1
        if produced_packet:
            self._push_ts(self._encode_ts)
        else:
            self.encode_zero_packet += 1

    # ── Response ────────────────────────────────────────────────────
    def record_response_expected(self) -> None:
        self.response_expected += 1

    def record_response_received(self) -> None:
        self.response_received += 1
        self._push_ts(self._response_ts)

    def record_response_timeout(self) -> None:
        self.response_timeout += 1

    def record_stale_response(self) -> None:
        self.stale_response += 1

    def record_invalid_response(self) -> None:
        self.invalid_response += 1

    # ── Artifact ────────────────────────────────────────────────────
    def record_artifact_enqueue(self) -> None:
        self.artifact_enqueue += 1

    def set_artifact_drops(self, n: int) -> None:
        self.artifact_drop = int(n)

    # ── Loop tick ───────────────────────────────────────────────────
    def record_loop_tick(self) -> None:
        now = time.perf_counter()
        if self._last_loop_ts is not None:
            gap_ms = (now - self._last_loop_ts) * 1000.0
            self._loop_gap_sum_ms += gap_ms
            self._loop_gap_count += 1
            if gap_ms > self.max_loop_gap_ms:
                self.max_loop_gap_ms = gap_ms
            if gap_ms > LONG_LOOP_GAP_MS:
                self.long_loop_gap_count += 1
        self._last_loop_ts = now

    # ── Rolling timestamp helpers ───────────────────────────────────
    def _push_ts(self, buf: Deque[float]) -> None:
        now = time.perf_counter()
        buf.append(now)
        cutoff = now - OBS_WINDOW_SEC
        while buf and buf[0] < cutoff:
            buf.popleft()

    @staticmethod
    def _fps_from_window(buf: Deque[float]) -> float:
        if len(buf) < 2:
            return 0.0
        span = buf[-1] - buf[0]
        if span <= 0:
            return 0.0
        return (len(buf) - 1) / span

    def observed_input_fps(self) -> float:
        return self._fps_from_window(self._capture_ts)

    def observed_encoded_fps(self) -> float:
        return self._fps_from_window(self._encode_ts)

    def observed_response_fps(self) -> float:
        return self._fps_from_window(self._response_ts)

    # ── Snapshots ───────────────────────────────────────────────────
    def snapshot_running(self) -> Dict[str, float]:
        """Running totals + instantaneous rates for metrics.csv."""
        return {
            "capture_drop_count_running": self.capture_drop,
            "processing_skip_count_running": self.processing_skip,
            "encode_zero_packet_count_running": self.encode_zero_packet,
            "response_timeout_count_running": self.response_timeout,
            "stale_response_count_running": self.stale_response,
            "artifact_drop_count_running": self.artifact_drop,
            "expected_fps": round(self.expected_fps, 3),
            "observed_input_fps": round(self.observed_input_fps(), 3),
            "observed_encoded_fps": round(self.observed_encoded_fps(), 3),
            "observed_response_fps": round(self.observed_response_fps(), 3),
        }

    def summary_dict(self) -> Dict[str, float]:
        """Flat dict suitable for summary.json merge."""
        total_loop_ticks = self._loop_gap_count
        mean_gap = (self._loop_gap_sum_ms / total_loop_ticks) if total_loop_ticks else 0.0

        def _rate(num: int, denom: int) -> float:
            return (num / denom) if denom > 0 else 0.0

        return {
            "expected_fps": self.expected_fps,
            # Capture
            "total_capture_attempt": self.capture_attempt,
            "total_frames_captured": self.capture_success,
            "total_capture_drops": self.capture_drop,
            "capture_drop_rate": _rate(self.capture_drop, self.capture_attempt),
            # Processing
            "total_processing_attempt": self.processing_attempt,
            "total_processing_skips": self.processing_skip,
            "processing_skip_rate": _rate(self.processing_skip, self.processing_attempt),
            # Encode
            "total_frames_encoded": self.encode_attempt - self.encode_zero_packet,
            "total_encode_zero_packet": self.encode_zero_packet,
            "encode_zero_packet_rate": _rate(self.encode_zero_packet, self.encode_attempt),
            # Response
            "total_frames_with_response": self.response_received,
            "total_response_timeouts": self.response_timeout,
            "total_stale_responses": self.stale_response,
            "total_invalid_responses": self.invalid_response,
            "response_timeout_rate": _rate(self.response_timeout, self.response_expected),
            "stale_response_rate": _rate(self.stale_response, self.response_received),
            # Artifact
            "total_artifact_drops": self.artifact_drop,
            "artifact_drop_rate": _rate(self.artifact_drop, self.artifact_enqueue),
            # Observed FPS (final moving-window values)
            "observed_input_fps": self.observed_input_fps(),
            "observed_encoded_fps": self.observed_encoded_fps(),
            "observed_response_fps": self.observed_response_fps(),
            # Loop / pause metrics
            "long_loop_gap_count": self.long_loop_gap_count,
            "max_loop_gap_ms": round(self.max_loop_gap_ms, 3),
            "mean_loop_gap_ms": round(mean_gap, 3),
        }
