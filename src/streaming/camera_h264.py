from __future__ import annotations

import argparse
import csv
import logging
import queue
import socket
import struct
import subprocess
import threading
import time
from collections import deque
from dataclasses import asdict
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from src.optical_flow.motion_masker import OpticalFlowMasker
from src.streaming.pid_controller import CrfPIDController, DiscreteStepCrfController


SERVER_IP = "localhost"
PORT = 9999
FIXED_CRF = None
WIDTH, HEIGHT = 640, 480
LOG_BANDWIDTH_EVERY_SEC = 10.0
INITIAL_CRF = 30

log = logging.getLogger("echostream.camera")

SHOW_YOLO_BOXES = True


class SegmentEncoder:
    """Encodes a list of raw BGR frames into a single H.264 segment via ffmpeg subprocess.
    GOP size matches FPS so each segment is exactly one keyframe-aligned group.
    """

    def __init__(self, width: int, height: int, fps: float, gop: int):
        self.width = width
        self.height = height
        # Frames-per-second for the rawvideo input to ffmpeg. This must match the
        # real capture cadence; tying it to GOP makes streams appear choppy/slow.
        self.fps = float(fps)
        self.gop = int(gop)
        log.info(
            "SegmentEncoder init width=%d height=%d fps=%.2f gop=%d",
            self.width, self.height, self.fps, self.gop,
        )

    def _build_cmd(self, crf: int) -> list[str]:
        return [
            "ffmpeg", "-loglevel", "quiet",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{self.width}x{self.height}",
            "-r", str(self.fps),
            "-i", "pipe:0",
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-tune", "zerolatency",
            "-crf", str(crf),
            "-g", str(self.gop),
            "-sc_threshold", "0",
            "-x264-params", "aq-mode=2:aq-strength=1.3",
            "-f", "h264",
            "pipe:1",
        ]

    def encode(self, frames: list[np.ndarray], crf: int) -> bytes:
        """Concatenate raw frames, pipe them through ffmpeg, and return the H.264 bytestream."""
        if not frames:
            return b""
        raw = b"".join(f.tobytes() for f in frames)
        proc = subprocess.Popen(
            self._build_cmd(crf),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
        )
        out, _ = proc.communicate(input=raw)
        return out


class ConfidenceListener:
    """Background thread that receives inference results from the server and
    derives the next CRF value from the reported confidence score.
    """

    # Cap how many resolved/pending segment_ids we retain so a long-running
    # session does not leak memory through the per-segment dicts.
    _SEGMENT_MEMORY = 256

    def __init__(self, sock: socket.socket, pid: CrfPIDController, counters=None):
        self.sock = sock
        self.counters = counters
        self._pid = pid
        self._current_crf = INITIAL_CRF
        self._next_crf = INITIAL_CRF
        self._crf_value = float(INITIAL_CRF)
        self._crf_integral = 0.0
        self._latest_conf = 0.5
        self._latest_heatmap = None
        self._latest_boxes = []
        self._latest_segment_id = -1
        self._responses = 0
        # Per-segment confidence tracking. _pending_segments holds segments
        # we have sent but not yet received a response for. _segment_confidences
        # is the resolved per-segment confidence indexed by segment_id.
        # _segment_events lets encode_and_send block briefly until the
        # response for a given segment_id arrives, so the conf logged in the
        # artifact row is the segment's actual conf and not a stale fallback.
        self._pending_segments: "dict[int, int]" = {}
        self._segment_confidences: "dict[int, float]" = {}
        self._segment_events: "dict[int, threading.Event]" = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._listen, daemon=True)
        self._thread.start()

    def _listen(self):
        """Receive inference result packets from the server in a loop.
        Each packet carries (segment_id, conf, heat_w, heat_h, num_boxes) +
        the heatmap and per-box payloads. The segment_id lets us attribute
        the response to the segment it came from rather than treating it as
        an undifferentiated "latest" update.
        """
        while not self._stop.is_set():
            try:
                header = _recv_exact(self.sock, 14)
                segment_id, conf, heat_w, heat_h, num_boxes = struct.unpack(
                    "!IfHHH", header,
                )
                if not np.isfinite(conf):
                    if self.counters is not None:
                        self.counters.record_invalid_response()
                    continue
                conf = max(0.0, min(1.0, conf))
                segment_id = int(segment_id)
                heat_w = int(heat_w)
                heat_h = int(heat_h)
                num_boxes = int(num_boxes)

                heat_bytes = _recv_exact(self.sock, heat_w * heat_h) if heat_w and heat_h else b""
                heatmap = None
                if heat_bytes and len(heat_bytes) == heat_w * heat_h:
                    heatmap = (
                        np.frombuffer(heat_bytes, dtype=np.uint8)
                        .reshape((heat_h, heat_w))
                        .astype(np.float32) / 255.0
                    )

                boxes = []
                if num_boxes:
                    box_bytes = _recv_exact(self.sock, num_boxes * 20)
                    for i in range(num_boxes):
                        off = i * 20
                        x1, y1, x2, y2, c = struct.unpack("!fffff", box_bytes[off:off + 20])
                        boxes.append((x1, y1, x2, y2, c))

                new_crf = self._pid.update(conf)
                event_to_set = None
                with self._lock:
                    frame_count = self._pending_segments.pop(segment_id, None)
                    self._segment_confidences[segment_id] = conf
                    # Trim old per-segment entries so the dicts can't grow
                    # unboundedly across a long run.
                    if len(self._segment_confidences) > self._SEGMENT_MEMORY:
                        oldest = min(self._segment_confidences)
                        self._segment_confidences.pop(oldest, None)
                    self._latest_conf = conf
                    self._latest_segment_id = int(segment_id)
                    self._segment_confidences[int(segment_id)] = conf
                    self._next_crf = new_crf
                    self._latest_heatmap = heatmap
                    self._latest_boxes = boxes
                    self._responses += 1
                    event_to_set = self._segment_events.pop(segment_id, None)
                if event_to_set is not None:
                    event_to_set.set()
                if self.counters is not None:
                    if frame_count is None:
                        self.counters.record_stale_response()
                    else:
                        self.counters.record_response_received(frame_count)
            except (ConnectionError, struct.error, OSError):
                break

    def stop(self):
        self._stop.set()

    def expect_segment(self, segment_id: int, frame_count: int) -> None:
        """Register that we have sent a segment and are waiting for its
        per-segment response. Idempotent for repeated calls with the same id.
        """
        frame_count = max(1, int(frame_count))
        with self._lock:
            self._pending_segments[int(segment_id)] = frame_count
            self._segment_events.setdefault(int(segment_id), threading.Event())
        if self.counters is not None:
            self.counters.record_response_expected(frame_count)

    def confidence_for_segment(self, segment_id: int,
                               default: "float | None" = None) -> float:
        """Return the per-segment confidence resolved for segment_id, or a
        fallback (default if given, else latest known conf) if the response
        for this segment has not yet arrived.
        """
        with self._lock:
            fallback = self._latest_conf if default is None else float(default)
            return float(
                self._segment_confidences.get(int(segment_id), fallback)
            )

    def wait_for_segment(self, segment_id: int, timeout: float) -> bool:
        """Block until the response for `segment_id` has been logged, or
        until `timeout` seconds elapse. Returns True if the segment was
        resolved within the timeout, False otherwise. Safe to call after
        the response has already arrived (returns True immediately).
        """
        with self._lock:
            if int(segment_id) in self._segment_confidences:
                return True
            event = self._segment_events.setdefault(int(segment_id), threading.Event())
        return bool(event.wait(timeout=float(timeout)))

    def get_next_crf(self) -> int:
        with self._lock:
            self._current_crf = self._next_crf
            return self._current_crf

    def latest_confidence(self) -> float:
        with self._lock:
            return float(self._latest_conf)

    def latest_object_score_map(self):
        with self._lock:
            return None if self._latest_heatmap is None else self._latest_heatmap.copy()

    def latest_boxes(self):
        with self._lock:
            return list(self._latest_boxes)

    def response_count(self) -> int:
        with self._lock:
            return int(self._responses)


def _recv_exact(sock: socket.socket, size: int) -> bytes:
    chunks, received = [], 0
    while received < size:
        chunk = sock.recv(size - received)
        if not chunk:
            raise ConnectionError("Socket closed while receiving.")
        chunks.append(chunk)
        received += len(chunk)
    return b"".join(chunks)


def estimate_baseline_bytes(frame: np.ndarray, crf: int) -> int:
    """Per-frame JPEG size estimator used as the denominator of the
    `[BW] saved=...%` log line.

    NOTE: this is *not* the same baseline as `[BW_BASE]`. This function
    encodes each raw frame as a standalone JPEG (no inter-frame
    compression) at a quality derived from the current PID-chosen CRF, so
    the denominator both lacks H.264's GOP gains and tracks the
    controller. The result is much larger than the actual H.264 baseline
    stream's bytes, which is why `[BW] saved` reads higher than the true
    EchoStream-vs-baseline ratio. Use `[BW_DELTA]` for the apples-to-
    apples comparison when the baseline stream is enabled.
    """
    jpeg_q = max(5, min(95, 100 - crf * 2))
    ok, enc = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_q])
    return len(enc) if ok else 0


def draw_hud(frame: np.ndarray, label: str, crf: int, conf: float,
             roi_ratio: float, sent_kb: float, mode: str) -> np.ndarray:
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 36), (0, 0, 0), -1)
    cv2.rectangle(overlay, (0, h - 48), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    quality_ratio = 1.0 - (crf - 18) / 33.0
    bar_w = int(w * quality_ratio)
    r = int(255 * (1.0 - quality_ratio))
    g = int(255 * quality_ratio)
    cv2.rectangle(frame, (0, 0), (bar_w, 4), (0, g, r), -1)

    cv2.putText(frame, label, (10, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    stats = f"CRF {crf}  Conf {conf:.2f}  ROI {roi_ratio*100:.0f}%  {sent_kb:.1f} KB  [{mode}]"
    cv2.putText(frame, stats, (10, h - 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 255, 120), 1, cv2.LINE_AA)
    return frame


def _open_input(spec: str, width: int, height: int):
    """Open a webcam index or video file path and return (cap, label, fps)."""
    is_index = spec.lstrip("-").isdigit()
    if is_index:
        idx = int(spec)
        cap = cv2.VideoCapture(idx)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, 15) # force a common FPS for webcams since many don't report it correctly; actual FPS may vary
        return cap, f"webcam:{idx}", 15
    cap = cv2.VideoCapture(spec)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {spec}")
    return cap, f"file:{spec}", cap.get(cv2.CAP_PROP_FPS)


def _decode_h264_segment(data: bytes, width: int, height: int) -> list[np.ndarray]:
    """Decode a raw H.264 bytestream into a list of BGR frames via ffmpeg.
    Used only for artifact logging, not the live display path.
    """
    if not data:
        return []
    cmd = [
        "ffmpeg", "-loglevel", "quiet",
        "-f", "h264",
        "-i", "pipe:0",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "pipe:1",
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    raw, _ = proc.communicate(input=data)
    frame_bytes = width * height * 3
    frames = []
    for off in range(0, len(raw) - frame_bytes + 1, frame_bytes):
        frames.append(
            np.frombuffer(raw[off:off + frame_bytes], dtype=np.uint8)
            .reshape((height, width, 3))
            .copy()
        )
    return frames


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="EchoStream compare camera (segmented H.264 + evaluator taps).",
    )
    p.add_argument("--config", default="configs/default.json",
                   help="JSON config file with defaults.")
    # Each override defaults to None / False so the JSON config's value is
    # only displaced when the flag is actually passed on the command line.
    p.add_argument("--input", default=None,
                   help="Override input source (webcam index or video path).")
    p.add_argument("--server-ip", dest="server_ip", default=None,
                   help="Override GPU server IP / hostname.")
    p.add_argument("--port", type=int, default=None,
                   help="Override server port.")
    p.add_argument("--width", type=int, default=None,
                   help="Override capture/encode width.")
    p.add_argument("--height", type=int, default=None,
                   help="Override capture/encode height.")
    p.add_argument("--classes", default=None,
                   help="Override prompted classes, e.g. 'person,wallet,bed'.")
    p.add_argument("--model", default=None,
                   help="Override model path recorded in session_config.json.")
    p.add_argument("--save-artifacts", dest="save_artifacts",
                   action="store_true",
                   help="Override save_artifacts=true (write run dir).")
    p.add_argument("--output-dir", dest="output_dir", default=None,
                   help="Override output directory for artifacts.")
    p.add_argument("--record-input", dest="record_input", default=None,
                   help="Override raw input recording path (webcam only).")
    p.add_argument("--record-input-fps", dest="record_input_fps",
                   type=float, default=None,
                   help="Override raw input recording fps.")
    p.add_argument("--record-input-max-frames", dest="record_input_max_frames",
                   type=int, default=None,
                   help="Override raw input max frames.")
    p.add_argument("--loop-video", dest="loop_video",
                   action="store_true",
                   help="Override loop_video=true for file inputs.")
    p.add_argument("--max-frames", dest="max_frames", type=int, default=None,
                   help="Override max_frames cap.")
    p.add_argument("--response-timeout-sec", dest="response_timeout_sec",
                   type=float, default=None,
                   help="Override response timeout (seconds).")
    p.add_argument("--no-preview", action="store_true",
                   help="Override no_preview=true (headless).")

    cli = p.parse_args()

    from src.common.config import load_json_config
    cfg = load_json_config(cli.config)
    block = cfg.get("camera_h264") if isinstance(cfg, dict) else None
    if not isinstance(block, dict):
        raise SystemExit(f"Missing camera_h264 section in config: {cli.config}")

    args = argparse.Namespace(**block)

    overrides = {
        "input": cli.input,
        "server_ip": cli.server_ip,
        "port": cli.port,
        "width": cli.width,
        "height": cli.height,
        "classes": cli.classes,
        "model": cli.model,
        "output_dir": cli.output_dir,
        "record_input": cli.record_input,
        "record_input_fps": cli.record_input_fps,
        "record_input_max_frames": cli.record_input_max_frames,
        "max_frames": cli.max_frames,
        "response_timeout_sec": cli.response_timeout_sec,
    }
    for key, value in overrides.items():
        if value is not None:
            setattr(args, key, value)
    if cli.save_artifacts:
        args.save_artifacts = True
    if cli.loop_video:
        args.loop_video = True
    if cli.no_preview:
        args.no_preview = True
    return args


def main():
    args = _parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    from src.eval.artifacts import SessionArtifacts, SessionConfig
    from src.eval.pipeline_counters import PipelineCounters
    from src.eval.recorded_input import RawInputRecorder
    from src.eval.video_metrics import write_summary
    from src.inference.detection import parse_classes

    classes = parse_classes(args.classes) or ["object"]
    cap, input_source, probed_fps = _open_input(args.input, args.width, args.height)
    fps = float(probed_fps)
    gop = fps//3 if fps >= 3 else 1 # SO IMPORTANT LIKE SO IMPORTANT

    # The "masked" socket carries the adaptive ROI-masked stream; this is the
    # original (and only) connection in the previous pipeline.
    masked_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    masked_socket.connect((args.server_ip, args.port))

    # Send the camera's actual FPS to the server immediately after connecting,
    # so the server can set its own FRAME_DURATION without relying on a hardcoded constant.
    masked_socket.sendall(struct.pack("!f", fps))
    log.info("sent FPS handshake to server: %.2f", fps)

    # Optional parallel "baseline" stream: a full-frame fixed-CRF copy sent to a
    # second server instance for A/B comparison against the masked stream.
    baseline_enabled = bool(getattr(args, "baseline_enabled", False))
    baseline_socket = None
    if baseline_enabled:
        baseline_port = int(getattr(args, "baseline_port", args.port))
        baseline_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        baseline_socket.connect((args.server_ip, baseline_port))
        baseline_socket.sendall(struct.pack("!f", fps))

    # -----set up shared state and background threads-----
    counters = PipelineCounters(expected_fps=fps)
    masker = OpticalFlowMasker(motion_threshold=3.0, min_contour_area=1200)
    encoder = SegmentEncoder(width=args.width, height=args.height, fps=fps, gop=int(gop))
    controller_type = str(getattr(args, "controller_type", "pid")).lower()

    def _build_controller():
        if controller_type == "discrete":
            return DiscreteStepCrfController(initial_crf=INITIAL_CRF)
        return CrfPIDController(
            target=float(getattr(args, "pid_target", 0.78)),
            kp=float(getattr(args, "pid_kp", 8.0)),
            ki=float(getattr(args, "pid_ki", 1.0)),
            kd=float(getattr(args, "pid_kd", 0.0)),
            max_step=float(getattr(args, "pid_max_step", 2.0)),
            deadband=float(getattr(args, "pid_deadband", 0.02)),
            crf_min=int(getattr(args, "pid_crf_min", 18)),
            crf_max=int(getattr(args, "pid_crf_max", 43)),
            initial_crf=INITIAL_CRF,
        )

    pid = _build_controller()
    if isinstance(pid, CrfPIDController):
        log.info(
            "PID adaptive CRF: target=%.2f kp=%.2f ki=%.2f kd=%.2f "
            "step=%.2f deadband=%.3f range=[%d,%d]",
            pid.target, pid.kp, pid.ki, pid.kd,
            pid.max_step, pid.deadband, pid.crf_min, pid.crf_max,
        )
        controller_meta = {
            "type": "pid",
            "initial_crf": INITIAL_CRF,
            "target": pid.target,
            "kp": pid.kp,
            "ki": pid.ki,
            "kd": pid.kd,
            "max_step": pid.max_step,
            "deadband": pid.deadband,
            "crf_min": pid.crf_min,
            "crf_max": pid.crf_max,
        }
    else:
        log.info(
            "Discrete-step CRF controller: levels=%s",
            getattr(pid, "LEVELS", []),
        )
        controller_meta = {
            "type": "discrete",
            "initial_crf": INITIAL_CRF,
            "levels": list(getattr(pid, "LEVELS", [])),
        }

    listener = ConfidenceListener(sock=masked_socket, pid=pid, counters=counters)
    baseline_listener = None
    if baseline_socket is not None:
        baseline_listener = ConfidenceListener(sock=baseline_socket, pid=_build_controller(), counters=None)

    run_dir = args.output_dir
    if args.save_artifacts and not run_dir:
        run_dir = str(Path("runs") / time.strftime("compare_%Y%m%d_%H%M%S"))
    artifacts = None
    if args.save_artifacts:
        artifacts = SessionArtifacts(
            run_dir=run_dir,
            width=args.width,
            height=args.height,
            fps=fps,
            enabled=True,
        )
        artifacts.write_session_config(SessionConfig(
            run_dir=run_dir,
            input_source=input_source,
            width=args.width,
            height=args.height,
            fps=fps,
            gop_size=gop,
            classes=classes,
            model=args.model,
            device="server",
            proactive_mode=False,
            max_frames=args.max_frames,
            loop_video=args.loop_video,
            seed=0,
            controller=controller_meta,
        ))

    # Sidecar CSV for the baseline (full-frame fixed-CRF) stream. One row per
    # encoded segment. Lives in the same run dir as metrics.csv so Streamlit can
    # overlay them on a common time axis. Only opened when both artifact saving
    # and the baseline stream are on; otherwise the writer stays None.
    baseline_csv_file = None
    baseline_csv_writer = None
    baseline_csv_lock = threading.Lock()
    baseline_csv_state = {"t0": None, "segment_index": 0, "cum_bytes": 0}
    if artifacts is not None and baseline_socket is not None:
        baseline_csv_path = artifacts.run_dir / "baseline_metrics.csv"
        try:
            baseline_csv_file = open(baseline_csv_path, "w", newline="")
            baseline_csv_writer = csv.DictWriter(
                baseline_csv_file,
                fieldnames=[
                    "segment_index", "wall_time_sec", "timestamp_sec",
                    "num_frames", "crf", "encoded_bytes",
                    "cumulative_encoded_bytes", "instantaneous_bitrate_bps",
                    "conf_metric",
                ],
            )
            baseline_csv_writer.writeheader()
            baseline_csv_file.flush()
            log.info("baseline metrics → %s", baseline_csv_path)
        except Exception as e:
            log.warning("baseline_metrics.csv open failed: %s", e)
            baseline_csv_file = None
            baseline_csv_writer = None

    # Optional: ship metrics.csv + baseline_metrics.csv + session_config.json
    # to a server-side mirror process so the server-hosted Streamlit can read
    # the comparison while the run is in progress. One-way TCP, best-effort.
    mirror_sender = None
    if artifacts is not None and bool(getattr(args, "mirror_enabled", False)):
        from src.eval.metrics_mirror import MetricsMirrorSender
        mirror_sender = MetricsMirrorSender(
            run_dir=artifacts.run_dir,
            host=str(getattr(args, "mirror_host", "localhost") or "localhost"),
            port=int(getattr(args, "mirror_port", 9997) or 9997),
        )
        mirror_sender.start()

    recorder = None
    recorded_input_meta = None
    if args.record_input and input_source.startswith("webcam:"):
        recorder = RawInputRecorder(
            output_path=args.record_input,
            width=args.width,
            height=args.height,
            capture_fps=float(args.record_input_fps or fps),
            input_source=input_source,
            max_frames=args.record_input_max_frames,
        )
    elif args.record_input:
        log.info("--record-input ignored for file input; use the file path for replay.")

    encode_queue = queue.Queue(maxsize=3)
    baseline_encode_queue = queue.Queue(maxsize=3) if baseline_socket is not None else None
    shared_lock = threading.Lock()
    last_raw = {"frame": None}
    last_masked = {"frame": None}
    last_roi = {"ratio": 0.0}
    last_sent_kb = {"kb": 0.0}
    last_crf = {"crf": INITIAL_CRF}
    last_baseline_sent_kb = {"kb": 0.0}
    last_baseline_crf = {"crf": int(getattr(args, "baseline_crf", 20) or 20)}
    shared_panel = {"frame": None}
    bw_lock = threading.Lock()
    bw_state = {"sent": 0, "baseline": 0}
    baseline_bw_state = {"sent": 0}
    mode = "Fixed" if FIXED_CRF is not None else "Adaptive"
    crf_transition_count = {"count": 0, "prev": None}

    log.info(
        "ready input=%s classes=%s preview=%s artifacts=%s",
        input_source, classes, "OFF" if args.no_preview else "ON",
        run_dir if artifacts else "OFF",
    )
    # ----- end of setup, start main loop and background threads -----

    def composer_loop():
        """Render the side-by-side raw+masked preview panel and push it to shared_panel
        for the main thread to display. Runs at the camera's native FPS.
        """
        while True:
            with shared_lock:
                raw = last_raw["frame"]
                masked = last_masked["frame"]
                roi = last_roi["ratio"]
                sent_kb = last_sent_kb["kb"]
                crf = last_crf["crf"]
                base_sent_kb = last_baseline_sent_kb["kb"]
                base_crf = last_baseline_crf["crf"]
            conf = listener.latest_confidence()
            boxes = listener.latest_boxes()
            base_conf = baseline_listener.latest_confidence() if baseline_listener is not None else None
            base_boxes = baseline_listener.latest_boxes() if baseline_listener is not None else []
            if raw is not None:
                raw_panel = raw.copy()
                if baseline_listener is not None:
                    # Left panel uses baseline server feedback when available.
                    draw_hud(
                        raw_panel,
                        "Baseline - full frame stream",
                        base_crf,
                        float(base_conf) if base_conf is not None else 0.0,
                        1.0,
                        base_sent_kb,
                        "Fixed",
                    )
                    if SHOW_YOLO_BOXES and base_boxes:
                        for (x1, y1, x2, y2, c) in base_boxes:
                            p1 = (int(x1), int(y1))
                            p2 = (int(x2), int(y2))
                            cv2.rectangle(raw_panel, p1, p2, (255, 255, 0), 2)
                            cv2.putText(
                                raw_panel,
                                f"person {c:.2f}",
                                (p1[0], max(0, p1[1] - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (255, 255, 0),
                                1,
                                cv2.LINE_AA,
                            )
                else:
                    draw_hud(raw_panel, "Source - raw camera feed",
                             crf, conf, roi, sent_kb, mode)
                if masked is not None:
                    masked_panel = masked.copy()
                    draw_hud(masked_panel, "Masked - ROI prepared stream",
                             crf, conf, roi, sent_kb, mode)
                    if SHOW_YOLO_BOXES:
                        for (x1, y1, x2, y2, c) in boxes:
                            p1 = (int(x1), int(y1))
                            p2 = (int(x2), int(y2))
                            cv2.rectangle(masked_panel, p1, p2, (0, 255, 255), 2)
                            cv2.putText(
                                masked_panel,
                                f"person {c:.2f}",
                                (p1[0], max(0, p1[1] - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (0, 255, 255),
                                1,
                                    cv2.LINE_AA,
                                )
                    temp_panel = cv2.hconcat([raw_panel, masked_panel])
                else:
                    temp_panel = raw_panel
                with shared_lock:
                    shared_panel["frame"] = temp_panel

            time.sleep(1.0 / max(fps, 1))

    threading.Thread(target=composer_loop, daemon=True).start()

    def encode_and_send(segment_id, segment_items, crf, baseline_total, transition_count):
        """Encode a completed GOP of masked frames and send it to the server
        with an 8-byte big-endian header (segment_id, payload_length). The
        segment_id is echoed back per-frame in the server's response so
        confidence values can be attributed to the segment that produced
        them rather than treated as undifferentiated "latest" updates.
        """
        try:
            frames = [item["masked"] for item in segment_items]
            raw_frames = [item["original"] for item in segment_items]
            t_enc = time.perf_counter()
            data = encoder.encode(frames, crf)
            encode_ms_total = (time.perf_counter() - t_enc) * 1000.0
            counters.record_encode(
                produced_packet=bool(data),
                frame_count=len(segment_items),
            )
            if not data:
                return

            header = struct.pack("!II", int(segment_id), len(data))
            listener.expect_segment(segment_id, len(segment_items))
            t_send = time.perf_counter()
            masked_socket.sendall(header + data)
            send_ms = (time.perf_counter() - t_send) * 1000.0

            with shared_lock:
                last_sent_kb["kb"] = len(data) / 1024
                last_crf["crf"] = crf
            with bw_lock:
                bw_state["sent"] += len(data)
                bw_state["baseline"] += baseline_total
                masked_bw_samples.append((time.time(), len(data)))

            if getattr(args, "measure_raw_bandwidth", False):
                # Expensive but accurate: encode raw frames with the same encoder settings.
                try:
                    raw_bytes = encoder.encode(raw_frames, crf)
                    with bw_lock:
                        raw_bw_samples.append((time.time(), len(raw_bytes)))
                except Exception:
                    pass

            if artifacts is None:
                return

            decoded_frames = _decode_h264_segment(data, args.width, args.height)
            per_frame_bytes = len(data) // max(len(segment_items), 1)
            remainder = len(data) - per_frame_bytes * len(segment_items)
            per_frame_encode_ms = encode_ms_total / max(len(segment_items), 1)
            # Block briefly so the segment's actual response has time to land
            # before we record its conf. Cap the wait so a stalled/dropped
            # response can't stall the encode worker.
            listener.wait_for_segment(segment_id, timeout=0.5)
            segment_conf = listener.confidence_for_segment(segment_id)
            for idx, item in enumerate(segment_items):
                encoded_bytes = per_frame_bytes + (1 if idx < remainder else 0)
                decoded = decoded_frames[idx] if idx < len(decoded_frames) else None
                artifacts.write_original(item["original"])
                artifacts.write_masked(item["masked"])
                artifacts.write_decoded(decoded)
                e2e_ms = (time.perf_counter() - item["loop_ts"]) * 1000.0
                artifacts.log_frame(
                    frame_index=item["frame_index"],
                    sequence_id=segment_id,
                    input_source=input_source,
                    roi_ratio=item["roi_ratio"],
                    conf_metric=segment_conf,
                    crf=crf,
                    encoded_bytes=encoded_bytes,
                    num_detections=0,
                    detected_classes=[],
                    restart_count=0,
                    source_timestamp_sec=item["source_ts"],
                    loop_timestamp_sec=item["loop_ts"],
                    capture_ok=True,
                    crf_transition_count=transition_count,
                    proactive_mode=False,
                    processing_time_ms=e2e_ms,
                    flow_ms=item["flow_ms"],
                    encode_ms=per_frame_encode_ms,
                    send_ms=send_ms if idx == len(segment_items) - 1 else None,
                    recv_ms=None,
                    parse_ms=None,
                    end_to_end_loop_ms=e2e_ms,
                    decode_us_server=None,
                    infer_us_server=None,
                    counter_snapshot=counters.snapshot_running(),
                )
        except Exception as e:
            log.warning("worker error: %s", e)

    def encode_and_send_baseline(segment_id, segment_items, crf):
        """Encode the unmodified frames at a fixed CRF and ship them to the
        parallel baseline server. Same 8-byte (segment_id, length) header
        the masked stream uses so a single server binary handles both. No-op
        when baseline mode is disabled.
        """
        if baseline_socket is None:
            return
        try:
            frames = [item["original"] for item in segment_items]
            data = encoder.encode(frames, crf)
            if not data:
                return
            header = struct.pack("!II", int(segment_id), len(data))
            if baseline_listener is not None:
                baseline_listener.expect_segment(segment_id, len(segment_items))
            baseline_socket.sendall(header + data)
            with shared_lock:
                last_baseline_sent_kb["kb"] = len(data) / 1024
                last_baseline_crf["crf"] = crf
            with bw_lock:
                baseline_bw_state["sent"] += len(data)

            # Sidecar per-segment row for Streamlit A/B overlay. Wait briefly
            # so the baseline conf logged here is for THIS segment, not the
            # previous one.
            if baseline_csv_writer is not None:
                wall_now = time.time()
                seg_dur = max(len(segment_items), 1) / max(float(fps), 1.0)
                inst_bps = (len(data) * 8.0) / seg_dur if seg_dur > 0 else 0.0
                seg_conf = 0.0
                if baseline_listener is not None:
                    baseline_listener.wait_for_segment(segment_id, timeout=0.5)
                    seg_conf = float(
                        baseline_listener.confidence_for_segment(segment_id)
                    )
                with baseline_csv_lock:
                    if baseline_csv_state["t0"] is None:
                        baseline_csv_state["t0"] = wall_now
                    baseline_csv_state["cum_bytes"] += len(data)
                    baseline_csv_state["segment_index"] += 1
                    row = {
                        "segment_index": baseline_csv_state["segment_index"],
                        "wall_time_sec": round(wall_now, 6),
                        "timestamp_sec": round(wall_now - baseline_csv_state["t0"], 6),
                        "num_frames": len(segment_items),
                        "crf": int(crf),
                        "encoded_bytes": len(data),
                        "cumulative_encoded_bytes": baseline_csv_state["cum_bytes"],
                        "instantaneous_bitrate_bps": round(inst_bps, 3),
                        "conf_metric": seg_conf,
                    }
                    try:
                        baseline_csv_writer.writerow(row)
                        baseline_csv_file.flush()
                    except Exception as e:
                        log.debug("baseline csv write failed: %s", e)
        except Exception as e:
            log.warning("baseline worker error: %s", e)

    def network_worker():
        """Drain the encode_queue in a dedicated thread so encoding never blocks
        the main capture loop. Sentinel None signals shutdown.
        """
        while True:
            item = encode_queue.get()
            if item is None:
                encode_queue.task_done()
                break
            encode_and_send(*item)
            encode_queue.task_done()

    worker = threading.Thread(target=network_worker, daemon=True)
    worker.start()

    baseline_worker = None
    if baseline_encode_queue is not None:
        def baseline_network_worker():
            while True:
                item = baseline_encode_queue.get()
                if item is None:
                    baseline_encode_queue.task_done()
                    break
                segment_id, segment_items, crf = item
                encode_and_send_baseline(segment_id, segment_items, crf)
                baseline_encode_queue.task_done()

        baseline_worker = threading.Thread(target=baseline_network_worker, daemon=True)
        baseline_worker.start()

    interval_start = time.time()
    stats_window_sec = 60.0
    baseline_conf_samples = deque()  # (ts, conf) from baseline (full-frame) server feedback
    masked_conf_samples = deque()    # (ts, conf)
    raw_bw_samples = deque()         # (ts, bytes)
    masked_bw_samples = deque()      # (ts, bytes)
    last_masked_response_count = 0
    last_baseline_response_count = 0
    segment_items = []
    segment_baseline = 0
    next_segment_id = 0
    current_segment_id: "int | None" = None
    current_segment_crf = None
    frame_index = 0
    baseline_crf = int(getattr(args, "baseline_crf", 20) or 20)

    try:
        while True:
            # -----logging and capture loop-----
            counters.record_loop_tick()
            if args.max_frames is not None and frame_index >= args.max_frames:
                break
            t_loop = time.perf_counter()
            ret, frame = cap.read()
            counters.record_capture_attempt(ok=bool(ret) and frame is not None)
            if not ret or frame is None:
                if args.loop_video and input_source.startswith("file:"):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break

            source_ts = None
            try:
                pos_ms = float(cap.get(cv2.CAP_PROP_POS_MSEC))
                source_ts = pos_ms / 1000.0 if pos_ms > 0 else None
            except Exception:
                source_ts = None

            try:
                frame = cv2.resize(frame, (args.width, args.height))
            except Exception as e:
                counters.record_processing_skip()
                log.debug("resize failed: %s", e)
                continue
            counters.record_processing_ok()

            if recorder is not None:
                recorder.write(frame)

            t_flow = time.perf_counter()
            object_score_map = listener.latest_object_score_map()
            masked_frame, roi_ratio = masker.apply(frame, object_score_map=object_score_map)
            flow_ms = (time.perf_counter() - t_flow) * 1000.0

            # Sample per-frame confidence from both streams whenever the server
            # delivers a new response. Used by the periodic [CONF] log.
            now = time.time()
            resp_count = listener.response_count()
            if resp_count != last_masked_response_count:
                last_masked_response_count = resp_count
                masked_conf_samples.append((now, float(listener.latest_confidence())))
            if baseline_listener is not None:
                bcnt = baseline_listener.response_count()
                if bcnt != last_baseline_response_count:
                    last_baseline_response_count = bcnt
                    baseline_conf_samples.append((now, float(baseline_listener.latest_confidence())))

            cutoff = now - stats_window_sec
            while baseline_conf_samples and baseline_conf_samples[0][0] < cutoff:
                baseline_conf_samples.popleft()
            while masked_conf_samples and masked_conf_samples[0][0] < cutoff:
                masked_conf_samples.popleft()

            with shared_lock:
                last_raw["frame"] = frame.copy()
                last_masked["frame"] = masked_frame.copy()
                last_roi["ratio"] = roi_ratio

            if not segment_items:
                current_segment_id = next_segment_id
                next_segment_id += 1
                current_segment_crf = FIXED_CRF if FIXED_CRF is not None else listener.get_next_crf()
                if crf_transition_count["prev"] is None:
                    crf_transition_count["prev"] = current_segment_crf
                elif current_segment_crf != crf_transition_count["prev"]:
                    crf_transition_count["count"] += 1
                    crf_transition_count["prev"] = current_segment_crf

            segment_baseline += estimate_baseline_bytes(frame, current_segment_crf)
            segment_items.append({
                "frame_index": frame_index,
                "original": frame.copy(),
                "masked": masked_frame.copy(),
                "roi_ratio": roi_ratio,
                "source_ts": source_ts,
                "loop_ts": t_loop,
                "flow_ms": flow_ms,
            })

            # Flush a full GOP to the encode queue once we've accumulated enough frames.
            if len(segment_items) >= gop: # CONFIG: fps = 1 second segments, gop = half-second
                try:
                    encode_queue.put_nowait((
                        current_segment_id,
                        segment_items,
                        current_segment_crf,
                        segment_baseline,
                        crf_transition_count["count"],
                    ))
                except queue.Full:
                    log.warning("drop segment")
                if baseline_encode_queue is not None:
                    try:
                        baseline_encode_queue.put_nowait((current_segment_id, segment_items, baseline_crf))
                    except queue.Full:
                        log.warning("drop baseline segment")
                segment_items = []
                segment_baseline = 0
                current_segment_id = None
                current_segment_crf = None

            if not args.no_preview:
                with shared_lock:
                    display_frame = shared_panel["frame"]
                if display_frame is not None:
                    cv2.imshow("Camera Node - Raw Feed", display_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if time.time() - interval_start >= LOG_BANDWIDTH_EVERY_SEC:
                with bw_lock:
                    sent = bw_state["sent"]
                    baseline = bw_state["baseline"]
                    bw_state["sent"] = bw_state["baseline"] = 0
                    base_sent = baseline_bw_state["sent"]
                    baseline_bw_state["sent"] = 0
                    cutoff = time.time() - stats_window_sec
                    while masked_bw_samples and masked_bw_samples[0][0] < cutoff:
                        masked_bw_samples.popleft()
                    while raw_bw_samples and raw_bw_samples[0][0] < cutoff:
                        raw_bw_samples.popleft()
                if baseline > 0:
                    # Denominator here is the synthetic per-frame JPEG
                    # estimator (estimate_baseline_bytes), NOT the parallel
                    # baseline H.264 stream. See [BW_DELTA] below for the
                    # apples-to-apples comparison against [BW_BASE].
                    log.info(
                        "[BW] sent=%.2fMB saved=%.1f%% vs per-frame JPEG estimator",
                        sent / 1024 / 1024,
                        (baseline - sent) / baseline * 100,
                    )
                if baseline_socket is not None:
                    log.info(
                        "[BW_BASE] sent=%.2fMB (full-frame fixed CRF=%d)",
                        base_sent / 1024 / 1024, baseline_crf,
                    )
                    # Apples-to-apples: actual EchoStream bytes vs the actual
                    # full-frame fixed-CRF baseline stream's bytes over the
                    # same window. This is the correct number for any
                    # "EchoStream saves X% vs baseline" claim.
                    if base_sent > 0:
                        log.info(
                            "[BW_DELTA] EchoStream=%.2fMB vs Baseline=%.2fMB "
                            "→ saved=%.1f%% vs baseline H.264",
                            sent / 1024 / 1024,
                            base_sent / 1024 / 1024,
                            (base_sent - sent) / base_sent * 100,
                        )

                def _stats(samples):
                    if not samples:
                        return {
                            "avg": 0.0, "p10": 0.0, "p50": 0.0, "p90": 0.0,
                            "pct_lt_05": 0.0, "n": 0,
                        }
                    vals = np.array([v for _t, v in samples], dtype=np.float32)
                    return {
                        "avg": float(vals.mean()),
                        "p10": float(np.percentile(vals, 10)),
                        "p50": float(np.percentile(vals, 50)),
                        "p90": float(np.percentile(vals, 90)),
                        "pct_lt_05": float(np.mean(vals < 0.5) * 100.0),
                        "n": int(vals.size),
                    }

                raw_s = _stats(baseline_conf_samples)
                masked_s = _stats(masked_conf_samples)
                keep_ratio = (masked_s["avg"] / raw_s["avg"]) if raw_s["avg"] > 1e-6 else 0.0
                extra = ""
                if getattr(args, "measure_raw_bandwidth", False) and raw_bw_samples and masked_bw_samples:
                    avg_raw_bytes = sum(v for _t, v in raw_bw_samples) / len(raw_bw_samples)
                    avg_masked_bytes = sum(v for _t, v in masked_bw_samples) / len(masked_bw_samples)
                    extra = f" | avg_raw_seg={avg_raw_bytes/1024:.1f}KB avg_masked_seg={avg_masked_bytes/1024:.1f}KB"

                log.info(
                    "[CONF] avg_raw=%.3f p10=%.3f p50=%.3f p90=%.3f lt0.5=%.1f%% | "
                    "avg_masked=%.3f p10=%.3f p50=%.3f p90=%.3f lt0.5=%.1f%% | "
                    "keep=%.2f samples(raw=%d masked=%d)%s",
                    raw_s["avg"], raw_s["p10"], raw_s["p50"], raw_s["p90"], raw_s["pct_lt_05"],
                    masked_s["avg"], masked_s["p10"], masked_s["p50"], masked_s["p90"], masked_s["pct_lt_05"],
                    keep_ratio,
                    raw_s["n"],
                    masked_s["n"],
                    extra,
                )
                interval_start = time.time()
            frame_index += 1
    finally:
        if segment_items:
            try:
                active_crf = FIXED_CRF if FIXED_CRF is not None else listener.get_next_crf()
                trailing_segment_id = (
                    current_segment_id
                    if current_segment_id is not None
                    else next_segment_id
                )
                encode_queue.put((
                    trailing_segment_id,
                    segment_items,
                    current_segment_crf,
                    segment_baseline,
                    crf_transition_count["count"],
                ), timeout=2.0)
                if baseline_encode_queue is not None:
                    baseline_encode_queue.put(
                        (trailing_segment_id, segment_items, baseline_crf),
                        timeout=2.0,
                    )
            except Exception:
                pass
        encode_queue.put(None)
        encode_queue.join()
        worker.join(timeout=5.0)
        if baseline_encode_queue is not None:
            baseline_encode_queue.put(None)
            baseline_encode_queue.join()
        if baseline_worker is not None:
            baseline_worker.join(timeout=5.0)
        listener.stop()
        if baseline_listener is not None:
            baseline_listener.stop()
        cap.release()
        try:
            masked_socket.close()
        except Exception:
            pass
        if baseline_socket is not None:
            try:
                baseline_socket.close()
            except Exception:
                pass
        if baseline_csv_file is not None:
            try:
                baseline_csv_file.flush()
                baseline_csv_file.close()
            except Exception:
                pass
        if mirror_sender is not None:
            try:
                mirror_sender.stop()
            except Exception:
                pass
        cv2.destroyAllWindows()

        if recorder is not None:
            try:
                recorded_input_meta = asdict(recorder.close())
            except Exception as e:
                log.warning("raw recorder close failed: %s", e)

        if artifacts is not None:
            counters.set_artifact_drops(artifacts.dropped_artifact_frames)
            counters.artifact_enqueue = artifacts.enqueued_artifact_frames
            artifacts.close()
            try:
                artifacts.write_session_config(SessionConfig(
                    run_dir=run_dir,
                    input_source=input_source,
                    width=args.width,
                    height=args.height,
                    fps=fps,
                    gop_size=gop,
                    classes=classes,
                    model=args.model,
                    device="server",
                    proactive_mode=False,
                    max_frames=args.max_frames,
                    loop_video=args.loop_video,
                    seed=0,
                    recorded_input=recorded_input_meta,
                    pipeline_health=counters.summary_dict(),
                    controller=controller_meta,
                ))
                summary = write_summary(str(artifacts.run_dir))
                log.info(
                    "artifacts saved to %s; live bitrate %.1fkbps",
                    artifacts.run_dir,
                    summary.adaptive_avg_bitrate_bps_live / 1000,
                )
            except Exception as e:
                log.warning("summary write failed: %s", e)

        log.info("camera shutdown complete.")


if __name__ == "__main__":
    main()
