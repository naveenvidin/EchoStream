import argparse
import cv2
import logging
import os
import socket
import subprocess
import threading
import queue
import time
import numpy as np
from src.optical_flow.motion_masker import OpticalFlowMasker

# === CONFIGURATION ===
SERVER_IP = 'localhost'
PORT = 9999

FIXED_CRF = None          # Set an int (e.g. 28) to lock quality; None = adaptive
GOP_SIZE = 30             # Keyframe every N frames (1 sec at 30 fps)
LOG_BANDWIDTH_EVERY_SEC = 60
WIDTH, HEIGHT = 640, 480
SHOW_IMPORTANCE_WINDOW = False

log = logging.getLogger("echostream.camera")


# ─────────────────────────────────────────────
#  H.264 Encoder — persistent ffmpeg subprocess
# ─────────────────────────────────────────────
class H264Encoder:
    """
    Wraps a single long-lived ffmpeg process that reads raw BGR frames
    from stdin and writes H.264 NAL units to stdout using two threads:
    - Main thread calls encode(frame) to write raw frames into ffmpeg's stdin.
    - Background thread reads and drains ffmpeg's stdout and queues complete NAL units.
    Seperation is required for a constant flow of data without blocking 

    CRF changes require an encoder restart
    Restart takes ~5 ms and is only triggered when conf_score shifts meaningfully.
    """
    def __init__(self, width=640, height=480, crf=28, fps=30, gop=30):
        self.width = width
        self.height = height
        self.fps = fps
        self.gop = gop
        self._crf = crf
        self._lock = threading.Lock()
        self._out_q = queue.Queue()
        self._proc = None
        self._reader_thread = None
        self._force_keyframe = False # behind lock for proper timing
        self._start(crf)

    def _build_cmd(self, crf):
        return [
            'ffmpeg', '-loglevel', 'quiet',
            '-f', 'rawvideo',
            '-pix_fmt', 'bgr24',
            '-s', f'{self.width}x{self.height}',
            '-r', str(self.fps),
            '-i', 'pipe:0',
            '-c:v', 'libx264',
            '-preset', 'ultrafast',
            '-tune', 'zerolatency',
            '-crf', str(crf),
            '-g', str(self.gop),
            '-sc_threshold', '0',
            '-f', 'h264',
            'pipe:1',
        ]

    def _start(self, crf):
        self._proc = subprocess.Popen(
            # main thread
            self._build_cmd(crf),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
        )

        # reader thread to drain stdout and prevent OS pipe buffer blocking
        self._reader_thread = threading.Thread(
            target=self._drain_stdout, daemon=True
        )
        self._reader_thread.start()

    def _drain_stdout(self):
        # on reader thread
        buf = b''
        START = b'\x00\x00\x00\x01' # NAL unit start code
        while True:
            chunk = self._proc.stdout.read(8192) # arbitrary chunk size
            if not chunk:
                # EOF reached or process terminated
                break
            buf += chunk
            while True:
                idx = buf.find(START, 4)
                if idx == -1:
                    # if can't find next start, current hasn't finished
                    break
                self._out_q.put(buf[:idx])
                buf = buf[idx:] # might skip frames?
        if buf:
            self._out_q.put(buf) # add to q

    def _restart(self, crf):
        # kill immediately and start new process with new CRF; drop any queued NALs from old CRF
        self._proc.kill()
        while not self._out_q.empty():
            self._out_q.get_nowait()
        self._start(crf)

    def encode(self, frame: np.ndarray) -> bytes:
        # returns available NAL units from previous frame(s) as bytes; non-blocking
        with self._lock:
            if self._force_keyframe:
                self._restart(self._crf)
                self._force_keyframe = False
        try:
            self._proc.stdin.write(frame.tobytes())
            self._proc.stdin.flush()
        except BrokenPipeError:
            with self._lock:
                self._restart(self._crf)
            return b''
        nals = []
        while not self._out_q.empty():
            nals.append(self._out_q.get_nowait())
        return b''.join(nals)

    def set_crf(self, crf: int, force_keyframe: bool = False):
        crf = max(18, min(51, crf))
        absolute_change = abs(crf - self._crf)
        with self._lock:
            if absolute_change > 4 or force_keyframe:
                self._crf = crf
                self._force_keyframe = True

    def close(self):
        try:
            self._proc.stdin.close()
            self._proc.wait(timeout=3)
        except Exception:
            self._proc.kill()


# ─────────────────────────────────────────────
#  H.264 Decoder — local preview of encoded stream
# ─────────────────────────────────────────────
class H264Decoder:
    """
    Local decoder that consumes the same NAL bytes sent to the server.
    This lets the dashboard render exactly what the server sees —
    compression artefacts, masking, and CRF effects all included.

    push()      — feed NAL bytes in (non-blocking, call after encode())
    get_frame() — pull latest decoded BGR frame, or None if not ready
    """

    def __init__(self, width=640, height=480):
        self.width = width
        self.height = height
        self._frame_bytes = width * height * 3
        self._frame_q = queue.Queue(maxsize=4)
        self._proc = None
        self._reader_thread = None
        self.drops = 0
        self._start()

    def _start(self):
        cmd = [
            'ffmpeg', '-loglevel', 'quiet',
            '-f', 'h264',
            '-i', 'pipe:0',
            '-f', 'rawvideo',
            '-pix_fmt', 'bgr24',
            'pipe:1',
        ]
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
        )
        self._reader_thread = threading.Thread(
            target=self._drain_stdout, daemon=True
        )
        self._reader_thread.start()

    def _drain_stdout(self):
        # on reader thread
        while True:
            raw = self._proc.stdout.read(self._frame_bytes)
            if len(raw) < self._frame_bytes:
                break # EOF or process terminated
            frame = np.frombuffer(raw, dtype=np.uint8).reshape(
                (self.height, self.width, 3)
            ).copy()

            # Drop oldest if full — we prefer latency over backpressure // needs review
            if self._frame_q.full():
                try:
                    self._frame_q.get_nowait()
                    self.drops += 1
                except queue.Empty:
                    pass
            try:
                self._frame_q.put_nowait(frame)
            except queue.Full:
                pass

    def push(self, nal_bytes: bytes):
        """Feed NAL bytes into the decoder (non-blocking)."""
        try:
            self._proc.stdin.write(nal_bytes)
            self._proc.stdin.flush()
        except BrokenPipeError:
            pass

    def get_frame(self):
        """Return latest decoded BGR frame, or None if not ready yet."""
        try:
            return self._frame_q.get(timeout=0.03) # needs review
        except queue.Empty:
            return None

    def close(self):
        try:
            self._proc.stdin.close()
            self._proc.wait(timeout=3)
        except Exception:
            self._proc.kill()


# ─────────────────────────────────────────────
#  CRF ↔ confidence mapping
# ─────────────────────────────────────────────
def conf_to_crf(conf: float) -> int:
    if FIXED_CRF is not None:
        return FIXED_CRF
    # Non-linear: compress hard when YOLO is confident, send high quality when struggling
    if conf >= 0.8:
        return 42
    elif conf >= 0.5:
        return int(28 + (conf - 0.5) / 0.3 * 14)
    else:
        return int(18 + (conf / 0.5) * 10)


# ─────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────
def estimate_baseline_bytes(frame: np.ndarray, crf: int) -> int:
    jpeg_q = max(5, min(95, 100 - crf * 2))
    ok, enc = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_q])
    return len(enc) if ok else 0


def draw_hud(frame: np.ndarray, label: str, crf: int, conf: float,
             roi_ratio: float, sent_kb: float, mode: str) -> np.ndarray:
    """
    Draw a semi-transparent HUD bar at the top and bottom of the frame.
    Top bar: panel label.
    Bottom bar: live stats.
    Also draws a CRF quality bar along the top edge (green=quality, red=compressed).
    """
    h, w = frame.shape[:2]

    # Top bar background
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 36), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    # Bottom bar background
    overlay2 = frame.copy()
    cv2.rectangle(overlay2, (0, h - 48), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay2, 0.5, frame, 0.5, 0, frame)

    # CRF quality colour bar — runs across very top of frame (4px tall)
    # Width proportional to quality: full width = CRF 18 (best), zero = CRF 51 (worst)
    quality_ratio = 1.0 - (crf - 18) / 33.0
    bar_w = int(w * quality_ratio)
    r = int(255 * (1.0 - quality_ratio))
    g = int(255 * quality_ratio)
    cv2.rectangle(frame, (0, 0), (bar_w, 4), (0, g, r), -1)

    # Panel label
    cv2.putText(frame, label, (10, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    # Stats line
    stats = f"CRF {crf}  Conf {conf:.2f}  ROI {roi_ratio*100:.0f}%  {sent_kb:.1f} KB  [{mode}]"
    cv2.putText(frame, stats, (10, h - 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 255, 120), 1, cv2.LINE_AA)

    return frame


# ─────────────────────────────────────────────
#  Input source (webcam / video file)
# ─────────────────────────────────────────────
def _open_input(spec: str, width: int, height: int):
    """Open either an integer webcam index or a file path.

    Returns (cv2.VideoCapture, input_source_label, probed_fps).
    `input_source_label` is a human-readable string for logs/CSV.
    """
    is_index = spec.lstrip("-").isdigit()
    if is_index:
        idx = int(spec)
        cap = cv2.VideoCapture(idx)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        return cap, f"webcam:{idx}", (cap.get(cv2.CAP_PROP_FPS) or 30.0)
    cap = cv2.VideoCapture(spec)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {spec}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    return cap, f"file:{spec}", fps


# ─────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="EchoStream camera/edge node (H.264 + optical-flow masking).",
    )
    p.add_argument("--input", default="0",
                   help="Webcam index (e.g. '0') or path to a video file. "
                        "Short prompts like 'person,wallet,bed' are preferred over "
                        "long natural-language prompts for YOLO-World reliability.")
    p.add_argument("--loop-video", action="store_true",
                   help="Rewind to frame 0 on EOF (file inputs only).")
    p.add_argument("--max-frames", type=int, default=None,
                   help="Stop after processing N captured frames.")
    p.add_argument("--server-ip", default=SERVER_IP)
    p.add_argument("--port", type=int, default=PORT)
    p.add_argument("--width", type=int, default=WIDTH)
    p.add_argument("--height", type=int, default=HEIGHT)
    p.add_argument("--gop", type=int, default=GOP_SIZE)
    p.add_argument("--fps", type=float, default=30.0,
                   help="Encoder fps (also used for artifact video writers).")
    p.add_argument("--classes", default="person",
                   help="Comma-separated prompted classes for YOLO-World. Short "
                        "category labels (e.g. 'person,wallet,bed') are the most "
                        "reliable; longer natural-language prompts are accepted "
                        "but quality degrades.")
    p.add_argument("--output-dir", default=None,
                   help="Where to write benchmark artifacts when --save-artifacts.")
    p.add_argument("--save-artifacts", action="store_true",
                   help="Write original/masked/decoded_adaptive mp4s + metrics.csv.")
    p.add_argument("--no-preview", action="store_true",
                   help="Disable the GUI dashboard (headless / CI runs).")
    p.add_argument("--seed", type=int, default=0,
                   help="Deterministic seed for Python/NumPy/Torch.")
    p.add_argument("--strict-determinism", action="store_true",
                   help="Enable torch.use_deterministic_algorithms(True).")
    p.add_argument("--model", default="yolov8s-world.pt",
                   help="YOLO-World model (logged; actually loaded by server).")
    # ── Fixed-video recording (webcam only) ─────────────────────────────────
    p.add_argument("--record-input", default=None,
                   help="When the source is a webcam, tap the post-resize BGR "
                        "frames into this MP4 before masking/encoding. "
                        "Replay later with --input <path> for reproducible eval.")
    p.add_argument("--record-input-fps", type=float, default=None,
                   help="Override capture fps written into the recorded mp4 "
                        "container (default: probed webcam fps).")
    p.add_argument("--record-input-max-frames", type=int, default=None,
                   help="Stop recording after N frames (the main run continues).")
    # ── Network timing ──────────────────────────────────────────────────────
    p.add_argument("--response-timeout-sec", type=float, default=2.0,
                   help="Give up on a server reply after N seconds; counted as "
                        "a response_timeout.")
    return p.parse_args()


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────
def main():
    args = _parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    from src.codec import wire
    from src.codec.h264_backend import H264EncoderBackend, H264DecoderBackend
    from src.inference.detection import parse_classes
    from src.inference.protocol import (
        DEFAULT_HEATMAP_WH, pack_handshake, read_response,
    )
    from src.streaming.controller import AdaptiveCRFController
    from src.utils.reproducibility import set_seeds
    from src.eval.artifacts import SessionArtifacts, SessionConfig
    from src.eval.video_metrics import write_summary
    from src.eval.pipeline_counters import PipelineCounters
    from src.eval.recorded_input import RawInputRecorder
    from src.eval.environment import collect_environment
    from dataclasses import asdict

    repro = set_seeds(args.seed, strict=args.strict_determinism)
    log.info("determinism: %s", repro)

    classes = parse_classes(args.classes) or ["object"]
    log.info("prompted classes=%s", classes)
    heat_wh = DEFAULT_HEATMAP_WH

    cap, input_source, probed_fps = _open_input(args.input, args.width, args.height)
    fps = args.fps or probed_fps or 30.0
    probed_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    probed_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    log.info(
        "input=%s probed=%dx%d @ %.2ffps frames=%d (loop=%s, max=%s)",
        input_source, probed_w, probed_h, fps, frame_count,
        args.loop_video, args.max_frames,
    )

    masker = OpticalFlowMasker(motion_threshold=3.0, min_contour_area=1200)
    encoder = H264EncoderBackend(
        width=args.width, height=args.height, fps=int(round(fps)), gop=args.gop,
    )
    local_decoder = H264DecoderBackend(width=args.width, height=args.height)

    # ── Adaptive CRF controller ──────────────────────────────────────────────
    proactive_mode = os.environ.get("ECHOSTREAM_PROACTIVE", "0") == "1"
    controller = AdaptiveCRFController(
        initial_crf=encoder.display_quality,
        min_frames_between_restarts=max(args.gop // 2, 10),
    )
    controller_cfg = {
        "initial_crf": controller.current_crf,
        "crf_min": controller.crf_min,
        "crf_max": controller.crf_max,
        "ema_alpha": controller.ema_alpha,
        "dead_band_crf": controller.dead_band_crf,
        "min_frames_between_restarts": controller.min_frames_between_restarts,
        "zero_det_tolerance": controller.zero_det_tolerance,
        "big_jump_crf": controller.big_jump_crf,
    }

    # ── Pipeline health counters ─────────────────────────────────────────────
    counters = PipelineCounters(expected_fps=float(fps))

    # ── Raw-input recorder (webcam only, when --record-input is set) ────────
    recorder: RawInputRecorder | None = None
    recorded_input_meta: dict | None = None
    if args.record_input:
        if input_source.startswith("webcam:"):
            recorder = RawInputRecorder(
                output_path=args.record_input,
                width=args.width, height=args.height,
                capture_fps=float(args.record_input_fps or probed_fps or fps),
                input_source=input_source,
                max_frames=args.record_input_max_frames,
            )
        else:
            log.warning(
                "--record-input ignored: input is a file (already reproducible)."
            )

    # ── Artifacts ────────────────────────────────────────────────────────────
    artifacts: SessionArtifacts | None = None
    run_dir: str | None = None
    env_info: dict = {}
    if args.save_artifacts:
        run_dir = args.output_dir or os.path.join(
            "runs", f"benchmark_{time.strftime('%Y%m%d_%H%M%S')}"
        )
        artifacts = SessionArtifacts(
            run_dir=run_dir, width=args.width, height=args.height,
            fps=fps, enabled=True,
        )
        # Environment snapshot — git SHA, ffmpeg version, input sha256.
        input_video_for_env = (
            args.input if not input_source.startswith("webcam:") else None
        )
        env_info = collect_environment(input_video_path=input_video_for_env)
        artifacts.write_session_config(SessionConfig(
            run_dir=run_dir,
            input_source=input_source,
            width=args.width, height=args.height, fps=fps,
            gop_size=args.gop, classes=classes,
            model=args.model, device="auto",
            proactive_mode=proactive_mode,
            max_frames=args.max_frames, loop_video=args.loop_video,
            seed=args.seed,
            heatmap_width=heat_wh[0], heatmap_height=heat_wh[1],
            controller=controller_cfg,
            reproducibility=repro,
            environment=env_info,
            recorded_input=None,  # filled in at close
            response_timeout_sec=float(args.response_timeout_sec),
        ))

    # ── Connect + handshake ──────────────────────────────────────────────────
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((args.server_ip, args.port))
    try:
        client_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    except OSError:
        pass
    # Enforce response-timeout for blocking recvs. The whole-socket timeout
    # is simplest here because the camera is single-threaded w.r.t. the
    # socket and always does a full-response recv per sent packet.
    if args.response_timeout_sec and args.response_timeout_sec > 0:
        client_socket.settimeout(float(args.response_timeout_sec))
    client_socket.sendall(pack_handshake(classes, heat_wh=heat_wh))
    log.info("handshake sent: %d classes, heatmap=%dx%d",
             len(classes), heat_wh[0], heat_wh[1])

    # Monotonic correlation id for packet↔response matching. Starts at 1
    # so a 0 in logs/fields unambiguously means "no packet sent yet".
    next_sequence_id = 1
    last_received_seq = 0

    conf_score = 0.5
    prev_object_score_map = None
    last_detections = []  # list of (x1,y1,x2,y2,conf,cls_idx)
    last_decode_us_server = 0
    last_infer_us_server = 0
    interval_start = time.time()
    interval_sent = 0
    interval_baseline = 0
    interval_frames = 0
    interval_encode_ms: list[float] = []
    interval_drops_base = 0

    last_decoded: np.ndarray | None = None

    if FIXED_CRF is not None:
        mode_label = "Fixed"
    elif proactive_mode:
        mode_label = "Proactive"
    else:
        mode_label = "Adaptive"
    header_size = wire.HEADER_SIZE
    log.info("H.264 adaptive encoder active (mode=%s)", mode_label)
    # Preview and recording are independent switches. Logging both at
    # startup makes it obvious which knobs are active for this run.
    log.info(
        "preview=%s  record_input=%s  save_artifacts=%s",
        "OFF (--no-preview)" if args.no_preview else "ON",
        args.record_input if args.record_input else "—",
        "ON" if args.save_artifacts else "—",
    )

    frame_index = 0
    last_sent_sequence_id = 0
    try:
        while True:
            if args.max_frames is not None and frame_index >= args.max_frames:
                break

            counters.record_loop_tick()
            t_frame_start = time.perf_counter()

            # ── Capture ──────────────────────────────────────────────────────
            ret, frame = cap.read()
            counters.record_capture_attempt(ok=bool(ret) and frame is not None)
            if not ret or frame is None:
                if args.loop_video and input_source.startswith("file:"):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break

            source_ts_sec: float | None = None
            try:
                # CAP_PROP_POS_MSEC is 0 for webcams on many backends — we
                # treat 0 as "unknown" and fall back to loop_ts at log time.
                pos_ms = float(cap.get(cv2.CAP_PROP_POS_MSEC))
                source_ts_sec = pos_ms / 1000.0 if pos_ms > 0 else None
            except Exception:
                source_ts_sec = None

            # ── Processing (resize + optional raw-input recording) ───────────
            try:
                frame = cv2.resize(frame, (args.width, args.height))
            except Exception as e:
                counters.record_processing_skip()
                log.debug("resize failed: %s", e)
                continue
            counters.record_processing_ok()
            original_bgr = frame  # preserved for artifact writer

            # Tap the post-resize frame into the recorder BEFORE masking so
            # the recording is faithful to what the camera actually saw.
            if recorder is not None:
                recorder.write(original_bgr)

            t_flow_start = time.perf_counter()
            masked_frame, roi_ratio = masker.apply(
                frame, object_score_map=prev_object_score_map,
            )
            flow_ms = (time.perf_counter() - t_flow_start) * 1000.0

            # ── Encode ───────────────────────────────────────────────────────
            t_enc_start = time.perf_counter()
            packets = encoder.encode(masked_frame)
            encode_ms = (time.perf_counter() - t_enc_start) * 1000.0
            counters.record_encode(produced_packet=bool(packets))
            interval_encode_ms.append(encode_ms)
            interval_frames += 1
            active_crf = encoder.display_quality
            total_bytes = sum(len(p.data) for p in packets)
            sent_kb = total_bytes / 1024 if total_bytes else 0.0

            for pkt in packets:
                local_decoder.push(pkt)

            # ── Send + recv feedback (only on frames that produced packets) ──
            send_ms: float | None = None
            recv_ms: float | None = None
            parse_ms: float | None = None
            object_score_map = prev_object_score_map
            if packets:
                interval_sent += total_bytes + header_size * len(packets)
                interval_baseline += estimate_baseline_bytes(frame, active_crf) + header_size

                # Each packet gets its own sequence_id. The last packet's id
                # is the one we correlate the authoritative reply against.
                t_send_start = time.perf_counter()
                last_sent_sequence_id = 0
                for pkt in packets:
                    seq = next_sequence_id
                    # Skip zero on wraparound — keep 0 reserved for "unused".
                    next_sequence_id = (next_sequence_id + 1) & 0xFFFFFFFF or 1
                    client_socket.sendall(wire.pack_packet(pkt, sequence_id=seq))
                    last_sent_sequence_id = seq
                send_ms = (time.perf_counter() - t_send_start) * 1000.0

                counters.record_response_expected()

                try:
                    # Drain intermediate responses for the pre-final packets
                    # (N>1 case). Each is counted; stale detection fires if
                    # any reply's seq is below the watermark.
                    for _ in range(len(packets) - 1):
                        try:
                            (_m, _h, _d, _du, _iu, seq_mid) = \
                                read_response(client_socket)
                            counters.record_response_received()
                            if seq_mid < last_received_seq:
                                counters.record_stale_response()
                            last_received_seq = max(last_received_seq, seq_mid)
                        except socket.timeout:
                            counters.record_response_timeout()
                            raise

                    t_recv_start = time.perf_counter()
                    (conf_score, heatmap_low, detections,
                     decode_us, infer_us, resp_seq) = read_response(client_socket)
                    recv_ms = (time.perf_counter() - t_recv_start) * 1000.0
                    counters.record_response_received()

                    # Correlation checks. A lower seq than our watermark means
                    # a very old frame's reply arrived (stale). A mismatch
                    # against last_sent_sequence_id in a single-threaded
                    # blocking pipeline is unexpected — counted as invalid.
                    if resp_seq < last_received_seq:
                        counters.record_stale_response()
                    if resp_seq != last_sent_sequence_id:
                        counters.record_invalid_response()
                    last_received_seq = max(last_received_seq, resp_seq)

                    t_parse_start = time.perf_counter()
                    # Upsample the low-res server heatmap to frame size before
                    # fusing with optical flow. INTER_LINEAR gives a smooth
                    # edge on the importance field, which the motion masker
                    # prefers over the nearest-neighbour steps from raw boxes.
                    if (heatmap_low.shape[0] != args.height
                            or heatmap_low.shape[1] != args.width):
                        heatmap_full = cv2.resize(
                            heatmap_low, (args.width, args.height),
                            interpolation=cv2.INTER_LINEAR,
                        )
                    else:
                        heatmap_full = heatmap_low
                    object_score_map = heatmap_full.astype(np.float32) / 255.0
                    last_detections = detections
                    last_decode_us_server = int(decode_us)
                    last_infer_us_server = int(infer_us)

                    decision = controller.update(conf_score, len(detections))
                    if decision.should_apply:
                        encoder.set_quality(
                            conf_score, force_keyframe=decision.force_idr,
                        )
                    if hasattr(encoder, "observe"):
                        encoder.observe(conf_score, roi_ratio)
                    parse_ms = (time.perf_counter() - t_parse_start) * 1000.0
                except socket.timeout:
                    counters.record_response_timeout()
                    log.warning("response timeout (> %.2fs) — closing session",
                                args.response_timeout_sec)
                    break
                except (ConnectionError, OSError) as e:
                    log.warning("server disconnected: %s", e)
                    break
            prev_object_score_map = object_score_map

            # ── Pull locally decoded frame for preview/artifact ──────────────
            decoded = local_decoder.get_frame()
            if decoded is not None:
                last_decoded = decoded

            # ── Artifact logging (frame-level, not packet-level) ─────────────
            if artifacts is not None:
                artifacts.write_original(original_bgr)
                artifacts.write_masked(masked_frame)
                artifacts.write_decoded(last_decoded)
                det_classes_seen = [
                    classes[idx] if 0 <= idx < len(classes) else str(idx)
                    for (_, _, _, _, _, idx) in last_detections
                ]
                e2e_ms = (time.perf_counter() - t_frame_start) * 1000.0
                artifacts.log_frame(
                    frame_index=frame_index,
                    sequence_id=last_sent_sequence_id or None,
                    source_timestamp_sec=source_ts_sec,
                    loop_timestamp_sec=t_frame_start,
                    capture_ok=True,
                    input_source=input_source,
                    roi_ratio=roi_ratio,
                    conf_metric=conf_score,
                    crf=active_crf,
                    encoded_bytes=total_bytes,
                    num_detections=len(last_detections),
                    detected_classes=det_classes_seen,
                    restart_count=getattr(encoder, "restart_count", 0),
                    crf_transition_count=controller.transition_count,
                    proactive_mode=proactive_mode,
                    processing_time_ms=e2e_ms,
                    flow_ms=flow_ms,
                    encode_ms=encode_ms,
                    send_ms=send_ms,
                    recv_ms=recv_ms,
                    parse_ms=parse_ms,
                    end_to_end_loop_ms=e2e_ms,
                    decode_us_server=last_decode_us_server,
                    infer_us_server=last_infer_us_server,
                    counter_snapshot=counters.snapshot_running(),
                )

            # ── Preview ─────────────────────────────────────────────────────
            if not args.no_preview:
                left = original_bgr.copy()
                draw_hud(left, "Source — raw input", active_crf,
                         conf_score, roi_ratio, sent_kb, mode_label)

                if last_decoded is not None:
                    right = last_decoded.copy()
                    draw_hud(right, "Encoded stream — server view",
                             active_crf, conf_score, roi_ratio, sent_kb, mode_label)
                    for (x1, y1, x2, y2, conf, cls_idx) in last_detections:
                        x1i = int(max(0, min(right.shape[1] - 1, x1)))
                        y1i = int(max(0, min(right.shape[0] - 1, y1)))
                        x2i = int(max(0, min(right.shape[1] - 1, x2)))
                        y2i = int(max(0, min(right.shape[0] - 1, y2)))
                        name = (classes[cls_idx]
                                if 0 <= cls_idx < len(classes) else str(cls_idx))
                        cv2.rectangle(right, (x1i, y1i), (x2i, y2i), (0, 255, 255), 2)
                        cv2.putText(right, f"{name} {conf:.2f}",
                                    (x1i, max(0, y1i - 8)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    (0, 255, 255), 1, cv2.LINE_AA)
                    # Active prompt list footer
                    cv2.putText(right, f"classes: {','.join(classes)}",
                                (10, right.shape[0] - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                                (200, 200, 255), 1, cv2.LINE_AA)
                else:
                    right = np.zeros((args.height, args.width, 3), dtype=np.uint8)
                    cv2.putText(right, "Buffering first GOP...",
                                (110, args.height // 2 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                                (160, 160, 160), 1, cv2.LINE_AA)

                dashboard = cv2.hconcat([left, right])
                cv2.imshow("EchoStream — Edge Node", dashboard)
                if SHOW_IMPORTANCE_WINDOW and masker.last_importance is not None:
                    importance_u8 = (masker.last_importance * 255.0).astype(np.uint8)
                    importance_heat = cv2.applyColorMap(importance_u8, cv2.COLORMAP_JET)
                    cv2.imshow("Importance Heatmap", importance_heat)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            # ── Periodic bandwidth log ───────────────────────────────────────
            elapsed = time.time() - interval_start
            if elapsed >= LOG_BANDWIDTH_EVERY_SEC and interval_baseline > 0:
                saved = interval_baseline - interval_sent
                pct = saved / interval_baseline * 100
                sent_mbps = (interval_sent * 8) / elapsed / 1_000_000
                base_mbps = (interval_baseline * 8) / elapsed / 1_000_000
                log.info(
                    "[BW] %.1fs sent=%.2fMB baseline=%.2fMB saved=%.2fMB (%.1f%%)"
                    " | %.2fMbps vs %.2fMbps",
                    elapsed, interval_sent / 1024 / 1024,
                    interval_baseline / 1024 / 1024,
                    saved / 1024 / 1024, pct, sent_mbps, base_mbps,
                )
                fps_actual = interval_frames / elapsed if elapsed > 0 else 0.0
                drops_total = local_decoder.drops
                drops_delta = drops_total - interval_drops_base
                if interval_encode_ms:
                    enc_arr = np.asarray(interval_encode_ms, dtype=np.float32)
                    p50 = float(np.percentile(enc_arr, 50))
                    p99 = float(np.percentile(enc_arr, 99))
                    enc_max = float(enc_arr.max())
                else:
                    p50 = p99 = enc_max = 0.0
                log.info(
                    "[STATS] %.1fs frames=%d fps=%.1f encode_p50=%.2fms "
                    "encode_p99=%.2fms encode_max=%.2fms decoder_drops=%d (total=%d)",
                    elapsed, interval_frames, fps_actual,
                    p50, p99, enc_max, drops_delta, drops_total,
                )
                interval_start = time.time()
                interval_sent = 0
                interval_baseline = 0
                interval_frames = 0
                interval_encode_ms.clear()
                interval_drops_base = drops_total

            frame_index += 1

    finally:
        encoder.close()
        local_decoder.close()
        cap.release()
        try:
            client_socket.close()
        except Exception:
            pass
        cv2.destroyAllWindows()

        # ── Close the raw-input recorder before we touch session_config so
        #    the sidecar's sha256/frame_count are available to embed. ──
        if recorder is not None:
            try:
                meta = recorder.close()
                recorded_input_meta = asdict(meta)
            except Exception as e:
                log.warning("raw recorder close failed: %s", e)

        if artifacts is not None:
            # Propagate artifact writer backpressure stats into the pipeline
            # counters. Reading these *before* close is fine — they are
            # lock-protected running totals, and close() does not reset
            # them. Doing it before close also means they land in the
            # health summary we write next.
            counters.set_artifact_drops(artifacts.dropped_artifact_frames)
            counters.artifact_enqueue = artifacts.enqueued_artifact_frames

            artifacts.close()

            # Re-serialize session_config.json with the fields that only
            # existed after the run finished (recorded-input sidecar and
            # the final pipeline-health block). summarize_run() reads this
            # file, so the health block flows into summary.json from here.
            try:
                artifacts.write_session_config(SessionConfig(
                    run_dir=run_dir or str(artifacts.run_dir),
                    input_source=input_source,
                    width=args.width, height=args.height, fps=fps,
                    gop_size=args.gop, classes=classes,
                    model=args.model, device="auto",
                    proactive_mode=proactive_mode,
                    max_frames=args.max_frames, loop_video=args.loop_video,
                    seed=args.seed,
                    heatmap_width=heat_wh[0], heatmap_height=heat_wh[1],
                    controller=controller_cfg,
                    reproducibility=repro,
                    environment=env_info,
                    recorded_input=recorded_input_meta,
                    response_timeout_sec=float(args.response_timeout_sec),
                    pipeline_health=counters.summary_dict(),
                ))
            except Exception as e:
                log.warning("session_config rewrite failed: %s", e)

            try:
                summary = write_summary(artifacts.run_dir.as_posix())
                log.info("artifacts saved to %s", artifacts.run_dir)
                log.info(
                    "bitrate summary: original=%.1fkbps masked=%.1fkbps adaptive=%.1fkbps "
                    "(live=%.1fkbps)",
                    (summary.original.avg_bitrate_bps / 1000) if summary.original else 0.0,
                    (summary.masked.avg_bitrate_bps / 1000) if summary.masked else 0.0,
                    (summary.adaptive.avg_bitrate_bps / 1000) if summary.adaptive else 0.0,
                    summary.adaptive_avg_bitrate_bps_live / 1000,
                )
                if summary.adaptive_vs_original_savings_pct is not None:
                    log.info(
                        "savings: masked=%.1f%% adaptive=%.1f%%",
                        summary.masked_vs_original_savings_pct or 0.0,
                        summary.adaptive_vs_original_savings_pct or 0.0,
                    )
            except Exception as e:
                log.warning("summary write failed: %s", e)

        # ── Shutdown health snapshot ─────────────────────────────────────
        health = counters.summary_dict()
        log.info(
            "counter totals: captured=%d/%d (drops=%d, %.1f%%) | "
            "processing_skips=%d | encoded=%d (zero_pkt=%d) | "
            "responses=%d (timeout=%d, stale=%d, invalid=%d) | "
            "artifact_drops=%d | long_loop_gaps=%d max_gap=%.1fms | "
            "observed_fps in/enc/resp = %.1f/%.1f/%.1f",
            int(health["total_frames_captured"]),
            int(health["total_capture_attempt"]),
            int(health["total_capture_drops"]),
            100.0 * float(health["capture_drop_rate"]),
            int(health["total_processing_skips"]),
            int(health["total_frames_encoded"]),
            int(health["total_encode_zero_packet"]),
            int(health["total_frames_with_response"]),
            int(health["total_response_timeouts"]),
            int(health["total_stale_responses"]),
            int(health["total_invalid_responses"]),
            int(health["total_artifact_drops"]),
            int(health["long_loop_gap_count"]),
            float(health["max_loop_gap_ms"]),
            float(health["observed_input_fps"]),
            float(health["observed_encoded_fps"]),
            float(health["observed_response_fps"]),
        )
        log.info("camera shutdown complete.")


if __name__ == '__main__':
    main()
