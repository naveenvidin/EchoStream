from __future__ import annotations

import argparse
import logging
import queue
import socket
import struct
import subprocess
import threading
import time
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

from src.optical_flow.motion_masker import OpticalFlowMasker


SERVER_IP = "localhost"
PORT = 9999
FIXED_CRF = None
FRAMES_PER_SEGMENT = 30
WIDTH, HEIGHT = 640, 480
LOG_BANDWIDTH_EVERY_SEC = 60
INITIAL_CRF = 23

log = logging.getLogger("echostream.camera")


class SegmentEncoder:
    def __init__(self, width: int, height: int, gop: int = 30):
        self.width = width
        self.height = height
        self.fps = gop
        self.gop = gop

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
            "-f", "h264",
            "pipe:1",
        ]

    def encode(self, frames: list[np.ndarray], crf: int) -> bytes:
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
    def __init__(self, sock: socket.socket, counters=None):
        self.sock = sock
        self.counters = counters
        self._current_crf = INITIAL_CRF
        self._next_crf = INITIAL_CRF
        self._latest_conf = 0.5
        self._responses = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._listen, daemon=True)
        self._thread.start()

    def _conf_to_crf(self, conf: float) -> int:
        levels = [23, 28, 33, 38, 43]
        index = min(int(conf * 5), 4)
        return levels[index]

    def _listen(self):
        while not self._stop.is_set():
            try:
                data = _recv_exact(self.sock, 4)
                conf = struct.unpack("!f", data)[0]
                conf = max(0.0, min(1.0, conf))
                new_crf = self._conf_to_crf(conf)
                with self._lock:
                    self._latest_conf = conf
                    self._next_crf = new_crf
                    self._responses += 1
                if self.counters is not None:
                    self.counters.record_response_received()
            except (ConnectionError, struct.error, OSError):
                break

    def stop(self):
        self._stop.set()

    def get_next_crf(self) -> int:
        with self._lock:
            self._current_crf = self._next_crf
            return self._current_crf

    def latest_confidence(self) -> float:
        with self._lock:
            return float(self._latest_conf)


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
    return cap, f"file:{spec}", (cap.get(cv2.CAP_PROP_FPS) or 30.0)


def _decode_h264_segment(data: bytes, width: int, height: int) -> list[np.ndarray]:
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
    p.add_argument("--input", default="0",
                   help="Webcam index or video path for replay.")
    p.add_argument("--loop-video", action="store_true",
                   help="Rewind video input at EOF.")
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--server-ip", default=SERVER_IP)
    p.add_argument("--port", type=int, default=PORT)
    p.add_argument("--width", type=int, default=WIDTH)
    p.add_argument("--height", type=int, default=HEIGHT)
    p.add_argument("--gop", type=int, default=FRAMES_PER_SEGMENT)
    p.add_argument("--classes", default="person",
                   help="Prompt list logged with artifacts; server must use matching --classes.")
    p.add_argument("--model", default="yolov8s-world.pt",
                   help="Model name logged with artifacts; loaded by the server.")
    p.add_argument("--save-artifacts", action="store_true")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--record-input", default=None,
                   help="Write post-resize webcam input to an MP4 for replay.")
    p.add_argument("--record-input-fps", type=float, default=None)
    p.add_argument("--record-input-max-frames", type=int, default=None)
    p.add_argument("--response-timeout-sec", type=float, default=2.0,
                   help="Accepted for CLI compatibility; the compare path "
                        "receives feedback on its existing listener thread.")
    p.add_argument("--no-preview", action="store_true")
    return p.parse_args()


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
    fps = float(probed_fps or args.gop or 30.0)

    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((args.server_ip, args.port))

    counters = PipelineCounters(expected_fps=fps)
    masker = OpticalFlowMasker(motion_threshold=3.0, min_contour_area=1200)
    encoder = SegmentEncoder(width=args.width, height=args.height, gop=args.gop)
    listener = ConfidenceListener(sock=client_socket, counters=counters)

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
            gop_size=args.gop,
            classes=classes,
            model=args.model,
            device="server",
            proactive_mode=False,
            max_frames=args.max_frames,
            loop_video=args.loop_video,
            seed=0,
        ))

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
    shared_lock = threading.Lock()
    last_raw = {"frame": None}
    last_masked = {"frame": None}
    last_roi = {"ratio": 0.0}
    last_sent_kb = {"kb": 0.0}
    last_crf = {"crf": INITIAL_CRF}
    shared_panel = {"frame": None}
    bw_lock = threading.Lock()
    bw_state = {"sent": 0, "baseline": 0}
    mode = "Fixed" if FIXED_CRF is not None else "Adaptive"
    crf_transition_count = {"count": 0, "prev": None}

    log.info(
        "ready input=%s classes=%s preview=%s artifacts=%s",
        input_source, classes, "OFF" if args.no_preview else "ON",
        run_dir if artifacts else "OFF",
    )

    def composer_loop():
        while True:
            with shared_lock:
                raw = last_raw["frame"]
                masked = last_masked["frame"]
                roi = last_roi["ratio"]
                sent_kb = last_sent_kb["kb"]
                crf = last_crf["crf"]
            conf = listener.latest_confidence()
            if raw is not None:
                raw_panel = raw.copy()
                draw_hud(raw_panel, "Source - raw camera feed",
                         crf, conf, roi, sent_kb, mode)
                if masked is not None:
                    masked_panel = masked.copy()
                    draw_hud(masked_panel, "Masked - ROI prepared stream",
                             crf, conf, roi, sent_kb, mode)
                    temp_panel = cv2.hconcat([raw_panel, masked_panel])
                else:
                    temp_panel = raw_panel
                with shared_lock:
                    shared_panel["frame"] = temp_panel
            time.sleep(1.0 / max(args.gop, 1))

    threading.Thread(target=composer_loop, daemon=True).start()

    def encode_and_send(segment_items, crf, baseline_total, transition_count):
        try:
            frames = [item["masked"] for item in segment_items]
            t_enc = time.perf_counter()
            data = encoder.encode(frames, crf)
            encode_ms_total = (time.perf_counter() - t_enc) * 1000.0
            counters.record_encode(produced_packet=bool(data))
            if not data:
                return

            header = struct.pack("!I", len(data))
            t_send = time.perf_counter()
            client_socket.sendall(header + data)
            send_ms = (time.perf_counter() - t_send) * 1000.0
            counters.record_response_expected()

            with shared_lock:
                last_sent_kb["kb"] = len(data) / 1024
                last_crf["crf"] = crf
            with bw_lock:
                bw_state["sent"] += len(data)
                bw_state["baseline"] += baseline_total

            if artifacts is None:
                return

            decoded_frames = _decode_h264_segment(data, args.width, args.height)
            per_frame_bytes = len(data) // max(len(segment_items), 1)
            remainder = len(data) - per_frame_bytes * len(segment_items)
            per_frame_encode_ms = encode_ms_total / max(len(segment_items), 1)
            for idx, item in enumerate(segment_items):
                encoded_bytes = per_frame_bytes + (1 if idx < remainder else 0)
                decoded = decoded_frames[idx] if idx < len(decoded_frames) else None
                artifacts.write_original(item["original"])
                artifacts.write_masked(item["masked"])
                artifacts.write_decoded(decoded)
                e2e_ms = (time.perf_counter() - item["loop_ts"]) * 1000.0
                artifacts.log_frame(
                    frame_index=item["frame_index"],
                    input_source=input_source,
                    roi_ratio=item["roi_ratio"],
                    conf_metric=listener.latest_confidence(),
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

    def network_worker():
        while True:
            item = encode_queue.get()
            if item is None:
                encode_queue.task_done()
                break
            encode_and_send(*item)
            encode_queue.task_done()

    worker = threading.Thread(target=network_worker, daemon=True)
    worker.start()

    interval_start = time.time()
    segment_items = []
    segment_baseline = 0
    frame_index = 0

    try:
        while True:
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
            masked_frame, roi_ratio = masker.apply(frame)
            flow_ms = (time.perf_counter() - t_flow) * 1000.0

            with shared_lock:
                last_raw["frame"] = frame.copy()
                last_masked["frame"] = masked_frame.copy()
                last_roi["ratio"] = roi_ratio

            active_crf = FIXED_CRF if FIXED_CRF is not None else listener.get_next_crf()
            if crf_transition_count["prev"] is None:
                crf_transition_count["prev"] = active_crf
            elif active_crf != crf_transition_count["prev"]:
                crf_transition_count["count"] += 1
                crf_transition_count["prev"] = active_crf

            segment_baseline += estimate_baseline_bytes(frame, active_crf)
            segment_items.append({
                "frame_index": frame_index,
                "original": frame.copy(),
                "masked": masked_frame.copy(),
                "roi_ratio": roi_ratio,
                "source_ts": source_ts,
                "loop_ts": t_loop,
                "flow_ms": flow_ms,
            })

            if len(segment_items) >= args.gop:
                try:
                    encode_queue.put_nowait((
                        segment_items,
                        active_crf,
                        segment_baseline,
                        crf_transition_count["count"],
                    ))
                except queue.Full:
                    log.warning("drop segment")
                segment_items = []
                segment_baseline = 0

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
                if baseline > 0:
                    log.info(
                        "[BW] sent=%.2fMB saved=%.1f%%",
                        sent / 1024 / 1024,
                        (baseline - sent) / baseline * 100,
                    )
                interval_start = time.time()
            frame_index += 1
    finally:
        if segment_items:
            try:
                active_crf = FIXED_CRF if FIXED_CRF is not None else listener.get_next_crf()
                encode_queue.put((
                    segment_items,
                    active_crf,
                    segment_baseline,
                    crf_transition_count["count"],
                ), timeout=2.0)
            except Exception:
                pass
        encode_queue.put(None)
        encode_queue.join()
        worker.join(timeout=5.0)
        listener.stop()
        cap.release()
        try:
            client_socket.close()
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
                    gop_size=args.gop,
                    classes=classes,
                    model=args.model,
                    device="server",
                    proactive_mode=False,
                    max_frames=args.max_frames,
                    loop_video=args.loop_video,
                    seed=0,
                    recorded_input=recorded_input_meta,
                    pipeline_health=counters.summary_dict(),
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
