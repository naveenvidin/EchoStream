"""Edge inference server using segmented H.264 feedback.

Camera -> server:
    [4-byte big-endian float: camera FPS]           (handshake, sent once on connect)
    [uint32 segment_id][uint32 payload_length][raw H.264 segment]  (repeated)

Server -> camera (per decoded frame):
    [uint32 segment_id][float32 metric][uint16 heat_w][uint16 heat_h][uint16 num_boxes]
    [heatmap bytes: heat_w*heat_h uint8]
    [boxes bytes: num_boxes * (x1,y1,x2,y2,conf) float32]

The segment_id flows camera→server→camera so per-frame responses can be
attributed to the segment they came from, rather than being treated as
undifferentiated "latest" updates. Server assumes one segment carries
exactly `int(fps)` decoded frames (camera's GOP == fps).

Only the detector is additive: fixed YOLOv8n is replaced by YOLO-World
with prompted classes supplied on the server CLI.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import queue
import socket
import struct
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


PORT = 9999
WIDTH, HEIGHT = 640, 480

log = logging.getLogger("echostream.server")


class H264Decoder:
    """Persistent ffmpeg subprocess that accepts raw H.264 bytes and emits decoded
    BGR frames into an internal queue for downstream consumption.
    """

    def __init__(self, width: int = 640, height: int = 480, fps: float = 30.0):
        self.width = width
        self.height = height
        self.fps = fps
        self._frame_bytes = width * height * 3
        # Queue sized to hold ~3 seconds of frames at the negotiated FPS.
        self._frame_q = queue.Queue(maxsize=int(fps * 3))
        self._proc = None
        self._reader_thread = None
        self._start()

    def _start(self):
        cmd = [
            "ffmpeg", "-loglevel", "quiet",
            "-f", "h264",
            "-i", "pipe:0",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "pipe:1",
        ]
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
        )
        self._reader_thread = threading.Thread(
            target=self._drain_stdout, daemon=True,
        )
        self._reader_thread.start()

    def _drain_stdout(self):
        """Read decoded frames from ffmpeg stdout and push them to the frame queue.
        If the queue is full, drop the oldest frame to make room so the producer
        never blocks on a slow consumer.
        """
        while True:
            raw = self._proc.stdout.read(self._frame_bytes)
            if len(raw) < self._frame_bytes:
                break
            frame = np.frombuffer(raw, dtype=np.uint8).reshape(
                (self.height, self.width, 3),
            ).copy()
            if self._frame_q.full():
                log.debug("full, dropping frames in drain, bad")
                try:
                    self._frame_q.get_nowait()
                except queue.Empty:
                    pass
            try:
                self._frame_q.put_nowait(frame)
            except queue.Full:
                pass

    def push(self, nal_bytes: bytes):
        """Write a raw H.264 segment into the ffmpeg decoder's stdin."""
        try:
            self._proc.stdin.write(nal_bytes)
            self._proc.stdin.flush()
        except (BrokenPipeError, OSError):
            pass

    def get_frame(self):
        try:
            return self._frame_q.get_nowait()
        except queue.Empty:
            return None

    def close(self):
        try:
            self._proc.stdin.close()
            self._proc.wait(timeout=3)
        except Exception:
            try:
                self._proc.kill()
            except Exception:
                pass


class ServerArtifacts:
    """Server-side artifact writer.

    Mirrors the camera's `--save-artifacts` workflow but for what the
    server actually sees and produces:

      <output_dir>/decoded.mp4          decoded H.264 frames as received
      <output_dir>/annotated.mp4        decoded frames with detection boxes
      <output_dir>/server_metrics.csv   per-frame inference stats
      <output_dir>/server_config.json   run config snapshot

    Synchronous on the inference thread. Inference dominates the loop
    budget by orders of magnitude over disk on a GPU box, so the async
    worker pattern used by the camera-side artifacts is unnecessary
    here. Zero-overhead when `enabled=False`.
    """

    COLUMNS = [
        "frame_index", "timestamp_sec", "num_boxes",
        "conf_min", "conf_max", "conf_mean",
        "infer_ms",
    ]

    def __init__(self, output_dir: Optional[str], width: int, height: int,
                 fps: float, enabled: bool = True):
        self.enabled = bool(enabled and output_dir)
        if not self.enabled:
            return
        self.run_dir = Path(output_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps) if fps and fps > 0 else 30.0

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._decoded_writer = cv2.VideoWriter(
            str(self.run_dir / "decoded.mp4"), fourcc, self.fps,
            (self.width, self.height),
        )
        self._annotated_writer = cv2.VideoWriter(
            str(self.run_dir / "annotated.mp4"), fourcc, self.fps,
            (self.width, self.height),
        )
        self._csv_file = open(self.run_dir / "server_metrics.csv",
                              "w", newline="")
        self._csv = csv.DictWriter(self._csv_file, fieldnames=self.COLUMNS)
        self._csv.writeheader()

        self._t0 = time.time()
        self._frame_idx = 0
        log.info("server artifacts -> %s", self.run_dir)

    def write_session_config(self, cfg: dict) -> None:
        if not self.enabled:
            return
        try:
            (self.run_dir / "server_config.json").write_text(
                json.dumps(cfg, indent=2, default=str)
            )
        except Exception as e:
            log.debug("server_config write failed: %s", e)

    def write_frame(self, decoded: np.ndarray, boxes: list,
                    infer_ms: float) -> None:
        """Write decoded + annotated frames and a CSV row.

        `boxes` follows the wire format used by inference_loop:
        list of (x1, y1, x2, y2, conf) floats.
        """
        if not self.enabled or decoded is None:
            return
        try:
            annotated = decoded.copy()
            for x1, y1, x2, y2, c in boxes:
                p1 = (int(x1), int(y1))
                p2 = (int(x2), int(y2))
                cv2.rectangle(annotated, p1, p2, (0, 255, 255), 2)
                cv2.putText(
                    annotated, f"{c:.2f}",
                    (p1[0], max(0, p1[1] - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1,
                    cv2.LINE_AA,
                )
            self._decoded_writer.write(decoded)
            self._annotated_writer.write(annotated)
        except Exception as e:
            log.debug("server video write failed: %s", e)

        confs = [float(b[4]) for b in boxes] if boxes else []
        row = {
            "frame_index": self._frame_idx,
            "timestamp_sec": round(time.time() - self._t0, 6),
            "num_boxes": len(boxes),
            "conf_min": f"{min(confs):.6f}" if confs else "",
            "conf_max": f"{max(confs):.6f}" if confs else "",
            "conf_mean": f"{(sum(confs) / len(confs)):.6f}" if confs else "",
            "infer_ms": f"{float(infer_ms):.3f}",
        }
        try:
            self._csv.writerow(row)
        except Exception as e:
            log.debug("server csv write failed: %s", e)
        self._frame_idx += 1

    def close(self) -> None:
        if not self.enabled:
            return
        for w in (getattr(self, "_decoded_writer", None),
                  getattr(self, "_annotated_writer", None)):
            try:
                if w is not None:
                    w.release()
            except Exception:
                pass
        try:
            if self._csv_file is not None:
                self._csv_file.flush()
                self._csv_file.close()
        except Exception:
            pass


def _recv_exact(conn, size: int):
    chunks = []
    bytes_recvd = 0
    while bytes_recvd < size:
        chunk = conn.recv(min(size - bytes_recvd, 4096))
        if not chunk:
            return None
        chunks.append(chunk)
        bytes_recvd += len(chunk)
    return b"".join(chunks)


def receive_segment(conn):
    """Read one camera segment from the socket.

    Camera-side header is `!II` = (uint32 segment_id, uint32 payload_length),
    followed by `payload_length` raw H.264 bytes (one GOP).

    Returns (segment_id, segment_bytes) or None on EOF / connection loss.
    """
    header = _recv_exact(conn, 8)
    if not header:
        return None
    segment_id, payload_size = struct.unpack("!II", header)
    if payload_size <= 0:
        return int(segment_id), b""
    segment_data = _recv_exact(conn, int(payload_size))
    if segment_data is None:
        return None
    return int(segment_id), segment_data


def _decode_h264_segment(segment_data: bytes, width: int, height: int):
    """Decode one self-contained H.264 segment (1 GOP) into BGR frames.

    Runs a one-shot ffmpeg subprocess per segment: write the segment to stdin,
    read raw BGR frames from stdout. Returns [] on empty input or decode error.
    """
    if not segment_data:
        return []
    frame_bytes = int(width) * int(height) * 3
    cmd = [
        "ffmpeg", "-loglevel", "quiet",
        "-f", "h264", "-i", "pipe:0",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "pipe:1",
    ]
    try:
        proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        )
        raw, _ = proc.communicate(input=segment_data, timeout=10)
    except Exception as e:
        log.debug("segment decode failed: %s", e)
        return []
    frames = []
    for offset in range(0, len(raw) - frame_bytes + 1, frame_bytes):
        chunk = raw[offset:offset + frame_bytes]
        frame = np.frombuffer(chunk, dtype=np.uint8).reshape(
            (int(height), int(width), 3)
        ).copy()
        frames.append(frame)
    return frames


def receive_loop(conn, decoder, segment_id_queue):
    """Read (segment_id, length)-prefixed H.264 segments from the socket and
    push each into the decoder. Segment_ids are also pushed onto
    `segment_id_queue` so the inference loop can attribute decoded frames
    back to the segment they came from.
    """
    while True:
        try:
            header = _recv_exact(conn, 8)
            if not header:
                break
            segment_id, payload_size = struct.unpack("!II", header)
            segment_data = _recv_exact(conn, payload_size)
            if not segment_data:
                break
            decoder.push(segment_data)
            try:
                segment_id_queue.put_nowait(int(segment_id))
            except queue.Full:
                # Inference loop is far behind; drop oldest and retry once.
                try:
                    segment_id_queue.get_nowait()
                    segment_id_queue.put_nowait(int(segment_id))
                except (queue.Empty, queue.Full):
                    pass
        except (ConnectionError, OSError, struct.error) as e:
            log.warning("receiver connection lost: %s", e)
            break


def inference_loop(decoder, detector, conn, result_q, stop_event,
                   artifacts: Optional["ServerArtifacts"] = None,
                   segment_id_queue: Optional["queue.Queue"] = None,
                   frames_per_segment: int = 30):
    """Pull decoded frames from the decoder, run YOLO inference, send the result
    back to the camera over the socket, and push annotated frames to result_q for display.
    Drops frames from result_q when the display thread falls behind.

    When `artifacts` is provided, also writes the decoded frame, an annotated
    copy, and a per-frame CSV row. Detection/tracking/protocol behaviour is
    unchanged regardless.

    Each response header carries the `segment_id` of the segment that this
    decoded frame originated from. We assume `frames_per_segment` decoded
    frames per pushed segment (camera enforces GOP == fps); when that
    counter exhausts we pop the next id from `segment_id_queue`.
    """
    current_segment_id = 0
    remaining_in_segment = 0
    expected = max(int(frames_per_segment), 1)
    while not stop_event.is_set():
        frame = decoder.get_frame()
        if frame is None:
            log.debug("decoded queue empty, good thing")
            time.sleep(0.005)
            continue

        # Advance to the next segment_id when the previous segment's frame
        # budget has run out. Falls back to current_segment_id (0 on cold
        # start) when the receiver hasn't seen the next header yet.
        if remaining_in_segment <= 0 and segment_id_queue is not None:
            try:
                current_segment_id = int(segment_id_queue.get_nowait())
                remaining_in_segment = expected
            except queue.Empty:
                remaining_in_segment = 1
        if remaining_in_segment > 0:
            remaining_in_segment -= 1

        conf, _heatmap, detections, _infer_us = detector.infer(frame)
        infer_ms = (
            float(_infer_us) / 1000.0 if _infer_us else 0.0
        )
        person_dets = []
        if detector is not None and getattr(detector, "class_names", None):
            try:
                person_cls_idx = detector.class_names.index("person")
            except ValueError:
                person_cls_idx = None
        else:
            person_cls_idx = None

def person_boxes_for_wire(detector, detections, person_cls_idx):
    person_dets = []
    if person_cls_idx is not None:
        for x1, y1, x2, y2, c, cls_idx in detections:
            if cls_idx != person_cls_idx:
                continue
            box = np.array([x1, y1, x2, y2], dtype=np.float32)
            person_dets.append((box, float(c)))

    boxes_for_wire: list[tuple[float, float, float, float, float]] = []
    tracker = getattr(detector, "_echostream_tracker", None)
    if tracker is None:
        for box, c in person_dets:
            x1, y1, x2, y2 = box.tolist()
            boxes_for_wire.append((float(x1), float(y1), float(x2), float(y2), float(c)))
    else:
        tracks = tracker.update(person_dets)
        for t in tracks:
            x1, y1, x2, y2 = t.bbox_xyxy.tolist()
            boxes_for_wire.append((float(x1), float(y1), float(x2), float(y2), float(t.conf)))
    return boxes_for_wire


def boxes_to_heatmap(boxes_for_wire, frame_shape, heat_w: int, heat_h: int) -> np.ndarray:
    heatmap = np.zeros((heat_h, heat_w), dtype=np.uint8)
    frame_h, frame_w = frame_shape[:2]
    if boxes_for_wire:
        sx = heat_w / float(max(frame_w, 1))
        sy = heat_h / float(max(frame_h, 1))
        for x1, y1, x2, y2, _c in boxes_for_wire:
            hx1 = int(max(0, min(heat_w - 1, np.floor(x1 * sx))))
            hy1 = int(max(0, min(heat_h - 1, np.floor(y1 * sy))))
            hx2 = int(max(1, min(heat_w, np.ceil(x2 * sx))))
            hy2 = int(max(1, min(heat_h, np.ceil(y2 * sy))))
            if hx2 > hx1 and hy2 > hy1:
                heatmap[hy1:hy2, hx1:hx2] = 255
    return heatmap


def send_segment_feedback(conn, segment_id: int, conf: float, heatmap: np.ndarray, boxes_for_wire):
    heat_h, heat_w = heatmap.shape[:2]
    boxes_payload = b"".join(struct.pack("!fffff", *b) for b in boxes_for_wire)
    header = struct.pack(
        "!IfHHH",
        int(segment_id),
        float(conf),
        int(heat_w),
        int(heat_h),
        int(len(boxes_for_wire)),
    )
    conn.sendall(header + heatmap.tobytes() + boxes_payload)




def draw_hud(frame: np.ndarray, label: str, conf: float,
             detections: list, class_names: list[str], fps: float) -> np.ndarray:
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 36), (0, 0, 0), -1)
    cv2.rectangle(overlay, (0, h - 48), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    cv2.putText(frame, label, (10, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    stats = f"Conf {conf:.2f}  Processing FPS {fps:.1f}"
    cv2.putText(frame, stats, (10, h - 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 255, 120), 1, cv2.LINE_AA)

    for (x1, y1, x2, y2, c, cls_idx) in detections:
        p1 = (int(x1), int(y1))
        p2 = (int(x2), int(y2))
        name = class_names[cls_idx] if 0 <= cls_idx < len(class_names) else str(cls_idx)
        cv2.rectangle(frame, p1, p2, (0, 255, 255), 2)
        cv2.putText(frame, f"{name} {c:.2f}", (p1[0], max(0, p1[1] - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    return frame


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="EchoStream compare server (H.264 segments + YOLO-World).",
    )
    p.add_argument("--config", default="configs/default.json",
                   help="JSON config file with defaults.")
    # Each override defaults to None / False so the JSON config's value is
    # only displaced when the flag is actually passed on the command line.
    p.add_argument("--model", default=None,
                   help="Override model path (YOLO-World checkpoint).")
    p.add_argument("--classes", default=None,
                   help="Override prompted classes, e.g. 'person,wallet,bed'.")
    p.add_argument("--device", default=None,
                   help="Override device: auto, cuda, mps, or cpu.")
    p.add_argument("--conf-threshold", dest="conf_threshold",
                   type=float, default=None,
                   help="Override detector confidence floor.")
    p.add_argument("--nms-iou", dest="nms_iou", type=float, default=None,
                   help="Override NMS IoU threshold.")
    p.add_argument("--tracker", default=None, choices=("kalman", "none"),
                   help="Override tracker mode.")
    p.add_argument("--host", default=None,
                   help="Override bind host (config default 0.0.0.0).")
    p.add_argument("--port", type=int, default=None,
                   help="Override port from config.")
    p.add_argument("--width", type=int, default=None,
                   help="Override decoded frame width.")
    p.add_argument("--height", type=int, default=None,
                   help="Override decoded frame height.")
    p.add_argument("--show-window", action="store_true",
                   help="Override show_window=true (server-side preview).")
    p.add_argument("--save-artifacts", dest="save_artifacts",
                   action="store_true",
                   help="Write decoded.mp4, annotated.mp4, server_metrics.csv, "
                        "server_config.json under --output-dir.")
    p.add_argument("--output-dir", dest="output_dir", default=None,
                   help="Output directory for server-side artifacts.")

    cli = p.parse_args()

    from src.common.config import load_json_config
    cfg = load_json_config(cli.config)
    block = cfg.get("server_h264") if isinstance(cfg, dict) else None
    if not isinstance(block, dict):
        raise SystemExit(f"Missing server_h264 section in config: {cli.config}")

    args = argparse.Namespace(**block)
    # Backfill defaults so older config files without these keys still work.
    args.host = getattr(args, "host", None) or "0.0.0.0"
    args.save_artifacts = bool(getattr(args, "save_artifacts", False))
    args.output_dir = getattr(args, "output_dir", None)

    overrides = {
        "model": cli.model,
        "classes": cli.classes,
        "device": cli.device,
        "conf_threshold": cli.conf_threshold,
        "nms_iou": cli.nms_iou,
        "tracker": cli.tracker,
        "host": cli.host,
        "port": cli.port,
        "width": cli.width,
        "height": cli.height,
        "output_dir": cli.output_dir,
    }
    for key, value in overrides.items():
        if value is not None:
            setattr(args, key, value)
    if cli.show_window:
        args.show_window = True
    if cli.save_artifacts:
        args.save_artifacts = True

    if args.save_artifacts and not args.output_dir:
        args.output_dir = str(
            Path("runs") / time.strftime("server_%Y%m%d_%H%M%S")
        )
    return args


def main():
    args = _parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    from src.inference.detection import YoloWorldDetector, parse_classes
    from src.inference.tracking.kalman_tracker import KalmanPersonTracker

    # ----user defined classes and model setup----
    classes = parse_classes(args.classes) or ["object"]
    detector = YoloWorldDetector(
        model_path=args.model,
        device=args.device,
        conf_threshold=args.conf_threshold,
        iou_threshold=args.nms_iou,
        heatmap_wh=(80, 60),
    )
    detector.set_classes(classes)
    person_cls_idx = None
    try:
        person_cls_idx = detector.class_names.index("person")
    except ValueError:
        person_cls_idx = None

    tracker = None
    if args.tracker == "kalman":
        tracker = KalmanPersonTracker(iou_threshold=0.3, max_age=10)
    detector._echostream_tracker = tracker  # type: ignore[attr-defined]
    try:
        detector.warmup(height=args.height, width=args.width)
    except Exception as e:
        log.warning("detector warmup skipped: %s", e)

    bind_host = getattr(args, "host", None) or "0.0.0.0"
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((bind_host, args.port))
    server_socket.listen(1)
    log.info(
        "listening on %s:%d model=%s classes=%s device=%s",
        bind_host, args.port, args.model, classes, detector.device,
    )

    artifacts: Optional[ServerArtifacts] = None
    conn = None
    worker_thread = None
    stop_event = threading.Event()
    try:
        conn, addr = server_socket.accept()
        log.info("connection from %s", addr)

        # Receive the camera's actual FPS from the handshake and derive frame duration.
        fps_bytes = _recv_exact(conn, 4)
        fps = struct.unpack("!f", fps_bytes)[0]
        frame_duration_ms = int(1000 / fps)
        log.info("FPS handshake received: %.2f  frame_duration_ms=%d", fps, frame_duration_ms)

        if getattr(args, "save_artifacts", False):
            artifacts = ServerArtifacts(
                output_dir=args.output_dir,
                width=args.width,
                height=args.height,
                fps=fps,
                enabled=True,
            )
            artifacts.write_session_config({
                "model": args.model,
                "classes": classes,
                "device": detector.device,
                "conf_threshold": args.conf_threshold,
                "nms_iou": args.nms_iou,
                "tracker": args.tracker,
                "host": bind_host,
                "port": args.port,
                "width": args.width,
                "height": args.height,
                "fps": fps,
                "client_addr": f"{addr[0]}:{addr[1]}",
                "start_wall_time": time.time(),
            })

        stop_event = threading.Event()
        result_q = queue.Queue(maxsize=int(fps * 3))

        fps_counter = 0
        fps_timer = time.time()
        current_fps = 0.0

        def process_segments():
            while not stop_event.is_set():
                try:
                    received = receive_segment(conn)
                    if received is None:
                        break

                    segment_id, segment_data = received
                    frames = _decode_h264_segment(segment_data, args.width, args.height)
                    if not frames:
                        heat_w, heat_h = detector.heatmap_size
                        send_segment_feedback(
                            conn,
                            segment_id,
                            0.5,
                            np.zeros((heat_h, heat_w), dtype=np.uint8),
                            [],
                        )
                        continue

                    frame_confs = []
                    latest_heatmap = None
                    latest_boxes_for_wire = []
                    heat_w, heat_h = detector.heatmap_size

                    for frame in frames:
                        conf, _heatmap, detections, _infer_us = detector.infer(frame)
                        boxes_for_wire = person_boxes_for_wire(detector, detections, person_cls_idx)
                        if boxes_for_wire:
                            conf = float(min(b[4] for b in boxes_for_wire))

                        frame_confs.append(float(conf))
                        latest_boxes_for_wire = boxes_for_wire
                        latest_heatmap = boxes_to_heatmap(boxes_for_wire, frame.shape, heat_w, heat_h)

                        if args.show_window:
                            if result_q.full():
                                try:
                                    result_q.get_nowait()
                                except queue.Empty:
                                    pass
                            try:
                                result_q.put_nowait((frame, conf, boxes_for_wire))
                            except queue.Full:
                                pass

                    segment_conf = float(np.percentile(np.array(frame_confs, dtype=np.float32), 50))
                    if latest_heatmap is None:
                        latest_heatmap = np.zeros((heat_h, heat_w), dtype=np.uint8)
                    send_segment_feedback(
                        conn,
                        segment_id,
                        segment_conf,
                        latest_heatmap,
                        latest_boxes_for_wire,
                    )
                except (BrokenPipeError, ConnectionError, OSError, struct.error) as e:
                    log.warning("segment worker stopped: %s", e)
                    break
            stop_event.set()

        worker_thread = threading.Thread(target=process_segments, daemon=True)
        worker_thread.start()

        if args.show_window:
            while not stop_event.is_set():
                try:
                    frame, conf, boxes_for_wire = result_q.get(timeout=0.1)
                except queue.Empty:
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        stop_event.set()
                        break
                    continue

                fps_counter += 1
                if time.time() - fps_timer >= 1.0:
                    current_fps = fps_counter / max(time.time() - fps_timer, 1e-6)
                    fps_counter = 0
                    fps_timer = time.time()

                annotated = frame.copy()
                tracked_dets = []
                tracked_names = detector.class_names
                if person_cls_idx is not None:
                    for x1, y1, x2, y2, c in boxes_for_wire:
                        tracked_dets.append((x1, y1, x2, y2, float(c), int(person_cls_idx)))
                draw_hud(
                    annotated,
                    "Edge Server - YOLO-World",
                    conf,
                    tracked_dets,
                    tracked_names,
                    current_fps,
                )
                cv2.imshow("Edge Server - YOLO-World", annotated)

                # Pace display to the negotiated FPS without slowing segment feedback.
                if cv2.waitKey(frame_duration_ms) & 0xFF == ord("q"):
                    stop_event.set()
                    break
        else:
            while not stop_event.is_set() and worker_thread.is_alive():
                worker_thread.join(timeout=0.5)

    except Exception as e:
        log.warning("server error: %s", e)
    finally:
        stop_event.set()
        if worker_thread is not None:
            worker_thread.join(timeout=2.0)
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
        server_socket.close()
        cv2.destroyAllWindows()
        if artifacts is not None:
            artifacts.close()
        log.info("server shutdown complete.")


if __name__ == "__main__":
    main()
