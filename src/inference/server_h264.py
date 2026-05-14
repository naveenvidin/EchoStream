"""Edge inference server using segmented H.264 feedback.

Camera -> server:
    [4-byte big-endian float: camera FPS]           (handshake, sent once on connect)
    [uint32 segment_id][uint32 payload length][raw H.264 segment]  (repeated)

Server -> camera (per segment):
    [uint32 segment_id][float32 p10_conf][uint16 heat_w][uint16 heat_h][uint16 num_boxes]
    [heatmap bytes: heat_w*heat_h uint8]
    [boxes bytes: num_boxes * (x1,y1,x2,y2,conf) float32]

Only the detector is additive: fixed YOLOv8n is replaced by YOLO-World
with prompted classes supplied on the server CLI.
"""
from __future__ import annotations

import argparse
import logging
import queue
import socket
import struct
import subprocess
import threading
import time

import cv2
import numpy as np


PORT = 9999
WIDTH, HEIGHT = 640, 480

log = logging.getLogger("echostream.server")


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


def receive_segment(conn):
    header = _recv_exact(conn, 8)
    if not header:
        return None
    segment_id, payload_size = struct.unpack("!II", header)
    segment_data = _recv_exact(conn, payload_size)
    if not segment_data:
        return None
    return int(segment_id), segment_data


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
    p.add_argument("--port", type=int, default=None,
                   help="Override port from config.")
    p.add_argument("--show-window", action="store_true",
                   help="Override show_window=true (server-side preview).")

    cli = p.parse_args()

    from src.common.config import load_json_config
    cfg = load_json_config(cli.config)
    block = cfg.get("server_h264") if isinstance(cfg, dict) else None
    if not isinstance(block, dict):
        raise SystemExit(f"Missing server_h264 section in config: {cli.config}")

    args = argparse.Namespace(**block)
    if cli.port is not None:
        args.port = int(cli.port)
    if cli.show_window:
        args.show_window = True
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

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind(("0.0.0.0", args.port))
    server_socket.listen(1)
    log.info(
        "listening on 0.0.0.0:%d model=%s classes=%s device=%s",
        args.port, args.model, classes, detector.device,
    )

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

        result_q = queue.Queue(maxsize=int(max(fps, 1) * 3))

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
        log.info("server shutdown complete.")


if __name__ == "__main__":
    main()
