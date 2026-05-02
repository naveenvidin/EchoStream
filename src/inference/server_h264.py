"""Edge inference server using the compare folder's H.264 segment protocol.

Camera -> server:
    [4-byte big-endian payload length][raw H.264 segment]

Server -> camera (per decoded frame):
    [float32 metric][uint16 heat_w][uint16 heat_h][uint16 num_boxes]
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
FPS = 30
FRAME_DURATION = 1.0 / FPS

log = logging.getLogger("echostream.server")


class H264Decoder:
    def __init__(self, width: int = 640, height: int = 480):
        self.width = width
        self.height = height
        self._frame_bytes = width * height * 3
        self._frame_q = queue.Queue(maxsize=FPS * 2)
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
        while True:
            raw = self._proc.stdout.read(self._frame_bytes)
            if len(raw) < self._frame_bytes:
                break
            frame = np.frombuffer(raw, dtype=np.uint8).reshape(
                (self.height, self.width, 3),
            ).copy()
            if self._frame_q.full():
                try:
                    self._frame_q.get_nowait()
                except queue.Empty:
                    pass
            try:
                self._frame_q.put_nowait(frame)
            except queue.Full:
                pass

    def push(self, nal_bytes: bytes):
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


def receive_loop(conn, decoder):
    """Continuously feed length-prefixed H.264 segments into the decoder."""
    while True:
        try:
            header = _recv_exact(conn, 4)
            if not header:
                break
            payload_size = struct.unpack("!I", header)[0]
            segment_data = _recv_exact(conn, payload_size)
            if not segment_data:
                break
            decoder.push(segment_data)
        except (ConnectionError, OSError, struct.error) as e:
            log.warning("receiver connection lost: %s", e)
            break


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
    p.add_argument("--tracker", default="kalman",
                   choices=("kalman", "none"),
                   help="Box stabilizer: kalman (smooth + coast) or none (raw detections).")
    p.add_argument("--model", default="yolov8s-world.pt",
                   help="YOLO-World weights path.")
    p.add_argument("--classes", default="person",
                   help="Comma-separated prompted classes, e.g. person,wallet,bed.")
    p.add_argument("--device", default="auto",
                   choices=("auto", "cuda", "mps", "cpu"),
                   help="Inference device.")
    p.add_argument("--conf-threshold", type=float, default=0.3)
    p.add_argument("--nms-iou", type=float, default=0.45,
                   help="NMS IoU threshold (lower = fewer overlapping boxes).")
    p.add_argument("--port", type=int, default=PORT)
    p.add_argument("--width", type=int, default=WIDTH)
    p.add_argument("--height", type=int, default=HEIGHT)
    p.add_argument("--show-window", action="store_true",
                   help="Show server-side YOLO annotations.")
    return p.parse_args()


def main():
    args = _parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    from src.inference.detection import YoloWorldDetector, parse_classes
    from src.inference.tracking.kalman_tracker import KalmanPersonTracker

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
    try:
        detector.warmup(height=args.height, width=args.width)
    except Exception as e:
        log.warning("detector warmup skipped: %s", e)

    decoder = H264Decoder(width=args.width, height=args.height)
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind(("0.0.0.0", args.port))
    server_socket.listen(1)
    log.info(
        "listening on 0.0.0.0:%d model=%s classes=%s device=%s",
        args.port, args.model, classes, detector.device,
    )

    conn = None
    try:
        conn, addr = server_socket.accept()
        log.info("connection from %s", addr)

        recv_thread = threading.Thread(
            target=receive_loop, args=(conn, decoder), daemon=True,
        )
        recv_thread.start()

        fps_counter = 0
        fps_timer = time.time()
        current_fps = 0.0

        while True:
            frame = decoder.get_frame()
            if frame is None:
                time.sleep(0.005)
                if not recv_thread.is_alive():
                    break
                continue

            conf, _heatmap, detections, _infer_us = detector.infer(frame)
            person_dets = []
            if person_cls_idx is not None:
                for x1, y1, x2, y2, c, cls_idx in detections:
                    if cls_idx != person_cls_idx:
                        continue
                    box = np.array([x1, y1, x2, y2], dtype=np.float32)
                    person_dets.append((box, float(c)))

            # Choose between raw detections and Kalman-smoothed tracks.
            boxes_for_wire: list[tuple[float, float, float, float, float]] = []
            if tracker is None:
                for box, c in person_dets:
                    x1, y1, x2, y2 = box.tolist()
                    boxes_for_wire.append((float(x1), float(y1), float(x2), float(y2), float(c)))
            else:
                tracks = tracker.update(person_dets)
                for t in tracks:
                    x1, y1, x2, y2 = t.bbox_xyxy.tolist()
                    boxes_for_wire.append((float(x1), float(y1), float(x2), float(y2), float(t.conf)))

            heat_w, heat_h = detector.heatmap_size
            heatmap = np.zeros((heat_h, heat_w), dtype=np.uint8)
            frame_h, frame_w = frame.shape[:2]
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

            # Minimal stage: still send only a float back, but stabilize it
            # based on active tracks (coasts across short YOLO misses).
            if boxes_for_wire:
                conf = float(min(b[4] for b in boxes_for_wire))
            try:
                boxes_payload = b"".join(struct.pack("!fffff", *b) for b in boxes_for_wire)
                header = struct.pack("!fHHH", float(conf), int(heat_w), int(heat_h), int(len(boxes_for_wire)))
                conn.sendall(header + heatmap.tobytes() + boxes_payload)
            except (BrokenPipeError, OSError):
                break

            fps_counter += 1
            if time.time() - fps_timer >= 1.0:
                current_fps = fps_counter / max(time.time() - fps_timer, 1e-6)
                fps_counter = 0
                fps_timer = time.time()

            if args.show_window:
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
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            time.sleep(FRAME_DURATION)

    except Exception as e:
        log.warning("server error: %s", e)
    finally:
        decoder.close()
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
