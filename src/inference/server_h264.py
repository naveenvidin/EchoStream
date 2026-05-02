"""Edge inference server using the compare folder's H.264 segment protocol.

The camera still sends:
    [4-byte big-endian payload length][raw H.264 segment]

The server still replies with:
    [4-byte float confidence]

Only the detector is additive: fixed YOLOv8n is replaced by YOLO-World
with prompted classes supplied on the server CLI.

Threading model:
    Thread 1 - receive_loop:   reads H.264 segments from socket → decoder.push()
    Thread 2 - _drain_stdout:  reads decoded BGr frames from ffmpeg → frame_q
    Thread 3 - inference_loop: frame_q → detector.infer() → send conf → result_q
    Main thread:               result_q → draw_hud → cv2.imshow → waitKey(1/FPS - inference_time)
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
FRAME_DURATION = 1000 // (FPS/2)  # Aim for ~15 FPS inference to allow some display overhead

log = logging.getLogger("echostream.server")


class H264Decoder:
    def __init__(self, width: int = 640, height: int = 480):
        self.width = width
        self.height = height
        self._frame_bytes = width * height * 3
        self._frame_q = queue.Queue(maxsize=FPS * 3)
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


def inference_loop(decoder, detector, conn, result_q, stop_event):
    """Pull frames from decoder, run inference, send conf, push results for display."""
    while not stop_event.is_set():
        frame = decoder.get_frame()
        if frame is None:
            log.debug("decoded queue empty, good thing")
            time.sleep(0.005)
            continue

        conf, _heatmap, detections, _infer_us = detector.infer(frame)

        try:
            conn.sendall(struct.pack("!f", float(conf)))
        except (BrokenPipeError, OSError):
            stop_event.set()
            break

        # Drop stale results if display is falling behind
        if result_q.full():
            log.debug("full, dropping frames in inference, bad")
            try:
                result_q.get_nowait()
            except queue.Empty:
                pass
        try:
            result_q.put_nowait((frame, conf, detections))
        except queue.Full:
            pass


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
    p.add_argument("--model", default="yolov8s-world.pt",
                   help="YOLO-World weights path.")
    p.add_argument("--classes", default="person",
                   help="Comma-separated prompted classes, e.g. person,wallet,bed.")
    p.add_argument("--device", default="auto",
                   choices=("auto", "cuda", "mps", "cpu"),
                   help="Inference device.")
    p.add_argument("--conf-threshold", type=float, default=0.05)
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

    classes = parse_classes(args.classes) or ["object"]
    detector = YoloWorldDetector(
        model_path=args.model,
        device=args.device,
        conf_threshold=args.conf_threshold,
        heatmap_wh=(80, 60),
    )
    detector.set_classes(classes)
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

        stop_event = threading.Event()
        result_q = queue.Queue(maxsize=FPS * 3)

        recv_thread = threading.Thread(
            target=receive_loop, args=(conn, decoder), daemon=True,
        )
        infer_thread = threading.Thread(
            target=inference_loop,
            args=(decoder, detector, conn, result_q, stop_event),
            daemon=True,
        )
        recv_thread.start()
        infer_thread.start()

        fps_counter = 0
        fps_timer = time.time()
        current_fps = 0.0

        while not stop_event.is_set():
            infer_start = time.perf_counter()
            if not infer_thread.is_alive():
                break

            if args.show_window:
                try:
                    frame, conf, detections = result_q.get(timeout=0.1)
                except queue.Empty:
                    log.debug("display queue empty, good thing")
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                    continue

                fps_counter += 1
                if time.time() - fps_timer >= 1.0:
                    current_fps = fps_counter / max(time.time() - fps_timer, 1e-6)
                    fps_counter = 0
                    fps_timer = time.time()

                annotated = frame.copy()
                draw_hud(
                    annotated,
                    "Edge Server - YOLO-World",
                    conf,
                    detections,
                    detector.class_names,
                    current_fps,
                )
                cv2.imshow("Edge Server - YOLO-World", annotated)
                elapsed = time.perf_counter() - infer_start
                log.debug(f"Inference+display latency: {elapsed*1000:.1f} ms")
                wait_time = max(1, 66-int(elapsed*1000))  # Aim for ~15 FPS display
                log.debug(f"Frame duration - Inference: {wait_time:.1f} ms")
                if cv2.waitKey(wait_time) & 0xFF == ord("q"):
                    break
            else:
                # No display — just wait for inference thread to finish
                infer_thread.join(timeout=1.0)

    except Exception as e:
        log.warning("server error: %s", e)
    finally:
        stop_event.set()
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