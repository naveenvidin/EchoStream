import socket
import struct
import subprocess
import threading
import queue
import json
import cv2
import torch
import numpy as np
from ultralytics import YOLO

# === CONFIGURATION ===
PORT = 9999
WIDTH, HEIGHT = 640, 480
SHOW_SERVER_WINDOW = False


# ─────────────────────────────────────────────
#  H.264 Decoder — persistent ffmpeg subprocess
# ─────────────────────────────────────────────
class H264Decoder:
    """
    Wraps a single long-lived ffmpeg process that accepts raw H.264 NAL
    bytes on stdin and outputs BGR24 frames on stdout.

    Uses a background reader thread so stdout never fills its OS pipe
    buffer while we're waiting to write more input.
    """

    def __init__(self, width=640, height=480):
        self.width = width
        self.height = height
        self._frame_bytes = width * height * 3
        self._frame_q = queue.Queue(maxsize=8)
        self._proc = None
        self._reader_thread = None
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
        while True:
            raw = self._proc.stdout.read(self._frame_bytes)
            if len(raw) < self._frame_bytes:
                break
            frame = np.frombuffer(raw, dtype=np.uint8).reshape(
                (self.height, self.width, 3)
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

    def decode(self, nal_bytes: bytes):
        """Push NAL bytes in; return oldest decoded frame or None."""
        try:
            self._proc.stdin.write(nal_bytes)
            self._proc.stdin.flush()
        except BrokenPipeError:
            return None
        try:
            return self._frame_q.get(timeout=0.05)
        except queue.Empty:
            return None

    def close(self):
        try:
            self._proc.stdin.close()
            self._proc.wait(timeout=3)
        except Exception:
            self._proc.kill()


# ─────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────
def recv_exact(sock, size: int) -> bytes:
    chunks, received = [], 0
    while received < size:
        chunk = sock.recv(size - received)
        if not chunk:
            raise ConnectionError("Camera disconnected.")
        chunks.append(chunk)
        received += len(chunk)
    return b''.join(chunks)


def compute_metric(confidences: list) -> float:
    """
    Min confidence → high value = confident scene → camera compresses harder.
    Returns 0.5 (neutral) when no detections are present.
    """
    if not confidences:
        return 0.5
    return float(min(confidences))


def encode_response(metric: float, heatmap: np.ndarray, boxes: list) -> bytes:
    """
    Pack the response sent back to camera-h264.py.
    Format:
      - float32 metric
      - uint32 width
      - uint32 height
      - uint32 num_boxes
      - heatmap bytes (uint8, H*W)
      - boxes (num_boxes * 5 float32): x1,y1,x2,y2,conf
    """
    h, w = heatmap.shape[:2]
    payload = struct.pack('!fIII', float(metric), int(w), int(h), int(len(boxes)))
    payload += heatmap.tobytes()
    for b in boxes:
        payload += struct.pack('!fffff', *b)
    return payload


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────
def main():
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    model = YOLO('yolov8n.pt').to(device)

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind(('0.0.0.0', PORT))
    server_socket.listen(1)
    print(f"Edge Server: H.264 AI Inference Engine active on port {PORT}")

    decoder = H264Decoder(width=WIDTH, height=HEIGHT)
    header_size = struct.calcsize('Q')

    try:
        conn, addr = server_socket.accept()
        print(f"Edge Server: connection from {addr}")
        buf = b''

        while True:
            # ── Read frame header ────────────────────────────────────────────
            while len(buf) < header_size:
                packet = conn.recv(8192)
                if not packet:
                    raise ConnectionError("Camera disconnected.")
                buf += packet

            msg_size = struct.unpack('Q', buf[:header_size])[0]
            buf = buf[header_size:]

            # ── Read H.264 NAL payload ───────────────────────────────────────
            while len(buf) < msg_size:
                packet = conn.recv(65536)
                if not packet:
                    raise ConnectionError("Camera disconnected mid-frame.")
                buf += packet

            nal_bytes = buf[:msg_size]
            buf = buf[msg_size:]

            # ── Decode H.264 → BGR ───────────────────────────────────────────
            frame = decoder.decode(nal_bytes)

            if frame is None:
                # Still buffering initial GOP — send neutral metric + empty heatmap
                empty_heat = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
                conn.sendall(encode_response(0.5, empty_heat, []))
                continue

            # ── YOLO inference ───────────────────────────────────────────────
            results = model(frame, verbose=False, device=device)
            confidences = [
                box.conf[0].item()
                for r in results
                for box in r.boxes
            ]
            metric = compute_metric(confidences)

            # Build person heatmap (binary) and boxes list
            h, w = frame.shape[:2]
            heatmap = np.zeros((h, w), dtype=np.uint8)
            person_boxes = []
            if results and results[0].boxes is not None and len(results[0].boxes) > 0:
                boxes = results[0].boxes
                for i in range(len(boxes)):
                    cls_id = int(boxes.cls[i].item())
                    cls_name = results[0].names.get(cls_id, str(cls_id))
                    if cls_name != "person":
                        continue
                    conf = float(boxes.conf[i].item())
                    x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                    x1 = max(0, min(w - 1, int(x1)))
                    y1 = max(0, min(h - 1, int(y1)))
                    x2 = max(0, min(w, int(x2)))
                    y2 = max(0, min(h, int(y2)))
                    if x2 > x1 and y2 > y1:
                        heatmap[y1:y2, x1:x2] = 255
                        person_boxes.append((float(x1), float(y1), float(x2), float(y2), conf))

            # ── Send metric + heatmap + boxes back ───────────────────────────
            conn.sendall(encode_response(metric, heatmap, person_boxes))

            # ── Optional server-side display ─────────────────────────────────
            if SHOW_SERVER_WINDOW:
                annotated = results[0].plot()
                cv2.imshow('Edge Server: YOLO Inference', annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    except ConnectionError as e:
        print(f"Edge Server: {e}")
    finally:
        decoder.close()
        try:
            conn.close()
        except Exception:
            pass
        server_socket.close()
        cv2.destroyAllWindows()
        print("Edge Server: shutdown complete.")


if __name__ == '__main__':
    main()
