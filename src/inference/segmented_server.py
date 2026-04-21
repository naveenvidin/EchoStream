import socket
import struct
import subprocess
import threading
import queue
import time
import cv2
import torch
import numpy as np
from ultralytics import YOLO

# === CONFIGURATION ===
PORT = 9999
WIDTH, HEIGHT = 640, 480
SHOW_SERVER_WINDOW = True
FPS = 30


# ─────────────────────────────────────────────
#  H.264 Decoder — persistent ffmpeg subprocess
# ─────────────────────────────────────────────
class H264Decoder:
    """
    Persistent ffmpeg process that accepts a raw H.264 bytestream on stdin
    and outputs BGR24 frames on stdout.

    push()      — feed raw H.264 bytes in (call as bytes arrive from socket)
    get_frame() — pull latest decoded BGR frame, or None if not ready
    """

    def __init__(self, width: int = 640, height: int = 480):
        self.width = width
        self.height = height
        self._frame_bytes = width * height * 3
        self._frame_q = queue.Queue(maxsize=FPS)
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

    def push(self, nal_bytes: bytes):
        try:
            self._proc.stdin.write(nal_bytes)
            self._proc.stdin.flush()
        except BrokenPipeError:
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
            self._proc.kill()


# ─────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────
def compute_confidence(confidences: list) -> float:
    """Min confidence → low value = YOLO struggling → camera sends higher quality."""
    if not confidences:
        return 0.5
    return float(min(confidences))


def draw_hud(frame: np.ndarray, label: str, conf: float,
             person_boxes: list, fps: float) -> np.ndarray:
    h, w = frame.shape[:2]

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 36), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    overlay2 = frame.copy()
    cv2.rectangle(overlay2, (0, h - 48), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay2, 0.5, frame, 0.5, 0, frame)

    cv2.putText(frame, label, (10, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    stats = f"Conf {conf:.2f}  FPS {fps:.1f}"
    cv2.putText(frame, stats, (10, h - 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 255, 120), 1, cv2.LINE_AA)

    for (x1, y1, x2, y2, c) in person_boxes:
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
        cv2.putText(frame, f"person {c:.2f}", (int(x1), max(0, int(y1) - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

    return frame


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

    # Shared state between inference thread and display thread
    shared_lock = threading.Lock()
    shared_state = {
        'conf': 0.5,
        'person_boxes': [],
        'fps': 0.0,
    }

    # Dedicated single-slot queue for the first frame of each segment.
    # Receive loop puts one frame here per segment; inference thread
    # blocks waiting on it. No competition with the display queue.
    infer_q: queue.Queue = queue.Queue(maxsize=1)

    try:
        conn, addr = server_socket.accept()
        print(f"Edge Server: connection from {addr}")

        # ── Inference thread ─────────────────────────────────────────────────
        # Blocks on infer_q waiting for the first frame of each segment.
        # Runs YOLO on that one frame, sends confidence back, updates
        # shared_state. Never touches the decoder frame queue.
        def inference_loop():
            fps_counter = 0
            fps_timer = time.time()

            while True:
                frame = infer_q.get()  # blocks until receive_loop puts a frame in
                if frame is None:
                    break  # shutdown signal

                results = model(frame, verbose=False, device=device)
                confidences = [
                    box.conf[0].item()
                    for r in results
                    for box in r.boxes
                ]
                conf = compute_confidence(confidences)

                try:
                    conn.sendall(struct.pack('!f', conf))
                except (BrokenPipeError, OSError):
                    break

                person_boxes = []
                if results and results[0].boxes is not None:
                    boxes = results[0].boxes
                    for i in range(len(boxes)):
                        cls_id = int(boxes.cls[i].item())
                        cls_name = results[0].names.get(cls_id, '')
                        if cls_name != 'person':
                            continue
                        c = float(boxes.conf[i].item())
                        x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                        person_boxes.append((x1, y1, x2, y2, c))

                fps_counter += 1
                elapsed = time.time() - fps_timer
                if elapsed >= 1.0:
                    fps = fps_counter / elapsed
                    fps_counter = 0
                    fps_timer = time.time()
                else:
                    fps = shared_state['fps']

                with shared_lock:
                    shared_state['conf'] = conf
                    shared_state['person_boxes'] = person_boxes
                    shared_state['fps'] = fps

        inference_thread = threading.Thread(target=inference_loop, daemon=True)
        inference_thread.start()

        # ── Receive loop ─────────────────────────────────────────────────────
        # Reads one full segment at a time using the 4-byte length header.
        # Pushes the whole segment into the decoder, then waits for the
        # decoder to produce the first frame and routes it to infer_q.
        # The remaining 29 frames stay in the decoder queue for display.
        def recv_exact(n: int) -> bytes:
            chunks, received = [], 0
            while received < n:
                chunk = conn.recv(n - received)
                if not chunk:
                    raise ConnectionError("Socket closed mid-receive.")
                chunks.append(chunk)
                received += len(chunk)
            return b''.join(chunks)

        def receive_loop():
            while True:
                try:
                    header = recv_exact(4)
                    seg_len = struct.unpack('!I', header)[0]
                    seg_data = recv_exact(seg_len)
                    decoder.push(seg_data)
                    print(f"Server: received segment — {seg_len/1024:.1f} KB")

                    # Wait for the first decoded frame of this segment
                    # and route it to the inference thread.
                    # Spin briefly — ffmpeg decodes fast with ultrafast preset.
                    first_frame = None
                    while first_frame is None:
                        first_frame = decoder.get_frame()
                        if first_frame is None:
                            time.sleep(0.001)

                    # Drop the frame silently if inference is still busy
                    # with the previous segment (infer_q is maxsize=1).
                    try:
                        infer_q.put_nowait(first_frame)
                    except queue.Full:
                        pass

                except (ConnectionError, OSError):
                    print("Server: connection closed.")
                    infer_q.put(None)  # unblock inference thread for clean shutdown
                    break

        receive_thread = threading.Thread(target=receive_loop, daemon=True)
        receive_thread.start()

        # ── Display loop (main thread — owns imshow) ─────────────────────────
        # Pulls raw decoded frames directly from the decoder at full speed.
        # Overlays the latest YOLO annotations from shared_state — these may
        # lag a few frames behind if YOLO is slower than 30fps, but video
        # motion stays smooth because we're not waiting on inference.
        last_display_time = time.time()

        while True:
            if SHOW_SERVER_WINDOW:
                now = time.time()
                if now - last_display_time >= 1.0 / FPS:
                    frame = decoder.get_frame()
                    if frame is not None:
                        with shared_lock:
                            conf = shared_state['conf']
                            person_boxes = shared_state['person_boxes']
                            fps = shared_state['fps']
                        draw_hud(frame, "Edge Server — YOLO Inference", conf, person_boxes, fps)
                        cv2.imshow('Edge Server — YOLO Inference', frame)
                    last_display_time = now

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                time.sleep(0.1)
                if not receive_thread.is_alive():
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