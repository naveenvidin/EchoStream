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
FRAME_DURATION = 1.0 / FPS


# ─────────────────────────────────────────────
#  H.264 Decoder — persistent ffmpeg subprocess
# ─────────────────────────────────────────────
class H264Decoder:
    def __init__(self, width: int = 640, height: int = 480):
        self.width = width
        self.height = height
        self._frame_bytes = width * height * 3
        # Buffer 2 seconds worth of frames to handle network bursts
        self._frame_q = queue.Queue(maxsize=FPS * 2) 
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
            
            # If queue is full, drop oldest to keep feed "live"
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
            self._proc.kill()


# ─────────────────────────────────────────────
#  Networking Helpers
# ─────────────────────────────────────────────
def _recv_exact(conn, size):
    """Helper to ensure we don't get partial reads from the socket."""
    chunks = []
    bytes_recvd = 0
    while bytes_recvd < size:
        chunk = conn.recv(min(size - bytes_recvd, 4096))
        if not chunk:
            return None
        chunks.append(chunk)
        bytes_recvd += len(chunk)
    return b''.join(chunks)

def receive_loop(conn, decoder):
    """Expects [4-byte length prefix] + [H.264 data segment]"""
    while True:
        try:
            header = _recv_exact(conn, 4)
            if not header:
                break
            payload_size = struct.unpack('!I', header)[0]

            segment_data = _recv_exact(conn, payload_size)
            if not segment_data:
                break

            decoder.push(segment_data)
        except (ConnectionError, OSError, struct.error) as e:
            print(f"Receiver: Connection lost ({e})")
            break

# ─────────────────────────────────────────────
#  Inference/HUD Helpers
# ─────────────────────────────────────────────
def compute_confidence(confidences: list) -> float:
    if not confidences:
        return 0.5
    return float(min(confidences))

def draw_hud(frame: np.ndarray, label: str, conf: float,
             person_boxes: list, fps: float) -> np.ndarray:
    h, w = frame.shape[:2]
    # Header/Footer overlays
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 36), (0, 0, 0), -1)
    cv2.rectangle(overlay, (0, h - 48), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    cv2.putText(frame, label, (10, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    stats = f"Conf {conf:.2f}  Processing FPS {fps:.1f}"
    cv2.putText(frame, stats, (10, h - 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 255, 120), 1, cv2.LINE_AA)

    for (x1, y1, x2, y2, c) in person_boxes:
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
        cv2.putText(frame, f"person {c:.2f}", (int(x1), max(0, int(y1) - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    return frame


# ─────────────────────────────────────────────
#  Main Execution
# ─────────────────────────────────────────────
def main():
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    if torch.cuda.is_available(): device = 'cuda'
    
    model = YOLO('yolov8n.pt').to(device)
    decoder = H264Decoder(width=WIDTH, height=HEIGHT)

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind(('0.0.0.0', PORT))
    server_socket.listen(1)
    print(f"Edge Server: Listening on port {PORT} (Device: {device})")

    shared_lock = threading.Lock()
    shared_state = {
        'display_frame': None,
        'conf': 0.5,
        'person_boxes': [],
        'fps': 0.0,
    }

    try:
        conn, addr = server_socket.accept()
        print(f"Edge Server: connection from {addr}")

        # Start Networking Thread
        recv_thread = threading.Thread(target=receive_loop, args=(conn, decoder), daemon=True)
        recv_thread.start()

        def inference_loop():
            fps_counter = 0
            fps_timer = time.time()

            while True:
                loop_start = time.time() # Start timing the frame cadence

                frame = decoder.get_frame()
                if frame is None:
                    time.sleep(0.005) # Prevent CPU spinning
                    continue

                # Run Inference
                results = model(frame, verbose=False, device=device)
                confidences = [b.conf[0].item() for r in results for b in r.boxes]
                conf = compute_confidence(confidences)

                # Send quality feedback back to camera
                try:
                    conn.sendall(struct.pack('!f', conf))
                except (BrokenPipeError, OSError):
                    break

                # Process Bounding Boxes
                person_boxes = []
                if results and results[0].boxes is not None:
                    boxes = results[0].boxes
                    for i in range(len(boxes)):
                        if results[0].names.get(int(boxes.cls[i].item()), '') == 'person':
                            c = float(boxes.conf[i].item())
                            x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                            person_boxes.append((x1, y1, x2, y2, c))

                # Update Internal FPS metric
                fps_counter += 1
                if time.time() - fps_timer >= 1.0:
                    with shared_lock:
                        shared_state['fps'] = fps_counter / (time.time() - fps_timer)
                    fps_counter = 0
                    fps_timer = time.time()

                # Build annotated frame
                annotated = frame.copy()
                with shared_lock:
                    current_fps = shared_state['fps']
                draw_hud(annotated, "Edge Server — YOLO Inference", conf, person_boxes, current_fps)

                # Update state for main display thread
                with shared_lock:
                    shared_state['display_frame'] = annotated
                    shared_state['conf'] = conf
                    shared_state['person_boxes'] = person_boxes

                # --- PACING LOGIC ---
                # Ensure we don't process faster than real-time (30 FPS)
                elapsed = time.time() - loop_start
                sleep_time = FRAME_DURATION - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

        inf_thread = threading.Thread(target=inference_loop, daemon=True)
        inf_thread.start()

        # ── Main Thread: Display Loop ────────────────────────────────────────
        last_display_time = time.time()
        while True:
            if SHOW_SERVER_WINDOW:
                now = time.time()
                if now - last_display_time >= FRAME_DURATION:
                    with shared_lock:
                        display_frame = shared_state['display_frame']
                    if display_frame is not None:
                        cv2.imshow('Edge Server — YOLO Inference', display_frame)
                    last_display_time = now

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                time.sleep(0.1)
                if not recv_thread.is_alive():
                    break

    except Exception as e:
        print(f"Main Error: {e}")
    finally:
        decoder.close()
        server_socket.close()
        cv2.destroyAllWindows()
        print("Edge Server: shutdown complete.")

if __name__ == '__main__':
    main()