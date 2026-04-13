import cv2
import socket
import struct
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
def recv_exact(sock, size: int) -> bytes:
    chunks, received = [], 0
    while received < size:
        chunk = sock.recv(size - received)
        if not chunk:
            raise ConnectionError("Socket closed while receiving.")
        chunks.append(chunk)
        received += len(chunk)
    return b''.join(chunks)


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
#  Main
# ─────────────────────────────────────────────
def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((SERVER_IP, PORT))

    masker = OpticalFlowMasker(motion_threshold=3.0, min_contour_area=1200)
    encoder = H264Encoder(width=WIDTH, height=HEIGHT, crf=28, fps=30, gop=GOP_SIZE)

    # Local decoder: same NAL bytes as the server receives.
    # This is the fix for the black right panel — we decode locally so the
    # dashboard shows real compression artefacts, masking, and CRF effects.
    local_decoder = H264Decoder(width=WIDTH, height=HEIGHT)

    header_size = struct.calcsize('Q')
    conf_score = 0.5
    prev_conf = 0.5
    prev_object_score_map = None
    last_person_boxes = []
    interval_start = time.time()
    interval_sent = 0
    interval_baseline = 0

    # Persist last decoded frame so panel doesn't blank during GOP gaps
    last_decoded: np.ndarray | None = None

    mode = 'Fixed' if FIXED_CRF is not None else 'Adaptive'
    print("Camera Node: H.264 Adaptive Encoder Active.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (WIDTH, HEIGHT))
            masked_frame, roi_ratio = masker.apply(frame, object_score_map=prev_object_score_map)

            # ── Encode ──────────────────────────────────────────────────────
            data = encoder.encode(masked_frame)
            active_crf = encoder._crf
            sent_kb = len(data) / 1024 if data else 0.0

            # Push into local decoder immediately — before sending to server —
            # so the local preview pipeline runs in parallel with network I/O.
            if data:
                local_decoder.push(data)

            # ── Send to server ───────────────────────────────────────────────
            if data:
                interval_sent += len(data) + header_size
                interval_baseline += estimate_baseline_bytes(frame, active_crf) + header_size
                client_socket.sendall(struct.pack('Q', len(data)) + data)

                # ── Receive metric + heatmap + boxes ─────────────────────────
                try:
                    header = recv_exact(client_socket, 16)
                    conf_score, heat_w, heat_h, num_boxes = struct.unpack('!fIII', header)
                    conf_score = max(0.0, min(1.0, conf_score))

                    heat_bytes = recv_exact(client_socket, int(heat_w * heat_h))
                    heatmap = np.frombuffer(heat_bytes, dtype=np.uint8).reshape(
                        (int(heat_h), int(heat_w))
                    )
                    object_score_map = heatmap.astype(np.float32) / 255.0

                    boxes = []
                    if num_boxes:
                        box_bytes = recv_exact(client_socket, int(num_boxes * 20))
                        for i in range(int(num_boxes)):
                            offset = i * 20
                            x1, y1, x2, y2, conf = struct.unpack(
                                '!fffff',
                                box_bytes[offset:offset + 20]
                            )
                            boxes.append((x1, y1, x2, y2, conf))
                    last_person_boxes = boxes

                    new_crf = conf_to_crf(conf_score)
                    big_change = abs(conf_score - prev_conf) > 0.15
                    encoder.set_crf(new_crf, force_keyframe=big_change)
                    prev_conf = conf_score
                except (ConnectionError, struct.error):
                    conf_score = 0.5
                    object_score_map = None
                    last_person_boxes = []
                prev_object_score_map = object_score_map

            # ── Pull latest locally decoded frame ────────────────────────────
            decoded = local_decoder.get_frame()
            if decoded is not None:
                last_decoded = decoded

            # ── Build left panel: source ─────────────────────────────────────
            left = frame.copy()
            draw_hud(left, "Source — raw camera feed", active_crf,
                     conf_score, roi_ratio, sent_kb, mode)

            # ── Build right panel: server's view ─────────────────────────────
            if last_decoded is not None:
                right = last_decoded.copy()
                draw_hud(right, "Encoded stream — what server receives",
                         active_crf, conf_score, roi_ratio, sent_kb, mode)
                for (x1, y1, x2, y2, conf) in last_person_boxes:
                    x1i = int(max(0, min(right.shape[1] - 1, x1)))
                    y1i = int(max(0, min(right.shape[0] - 1, y1)))
                    x2i = int(max(0, min(right.shape[1] - 1, x2)))
                    y2i = int(max(0, min(right.shape[0] - 1, y2)))
                    cv2.rectangle(right, (x1i, y1i), (x2i, y2i), (0, 255, 255), 2)
                    cv2.putText(
                        right,
                        f"person {conf:.2f}",
                        (x1i, max(0, y1i - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 255),
                        1,
                        cv2.LINE_AA
                    )
            else:
                right = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
                cv2.putText(right, "Buffering first GOP...", (110, HEIGHT // 2 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (160, 160, 160), 1, cv2.LINE_AA)
                cv2.putText(right, f"(waiting for {GOP_SIZE} frames)", (150, HEIGHT // 2 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1, cv2.LINE_AA)

            dashboard = cv2.hconcat([left, right])
            cv2.imshow('EchoStream — Edge Node', dashboard)
            if SHOW_IMPORTANCE_WINDOW and masker.last_importance is not None:
                importance_u8 = (masker.last_importance * 255.0).astype(np.uint8)
                importance_heat = cv2.applyColorMap(importance_u8, cv2.COLORMAP_JET)
                cv2.imshow('Importance Heatmap', importance_heat)

            # ── Bandwidth log ────────────────────────────────────────────────
            elapsed = time.time() - interval_start
            if elapsed >= LOG_BANDWIDTH_EVERY_SEC and interval_baseline > 0:
                saved = interval_baseline - interval_sent
                pct = saved / interval_baseline * 100
                sent_mbps = (interval_sent * 8) / elapsed / 1_000_000
                base_mbps = (interval_baseline * 8) / elapsed / 1_000_000
                print(
                    f"[BW] {elapsed:.1f}s | "
                    f"sent={interval_sent/1024/1024:.2f} MB | "
                    f"baseline={interval_baseline/1024/1024:.2f} MB | "
                    f"saved={saved/1024/1024:.2f} MB ({pct:.1f}%) | "
                    f"sent={sent_mbps:.2f} Mbps | baseline={base_mbps:.2f} Mbps"
                )
                interval_start = time.time()
                interval_sent = 0
                interval_baseline = 0

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        encoder.close()
        local_decoder.close()
        cap.release()
        client_socket.close()
        cv2.destroyAllWindows()
        print("Camera Node: shutdown complete.")


if __name__ == '__main__':
    main()
