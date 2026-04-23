import cv2
import socket
import struct
import subprocess
import threading
import time
import queue
import numpy as np
from src.optical_flow.motion_masker import OpticalFlowMasker

# === CONFIGURATION ===
SERVER_IP = 'localhost'
PORT = 9999
FIXED_CRF = None
FRAMES_PER_SEGMENT = 30 # 1 second segments at 30 FPS
WIDTH, HEIGHT = 640, 480
LOG_BANDWIDTH_EVERY_SEC = 60
INITIAL_CRF = 23


# ─────────────────────────────────────────────
#  Segment Encoder
# ─────────────────────────────────────────────
class SegmentEncoder:
    def __init__(self, width: int, height: int, gop: int = 30):
        self.width = width
        self.height = height
        self.fps = gop # should be same as gop, changed to fps for readability
        self.gop = gop

    def _build_cmd(self, crf: int) -> list[str]:
        #ffmpeg commandline, each second has its own encoder instance so we can adjust CRF per segment
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

    def encode(self, frames: list[np.ndarray], crf: int) -> bytes:
        if not frames:
            return b''
        raw = b''.join(f.tobytes() for f in frames)
        proc = subprocess.Popen(
            self._build_cmd(crf),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
        )
        out, _ = proc.communicate(input=raw)
        return out


# ─────────────────────────────────────────────
#  Confidence Listener
# ─────────────────────────────────────────────
class ConfidenceListener:
    # responsible for receiving confidence scores from the server and translating them to CRF values for the encoder
    def __init__(self, sock: socket.socket):
        self.sock = sock
        self._current_crf = INITIAL_CRF
        self._next_crf = INITIAL_CRF
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._listen, daemon=True)
        self._thread.start() # start background listening immediately

    def _conf_to_crf(self, conf: float) -> int:
        # evenly map confidence [0.0, 1.0] to CRF [23, 28, 33, 38, 43]
        levels = [23, 28, 33, 38, 43]
        index = min(int(conf * 5), 4)
        return levels[index]

    def _listen(self):
        while not self._stop.is_set():
            try:
                data = _recv_exact(self.sock, 4)
                conf = struct.unpack('!f', data)[0]
                conf = max(0.0, min(1.0, conf))
                new_crf = self._conf_to_crf(conf)
                with self._lock:
                    # in bg thread
                    self._next_crf = new_crf
            except (ConnectionError, struct.error, OSError):
                break

    def stop(self):
        # for clean shutdown
        self._stop.set()

    def get_next_crf(self) -> int:
        with self._lock:
            # in main thread, called by composer and encoder workers
            self._current_crf = self._next_crf
            return self._current_crf


# ─────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────
def _recv_exact(sock: socket.socket, size: int) -> bytes:
    chunks, received = [], 0
    while received < size:
        chunk = sock.recv(size - received)
        if not chunk:
            raise ConnectionError("Socket closed while receiving.")
        chunks.append(chunk)
        received += len(chunk)
    return b''.join(chunks)

# TODO: make this optional to the user, adds overhead but has useful info
def estimate_baseline_bytes(frame: np.ndarray, crf: int) -> int:
    '''
    rough estimation of what the segment size would be if encoded at a baseline CRF (for bandwidth comparison)
    still uses opencv jpeg in relation to the crf, but can still be misleading
    '''
    jpeg_q = max(5, min(95, 100 - crf * 2))
    ok, enc = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_q])
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

# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────
def main():
    # setup camera, network, OpticalFlowMasker, SegmentEncoder, ConfidenceListener
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((SERVER_IP, PORT))

    masker = OpticalFlowMasker(motion_threshold=3.0, min_contour_area=1200)
    encoder = SegmentEncoder(width=WIDTH, height=HEIGHT, gop=FRAMES_PER_SEGMENT)
    listener = ConfidenceListener(sock=client_socket)

    # setup State management
    encode_queue = queue.Queue(maxsize=3) # stores up to 3 segments to encode, drops if full to avoid memory bloat and latency spikes
    
    # TODO: consider deleting this, could be unnecessary when on separate camera
    shared_lock = threading.Lock() # hud drawing stuff
    last_raw = {'frame': None}
    last_roi = {'ratio': 0.0}
    last_sent_kb = {'kb': 0.0}
    last_crf = {'crf': INITIAL_CRF}
    shared_panel = {'frame': None}

    conf_lock = threading.Lock()
    conf_state = {'score': 0.5}

    bw_lock = threading.Lock()
    bw_state = {'sent': 0, 'baseline': 0}

    mode = 'Fixed' if FIXED_CRF is not None else 'Adaptive'
    print("Camera Node: Ready.")

    # ── Background Worker: HUD Composer ──────────────────────────────────────
    def composer_loop():
        while True:
            with shared_lock:
                raw = last_raw['frame']
                roi = last_roi['ratio']
                sent_kb = last_sent_kb['kb']
                crf = last_crf['crf']
            with conf_lock:
                conf = conf_state['score']

            if raw is not None:
                temp_panel = raw.copy()
                draw_hud(temp_panel, "Source — raw camera feed", crf, conf, roi, sent_kb, mode)
                with shared_lock:
                    shared_panel['frame'] = temp_panel
            time.sleep(1.0 / FRAMES_PER_SEGMENT)

    threading.Thread(target=composer_loop, daemon=True).start()

    # ── Background Worker: Encoder + Network ─────────────────────────────────
    def encode_and_send(frames, crf, baseline_total):
        try:
            data = encoder.encode(frames, crf)
            if not data: return
            
            header = struct.pack('!I', len(data)) # for recv exact on server side
            client_socket.sendall(header + data)

            # hud stuff
            with conf_lock:
                conf_state['score'] = listener.get_next_crf() / 51.0
            with shared_lock:
                last_sent_kb['kb'] = len(data) / 1024
                last_crf['crf'] = crf
            with bw_lock:
                bw_state['sent'] += len(data)
                bw_state['baseline'] += baseline_total
        except Exception as e:
            print(f"[Worker Error] {e}")

    def network_worker():
        while True:
            item = encode_queue.get()
            if item is None: break
            f, c, b = item
            encode_and_send(f, c, b)
            encode_queue.task_done()

    threading.Thread(target=network_worker, daemon=True).start()

    # ── Main Capture & UI Loop ───────────────────────────────────────────────
    interval_start = time.time()
    segment_frames = []
    segment_baseline = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret: break

            frame = cv2.resize(frame, (WIDTH, HEIGHT))
            masked_frame, roi_ratio = masker.apply(frame)

            # Update state for composer thread
            with shared_lock:
                last_raw['frame'] = frame.copy()
                last_roi['ratio'] = roi_ratio

            # Segment logic
            active_crf = FIXED_CRF if FIXED_CRF is not None else listener.get_next_crf()
            segment_baseline += estimate_baseline_bytes(frame, active_crf) # accumulate baseline for the whole segment
            segment_frames.append(masked_frame)

            if len(segment_frames) >= FRAMES_PER_SEGMENT:
                # a full second is ready, add it to the queue
                try:
                    encode_queue.put_nowait((segment_frames, active_crf, segment_baseline))
                except queue.Full:
                    print("[WARNING] Drop segment")
                segment_frames = []
                segment_baseline = 0

            # UI Update (MUST BE IN MAIN THREAD)
            with shared_lock:
                display_frame = shared_panel['frame']
            
            if display_frame is not None:
                cv2.imshow('Camera Node — Raw Feed', display_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # Logging
            if time.time() - interval_start >= LOG_BANDWIDTH_EVERY_SEC:
                with bw_lock:
                    s, b = bw_state['sent'], bw_state['baseline']
                    bw_state['sent'] = bw_state['baseline'] = 0
                if b > 0:
                    print(f"[BW] Sent: {s/1024/1024:.2f}MB | Saved: {(b-s)/b*100:.1f}%")
                interval_start = time.time()
    finally:
        encode_queue.put(None)
        listener.stop()       # signal thread first
        cap.release()
        client_socket.close() # now safe to close
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()