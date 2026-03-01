import cv2
import socket
import struct
import subprocess
import numpy as np
import time
import shutil

class OpticalFlowMasker:
    def __init__(self, motion_threshold=3.0, min_contour_area=1200, morph_kernel_size=7):
        self.motion_threshold = motion_threshold
        self.min_contour_area = min_contour_area
        self.kernel = np.ones((morph_kernel_size, morph_kernel_size), np.uint8)
        self.prev_gray = None

    def apply(self, frame_bgr):
        """
        Returns:
            masked_frame: motion-ROI-only BGR frame
            roi_ratio: fraction of image area covered by selected ROIs
        """
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        if self.prev_gray is None:
            self.prev_gray = gray
            return frame_bgr.copy(), 1.0

        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray, gray, None,
            0.5, 2, 15, 2, 5, 1.1, 0
        )

        magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        motion_mask = (magnitude > self.motion_threshold).astype(np.uint8) * 255
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, self.kernel)

        contours, _ = cv2.findContours(
            motion_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        masked = np.zeros_like(frame_bgr)
        frame_area = frame_bgr.shape[0] * frame_bgr.shape[1]
        roi_area = 0

        for cnt in contours:
            if cv2.contourArea(cnt) > self.min_contour_area:
                x, y, w, h = cv2.boundingRect(cnt)
                masked[y:y + h, x:x + w] = frame_bgr[y:y + h, x:x + w]
                roi_area += w * h

        self.prev_gray = gray
        roi_ratio = (roi_area / frame_area) if frame_area > 0 else 0.0
        return masked, roi_ratio

# === CONFIGURATION ===
SERVER_IP = "100.88.178.33"
PORT = 9999
USE_MASKING = True
BG_ALPHA = 0.15

# CRF-like knob (0~51 in ffmpeg CRF world; here we map roughly to JPEG bitrate quality behavior)
current_crf = 45

# Headless mode on Pi (no GUI)
HEADLESS = True

# Optional: choose camera index (0 is most common for USB webcam)
CAMERA_INDEX = 0

# Frame size to send
FRAME_W = 640
FRAME_H = 480


def find_ffmpeg():
    """Return ffmpeg executable path if available, else None."""
    ff = shutil.which("ffmpeg")
    return ff


def connect_with_retry(server_ip, port, retries=30, delay_sec=1.0):
    """Retry TCP connect so camera node can start before/after server."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    last_err = None
    for i in range(1, retries + 1):
        try:
            print(f"[CONNECT] Try {i}/{retries} -> {server_ip}:{port}")
            sock.connect((server_ip, port))
            print("[CONNECT] Connected to server.")
            return sock
        except OSError as e:
            last_err = e
            time.sleep(delay_sec)
    raise RuntimeError(f"Cannot connect to server {server_ip}:{port}. Last error: {last_err}")


def ffmpeg_compress(frame, crf_val, ffmpeg_path):
    """
    Compress one frame using FFmpeg -> MJPEG bytes (single frame).
    """
    raw_frame = frame.tobytes()
    height, width, _ = frame.shape

    # We use MJPEG single-frame output and map crf_val to a bitrate-ish knob.
    ffmpeg_cmd = [
        ffmpeg_path,
        '-y',
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-s', f'{width}x{height}',
        '-pix_fmt', 'bgr24',
        '-i', '-',  # stdin
        '-vframes', '1',
        # '-b:v', str(max(1, int(crf_val / 2))),  # simple mapping
        '-q:v', str(max(2, min(31, int(crf_val / 2)))),
        '-f', 'mjpeg',
        '-loglevel', 'quiet',
        '-'        # stdout
    ]

    process = subprocess.Popen(
        ffmpeg_cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    out, err = process.communicate(input=raw_frame)

    if process.returncode != 0 or not out:
        err_msg = err.decode(errors='ignore') if err else "Unknown ffmpeg error"
        raise RuntimeError(f"FFmpeg compression failed: {err_msg}")

    return out


def main():
    global current_crf

    ffmpeg_path = find_ffmpeg()
    if not ffmpeg_path:
        raise RuntimeError("ffmpeg not found. Install it on Pi: sudo apt install -y ffmpeg")

    print(f"[INIT] ffmpeg: {ffmpeg_path}")

    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {CAMERA_INDEX}. Try 1 or check webcam connection.")

    # Set camera resolution (best effort)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

    client_socket = None
    frame_count = 0
    t0 = time.time()

    try:
        client_socket = connect_with_retry(SERVER_IP, PORT, retries=60, delay_sec=1.0)
        client_socket.settimeout(5.0)
        print("[RUN] Camera Node: FFmpeg Subprocess Controller Active (Headless).")
        
        masker = OpticalFlowMasker(
            motion_threshold=3.0,
            min_contour_area=1200,
            morph_kernel_size=7
    )

        conf_score = None

        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("[WARN] Failed to read frame from camera.")
                time.sleep(0.1)
                continue

            frame = cv2.resize(frame, (FRAME_W, FRAME_H))

            # 0) ROI masking (optional)
            if USE_MASKING:
                masked, roi_ratio = masker.apply(frame)
                if BG_ALPHA > 0:
                    frame_to_send = (masked.astype(np.float32) + BG_ALPHA * frame.astype(np.float32)).clip(0, 255).astype(np.uint8)
                else:
                    frame_to_send = masked
            else:
                roi_ratio = 1.0
                frame_to_send = frame

            if USE_MASKING and roi_ratio < 0.01:
                frame_to_send = frame
                roi_ratio = 1.0

            # 1) Compress frame with ffmpeg
            data = ffmpeg_compress(frame_to_send, current_crf, ffmpeg_path)

            # 2) Send length + payload
            client_socket.sendall(struct.pack("Q", len(data)) + data)

            # 3) Receive AI metric (confidence) from server
            try:
                resp = client_socket.recv(1024).decode(errors="ignore").strip()
                conf_score = float(resp)
            except Exception:
                conf_score = None

            # 4) Update CRF using conf + roi_ratio (smooth)
            # base from conf: low conf -> better quality (lower CRF)
            if conf_score is not None:
                base = int((1.0 - conf_score) * 50)
            else:
                base = current_crf  # keep if no signal

            # ROI bonus: ROI smaller -> allow more compression (higher CRF)
            roi_bonus = int((1.0 - max(0.0, min(1.0, roi_ratio))) * 10)
            target = base + roi_bonus

            # smooth to avoid oscillation
            current_crf = int(0.8 * current_crf + 0.2 * target)
            current_crf = max(5, min(50, current_crf))

            # Log every N frames
            frame_count += 1
            if frame_count % 10 == 0:
                dt = time.time() - t0
                fps = frame_count / dt if dt > 0 else 0.0
                size_kb = len(data) / 1024.0
                print(
                    f"[STAT] frame={frame_count:5d} fps={fps:5.2f} "
                    f"payload={size_kb:7.1f} KB crf={current_crf:2d} "
                    f"conf={conf_score} roi={roi_ratio:.3f}"
                )

            if not HEADLESS:
                cv2.putText(frame, f"CRF: {current_crf} roi:{roi_ratio:.2f} conf:{conf_score}", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                cv2.imshow("Edge Node: Adaptive Frame Processing", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    except KeyboardInterrupt:
        print("\n[STOP] KeyboardInterrupt received. Exiting...")
    finally:
        try:
            if cap:
                cap.release()
        except Exception:
            pass
        try:
            if client_socket:
                client_socket.close()
        except Exception:
            pass
        if not HEADLESS:
            cv2.destroyAllWindows()
        print("[CLEANUP] Camera node stopped.")


def run_forever():
    while True:
        try:
            main()
        except (BrokenPipeError, ConnectionResetError, ConnectionRefusedError, TimeoutError, OSError) as e:
            print(f"[RECONNECT] network/socket error: {e}. retrying in 3s...")
            time.sleep(3)
        except Exception as e:

            print(f"[RECONNECT] unexpected error: {e}. retrying in 3s...")
            time.sleep(3)

if __name__ == "__main__":
    run_forever()
