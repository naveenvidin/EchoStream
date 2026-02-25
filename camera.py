import cv2
import socket
import struct
import subprocess
import numpy as np
import time
import shutil

# === CONFIGURATION ===
SERVER_IP = '192.168.7.203'
PORT = 9999

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
        '-i', '-',          # stdin
        '-vframes', '1',
        '-b:v', str(max(1, int(crf_val / 2))),  # simple mapping
        '-f', 'mjpeg',
        '-loglevel', 'quiet',
        '-'                 # stdout
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
        print("[RUN] Camera Node: FFmpeg Subprocess Controller Active (Headless).")

        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("[WARN] Failed to read frame from camera.")
                time.sleep(0.1)
                continue

            frame = cv2.resize(frame, (FRAME_W, FRAME_H))

            # 1) Compress frame with ffmpeg
            data = ffmpeg_compress(frame, current_crf, ffmpeg_path)

            # 2) Send length + payload
            payload = struct.pack("Q", len(data)) + data
            client_socket.sendall(payload)

            # 3) Receive confidence metric from server
            response = client_socket.recv(1024).decode(errors='ignore').strip()

            conf_score = None
            try:
                conf_score = float(response)
                # Inverse mapping:
                # high confidence -> allow higher CRF (more compression)
                # low confidence  -> lower CRF (better quality)
                current_crf = int((1.0 - conf_score) * 50)
                current_crf = max(5, min(50, current_crf))
            except ValueError:
                print(f"[WARN] Non-float response from server: {response!r}")

            # Log every N frames
            frame_count += 1
            if frame_count % 10 == 0:
                dt = time.time() - t0
                fps = frame_count / dt if dt > 0 else 0.0
                size_kb = len(data) / 1024.0
                print(
                    f"[STAT] frame={frame_count:5d}  fps={fps:5.2f}  "
                    f"payload={size_kb:7.1f} KB  crf={current_crf:2d}  conf={conf_score}"
                )

            # Headless mode: no cv2.imshow / waitKey
            if not HEADLESS:
                cv2.putText(frame, f"FFmpeg CRF Knob: {current_crf}", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
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


if __name__ == "__main__":
    main()
