import cv2
import socket
import struct
import ffmpeg
import time
import numpy as np
from motion_masker import OpticalFlowMasker

# === CONFIGURATION ===
SERVER_IP = 'localhost'
PORT = 9999
current_crf = 45  # 0 (best) → 51 (worst)
LOG_BANDWIDTH_EVERY_SEC = 60
FIXED_CRF = None  # Set None to enable adaptive CRF.

cap = cv2.VideoCapture(0)

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((SERVER_IP, PORT))
masker = OpticalFlowMasker(motion_threshold=3.0, min_contour_area=1200)


def recv_exact(sock, size):
    """
    Receive exactly `size` bytes from TCP stream.
    """
    chunks = []
    received = 0
    while received < size:
        chunk = sock.recv(size - received)
        if not chunk:
            raise ConnectionError("Socket closed while receiving confidence metric.")
        chunks.append(chunk)
        received += len(chunk)
    return b"".join(chunks)


def ffmpeg_compress(frame, crf_val):
    """
    Compress a single frame using ffmpeg-python.
    """

    height, width, _ = frame.shape
    raw_bytes = frame.tobytes()

    # Map CRF-like value to MJPEG quantizer (2=best, 31=worst).
    mjpeg_q = int(np.interp(float(crf_val), [0.0, 51.0], [2.0, 31.0]))
    mjpeg_q = max(2, min(31, mjpeg_q))

    process = (
        ffmpeg
        .input(
            'pipe:',
            format='rawvideo',
            pix_fmt='bgr24',
            s=f'{width}x{height}'
        )
        .output(
            'pipe:',
            vframes=1,
            format='mjpeg',
            **{'q:v': str(mjpeg_q)}
        )
        .global_args('-loglevel', 'quiet')
        .run_async(pipe_stdin=True, pipe_stdout=True)
    )

    out, _ = process.communicate(input=raw_bytes)
    return out


def estimate_full_frame_bytes(frame, crf_val):
    """
    Estimate full-frame JPEG size (for baseline comparison).
    """
    jpeg_quality = max(5, min(95, 100 - int(crf_val) * 2))
    ok, encoded = cv2.imencode(
        ".jpg",
        frame,
        [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
    )
    return len(encoded) if ok else 0


print("Camera Node: FFmpeg-Python Adaptive Controller Active.")

try:
    conf_score = 0.0
    header_bytes = struct.calcsize("Q")
    interval_start_ts = time.time()
    interval_sent_bytes = 0
    interval_baseline_bytes = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 480))
        masked_frame, roi_ratio = masker.apply(frame)
        active_crf = FIXED_CRF if FIXED_CRF is not None else current_crf

        # Compress masked frame and send it to server
        data = ffmpeg_compress(masked_frame, active_crf)
        sent_bytes = len(data) + header_bytes

        # Baseline: what we'd approximately send if we streamed full frame
        full_est_bytes = estimate_full_frame_bytes(frame, active_crf) + header_bytes
        interval_sent_bytes += sent_bytes
        interval_baseline_bytes += full_est_bytes

        # Send compressed frame
        client_socket.sendall(struct.pack("Q", len(data)) + data)

        # Decode actual transmitted bytes for visual quality comparison.
        compressed_frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if compressed_frame is None:
            compressed_frame = np.zeros_like(frame)

        # Receive AI metric (4-byte network-order float).
        try:
            conf_payload = recv_exact(client_socket, 4)
            conf_score = struct.unpack("!f", conf_payload)[0]
            conf_score = max(0.0, min(1.0, conf_score))

            # High confidence -> high CRF (lower quality)
            # Low confidence -> low CRF (higher quality)
            if FIXED_CRF is None:
                current_crf = int(conf_score * 50)
                current_crf = max(5, min(50, current_crf))
        except (ConnectionError, struct.error):
            conf_score = 0.0

        # Single-window dashboard UI
        cv2.putText(frame, "Source", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(compressed_frame, "Decoded Compressed Stream", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        dashboard = cv2.hconcat([frame, compressed_frame])
        cv2.putText(dashboard, f"ROI: {roi_ratio*100:.1f}%", (20, 470),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(dashboard, f"FFmpeg CRF Knob: {active_crf}", (260, 470),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        cv2.putText(dashboard, f"Server Min Conf: {conf_score:.2f}", (620, 470),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
        cv2.putText(
            dashboard,
            f"Mode: {'Fixed' if FIXED_CRF is not None else 'Adaptive'}",
            (860, 470),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (200, 255, 0),
            2
        )

        cv2.imshow("Edge Node Dashboard", dashboard)

        elapsed = time.time() - interval_start_ts
        if elapsed >= LOG_BANDWIDTH_EVERY_SEC:
            saved_bytes = interval_baseline_bytes - interval_sent_bytes
            saved_pct = (
                (saved_bytes / interval_baseline_bytes) * 100.0
                if interval_baseline_bytes > 0 else 0.0
            )
            sent_mbps = (interval_sent_bytes * 8.0) / elapsed / 1_000_000
            baseline_mbps = (interval_baseline_bytes * 8.0) / elapsed / 1_000_000
            print(
                "[BW] "
                f"{elapsed:.1f}s | sent={interval_sent_bytes/1024/1024:.2f} MB "
                f"| baseline={interval_baseline_bytes/1024/1024:.2f} MB "
                f"| saved={saved_bytes/1024/1024:.2f} MB ({saved_pct:.2f}%) "
                f"| sent_rate={sent_mbps:.2f} Mbps "
                f"| baseline_rate={baseline_mbps:.2f} Mbps"
            )
            interval_start_ts = time.time()
            interval_sent_bytes = 0
            interval_baseline_bytes = 0

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    client_socket.close()
    cv2.destroyAllWindows()
