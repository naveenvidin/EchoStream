import cv2
import socket
import struct
import subprocess
import select
import time
import numpy as np
from motion_masker import OpticalFlowMasker

# === CONFIGURATION ===
SERVER_IP = 'localhost'
PORT = 9999
current_crf = 45        # 0 (best) → 51 (worst)
last_crf = current_crf  # tracks last CRF used to detect when restart is needed
LOG_BANDWIDTH_EVERY_SEC = 60
FIXED_CRF = None        # Set None to enable adaptive CRF
FRAME_W = 640
FRAME_H = 480

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


def start_encoder(crf):
    """
    Spawns a persistent ffmpeg process that:
    - reads raw BGR frames from stdin
    - encodes them as H.264 with the given CRF
    - outputs a raw H.264 bytestream to stdout
    """
    return subprocess.Popen(
        [
            'ffmpeg', '-y',
            '-f', 'rawvideo',
            '-pix_fmt', 'bgr24',
            '-s', f'{FRAME_W}x{FRAME_H}',
            '-r', '30',
            '-i', 'pipe:0',
            '-c:v', 'libx264',
            '-crf', str(crf),
            '-preset', 'ultrafast',
            '-tune', 'zerolatency',
            '-f', 'h264',
            'pipe:1'
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL
    )


def compress_frame(encoder, frame):
    """
    Writes one raw BGR frame into the encoder stdin.
    Returns whatever H.264 bytes are ready immediately — may be empty.
    """
    encoder.stdin.write(frame.tobytes())
    encoder.stdin.flush()

    readable, _, _ = select.select([encoder.stdout], [], [], 0)
    if readable:
        return encoder.stdout.read1()
    return b""


def estimate_full_frame_bytes(frame, crf_val):
    """
    Estimate full-frame JPEG size as a baseline BWC comparison.
    Note: this is JPEG-based so not apples-to-apples with H.264,
    but kept for relative trending across frames.
    """
    jpeg_quality = max(5, min(95, 100 - int(crf_val) * 2))
    ok, encoded = cv2.imencode(
        ".jpg",
        frame,
        [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
    )
    return len(encoded) if ok else 0


encoder = start_encoder(current_crf)

print("Camera Node: H.264 Adaptive Controller Active.")

try:
    conf_score = 0.0
    interval_start_ts = time.time()
    interval_sent_bytes = 0
    interval_baseline_bytes = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (FRAME_W, FRAME_H))
        masked_frame, roi_ratio = masker.apply(frame)
        active_crf = FIXED_CRF if FIXED_CRF is not None else current_crf

        # Restart encoder only if CRF has changed since last frame
        if active_crf != last_crf:
            encoder.stdin.close()
            encoder.wait()
            encoder = start_encoder(active_crf)
            last_crf = active_crf

        # Compress masked frame and send to server
        data = compress_frame(encoder, masked_frame)
        sent_bytes = len(data)
        interval_sent_bytes += sent_bytes

        # Baseline: approximate what full-frame MJPEG would have cost
        full_est_bytes = estimate_full_frame_bytes(frame, active_crf)
        interval_baseline_bytes += full_est_bytes

        # Send raw H.264 bytes — no size header framing
        if data:
            client_socket.sendall(data)

        # Dashboard: show masked frame in place of decoded stream
        # cv2.imdecode cannot decode raw H.264 chunks
        compressed_frame = masked_frame

        # Receive AI metric (4-byte network-order float)
        try:
            conf_payload = recv_exact(client_socket, 4)
            conf_score = struct.unpack("!f", conf_payload)[0]
            conf_score = max(0.0, min(1.0, conf_score))

            # High confidence -> high CRF (lower quality, less bandwidth)
            # Low confidence -> low CRF (higher quality, more bandwidth)
            if FIXED_CRF is None:
                current_crf = int(conf_score * 50)
                current_crf = max(5, min(50, current_crf))
        except (ConnectionError, struct.error):
            conf_score = 0.0

        # Single-window dashboard UI
        cv2.putText(frame, "Source", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(compressed_frame, "Masked Frame (uncompressed)", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        dashboard = cv2.hconcat([frame, compressed_frame])
        cv2.putText(dashboard, f"ROI: {roi_ratio*100:.1f}%", (20, 470),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(dashboard, f"CRF: {active_crf}", (260, 470),
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
    encoder.stdin.close()
    encoder.wait()
    cap.release()
    client_socket.close()
    cv2.destroyAllWindows()