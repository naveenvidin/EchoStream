import cv2
import socket
import struct
import time
import numpy as np
from motion_masker import OpticalFlowMasker
from neural_codec import NeuralCodec

# === CONFIGURATION ===
SERVER_IP = "localhost"
PORT = 9999
current_crf = 45              # 0 (best) -> 51 (worst)
LOG_BANDWIDTH_EVERY_SEC = 60
FIXED_CRF = None              # Set None to enable adaptive CRF
GOP_SIZE = 10                  # I-frame interval (frames between keyframes)
CODEC_DEVICE = None            # None = auto-detect (CUDA > CPU)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open camera index 0. Check that your webcam is connected and not in use.")

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((SERVER_IP, PORT))
masker = OpticalFlowMasker(motion_threshold=3.0, min_contour_area=1200)

# Initialize neural codec
initial_quality = NeuralCodec.crf_to_quality(current_crf)
codec = NeuralCodec(quality=initial_quality, device=CODEC_DEVICE, gop_size=GOP_SIZE)


def recv_exact(sock, size):
    """Receive exactly `size` bytes from TCP stream."""
    chunks = []
    received = 0
    while received < size:
        chunk = sock.recv(size - received)
        if not chunk:
            raise ConnectionError("Socket closed while receiving confidence metric.")
        chunks.append(chunk)
        received += len(chunk)
    return b"".join(chunks)


def estimate_full_frame_bytes(frame, crf_val):
    """Estimate full-frame JPEG size (for baseline comparison)."""
    jpeg_quality = max(5, min(95, 100 - int(crf_val) * 2))
    ok, encoded = cv2.imencode(
        ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
    )
    return len(encoded) if ok else 0


print("Camera Node: Neural Codec (DCVC-RT style) Adaptive Controller Active.")

try:
    conf_score = 0.0
    header_bytes = struct.calcsize("Q")
    interval_start_ts = time.time()
    interval_sent_bytes = 0
    interval_baseline_bytes = 0
    interval_iframe_count = 0
    interval_pframe_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera read failed — no frame returned.")
            break

        frame = cv2.resize(frame, (640, 480))
        masked_frame, roi_ratio = masker.apply(frame)
        active_crf = FIXED_CRF if FIXED_CRF is not None else current_crf

        # Adapt neural codec quality from CRF
        target_quality = NeuralCodec.crf_to_quality(active_crf)
        codec.set_quality(target_quality)

        # Encode with neural codec (I-frame or P-frame decided internally)
        data, frame_type = codec.encode(masked_frame)
        sent_bytes = len(data) + header_bytes

        if frame_type == 0:
            interval_iframe_count += 1
        else:
            interval_pframe_count += 1

        # Baseline: approximate JPEG full-frame cost
        full_est_bytes = estimate_full_frame_bytes(frame, active_crf) + header_bytes
        interval_sent_bytes += sent_bytes
        interval_baseline_bytes += full_est_bytes

        # Send compressed frame
        client_socket.sendall(struct.pack("Q", len(data)) + data)

        # Receive AI metric (4-byte network-order float)
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

        # Decode locally for dashboard preview
        try:
            compressed_frame = codec.decode(data)
        except Exception:
            compressed_frame = np.zeros_like(frame)

        # Single-window dashboard UI
        cv2.putText(
            frame, "Source", (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2,
        )
        cv2.putText(
            compressed_frame, "Neural Codec Decoded", (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2,
        )

        dashboard = cv2.hconcat([frame, compressed_frame])
        y0 = 470
        cv2.putText(
            dashboard, f"ROI: {roi_ratio*100:.1f}%", (20, y0),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
        )
        cv2.putText(
            dashboard, f"Q: {target_quality} (CRF {active_crf})", (200, y0),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2,
        )
        ftype_label = "I" if frame_type == 0 else "P"
        cv2.putText(
            dashboard, f"Frame: {ftype_label}", (480, y0),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2,
        )
        cv2.putText(
            dashboard, f"Conf: {conf_score:.2f}", (640, y0),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2,
        )
        cv2.putText(
            dashboard,
            f"Mode: {'Fixed' if FIXED_CRF is not None else 'Adaptive'}",
            (820, y0),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 255, 0), 2,
        )
        cv2.putText(
            dashboard, f"Size: {len(data)/1024:.1f}KB", (1020, y0),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2,
        )

        cv2.imshow("Edge Node Dashboard", dashboard)

        elapsed = time.time() - interval_start_ts
        if elapsed >= LOG_BANDWIDTH_EVERY_SEC:
            saved_bytes = interval_baseline_bytes - interval_sent_bytes
            saved_pct = (
                (saved_bytes / interval_baseline_bytes) * 100.0
                if interval_baseline_bytes > 0
                else 0.0
            )
            sent_mbps = (interval_sent_bytes * 8.0) / elapsed / 1_000_000
            baseline_mbps = (interval_baseline_bytes * 8.0) / elapsed / 1_000_000
            total_frames = interval_iframe_count + interval_pframe_count
            print(
                f"[BW] {elapsed:.1f}s "
                f"| sent={interval_sent_bytes/1024/1024:.2f} MB "
                f"| baseline={interval_baseline_bytes/1024/1024:.2f} MB "
                f"| saved={saved_bytes/1024/1024:.2f} MB ({saved_pct:.1f}%) "
                f"| sent_rate={sent_mbps:.2f} Mbps "
                f"| baseline_rate={baseline_mbps:.2f} Mbps "
                f"| frames={total_frames} (I={interval_iframe_count} P={interval_pframe_count})"
            )
            interval_start_ts = time.time()
            interval_sent_bytes = 0
            interval_baseline_bytes = 0
            interval_iframe_count = 0
            interval_pframe_count = 0

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

except Exception as e:
    print(f"\n*** ERROR: {e}")
    import traceback; traceback.print_exc()
finally:
    cap.release()
    client_socket.close()
    cv2.destroyAllWindows()
