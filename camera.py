import cv2
import socket
import struct
import numpy as np
import ffmpeg

# === CONFIGURATION ===
SERVER_IP = 'localhost'
PORT = 9999
current_crf = 45  # 0 (best) → 51 (worst)

cap = cv2.VideoCapture(0)

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((SERVER_IP, PORT))


def ffmpeg_compress(frame, crf_val):
    """
    Compress a single frame using ffmpeg-python.
    """

    height, width, _ = frame.shape
    raw_bytes = frame.tobytes()

    # Map CRF-like value to MJPEG quality scale
    bitrate_val = str(max(1, int(crf_val / 2)))

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
            **{'b:v': bitrate_val}
        )
        .global_args('-loglevel', 'quiet')
        .run_async(pipe_stdin=True, pipe_stdout=True)
    )

    out, _ = process.communicate(input=raw_bytes)
    return out


print("Camera Node: FFmpeg-Python Adaptive Controller Active.")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 480))

        # Compress frame using FFmpeg wrapper
        data = ffmpeg_compress(frame, current_crf)

        # Send compressed frame
        client_socket.sendall(struct.pack("Q", len(data)) + data)

        # Receive AI metric
        response = client_socket.recv(1024).decode()

        try:
            conf_score = float(response)

            # Inverse Mapping:
            # High confidence → high CRF (lower quality)
            # Low confidence → low CRF (higher quality)
            current_crf = int((1.0 - conf_score) * 50)
            current_crf = max(5, min(50, current_crf))

        except ValueError:
            conf_score = 0.0

        # UI
        cv2.putText(frame, f"FFmpeg CRF Knob: {current_crf}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 0, 0),
                    2)

        cv2.imshow("Edge Node: Adaptive Frame Processing", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    client_socket.close()
    cv2.destroyAllWindows()