import cv2
import socket
import struct
import subprocess
import numpy as np

# === CONFIGURATION ===
SERVER_IP = 'localhost'
PORT = 9999
current_crf = 45  # Standard FFmpeg CRF range is 0 (lossless) to 51 (worst)

cap = cv2.VideoCapture(0)
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((SERVER_IP, PORT))

def ffmpeg_compress(frame, crf_val):
    """
    Pipes a raw frame into FFmpeg to apply actual CRF compression.
    Matches the 'Video Compression Controller' in the HDL.
    """
    # Convert frame to raw bytes
    raw_frame = frame.tobytes()
    height, width, _ = frame.shape

    # FFmpeg command: Read raw pixels from stdin -> Output compressed image to stdout
    # We use mjpeg format here for single-frame simplicity, but with the -q:v (quality) flag
    # In FFmpeg, lower -q:v is better quality.
    ffmpeg_cmd = [
        'ffmpeg',
        '-y',
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-s', f'{width}x{height}',
        '-pix_fmt', 'bgr24',
        '-i', '-',  # Input from pipe
        '-vframes', '1',
        '-b:v', str(max(1, int(crf_val / 2))), # Mapping CRF-like logic to JPEG quality scales
        '-f', 'mjpeg',
        '-loglevel', 'quiet',
        '-' # Output to pipe
    ]

    process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    out, _ = process.communicate(input=raw_frame)
    return out

print("Camera Node: FFmpeg Subprocess Controller Active.")

try:
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.resize(frame, (640, 480))

        # 1. Video Compression Controller: Real FFmpeg Compression
        data = ffmpeg_compress(frame, current_crf)

        # 2. Send via Stream Client
        client_socket.sendall(struct.pack("Q", len(data)) + data)
        
        # 3. Receive AI Metric (Confidence Score) from Server
        response = client_socket.recv(1024).decode()
        try:
            conf_score = float(response)
            # Inverse Mapping: High Confidence -> High CRF (Low Quality)
            # Low Confidence -> Low CRF (High Quality)
            current_crf = int((1.0 - conf_score) * 50)
            current_crf = max(5, min(50, current_crf))
        except ValueError:
            conf_score = 0.0

        # UI
        cv2.putText(frame, f"FFmpeg CRF Knob: {current_crf}", (20, 40), 1, 1.5, (255, 0, 0), 2)
        cv2.imshow("Edge Node: Adaptive Frame Processing", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'): break
finally:
    cap.release()
    client_socket.close()
    cv2.destroyAllWindows()