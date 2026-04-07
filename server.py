import socket
import struct
import subprocess
import select
import threading
import cv2
import torch
import numpy as np
from ultralytics import YOLO

# === CONFIGURATION ===
FRAME_W = 640
FRAME_H = 480
FRAME_BYTES = FRAME_W * FRAME_H * 3  # exact byte size of one raw BGR frame
SHOW_SERVER_WINDOW = False

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
model = YOLO('yolov8n.pt').to(device)

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server_socket.bind(('0.0.0.0', 9999))
server_socket.listen(1)

latest_conf = 0.5
conf_lock = threading.Lock()


def start_decoder():
    """
    Spawns a persistent ffmpeg process that:
    - reads a raw H.264 bytestream from stdin
    - outputs raw BGR frames of fixed size to stdout
    """
    return subprocess.Popen(
        [
            'ffmpeg', '-y',
            '-f', 'h264',
            '-i', 'pipe:0',
            '-f', 'rawvideo',
            '-pix_fmt', 'bgr24',
            'pipe:1'
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL
    )


def read_exact_frame(decoder_stdout):
    """
    Reads exactly FRAME_BYTES from the decoder stdout.
    Accumulates chunks until a full frame is available.
    Returns the raw bytes, or None if the stream stalls or closes.
    """
    raw_frame = b""
    while len(raw_frame) < FRAME_BYTES:
        readable, _, _ = select.select([decoder_stdout], [], [], 0.1)
        if not readable:
            # Nothing arrived within 100ms — decoder still buffering
            return None
        chunk = decoder_stdout.read(FRAME_BYTES - len(raw_frame))
        if not chunk:
            # Stream closed
            return None
        raw_frame += chunk
    return raw_frame


def decode_and_infer(conn, decoder):
    """
    Background thread that:
    - reads raw H.264 bytes from the socket and feeds them to the decoder
    - reads decoded BGR frames from the decoder stdout
    - runs YOLO on each frame
    - sends confidence score back to camera
    """
    global latest_conf

    while True:
        # Feed incoming H.264 bytes to decoder stdin
        try:
            chunk = conn.recv(8192)
        except OSError:
            break
        if not chunk:
            break

        try:
            decoder.stdin.write(chunk)
            decoder.stdin.flush()
        except BrokenPipeError:
            break

        # Try to read a full decoded frame
        raw_frame = read_exact_frame(decoder.stdout)
        if raw_frame is None:
            # Decoder still buffering, keep feeding bytes
            continue

        # Reconstruct BGR frame for YOLO
        frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape((FRAME_H, FRAME_W, 3))

        results = model(frame, verbose=False, device=device)
        confidences = [box.conf[0].item() for r in results for box in r.boxes]
        min_conf = min(confidences) if confidences else 0.5

        # Send confidence back to camera
        try:
            conn.sendall(struct.pack("!f", float(min_conf)))
        except (BrokenPipeError, OSError):
            break

        with conf_lock:
            latest_conf = min_conf

        if SHOW_SERVER_WINDOW:
            cv2.imshow("Edge Server: YOLO Inference", results[0].plot())
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


print("Edge Server: AI Inference Engine Active")

try:
    conn, addr = server_socket.accept()
    print(f"Connection from {addr}")

    decoder = start_decoder()

    thread = threading.Thread(target=decode_and_infer, args=(conn, decoder), daemon=True)
    thread.start()
    thread.join()

finally:
    decoder.stdin.close()
    decoder.wait()
    conn.close()
    server_socket.close()
    cv2.destroyAllWindows()