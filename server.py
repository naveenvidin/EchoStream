import socket
import struct
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from neural_codec import NeuralCodec

# === CONFIGURATION ===
CODEC_DEVICE = None  # None = auto-detect (CUDA > MPS > CPU)
SHOW_SERVER_WINDOW = False

# Resolve device for YOLO
if torch.cuda.is_available():
    yolo_device = "cuda"
elif torch.backends.mps.is_available():
    yolo_device = "mps"
else:
    yolo_device = "cpu"

model = YOLO("yolov8n.pt").to(yolo_device)

# Neural codec decoder (quality set dynamically from bitstream header)
codec = NeuralCodec(quality=3, device=CODEC_DEVICE)

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server_socket.bind(("0.0.0.0", 9999))
server_socket.listen(1)

print("Edge Server: Neural Codec AI Inference Engine Active")

try:
    conn, addr = server_socket.accept()
    print(f"Connected: {addr}")
    data = b""
    payload_size = struct.calcsize("Q")

    while True:
        # Receive frame length header
        try:
            while len(data) < payload_size:
                packet = conn.recv(8192)
                if not packet:
                    break
                data += packet
        except ConnectionResetError:
            break
        if len(data) < payload_size:
            break

        packed_msg_size = data[:payload_size]
        data = data[payload_size:]
        msg_size = struct.unpack("Q", packed_msg_size)[0]

        # Receive frame payload
        try:
            while len(data) < msg_size:
                packet = conn.recv(8192)
                if not packet:
                    break
                data += packet
        except ConnectionResetError:
            break
        if len(data) < msg_size:
            break

        frame_data = data[:msg_size]
        data = data[msg_size:]

        # Decode with neural codec
        try:
            frame = codec.decode(frame_data)
        except Exception as e:
            print(f"Decode error: {e}")
            conn.sendall(struct.pack("!f", 0.5))
            continue

        if frame is not None:
            results = model(frame, verbose=False, device=yolo_device)
            confidences = [box.conf[0].item() for r in results for box in r.boxes]

            # Send metric: min confidence score
            min_conf = min(confidences) if confidences else 0.5
            conn.sendall(struct.pack("!f", float(min_conf)))

            if SHOW_SERVER_WINDOW:
                cv2.imshow("Edge Server: YOLO Inference", results[0].plot())

        if SHOW_SERVER_WINDOW and cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    conn.close()
    server_socket.close()
    cv2.destroyAllWindows()
