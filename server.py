import subprocess
import numpy as np
import cv2
import socket
from ultralytics import YOLO
import torch

STREAM_PORT = 5004
CONTROL_PORT = 6000

WIDTH = 640
HEIGHT = 480

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
model = YOLO("yolov8n.pt").to(device)

# ---------- CONTROL SOCKET ----------
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(("0.0.0.0", CONTROL_PORT))
server.listen(1)
print("Waiting for camera control connection...")
conn, addr = server.accept()
print("Camera connected for control.")

# ---------- START FFMPEG DECODER ----------
ffmpeg = subprocess.Popen([
    "ffmpeg",
    "-loglevel", "error",
    "-i", f"udp://127.0.0.1:{STREAM_PORT}",
    "-f", "rawvideo",
    "-pix_fmt", "bgr24",
    "-"
], stdout=subprocess.PIPE)

print("Receiving video stream...")

try:
    while True:
        raw = ffmpeg.stdout.read(WIDTH * HEIGHT * 3)
        if not raw:
            continue

        frame = np.frombuffer(raw, np.uint8).reshape((HEIGHT, WIDTH, 3))

        # ---------- YOLO INFERENCE ----------
        results = model(frame, verbose=False, device=device)

        human_detected = False
        for r in results:
            for box in r.boxes:
                if int(box.cls[0]) == 0:  # class 0 = person
                    human_detected = True

        feedback = "HIGH" if human_detected else "LOW"
        conn.sendall(feedback.encode())

        annotated = results[0].plot()
        cv2.putText(annotated, f"MODE: {feedback}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Server YOLO View", annotated)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    ffmpeg.terminate()
    conn.close()
    server.close()
    cv2.destroyAllWindows()
