import ffmpeg
import numpy as np
import cv2
import torch
import socket
from ultralytics import YOLO

WIDTH, HEIGHT = 640, 480
VIDEO_PORT = 10000
CONTROL_PORT = 10001
device = 'mps' if torch.backends.mps.is_available() else 'cpu'

print("[SERVER] Loading YOLO...")
model = YOLO("yolov8n.pt").to(device)

# ---- Control Socket ----
control_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
control_sock.bind(("0.0.0.0", CONTROL_PORT))
control_sock.listen(1)
print("[SERVER] Waiting for control connection...")
control_conn, _ = control_sock.accept()
print("[SERVER] Control connected.")

# ---- Video Stream ----
print("[SERVER] Waiting for video stream...")
process = (
    ffmpeg
    .input(f'tcp://0.0.0.0:{VIDEO_PORT}?listen=1', format='mpegts')
    .output('pipe:', format='rawvideo', pix_fmt='bgr24')
    .run_async(pipe_stdout=True)
)

frame_size = WIDTH * HEIGHT * 3
print("[SERVER] Video stream connected.")

try:
    while True:
        in_bytes = process.stdout.read(frame_size)
        if len(in_bytes) != frame_size:
            continue

        frame = np.frombuffer(in_bytes, np.uint8).reshape((HEIGHT, WIDTH, 3))

        results = model(frame, verbose=False, device=device)

        confidences = [box.conf[0].item() for r in results for box in r.boxes]
        mean_conf = np.mean(confidences) if confidences else 0

        print(f"[SERVER] Mean Confidence: {mean_conf:.3f}")

        # ---- Simple Adaptive CRF Logic ----
        if mean_conf < 0.6:
            new_crf = 20   # higher quality
        else:
            new_crf = 35   # lower quality

        control_conn.sendall(str(new_crf).encode())

        annotated = results[0].plot()
        cv2.putText(annotated, f"Conf: {mean_conf:.2f} | CRF: {new_crf}",
                    (20,40), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0,255,0), 2)

        cv2.imshow("Server AI View", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    process.stdout.close()
    process.wait()
    control_conn.close()
    cv2.destroyAllWindows()
