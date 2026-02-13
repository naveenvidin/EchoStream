import cv2
import subprocess
import socket
import threading

STREAM_PORT = 5004
CONTROL_PORT = 6000

WIDTH = 640
HEIGHT = 480
FPS = 30

CRF_HIGH = 18
CRF_LOW = 30

mode = "LOW"
current_crf = CRF_LOW

# ---------- CONTROL LISTENER ----------
def control_listener():
    global mode
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect(("127.0.0.1", CONTROL_PORT))
    while True:
        feedback = sock.recv(1024).decode()
        if feedback in ["HIGH", "LOW"]:
            mode = feedback

threading.Thread(target=control_listener, daemon=True).start()

# ---------- START FFMPEG ----------
def start_ffmpeg(crf):
    print(f"Starting encoder with CRF {crf}")
    return subprocess.Popen([
        "ffmpeg",
        "-loglevel", "error",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{WIDTH}x{HEIGHT}",
        "-r", str(FPS),
        "-i", "-",
        "-an",
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-tune", "zerolatency",
        "-crf", str(crf),
        "-f", "mpegts",
        f"udp://127.0.0.1:{STREAM_PORT}"
    ], stdin=subprocess.PIPE)

cap = cv2.VideoCapture(0)
ffmpeg = start_ffmpeg(current_crf)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Check if CRF needs changing
        desired_crf = CRF_HIGH if mode == "HIGH" else CRF_LOW
        if desired_crf != current_crf:
            ffmpeg.stdin.close()
            ffmpeg.terminate()
            ffmpeg.wait()
            current_crf = desired_crf
            ffmpeg = start_ffmpeg(current_crf)

        ffmpeg.stdin.write(frame.tobytes())

        cv2.imshow("Camera (Local View)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    ffmpeg.terminate()
    cv2.destroyAllWindows()
