import cv2
import ffmpeg
import socket

WIDTH, HEIGHT = 640, 480
FPS = 30
SERVER_IP = "127.0.0.1"
VIDEO_PORT = 10000
CONTROL_PORT = 10001

current_crf = 28

print("[CAMERA] Connecting control socket...")
control_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
control_sock.connect((SERVER_IP, CONTROL_PORT))
print("[CAMERA] Control connected.")

def start_encoder(crf):
    print(f"[CAMERA] Starting encoder with CRF {crf}")
    return (
        ffmpeg
        .input(
            'pipe:',
            format='rawvideo',
            pix_fmt='bgr24',
            s=f'{WIDTH}x{HEIGHT}',
            framerate=FPS
        )
        .output(
            f'tcp://{SERVER_IP}:{VIDEO_PORT}?listen=0',
            format='mpegts',
            vcodec='libx264',
            preset='ultrafast',
            tune='zerolatency',
            pix_fmt='yuv420p',
            g=30,
            bf=0,
            crf=crf
        )
        .run_async(pipe_stdin=True)
    )

encoder = start_encoder(current_crf)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Camera failed to open")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (WIDTH, HEIGHT))
        encoder.stdin.write(frame.tobytes())

        # ---- Check for CRF update ----
        control_sock.setblocking(False)
        try:
            data = control_sock.recv(1024)
            if data:
                new_crf = int(data.decode())
                if new_crf != current_crf:
                    current_crf = new_crf
                    encoder.stdin.close()
                    encoder.wait()
                    encoder = start_encoder(current_crf)
        except:
            pass

        cv2.imshow("Camera Node", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    encoder.stdin.close()
    encoder.wait()
    cap.release()
    control_sock.close()
    cv2.destroyAllWindows()
