import cv2
import socket
import struct
import subprocess
import threading
import time
import numpy as np

# === CONFIGURATION ===
SERVER_IP = 'localhost'
PORT = 9999
WIDTH, HEIGHT = 640, 480
FPS = 30
FRAMES_PER_SEGMENT = 30
CRF = 28  # fixed for now — YOLO-adaptive CRF comes later


# ─────────────────────────────────────────────
#  Segment Encoder
#  Takes a list of 30 BGR frames, encodes them
#  as a single H.264 blob via a one-shot ffmpeg
#  subprocess, and returns the raw bytes.
# ─────────────────────────────────────────────
class SegmentEncoder:
    def __init__(self, width: int, height: int, fps: int, gop: int):
        self.width = width
        self.height = height
        self.fps = fps
        self.gop = gop  # GOP = FRAMES_PER_SEGMENT so each segment is self-contained

    def encode(self, frames: list[np.ndarray], crf: int) -> bytes:
        """
        Concatenate all frames into one raw byte blob, pipe into ffmpeg,
        get back a self-contained H.264 bytestream.

        -g (GOP size) == number of frames so there's one keyframe per segment.
        -sc_threshold 0 disables scene-cut detection so ffmpeg doesn't
        insert extra keyframes mid-segment.
        """
        if not frames:
            return b''

        raw = b''.join(f.tobytes() for f in frames)

        cmd = [
            'ffmpeg', '-loglevel', 'quiet',
            '-f', 'rawvideo',
            '-pix_fmt', 'bgr24',
            '-s', f'{self.width}x{self.height}',
            '-r', str(self.fps),
            '-i', 'pipe:0',
            '-c:v', 'libx264',
            '-preset', 'ultrafast',
            '-tune', 'zerolatency',
            '-crf', str(crf),
            '-g', str(self.gop),
            '-sc_threshold', '0',
            '-f', 'h264',
            'pipe:1',
        ]

        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
        )
        out, _ = proc.communicate(input=raw)
        return out


# ─────────────────────────────────────────────
#  Send with 4-byte length header
#  Protocol: [uint32 big-endian length][H.264 bytes]
#  The server uses this to know exactly how many
#  bytes to read for one segment — needed later
#  for the YOLO confidence handshake.
# ─────────────────────────────────────────────
def send_segment(sock: socket.socket, data: bytes):
    header = struct.pack('!I', len(data))  # 4-byte big-endian unsigned int
    sock.sendall(header + data)


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────
def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((SERVER_IP, PORT))
    print(f"Camera: connected to {SERVER_IP}:{PORT}")

    encoder = SegmentEncoder(width=WIDTH, height=HEIGHT, fps=FPS, gop=FRAMES_PER_SEGMENT)

    # Semaphore(1): only one encode+send thread allowed at a time.
    # If encoding segment N is still running when segment N+1 is ready,
    # we block here briefly rather than launching a second thread.
    # In practice at 30fps/segment this should never block — encoding
    # 30 ultrafast H.264 frames takes well under 1 second.
    encode_sem = threading.Semaphore(1)

    def encode_and_send(frames: list[np.ndarray]):
        try:
            data = encoder.encode(frames, CRF)
            if data:
                send_segment(sock, data)
                print(f"Camera: sent segment — {len(data)/1024:.1f} KB")
        finally:
            encode_sem.release()

    current_segment: list[np.ndarray] = []

    print("Camera: capturing. Press 'q' to quit.")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (WIDTH, HEIGHT))
            current_segment.append(frame)

            # Show local raw feed so we can compare against server display
            cv2.imshow('Camera — Raw Feed', frame)

            if len(current_segment) >= FRAMES_PER_SEGMENT:
                # Hand off completed segment to encode thread
                segment_to_encode = current_segment
                current_segment = []  # immediately start filling next segment

                encode_sem.acquire() # code sits here if locked
                t = threading.Thread(
                    target=encode_and_send,
                    args=(segment_to_encode,),
                    daemon=True,
                )
                t.start()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        sock.close()
        cv2.destroyAllWindows()
        print("Camera: shutdown.")


if __name__ == '__main__':
    main()