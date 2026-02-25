import socket, struct, cv2, torch
import numpy as np
from ultralytics import YOLO


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
model = YOLO('yolo26n.pt').to(device)


server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server_socket.bind(('0.0.0.0', 9999))
server_socket.listen(1)

print("Edge Server: AI Inference Engine Active")

try:
    conn, addr = server_socket.accept()
    data = b""; payload_size = struct.calcsize("Q")

    while True:
        while len(data) < payload_size:
            packet = conn.recv(8192)
            if not packet: break
            data += packet
        if len(data) < payload_size: break
        
        packed_msg_size = data[:payload_size]
        data = data[payload_size:]; msg_size = struct.unpack("Q", packed_msg_size)[0]

        while len(data) < msg_size:
            packet = conn.recv(8192); data += packet
        
        frame_data = data[:msg_size]; data = data[msg_size:]
        nparr = np.frombuffer(frame_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is not None:
            results = model(frame, verbose=False, device=device)
            confidences = [box.conf[0].item() for r in results for box in r.boxes]
            
            # Send Metric: Min confidence score (Inference Engine logic)
            min_conf = min(confidences) if confidences else 1.0
            conn.sendall(str(round(min_conf, 2)).encode())

            cv2.imshow("Edge Server: YOLO Inference", results[0].plot())
            
        if cv2.waitKey(1) & 0xFF == ord('q'): break
finally:
    conn.close(); server_socket.close(); cv2.destroyAllWindows()
