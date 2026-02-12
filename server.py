import socket, struct, cv2, torch, time, csv
import numpy as np
from ultralytics import YOLO

# === CONFIGURATION ===
MODE = "SMART"  # Toggle between "SMART" and "BASELINE"
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
# =====================

model = YOLO('yolov8n.pt').to(device)
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server_socket.bind(('0.0.0.0', 9999))
server_socket.listen(1)

# Accuracy Logging
log_file = open(f'accuracy_log_{MODE}.csv', mode='w', newline='')
log_writer = csv.writer(log_file)
log_writer.writerow(['Timestamp', 'Avg_Confidence', 'Object_Count', 'Mode'])

print(f"Server [{MODE} MODE] active. Waiting for Camera...")

try:
    conn, addr = server_socket.accept()
    data = b""
    payload_size = struct.calcsize("Q")
    start_time = time.time()

    while True:
        # --- RECEIVE DATA ---
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

        # --- INFERENCE ---
        nparr = np.frombuffer(frame_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is not None:
            results = model(frame, verbose=False, device=device)
            
            # Calculate Accuracy Metrics
            confidences = [box.conf[0].item() for r in results for box in r.boxes]
            avg_conf = np.mean(confidences) if confidences else 0
            obj_count = len(confidences)
            
            # Logic for Feedback
            if MODE == "BASELINE":
                feedback = "HIGH" # Ground Truth mode
            else:
                feedback = "HIGH" if any(c < 0.65 for c in confidences) or obj_count == 0 else "LOW"

            conn.sendall(feedback.encode())
            log_writer.writerow([time.time() - start_time, avg_conf, obj_count, feedback])

            # Visuals
            annotated_frame = results[0].plot()
            cv2.putText(annotated_frame, f"MODE: {MODE} | Send: {feedback}", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Server AI View", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'): break
finally:
    log_file.close()
    conn.close()
    server_socket.close()
    cv2.destroyAllWindows()