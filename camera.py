import cv2, socket, struct, time, csv

# === CONFIGURATION ===
TEST_NAME = "SMART" # Change this to match the server mode
# =====================

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect(('localhost', 9999))

log_file = open(f'bandwidth_log_{TEST_NAME}.csv', mode='w', newline='')
log_writer = csv.writer(log_file)
log_writer.writerow(['Timestamp', 'Size_KB', 'Quality'])

cap = cv2.VideoCapture(0)
quality = 15; total_bytes = 0; frame_count = 0; start_time = time.time()

try:
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.resize(frame, (640, 480))

        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, img_encoded = cv2.imencode('.jpg', frame, encode_param)
        data = img_encoded.tobytes()

        # Log & Send
        log_writer.writerow([time.time() - start_time, len(data)/1024, quality])
        total_bytes += len(data); frame_count += 1
        client_socket.sendall(struct.pack("Q", len(data)) + data)
        
        feedback = client_socket.recv(1024).decode()
        quality = 90 if feedback == "HIGH" else 15

        cv2.imshow("Camera Node", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
finally:
    print(f"\nTest {TEST_NAME} Complete. Avg Size: {(total_bytes/1024)/frame_count:.2f} KB")
    cap.release(); client_socket.close(); log_file.close(); cv2.destroyAllWindows()