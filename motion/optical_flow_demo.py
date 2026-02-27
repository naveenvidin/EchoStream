import cv2
import numpy as np

cap = cv2.VideoCapture(0)

ret, frame1 = cap.read()
if not ret:
    print("Camera not accessible")
    exit()

prev_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

original_bitrate_est = 5  # Mbps baseline assumption

while True:
    ret, frame2 = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # ---------- Optical Flow ----------
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, gray, None,
        0.5, 2, 15, 2, 5, 1.1, 0
    )

    magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # ---------- Motion Threshold ----------
    motion_mask = (magnitude > 3).astype(np.uint8) * 255

    kernel = np.ones((7, 7), np.uint8)
    motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        motion_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    output = np.zeros_like(frame2)

    frame_area = frame2.shape[0] * frame2.shape[1]
    roi_area = 0

    for cnt in contours:
        if cv2.contourArea(cnt) > 1200:
            x, y, w, h = cv2.boundingRect(cnt)
            roi_area += w * h
            output[y:y+h, x:x+w] = frame2[y:y+h, x:x+w]
            cv2.rectangle(output, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # ---------- Real Compression Comparison ----------
    _, encoded_full = cv2.imencode('.jpg', frame2)
    _, encoded_masked = cv2.imencode('.jpg', output)

    full_size = len(encoded_full)
    masked_size = len(encoded_masked)

    if full_size > 0:
        saved_percent = (1 - masked_size / full_size) * 100
    else:
        saved_percent = 0

    roi_ratio = roi_area / frame_area if frame_area > 0 else 0

    # ---------- Overlay Stats ----------
    cv2.putText(output, f"ROI: {roi_ratio*100:.2f}%", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    cv2.putText(output, f"Bandwidth Saved: {saved_percent:.2f}%", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    cv2.imshow("Adaptive Motion Stream", output)

    prev_gray = gray

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
