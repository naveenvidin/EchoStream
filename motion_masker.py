import cv2
import numpy as np


class OpticalFlowMasker:
    def __init__(self, motion_threshold=3.0, min_contour_area=1200, morph_kernel_size=7):
        self.motion_threshold = motion_threshold
        self.min_contour_area = min_contour_area
        self.kernel = np.ones((morph_kernel_size, morph_kernel_size), np.uint8)
        self.prev_gray = None

    def apply(self, frame_bgr):
        """
        Returns:
            masked_frame: motion-ROI-only BGR frame
            roi_ratio: fraction of image area covered by selected ROIs
        """
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        if self.prev_gray is None:
            self.prev_gray = gray
            return frame_bgr.copy(), 1.0

        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray, gray, None,
            0.5, 2, 15, 2, 5, 1.1, 0
        )

        magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        motion_mask = (magnitude > self.motion_threshold).astype(np.uint8) * 255
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, self.kernel)

        contours, _ = cv2.findContours(
            motion_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        masked = np.zeros_like(frame_bgr)
        frame_area = frame_bgr.shape[0] * frame_bgr.shape[1]
        roi_area = 0

        for cnt in contours:
            if cv2.contourArea(cnt) > self.min_contour_area:
                x, y, w, h = cv2.boundingRect(cnt)
                masked[y:y + h, x:x + w] = frame_bgr[y:y + h, x:x + w]
                roi_area += w * h

        self.prev_gray = gray
        roi_ratio = (roi_area / frame_area) if frame_area > 0 else 0.0
        return masked, roi_ratio
