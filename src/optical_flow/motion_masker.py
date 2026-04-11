import cv2
import numpy as np

from src.optical_flow.blur import make_blurred
from src.optical_flow.flow import compute_flow
from src.optical_flow.importance import (
    apply_soft_cap,
    combine_importance,
    composite_frame,
    normalize_motion,
    resize_object_map,
)


class OpticalFlowMasker:
    def __init__(
        self,
        motion_threshold=3.0,
        min_contour_area=1200,
        morph_kernel_size=7,
        dis_preset=cv2.DISOPTICAL_FLOW_PRESET_MEDIUM,
        compensate_global_motion=True,
        global_flow_mode="affine",
        max_roi_ratio=0.4,
        soft_cap_percentile=95.0,
        blur_kernel_size=21,
        importance_alpha=0.3,
        importance_beta=0.7,
        importance_low_thresh=0.1,
        importance_high_thresh=0.3,
        motion_norm_percentile=90.0,
    ):
        self.motion_threshold = motion_threshold
        self.min_contour_area = min_contour_area
        self.kernel = np.ones((morph_kernel_size, morph_kernel_size), np.uint8)
        self.prev_gray = None
        self.dis = cv2.DISOpticalFlow_create(dis_preset)
        self.compensate_global_motion = compensate_global_motion
        self.global_flow_mode = global_flow_mode
        self.max_roi_ratio = max_roi_ratio
        self.soft_cap_percentile = soft_cap_percentile
        self.blur_kernel_size = blur_kernel_size
        self.importance_alpha = importance_alpha
        self.importance_beta = importance_beta
        self.importance_low_thresh = importance_low_thresh
        self.importance_high_thresh = importance_high_thresh
        self.motion_norm_percentile = motion_norm_percentile

    def apply(self, frame_bgr, object_score_map=None):
        """
        Returns:
            masked_frame: composite frame with black/blur/sharp regions
            roi_ratio: fraction of image area covered by mid+high importance
        """
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        blurred = make_blurred(frame_bgr, self.blur_kernel_size)

        if self.prev_gray is None:
            self.prev_gray = gray
            return blurred.copy(), 0.0

        flow = compute_flow(
            self.prev_gray,
            gray,
            self.dis,
            compensate_global_motion=self.compensate_global_motion,
            global_flow_mode=self.global_flow_mode,
        )

        magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        frame_area = frame_bgr.shape[0] * frame_bgr.shape[1]

        motion_norm = normalize_motion(magnitude, self.motion_norm_percentile)
        obj_norm = resize_object_map(object_score_map, motion_norm.shape)

        importance = combine_importance(
            motion_norm,
            obj_norm,
            self.importance_alpha,
            self.importance_beta,
        )

        low_thresh, high_thresh = apply_soft_cap(
            importance,
            self.importance_low_thresh,
            self.importance_high_thresh,
            self.max_roi_ratio,
            self.soft_cap_percentile,
        )

        masked, roi_ratio = composite_frame(
            frame_bgr,
            blurred,
            importance,
            obj_norm,
            low_thresh,
            high_thresh,
        )

        self.prev_gray = gray
        return masked, roi_ratio
