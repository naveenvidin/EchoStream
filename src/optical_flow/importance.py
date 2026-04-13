import cv2
import numpy as np


def normalize_motion(magnitude, percentile):
    denom = float(np.percentile(magnitude, percentile))
    if denom <= 1e-6:
        denom = 1.0
    return np.clip(magnitude / denom, 0.0, 1.0)


def resize_object_map(object_score_map, target_shape):
    if object_score_map is None:
        return 0.0
    h, w = target_shape
    obj_norm = cv2.resize(
        object_score_map,
        (w, h),
        interpolation=cv2.INTER_LINEAR,
    )
    return np.clip(obj_norm, 0.0, 1.0)


def combine_importance(motion_norm, obj_norm, alpha, beta):
    importance = (alpha * motion_norm) + (beta * obj_norm)
    return np.clip(importance, 0.0, 1.0)


def apply_soft_cap(importance, low_thresh, high_thresh, max_roi_ratio, soft_cap_percentile):
    if soft_cap_percentile is None:
        return low_thresh, high_thresh
    broad_ratio = float(np.mean(importance >= low_thresh))
    if broad_ratio > max_roi_ratio:
        tighten = float(np.percentile(importance, soft_cap_percentile))
        low_thresh = max(low_thresh, tighten)
        high_thresh = max(high_thresh, low_thresh + 0.05)
    return low_thresh, high_thresh


def composite_frame(frame_bgr, blurred, importance, obj_norm, low_thresh, high_thresh):
    low_mask = importance < low_thresh
    high_mask = importance >= high_thresh

    masked = blurred.copy()
    masked[high_mask] = frame_bgr[high_mask]

    if isinstance(obj_norm, np.ndarray):
        obj_mask = obj_norm > 0.0
        masked[obj_mask] = frame_bgr[obj_mask]

    roi_ratio = float(np.mean(~low_mask)) if importance.size > 0 else 0.0
    return masked, roi_ratio
