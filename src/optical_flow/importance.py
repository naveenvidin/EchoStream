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


def apply_soft_cap(
    importance,
    low_thresh,
    high_thresh,
    max_roi_ratio,
    soft_cap_percentile,
    valid_mask=None,
):
    if soft_cap_percentile is None:
        return low_thresh, high_thresh

    values = importance
    if valid_mask is not None:
        try:
            values = importance[valid_mask]
        except Exception:
            values = importance
    if isinstance(values, np.ndarray) and values.size == 0:
        values = importance

    broad_ratio = float(np.mean(values >= low_thresh))
    if broad_ratio > max_roi_ratio:
        tighten = float(np.percentile(values, soft_cap_percentile))
        # Only tighten the low threshold. Keep high_thresh stable so
        # "high importance" remains an absolute band.
        low_thresh = max(low_thresh, tighten)
        low_thresh = min(low_thresh, float(high_thresh) - 1e-2)
    return low_thresh, high_thresh


def composite_frame(frame_bgr, blurred_mid, blurred_low, importance, obj_norm, low_thresh, high_thresh):
    low_mask = importance < low_thresh
    high_mask = importance >= high_thresh

    masked = blurred_mid.copy()
    masked[low_mask] = blurred_low[low_mask]
    masked[high_mask] = frame_bgr[high_mask]

    roi_ratio = float(np.mean(~low_mask)) if importance.size > 0 else 0.0
    return masked, roi_ratio
