from dataclasses import dataclass


@dataclass
class MaskerConfig:
    motion_threshold: float = 3.0
    min_contour_area: int = 1200
    morph_kernel_size: int = 7
    dis_preset: int = 1  # cv2.DISOPTICAL_FLOW_PRESET_MEDIUM
    compensate_global_motion: bool = True
    global_flow_mode: str = "affine"
    max_roi_ratio: float = 0.4
    soft_cap_percentile: float = 95.0
    blur_kernel_size: int = 21
    importance_alpha: float = 0.3
    importance_beta: float = 0.7
    importance_low_thresh: float = 0.1
    importance_high_thresh: float = 0.3
    motion_norm_percentile: float = 90.0
