import cv2
import numpy as np

from src.optical_flow.motion_comp import estimate_affine_warp


def compute_flow(prev_gray, curr_gray, dis, compensate_global_motion=True, global_flow_mode="affine"):
    flow = None
    if compensate_global_motion and global_flow_mode == "affine":
        warped_prev = estimate_affine_warp(prev_gray, curr_gray)
        if warped_prev is not None:
            flow = dis.calc(warped_prev, curr_gray, None)

    if flow is None:
        flow = dis.calc(prev_gray, curr_gray, None)

    if compensate_global_motion and global_flow_mode in ("mean", "median"):
        if global_flow_mode == "mean":
            global_flow = np.mean(flow.reshape(-1, 2), axis=0)
        else:
            global_flow = np.median(flow.reshape(-1, 2), axis=0)
        flow = flow - global_flow

    return flow
