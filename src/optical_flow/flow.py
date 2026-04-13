import cv2
import numpy as np

from src.optical_flow.motion_comp import estimate_affine_warp


def warp_with_flow(prev_gray, flow):
    h, w = prev_gray.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (grid_x + flow[..., 0]).astype(np.float32)
    map_y = (grid_y + flow[..., 1]).astype(np.float32)
    return cv2.remap(
        prev_gray,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )


def compute_flow(
    prev_gray,
    curr_gray,
    dis,
    compensate_global_motion=True,
    global_flow_mode="affine",
    warp_mode="affine",
):
    flow = None
    if compensate_global_motion and warp_mode == "affine" and global_flow_mode == "affine":
        warped_prev = estimate_affine_warp(prev_gray, curr_gray)
        if warped_prev is not None:
            flow = dis.calc(warped_prev, curr_gray, None)

    if flow is None:
        flow = dis.calc(prev_gray, curr_gray, None)

    if compensate_global_motion and warp_mode == "flow":
        warped_prev = warp_with_flow(prev_gray, flow)
        flow = dis.calc(warped_prev, curr_gray, None)

    if compensate_global_motion and warp_mode != "flow" and global_flow_mode in ("mean", "median"):
        if global_flow_mode == "mean":
            global_flow = np.mean(flow.reshape(-1, 2), axis=0)
        else:
            global_flow = np.median(flow.reshape(-1, 2), axis=0)
        flow = flow - global_flow

    return flow
