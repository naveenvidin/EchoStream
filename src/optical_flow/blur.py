import cv2


def make_blurred(frame_bgr, blur_kernel_size):
    k = blur_kernel_size
    if k % 2 == 0:
        k += 1
    if k < 3:
        k = 3
    return cv2.GaussianBlur(frame_bgr, (k, k), 0)
