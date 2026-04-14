import cv2


def estimate_affine_warp(prev_gray, curr_gray):
    p0 = cv2.goodFeaturesToTrack(
        prev_gray,
        maxCorners=200,
        qualityLevel=0.01,
        minDistance=7,
        blockSize=7,
    )
    if p0 is None:
        return None

    p1, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, p0, None)
    if p1 is None or st is None:
        return None

    good0 = p0[st.flatten() == 1]
    good1 = p1[st.flatten() == 1]
    if good0.shape[0] < 6:
        return None

    M, _ = cv2.estimateAffinePartial2D(
        good0,
        good1,
        method=cv2.RANSAC,
        ransacReprojThreshold=3.0,
    )
    if M is None:
        return None

    warped_prev = cv2.warpAffine(
        prev_gray,
        M,
        (prev_gray.shape[1], prev_gray.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )
    return warped_prev
