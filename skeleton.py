for frame in stream:
    detections = yolo_detect(frame)  # person bbox

    tracks = tracker.update(detections)  # optional but recommended

    pose_results = []
    for trk in tracks:
        if trk.cls != "person":
            continue
        if trk.conf < 0.4:
            continue

        x1, y1, x2, y2 = trk.bbox
        w, h = x2 - x1, y2 - y1
        if h < 64 or w < 24:
            continue  # too small for reliable pose

        roi = frame[y1:y2, x1:x2]
        keypoints, kp_conf = pose_model(roi)

        # map ROI coords back to frame coords
        keypoints_global = remap_to_frame(keypoints, x1, y1)

        motion_score = compute_motion_score(
            track_id=trk.id,
            keypoints=keypoints_global,
            conf=kp_conf,
            bbox=(x1, y1, x2, y2)
        )

        pose_results.append({
            "track_id": trk.id,
            "pose_conf": kp_conf.mean(),
            "motion_score": motion_score
        })

    stream_policy = adaptive_controller.update(
        detections=detections,
        pose_results=pose_results,
        network_stats=get_network_stats()
    )

    ffmpeg_encoder.apply_policy(stream_policy)