"""Edge inference server — H.264 receiver + open-vocab YOLO-World detector.

Loop:
  1. Listen on a TCP port, accept one camera.
  2. Read the v3 handshake (class list + requested heatmap resolution).
  3. Apply the prompts and heatmap size to the detector, warm it up.
  4. On each received H.264 packet: decode → infer → reply with
     (metric, low-res heatmap, detections, decode_us, infer_us, sequence_id).

CLI:
    python -m src.inference.server_h264 \
        --model yolov8s-world.pt \
        --classes "person,car,dog" \
        --device auto

The --classes flag is the fallback used if the camera handshake is
empty. Otherwise the camera is source-of-truth for prompts.
"""
from __future__ import annotations

import argparse
import logging
import socket
import time

import cv2
import numpy as np


# === CONFIGURATION ===
PORT = 9999
WIDTH, HEIGHT = 640, 480


log = logging.getLogger("echostream.server")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="EchoStream edge inference server (YOLO-World)."
    )
    p.add_argument("--model", default="yolov8s-world.pt",
                   help="YOLO-World weights path (e.g. yolov8s-world.pt).")
    p.add_argument("--classes", default="person",
                   help="Comma-separated fallback class prompts.")
    p.add_argument("--device", default="auto",
                   choices=("auto", "cuda", "mps", "cpu"),
                   help="Inference device. 'auto' picks CUDA→MPS→CPU.")
    p.add_argument("--port", type=int, default=PORT)
    p.add_argument("--width", type=int, default=WIDTH)
    p.add_argument("--height", type=int, default=HEIGHT)
    p.add_argument("--conf-threshold", type=float, default=0.05,
                   help="Low-pass filter on detector confidence.")
    p.add_argument("--show-window", action="store_true",
                   help="Show server-side YOLO annotations window.")
    return p.parse_args()


def main():
    args = _parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Imports deferred so `--help` is cheap.
    from src.codec import wire
    from src.codec.h264_backend import H264DecoderBackend
    from src.inference.detection import YoloWorldDetector, parse_classes
    from src.inference.protocol import (
        DEFAULT_HEATMAP_WH, encode_response, read_handshake,
    )

    fallback_classes = parse_classes(args.classes) or ["object"]

    log.info("loading YOLO-World model=%s device=%s", args.model, args.device)
    detector = YoloWorldDetector(
        model_path=args.model, device=args.device,
        conf_threshold=args.conf_threshold,
        heatmap_wh=DEFAULT_HEATMAP_WH,
    )
    detector.set_classes(fallback_classes)
    log.info("fallback classes=%s (device=%s)", fallback_classes, detector.device)

    decoder = H264DecoderBackend(width=args.width, height=args.height)

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind(("0.0.0.0", args.port))
    server_socket.listen(1)
    log.info("EchoStream server listening on 0.0.0.0:%d", args.port)

    conn = None
    try:
        conn, addr = server_socket.accept()
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        log.info("accepted connection from %s", addr)

        # ── Handshake v3 ────────────────────────────────────────────────────
        heat_wh = DEFAULT_HEATMAP_WH
        try:
            camera_classes, heat_wh = read_handshake(conn)
            if camera_classes:
                detector.set_classes(camera_classes)
                log.info("camera prompted classes=%s", camera_classes)
            else:
                log.warning("empty class list from camera; keeping fallback")
            if heat_wh[0] > 0 and heat_wh[1] > 0:
                detector.set_heatmap_size(*heat_wh)
            log.info("reply heatmap size=%dx%d", *detector.heatmap_size)
        except ConnectionError as e:
            log.warning("handshake failed: %s (using fallback classes)", e)

        # Warmup so the first real inference doesn't include autotune time.
        try:
            detector.warmup(height=args.height, width=args.width)
            log.info("detector warmup complete")
        except Exception as e:
            log.warning("detector warmup skipped: %s", e)

        frame_idx = 0
        heat_w, heat_h = detector.heatmap_size
        empty_heat = np.zeros((heat_h, heat_w), dtype=np.uint8)

        while True:
            # ── Read one H.264 packet (with correlation id) ─────────────────
            pkt, seq_id = wire.read_packet(conn, frame_index=frame_idx)
            frame_idx += 1

            # ── Decode → BGR (may be None during GOP warmup) ─────────────────
            t_decode_start = time.perf_counter()
            decoder.push(pkt)
            frame = decoder.get_frame()
            decode_us = int((time.perf_counter() - t_decode_start) * 1_000_000)

            if frame is None:
                conn.sendall(encode_response(
                    0.5, empty_heat, [],
                    decode_us=decode_us, infer_us=0,
                    sequence_id=seq_id,
                ))
                continue

            # ── Inference ────────────────────────────────────────────────────
            metric, heatmap, detections, infer_us = detector.infer(frame)
            conn.sendall(encode_response(
                metric, heatmap, detections,
                decode_us=decode_us, infer_us=infer_us,
                sequence_id=seq_id,
            ))

            # ── Optional server-side preview ─────────────────────────────────
            if args.show_window:
                vis = frame.copy()
                for (x1, y1, x2, y2, conf, cls_idx) in detections:
                    p1 = (int(x1), int(y1))
                    p2 = (int(x2), int(y2))
                    name = detector.class_names[cls_idx] \
                        if 0 <= cls_idx < len(detector.class_names) \
                        else str(cls_idx)
                    cv2.rectangle(vis, p1, p2, (0, 255, 255), 2)
                    cv2.putText(
                        vis, f"{name} {conf:.2f}",
                        (p1[0], max(0, p1[1] - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1,
                        cv2.LINE_AA,
                    )
                cv2.imshow("Edge Server: YOLO-World", vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    except ConnectionError as e:
        log.warning("connection closed: %s", e)
    finally:
        try:
            decoder.close()
        except Exception:
            pass
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
        server_socket.close()
        cv2.destroyAllWindows()
        log.info("server shutdown complete.")


if __name__ == "__main__":
    main()
