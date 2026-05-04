#!/usr/bin/env python3
import argparse
import csv
import os
import sys
import tempfile
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="matplotlib-"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.streaming.camera_h264 import GOP_SIZE, HEIGHT, H264Decoder, H264Encoder, WIDTH


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep fixed CRF values and plot a BPP vs PSNR rate-distortion curve."
    )
    parser.add_argument(
        "--source",
        default="0",
        help="Video source path or camera index. Default: 0",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=None,
        help="Number of measured frames per CRF. Default: full file for video inputs, 180 for live camera.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=30,
        help="Warmup frames to skip before measurement. Default: 30",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Fallback FPS if capture does not report one. Default: 30",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=WIDTH,
        help=f"Frame width. Default: {WIDTH}",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=HEIGHT,
        help=f"Frame height. Default: {HEIGHT}",
    )
    parser.add_argument(
        "--gop",
        type=int,
        default=GOP_SIZE,
        help=f"GOP size. Default: {GOP_SIZE}",
    )
    parser.add_argument(
        "--crfs",
        type=int,
        nargs="+",
        default=[18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50],
        help="Fixed CRF values to test.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Output CSV path. Default: logs/crf_rd_<timestamp>.csv",
    )
    parser.add_argument(
        "--plot",
        type=Path,
        default=None,
        help="Output PNG path. Default: plots/crf_rd_<timestamp>.png",
    )
    return parser.parse_args()


def parse_source(source: str):
    return int(source) if source.isdigit() else source


def default_csv_path() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("logs") / f"crf_rd_{stamp}.csv"


def default_plot_path() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("plots") / f"crf_rd_{stamp}.png"


def open_capture(source, width: int, height: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(source)
    if isinstance(source, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def resolve_frame_budget(source, requested_frames: int | None) -> int | None:
    if requested_frames is not None:
        return requested_frames
    if isinstance(source, int):
        return 180
    return None


def average_psnr(total_psnr: float, frame_count: int) -> float:
    if frame_count == 0:
        raise RuntimeError("No decoded frames were matched for PSNR measurement.")
    return total_psnr / frame_count


def measure_rd_for_crf(
    source,
    crf: int,
    width: int,
    height: int,
    fps: float,
    gop: int,
    warmup: int,
    frames: int | None,
) -> dict[str, float]:
    cap = open_capture(source, width, height)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {source}")

    capture_fps = cap.get(cv2.CAP_PROP_FPS)
    encoder_fps = capture_fps if capture_fps and capture_fps > 1 else fps
    encoder = H264Encoder(
        width=width,
        height=height,
        crf=crf,
        fps=int(round(encoder_fps)),
        gop=gop,
    )
    decoder = H264Decoder(width=width, height=height)

    total_bytes = 0
    measured_frames = 0
    matched_frames = 0
    total_psnr = 0.0
    originals = deque()
    start = time.time()

    try:
        target_frames = None if frames is None else warmup + frames
        frame_idx = 0
        while True:
            if target_frames is not None and frame_idx >= target_frames:
                break
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1

            frame = cv2.resize(frame, (width, height))
            data = encoder.encode(frame)

            if data:
                decoder.push(data)

            decoded = decoder.get_frame()
            if decoded is not None and originals:
                original = originals.popleft()
                total_psnr += cv2.PSNR(original, decoded)
                matched_frames += 1

            if frame_idx < warmup:
                continue

            total_bytes += len(data)
            measured_frames += 1
            originals.append(frame.copy())

        # Drain a small tail so frames buffered by the decoder are still counted.
        drain_tries = max(gop * 2, 30)
        for _ in range(drain_tries):
            if not originals:
                break
            decoded = decoder.get_frame()
            if decoded is None:
                continue
            original = originals.popleft()
            total_psnr += cv2.PSNR(original, decoded)
            matched_frames += 1
    finally:
        decoder.close()
        encoder.close()
        cap.release()

    if measured_frames == 0:
        raise RuntimeError(
            f"No frames measured for CRF {crf}. Try a longer source or smaller --warmup."
        )

    elapsed = time.time() - start
    total_pixels = measured_frames * width * height
    total_bits = total_bytes * 8
    bpp = total_bits / total_pixels
    avg_bytes_per_frame = total_bytes / measured_frames
    psnr_avg = average_psnr(total_psnr, matched_frames)

    return {
        "crf": float(crf),
        "measured_frames": float(measured_frames),
        "matched_frames": float(matched_frames),
        "fps_used": float(encoder_fps),
        "total_bytes": float(total_bytes),
        "avg_bytes_per_frame": float(avg_bytes_per_frame),
        "bpp": float(bpp),
        "psnr_db": float(psnr_avg),
        "elapsed_sec": float(elapsed),
    }


def write_csv(rows: list[dict[str, float]], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "crf",
        "measured_frames",
        "matched_frames",
        "fps_used",
        "total_bytes",
        "avg_bytes_per_frame",
        "bpp",
        "psnr_db",
        "elapsed_sec",
    ]
    with csv_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_curve(rows: list[dict[str, float]], plot_path: Path) -> None:
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    bpps = [row["bpp"] for row in rows]
    psnrs = [row["psnr_db"] for row in rows]
    crfs = [int(row["crf"]) for row in rows]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(bpps, psnrs, marker="o", linewidth=2.2)
    for bpp, psnr, crf in zip(bpps, psnrs, crfs):
        ax.annotate(f"CRF {crf}", (bpp, psnr), textcoords="offset points", xytext=(5, 5))
    ax.set_title("H.264 Rate-Distortion Curve")
    ax.set_xlabel("Bits Per Pixel (bpp)")
    ax.set_ylabel("PSNR (dB)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    source = parse_source(args.source)
    csv_path = args.csv or default_csv_path()
    plot_path = args.plot or default_plot_path()
    frame_budget = resolve_frame_budget(source, args.frames)

    if isinstance(source, int):
        print(
            "[crf-rd] using a live camera source; results will vary between CRFs "
            "because each sweep sees different frames"
        )
        print("[crf-rd] for a cleaner rate-distortion curve, prefer --source path/to/video.mp4")
    else:
        if frame_budget is None:
            print("[crf-rd] using the full video for every CRF sweep")
        else:
            print(f"[crf-rd] using {frame_budget} measured frames from the video for every CRF sweep")

    rows = []
    for crf in args.crfs:
        print(f"[crf-rd] measuring CRF {crf}...")
        row = measure_rd_for_crf(
            source=source,
            crf=crf,
            width=args.width,
            height=args.height,
            fps=args.fps,
            gop=args.gop,
            warmup=args.warmup,
            frames=frame_budget,
        )
        rows.append(row)
        print(
            f"[crf-rd] CRF {crf} | "
            f"frames={int(row['measured_frames'])} | "
            f"matched={int(row['matched_frames'])} | "
            f"bpp={row['bpp']:.4f} | "
            f"psnr={row['psnr_db']:.2f} dB"
        )

    rows.sort(key=lambda row: row["crf"])
    write_csv(rows, csv_path)
    plot_curve(rows, plot_path)

    print(f"[crf-rd] csv: {csv_path}")
    print(f"[crf-rd] plot: {plot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
