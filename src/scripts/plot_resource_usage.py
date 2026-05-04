#!/usr/bin/env python3
import argparse
import csv
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="matplotlib-"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot CPU and memory graphs from a resource usage CSV log."
    )
    parser.add_argument(
        "csv_path",
        nargs="?",
        type=Path,
        default=None,
        help="Path to a resource usage CSV. Defaults to the newest file in logs/.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("plots"),
        help="Directory for output PNGs. Default: plots",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the plots interactively in addition to saving them.",
    )
    return parser.parse_args()


def latest_csv(log_dir: Path) -> Path:
    candidates = sorted(log_dir.glob("resource_usage_*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No resource usage CSV files found in {log_dir}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def load_rows(csv_path: Path) -> dict[str, list[float]]:
    series: dict[str, list[float]] = {}
    with csv_path.open(newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            for field, value in row.items():
                if field == "timestamp":
                    continue
                if value is None or value == "":
                    continue
                try:
                    numeric_value = float(value)
                except ValueError:
                    continue
                series.setdefault(field, []).append(numeric_value)
    return series


def plot_cpu(series: dict[str, list[float]], title_suffix: str, out_path: Path):
    t = series["elapsed_sec"]
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(t, series["cpu_percent"], label="Total CPU", linewidth=2.4)
    ax.plot(
        t,
        series["cpu_percent_normalized"],
        label="Total CPU (normalized)",
        linewidth=2.0,
        linestyle="--",
    )
    ax.plot(t, series["server_cpu_percent"], label="Server CPU", linewidth=1.8)
    ax.plot(t, series["camera_cpu_percent"], label="Camera CPU", linewidth=1.8)
    ax.set_title(f"CPU Usage Over Time{title_suffix}")
    ax.set_xlabel("Elapsed Time (s)")
    ax.set_ylabel("CPU (%)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    return fig


def plot_memory(series: dict[str, list[float]], title_suffix: str, out_path: Path):
    t = series["elapsed_sec"]
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(t, series["rss_mb"], label="Total Memory Consumption", linewidth=2.4)
    ax.plot(t, series["server_rss_mb"], label="Server Memory Consumption", linewidth=1.8)
    ax.plot(t, series["camera_rss_mb"], label="Camera Memory Consumption", linewidth=1.8)
    ax.set_title(f"Memory Consumption Over Time{title_suffix}")
    ax.set_xlabel("Elapsed Time (s)")
    ax.set_ylabel("Memory Consumption (MB)")
    ax.ticklabel_format(style="plain", axis="y")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    return fig


def main() -> int:
    args = parse_args()
    csv_path = args.csv_path or latest_csv(Path("logs"))
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    args.outdir.mkdir(parents=True, exist_ok=True)
    series = load_rows(csv_path)
    stem = csv_path.stem
    title_suffix = f" ({stem})"
    cpu_path = args.outdir / f"{stem}_cpu.png"
    mem_path = args.outdir / f"{stem}_memory.png"

    cpu_fig = plot_cpu(series, title_suffix, cpu_path)
    mem_fig = plot_memory(series, title_suffix, mem_path)

    print(f"[plot] csv: {csv_path}")
    print(f"[plot] cpu graph: {cpu_path}")
    print(f"[plot] memory graph: {mem_path}")

    if args.show:
        plt.show()
    else:
        plt.close(cpu_fig)
        plt.close(mem_fig)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
