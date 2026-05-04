#!/usr/bin/env python3
import argparse
import csv
import os
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import psutil


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Launch the inference server, wait for it to be ready, then launch "
            "the camera app and record CPU and memory usage over time."
        )
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Sampling interval in seconds. Default: 1.0",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="CSV file path. Default: logs/resource_usage_<timestamp>.csv",
    )
    parser.add_argument(
        "--no-children",
        action="store_true",
        help="Only measure the top-level process and exclude child processes.",
    )
    parser.add_argument(
        "--server-port",
        type=int,
        default=9999,
        help="TCP port to wait on before launching the camera app. Default: 9999",
    )
    parser.add_argument(
        "--server-timeout",
        type=float,
        default=30.0,
        help="Seconds to wait for the server to start listening. Default: 30",
    )
    parser.add_argument(
        "camera_command",
        nargs=argparse.REMAINDER,
        help=(
            "Camera command to run. If omitted, defaults to "
            "'python -m src.streaming.camera_h264'."
        ),
    )
    return parser.parse_args()


def default_camera_command() -> list[str]:
    return [sys.executable, "-u", "-m", "src.streaming.camera_h264"]


def default_server_command() -> list[str]:
    return [sys.executable, "-u", "-m", "src.inference.server_h264"]


def default_output_path() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("logs") / f"resource_usage_{timestamp}.csv"


def safe_name(proc: psutil.Process) -> str:
    try:
        return proc.name()
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return "unknown"


def collect_processes(root: psutil.Process, include_children: bool) -> list[psutil.Process]:
    processes = [root]
    if include_children:
        try:
            processes.extend(root.children(recursive=True))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return processes


def collect_roots(server_root: psutil.Process | None, camera_root: psutil.Process | None) -> list[psutil.Process]:
    roots = []
    if server_root is not None:
        roots.append(server_root)
    if camera_root is not None:
        roots.append(camera_root)
    return roots


def empty_sample() -> dict[str, float]:
    return {
        "cpu_percent": 0.0,
        "rss_mb": 0.0,
        "vms_mb": 0.0,
        "process_count": 0,
    }


def prime_cpu_counters(processes: list[psutil.Process]) -> None:
    for proc in processes:
        try:
            proc.cpu_percent(interval=None)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue


def sample_usage(
    root: psutil.Process,
    include_children: bool,
    primed_pids: set[int],
) -> dict[str, float]:
    cpu_percent = 0.0
    rss_bytes = 0
    vms_bytes = 0
    process_count = 0

    for proc in collect_processes(root, include_children):
        try:
            if proc.pid not in primed_pids:
                proc.cpu_percent(interval=None)
                primed_pids.add(proc.pid)
                proc_cpu = 0.0
            else:
                proc_cpu = proc.cpu_percent(interval=None)
            cpu_percent += proc_cpu
            mem = proc.memory_info()
            rss_bytes += mem.rss
            vms_bytes += mem.vms
            process_count += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    return {
        "cpu_percent": cpu_percent,
        "rss_mb": rss_bytes / (1024 * 1024),
        "vms_mb": vms_bytes / (1024 * 1024),
        "process_count": process_count,
    }


def combine_samples(*samples: dict[str, float]) -> dict[str, float]:
    return {
        "cpu_percent": sum(sample["cpu_percent"] for sample in samples),
        "rss_mb": sum(sample["rss_mb"] for sample in samples),
        "vms_mb": sum(sample["vms_mb"] for sample in samples),
        "process_count": sum(int(sample["process_count"]) for sample in samples),
    }


def normalize_cpu_percent(cpu_percent: float) -> float:
    cpu_count = psutil.cpu_count() or 1
    return cpu_percent / cpu_count


def terminate_process_tree(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return

    try:
        if os.name == "nt":
            proc.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            proc.send_signal(signal.SIGINT)
        proc.wait(timeout=5)
        return
    except Exception:
        pass

    try:
        parent = psutil.Process(proc.pid)
        children = parent.children(recursive=True)
        for child in children:
            try:
                child.terminate()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        parent.terminate()
        psutil.wait_procs(children + [parent], timeout=5)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass

    if proc.poll() is None:
        proc.kill()


def stream_output(
    proc: subprocess.Popen,
    ready_event: threading.Event | None = None,
    ready_text: str | None = None,
) -> None:
    if proc.stdout is None:
        return

    try:
        for line in proc.stdout:
            print(line, end="")
            if ready_event is not None and ready_text is not None and ready_text in line:
                ready_event.set()
    finally:
        proc.stdout.close()


def wait_for_ready_event(
    proc: subprocess.Popen,
    ready_event: threading.Event,
    timeout: float,
) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if ready_event.is_set():
            return True
        if proc.poll() is not None:
            return False
        time.sleep(0.1)
    return ready_event.is_set()


def main() -> int:
    args = parse_args()
    server_command = default_server_command()
    camera_command = args.camera_command if args.camera_command else default_camera_command()
    output_path = args.output or default_output_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    include_children = not args.no_children

    print(f"[monitor] launching server: {' '.join(server_command)}")
    print(f"[monitor] camera command: {' '.join(camera_command)}")
    print(f"[monitor] writing samples to: {output_path}")
    if include_children:
        print("[monitor] including child processes such as ffmpeg")
    print(f"[monitor] waiting for server readiness on port {args.server_port}")

    server_ready = threading.Event()
    server_proc = subprocess.Popen(
        server_command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    server_root = psutil.Process(server_proc.pid)
    server_output_thread = threading.Thread(
        target=stream_output,
        args=(
            server_proc,
            server_ready,
            f"Edge Server: H.264 AI Inference Engine active on port {args.server_port}",
        ),
        daemon=True,
    )
    server_output_thread.start()
    camera_proc = None
    camera_root = None

    try:
        if not wait_for_ready_event(server_proc, server_ready, args.server_timeout):
            print(
                f"[monitor] server did not become ready within "
                f"{args.server_timeout:.1f}s"
            )
            terminate_process_tree(server_proc)
            return 1

        print("[monitor] server is ready, launching camera")
        camera_proc = subprocess.Popen(camera_command)
        camera_root = psutil.Process(camera_proc.pid)

        prime_cpu_counters(collect_processes(server_root, include_children))
        prime_cpu_counters(collect_processes(camera_root, include_children))
        primed_pids = {
            proc.pid
            for proc in collect_processes(server_root, include_children)
            + collect_processes(camera_root, include_children)
        }
        start = time.time()

        fieldnames = [
            "elapsed_sec",
            "timestamp",
            "server_pid",
            "server_name",
            "camera_pid",
            "camera_name",
            "process_count",
            "cpu_percent",
            "cpu_percent_normalized",
            "rss_mb",
            "vms_mb",
            "server_cpu_percent",
            "server_cpu_percent_normalized",
            "server_rss_mb",
            "server_vms_mb",
            "server_process_count",
            "camera_cpu_percent",
            "camera_cpu_percent_normalized",
            "camera_rss_mb",
            "camera_vms_mb",
            "camera_process_count",
            "server_alive",
            "camera_alive",
        ]

        with output_path.open("w", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()

            while True:
                server_alive = server_proc.poll() is None
                camera_alive = camera_proc.poll() is None
                if not server_alive and not camera_alive:
                    break

                time.sleep(args.interval)
                server_sample = (
                    sample_usage(server_root, include_children, primed_pids)
                    if server_alive else empty_sample()
                )
                camera_sample = (
                    sample_usage(camera_root, include_children, primed_pids)
                    if camera_alive else empty_sample()
                )
                sample = combine_samples(server_sample, camera_sample)
                row = {
                    "elapsed_sec": round(time.time() - start, 3),
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "server_pid": server_proc.pid,
                    "server_name": safe_name(server_root),
                    "camera_pid": camera_proc.pid,
                    "camera_name": safe_name(camera_root),
                    "process_count": sample["process_count"],
                    "cpu_percent": round(sample["cpu_percent"], 2),
                    "cpu_percent_normalized": round(normalize_cpu_percent(sample["cpu_percent"]), 2),
                    "rss_mb": round(sample["rss_mb"], 2),
                    "vms_mb": round(sample["vms_mb"], 2),
                    "server_cpu_percent": round(server_sample["cpu_percent"], 2),
                    "server_cpu_percent_normalized": round(
                        normalize_cpu_percent(server_sample["cpu_percent"]), 2
                    ),
                    "server_rss_mb": round(server_sample["rss_mb"], 2),
                    "server_vms_mb": round(server_sample["vms_mb"], 2),
                    "server_process_count": server_sample["process_count"],
                    "camera_cpu_percent": round(camera_sample["cpu_percent"], 2),
                    "camera_cpu_percent_normalized": round(
                        normalize_cpu_percent(camera_sample["cpu_percent"]), 2
                    ),
                    "camera_rss_mb": round(camera_sample["rss_mb"], 2),
                    "camera_vms_mb": round(camera_sample["vms_mb"], 2),
                    "camera_process_count": camera_sample["process_count"],
                    "server_alive": int(server_alive),
                    "camera_alive": int(camera_alive),
                }
                writer.writerow(row)
                csv_file.flush()
                print(
                    "[monitor] "
                    f"t={row['elapsed_sec']:.1f}s | "
                    f"cpu={row['cpu_percent']:.1f}% "
                    f"({row['cpu_percent_normalized']:.1f}% normalized) | "
                    f"rss={row['rss_mb']:.1f} MB | "
                    f"server={row['server_cpu_percent']:.1f}% | "
                    f"camera={row['camera_cpu_percent']:.1f}% | "
                    f"procs={row['process_count']} | "
                    f"server={'up' if server_alive else 'down'} | "
                    f"camera={'up' if camera_alive else 'down'}"
                )
    except KeyboardInterrupt:
        print("\n[monitor] interrupt received, stopping camera and server...")
    finally:
        if camera_proc is not None:
            terminate_process_tree(camera_proc)
        terminate_process_tree(server_proc)

    server_code = server_proc.wait()
    camera_code = camera_proc.wait() if camera_proc is not None else 0
    return camera_code if camera_code != 0 else server_code


if __name__ == "__main__":
    raise SystemExit(main())
