"""EchoStream Demo UI — two-screen Tkinter app.

Screen 1: Configuration form. Writes a temp runtime config and launches
          both inference servers + camera as subprocesses.
Screen 2: Live display. Shows adaptive (port 9999) and baseline (port 9998)
          feeds side-by-side via FrameRelayClient, plus a Stop button and
          save-status message.

Launch with:
    python -m src.ui.app
"""
from __future__ import annotations

import queue
import socket
import struct
import subprocess
import threading
import time
import tkinter as tk
import logging
from collections import deque
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

from src.ui.bandwidth_relay import BandwidthRelayReceiver
from src.ui.config_writer import write_runtime_config

# ---------------------------------------------------------------------------
# Relay ports — must match what app passes to --relay-port on each server
# ---------------------------------------------------------------------------
RELAY_PORT_ADAPTIVE = 9997   # server on 9999 sends frames here
RELAY_PORT_BASELINE  = 9996  # server on 9998 sends frames here
BANDWIDTH_RELAY_PORT = 9995  # camera sends live baseline-vs-adaptive samples here

# Inference server ports (camera connects to these, unchanged)
SERVER_PORT_ADAPTIVE = 9999
SERVER_PORT_BASELINE = 9998

# How often the UI polls each relay queue for a new frame (ms)
POLL_INTERVAL_MS = 33  # ~30 fps

# Display dimensions for each panel in Screen 2
RELAY_W = 640
RELAY_H = 480
PANEL_W = 480
PANEL_H = 360

DEFAULT_CONFIG = "configs/default.json"

log = logging.getLogger("echostream.app")


def _setup_app_logging() -> None:
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    handler = logging.FileHandler(log_dir / "app.log", mode="a")
    handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    ))
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(handler)


# ---------------------------------------------------------------------------
# FrameRelayClient — runs in a background thread, fills a queue for the UI
# ---------------------------------------------------------------------------

class FrameRelayClient:
    """Connects to a FrameRelayServer and decodes incoming frames into a queue.

    The UI polls self.latest() on a Tkinter after() timer — no blocking on
    the main thread.
    """

    def __init__(self, port: int, width: int = RELAY_W, height: int = RELAY_H):
        self.port = port
        self.width = width
        self.height = height
        self._q: queue.Queue[np.ndarray] = queue.Queue(maxsize=4)
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._recv_loop, daemon=True, name=f"relay-client-{port}"
        )

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def latest(self) -> np.ndarray | None:
        """Return the most recent frame, draining stale ones. Non-blocking."""
        frame = None
        while True:
            try:
                frame = self._q.get_nowait()
            except queue.Empty:
                break
        return frame

    def _recv_loop(self) -> None:
        frame_bytes = self.width * self.height * 3
        while not self._stop.is_set():
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2.0)
            try:
                sock.connect(("127.0.0.1", self.port))
            except (ConnectionRefusedError, socket.timeout, OSError):
                # Server not ready yet — retry after a short wait
                sock.close()
                time.sleep(0.5)
                continue

            try:
                while not self._stop.is_set():
                    # Read 4-byte length header
                    header = self._recv_exact(sock, 4)
                    if header is None:
                        break
                    (length,) = struct.unpack("!I", header)
                    if length != frame_bytes:
                        # Unexpected size — drain and reconnect
                        break
                    raw = self._recv_exact(sock, length)
                    if raw is None:
                        break
                    frame = (
                        np.frombuffer(raw, dtype=np.uint8)
                        .reshape((self.height, self.width, 3))
                        .copy()
                    )
                    # Drop oldest if UI can't keep up
                    try:
                        self._q.put_nowait(frame)
                    except queue.Full:
                        try:
                            self._q.get_nowait()
                        except queue.Empty:
                            pass
                        try:
                            self._q.put_nowait(frame)
                        except queue.Full:
                            pass
            except (OSError, struct.error):
                pass
            finally:
                try:
                    sock.close()
                except OSError:
                    pass

    def _recv_exact(self, sock: socket.socket, n: int) -> bytes | None:
        buf = b""
        while len(buf) < n:
            if self._stop.is_set():
                return None
            try:
                chunk = sock.recv(n - len(buf))
            except socket.timeout:
                continue
            except OSError:
                return None
            if not chunk:
                return None
            buf += chunk
        return buf


# ---------------------------------------------------------------------------
# Subprocess launcher helpers  (mirrors run.sh logic)
# ---------------------------------------------------------------------------

def _wait_for_ready(log_path: Path, pid: int, port: int, timeout: float = 30.0) -> bool:
    """Poll log file for 'listening on .*:<port>' — same as run.sh wait_for_server."""
    pattern = f"listening on"
    port_str = f":{port}"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            text = log_path.read_text(errors="replace")
            for line in text.splitlines():
                if pattern in line and port_str in line:
                    return True
        except OSError:
            pass
        # Check that the process is still alive
        try:
            import os
            os.kill(pid, 0)
        except OSError:
            return False
        time.sleep(0.25)
    return False


def launch_server(config_path: str, port: int, relay_port: int,
                  log_path: Path, save_artifacts: bool,
                  output_dir: str | None) -> subprocess.Popen:
    cmd = [
        "python", "-u", "-m", "src.inference.server_h264",
        "--config", config_path,
        "--port", str(port),
        "--relay-port", str(relay_port),
    ]
    if save_artifacts and output_dir:
        cmd += ["--save-artifacts", "--output-dir", output_dir]
    log_file = open(log_path, "w")
    return subprocess.Popen(cmd, stdout=log_file, stderr=log_file)


def launch_camera(config_path: str, log_path: Path,
                  loop_video: bool, bandwidth_relay_port: int | None) -> subprocess.Popen:
    cmd = [
        "python", "-u", "-m", "src.streaming.camera_h264",
        "--config", config_path,
        "--no-preview",   # camera raw feed handled by the existing cv2 window or suppressed
    ]
    if loop_video:
        cmd.append("--loop-video")
    if bandwidth_relay_port is not None:
        cmd += ["--bandwidth-relay-port", str(bandwidth_relay_port)]
    log_file = open(log_path, "w")
    return subprocess.Popen(cmd, stdout=log_file, stderr=log_file)


# ---------------------------------------------------------------------------
# Screen 1 — Configuration
# ---------------------------------------------------------------------------

class ConfigScreen(tk.Frame):

    def __init__(self, master: tk.Tk, on_start):
        super().__init__(master, padx=24, pady=24)
        self._on_start = on_start
        self._source_var = tk.StringVar(value="live")
        self._file_path_var = tk.StringVar()
        self._loop_var = tk.BooleanVar(value=False)
        self._classes_var = tk.StringVar(value="person")
        self._baseline_var = tk.BooleanVar(value=True)
        self._mask_blur_var = tk.StringVar(value="low")
        self._save_var = tk.BooleanVar(value=False)
        self._output_dir_var = tk.StringVar()
        self._build()

    def _build(self):
        # Title
        tk.Label(self, text="StreamSense Demo", font=("Helvetica", 18, "bold")).grid(
            row=0, column=0, columnspan=3, pady=(0, 20), sticky="w"
        )

        # --- Video source ---
        tk.Label(self, text="Video source", font=("Helvetica", 11, "bold")).grid(
            row=1, column=0, sticky="w", pady=(0, 4)
        )
        src_frame = tk.Frame(self)
        src_frame.grid(row=2, column=0, columnspan=3, sticky="w", pady=(0, 12))

        tk.Radiobutton(
            src_frame, text="Live camera", variable=self._source_var,
            value="live", command=self._on_source_change
        ).pack(side="left")
        tk.Radiobutton(
            src_frame, text="Video file", variable=self._source_var,
            value="file", command=self._on_source_change
        ).pack(side="left", padx=(16, 0))

        # File path row (hidden until "file" selected)
        self._file_frame = tk.Frame(self)
        self._file_frame.grid(row=3, column=0, columnspan=3, sticky="ew", pady=(0, 4))
        tk.Label(self._file_frame, text="File path").pack(side="left")
        self._file_entry = tk.Entry(
            self._file_frame, textvariable=self._file_path_var, width=40
        )
        self._file_entry.pack(side="left", padx=(8, 4))
        tk.Button(
            self._file_frame, text="Browse…", command=self._browse_file
        ).pack(side="left")
        tk.Checkbutton(
            self._file_frame, text="Loop", variable=self._loop_var
        ).pack(side="left", padx=(16, 0))
        self._file_frame.grid_remove()  # hidden by default

        # --- Detection prompt ---
        tk.Label(self, text="Detection prompt", font=("Helvetica", 11, "bold")).grid(
            row=4, column=0, sticky="w", pady=(8, 4)
        )
        tk.Label(
            self, text="Comma-separated classes, e.g. person,wallet",
            fg="gray", font=("Helvetica", 9)
        ).grid(row=5, column=0, columnspan=3, sticky="w")
        tk.Entry(self, textvariable=self._classes_var, width=40).grid(
            row=6, column=0, columnspan=2, sticky="w", pady=(2, 12)
        )

        # --- Baseline ---
        tk.Label(self, text="Baseline", font=("Helvetica", 11, "bold")).grid(
            row=7, column=0, sticky="w", pady=(0, 4)
        )
        tk.Checkbutton(
            self, text="Enable baseline comparison (port 9998)",
            variable=self._baseline_var
        ).grid(row=8, column=0, columnspan=3, sticky="w", pady=(0, 12))

        # --- Mask blur ---
        tk.Label(self, text="Masking Strength", font=("Helvetica", 11, "bold")).grid(
            row=9, column=0, sticky="w", pady=(0, 4)
        )
        blur_frame = tk.Frame(self)
        blur_frame.grid(row=10, column=0, columnspan=3, sticky="w", pady=(0, 12))
        for idx, level in enumerate(("low", "medium", "high")):
            tk.Radiobutton(
                blur_frame,
                text=level.title(),
                variable=self._mask_blur_var,
                value=level,
            ).pack(side="left", padx=(0 if idx == 0 else 16, 0))

        # --- Save artifacts ---
        tk.Label(self, text="Artifacts", font=("Helvetica", 11, "bold")).grid(
            row=11, column=0, sticky="w", pady=(0, 4)
        )
        tk.Checkbutton(
            self, text="Save artifacts", variable=self._save_var,
            command=self._on_save_toggle
        ).grid(row=12, column=0, sticky="w")

        self._output_frame = tk.Frame(self)
        self._output_frame.grid(row=13, column=0, columnspan=3, sticky="ew", pady=(4, 12))
        tk.Label(self._output_frame, text="Output folder").pack(side="left")
        tk.Entry(
            self._output_frame, textvariable=self._output_dir_var, width=34
        ).pack(side="left", padx=(8, 4))
        tk.Button(
            self._output_frame, text="Browse…", command=self._browse_output
        ).pack(side="left")
        tk.Label(
            self._output_frame, text="(leave blank for auto)", fg="gray",
            font=("Helvetica", 9)
        ).pack(side="left", padx=(8, 0))
        self._output_frame.grid_remove()

        # --- Start button ---
        ttk.Separator(self, orient="horizontal").grid(
            row=14, column=0, columnspan=3, sticky="ew", pady=16
        )
        tk.Button(
            self, text="Start", font=("Helvetica", 13, "bold"),
            bg="#2d7d46", fg="white", padx=20, pady=8,
            command=self._start
        ).grid(row=15, column=0, sticky="w")

    # --- event handlers ---

    def _on_source_change(self):
        if self._source_var.get() == "file":
            self._file_frame.grid()
        else:
            self._file_frame.grid_remove()

    def _on_save_toggle(self):
        if self._save_var.get():
            self._output_frame.grid()
        else:
            self._output_frame.grid_remove()

    def _browse_file(self):
        path = filedialog.askopenfilename(
            title="Select video file",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
        )
        if path:
            self._file_path_var.set(path)

    def _browse_output(self):
        path = filedialog.askdirectory(title="Select output folder")
        if path:
            self._output_dir_var.set(path)

    def _start(self):
        # Validate
        classes = self._classes_var.get().strip()
        if not classes:
            messagebox.showerror("Validation", "Detection prompt cannot be empty.")
            return
        if self._source_var.get() == "file":
            file_path = self._file_path_var.get().strip()
            if not file_path:
                messagebox.showerror("Validation", "Please select a video file.")
                return
            input_source = file_path
        else:
            input_source = "0"  # default webcam index

        self._on_start(
            input_source=input_source,
            loop_video=self._loop_var.get(),
            classes=classes,
            baseline_enabled=self._baseline_var.get(),
            mask_blur_level=self._mask_blur_var.get(),
            save_artifacts=self._save_var.get(),
            output_dir=self._output_dir_var.get().strip() or None,
        )


# ---------------------------------------------------------------------------
# Screen 2 — Running session
# ---------------------------------------------------------------------------

class RunningScreen(tk.Frame):

    def __init__(self, master: tk.Tk, baseline_enabled: bool,
                 save_artifacts: bool, output_dir: str | None,
                 on_stop):
        super().__init__(master, padx=16, pady=16)
        self._baseline_enabled = baseline_enabled
        self._save_artifacts = save_artifacts
        self._output_dir = output_dir
        self._on_stop = on_stop
        self._stopped = False

        # Relay clients
        self._adaptive_client = FrameRelayClient(RELAY_PORT_ADAPTIVE)
        self._baseline_client = (
            FrameRelayClient(RELAY_PORT_BASELINE) if baseline_enabled else None
        )
        self._bandwidth_receiver = (
            BandwidthRelayReceiver(BANDWIDTH_RELAY_PORT)
            if baseline_enabled
            else None
        )
        self._bandwidth_samples = deque(maxlen=120)
        self._bandwidth_totals = {
            "adaptive_bytes": 0.0,
            "baseline_bytes": 0.0,
            "conf_sum": 0.0,
            "conf_count": 0,
        }
        self._metric_vars: dict[str, tk.StringVar] = {}
        self._adaptive_title_var = tk.StringVar(value="Adaptive (StreamSense)")

        # Placeholder image (black panel)
        self._blank = ImageTk.PhotoImage(
            Image.fromarray(np.zeros((PANEL_H, PANEL_W, 3), dtype=np.uint8))
        )

        self._build()
        self._adaptive_client.start()
        if self._baseline_client:
            self._baseline_client.start()
        if self._bandwidth_receiver:
            self._bandwidth_receiver.start()

        # Start polling
        self.after(POLL_INTERVAL_MS, self._poll_frames)
        self.after(250, self._poll_bandwidth)

    def _build(self):
        tk.Label(self, text="StreamSense — Live Session",
                 font=("Helvetica", 16, "bold")).grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 12)
        )


        # Stats panel (moved to top)
        self._build_stats_panel(row=1)
        self._update_stats_panel()

        # Video panels
        baseline_label_text = "Baseline (fixed CRF)"

        tk.Label(self, text="Adaptive (StreamSense)",
                 font=("Helvetica", 11, "bold")).grid(
            row=2, column=0, pady=(0, 4)
        )
        self._adaptive_panel = tk.Label(self, image=self._blank, bg="black")
        self._adaptive_panel.grid(row=3, column=0, padx=(0, 8), sticky="nsew")

        if self._baseline_enabled:
            tk.Label(self, text=baseline_label_text,
                     font=("Helvetica", 11, "bold")).grid(
                row=2, column=1, pady=(0, 4)
            )
            self._baseline_panel = tk.Label(self, image=self._blank, bg="black")
            self._baseline_panel.grid(row=3, column=1, sticky="nsew")
        else:
            self._baseline_panel = None

        # Live baseline-vs-adaptive bandwidth graph
        self._bandwidth_canvas = tk.Canvas(
            self,
            width=(PANEL_W * 2 + 8) if self._baseline_enabled else PANEL_W,
            height=170,
            bg="#111827",
            highlightthickness=1,
            highlightbackground="#243047",
        )
        self._bandwidth_canvas.grid(
            row=4, column=0, columnspan=2, sticky="ew", pady=(12, 4)
        )
        # Make layout responsive: allocate more space to video panels
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        # Rows: title(0)=auto, stats(1)=small, labels(2)=auto, video(3)=big, graph(4)=medium
        self.grid_rowconfigure(0, weight=0)
        self.grid_rowconfigure(1, weight=1)
        self.grid_rowconfigure(2, weight=0)
        self.grid_rowconfigure(3, weight=6)
        self.grid_rowconfigure(4, weight=2)
        self._draw_bandwidth_chart()
        

        # Status bar
        self._status_var = tk.StringVar(value="Session running…")
        tk.Label(self, textvariable=self._status_var, fg="gray",
                 font=("Helvetica", 10)).grid(
            row=5, column=0, columnspan=2, sticky="w", pady=(8, 4)
        )

        # Stop button
        tk.Button(
            self, text="Stop", font=("Helvetica", 12, "bold"),
            bg="#c0392b", fg="white", padx=16, pady=6,
            command=self._stop
        ).grid(row=6, column=0, sticky="w", pady=(4, 0))

    def _poll_frames(self):
        """Called by Tkinter's after() loop — updates panels with latest frames."""
        if self._stopped:
            return
        self._update_panel(self._adaptive_panel, self._adaptive_client)
        if self._baseline_client and self._baseline_panel:
            self._update_panel(self._baseline_panel, self._baseline_client)
        self.after(POLL_INTERVAL_MS, self._poll_frames)

    def _poll_bandwidth(self):
        if self._stopped:
            return
        if self._bandwidth_receiver is not None:
            for sample in self._bandwidth_receiver.drain():
                self._bandwidth_samples.append(sample)
                self._accumulate_live_metrics(sample)
            self._draw_bandwidth_chart()
            self._update_stats_panel()
        self.after(250, self._poll_bandwidth)

    def _update_panel(self, panel: tk.Label, client: FrameRelayClient):
        frame = client.latest()
        if frame is None:
            return
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Get actual panel dimensions; fall back to PANEL_W/PANEL_H if not yet rendered
        panel_width = panel.winfo_width()
        panel_height = panel.winfo_height()
        if panel_width <= 1 or panel_height <= 1:
            panel_width, panel_height = PANEL_W, PANEL_H
        
        img = Image.fromarray(rgb).resize((panel_width, panel_height), Image.LANCZOS)
        photo = ImageTk.PhotoImage(img)
        panel.configure(image=photo)
        panel.image = photo  # prevent GC

    def _stop(self):
        self._stopped = True
        self._adaptive_client.stop()
        if self._baseline_client:
            self._baseline_client.stop()
        if self._bandwidth_receiver:
            self._bandwidth_receiver.stop()
        self._update_save_status()
        self._on_stop()

    def _draw_bandwidth_chart(self):
        canvas = self._bandwidth_canvas
        canvas.delete("all")
        width = max(int(canvas.winfo_width()), int(canvas["width"]))
        height = max(int(canvas.winfo_height()), int(canvas["height"]))
        left, right, top, bottom = 58, width - 20, 34, height - 38
        plot_bg = "#0f172a"
        grid = "#243047"
        axis = "#475569"
        label = "#94a3b8"
        title = "#f8fafc"
        baseline_color = "#fb7185"
        adaptive_color = "#38bdf8"

        canvas.create_rectangle(
            left,
            top,
            right,
            bottom,
            fill=plot_bg,
            outline="#1f2a44",
        )

        canvas.create_text(
            left,
            16,
            text="Live bandwidth consumption: baseline vs adaptive",
            anchor="w",
            font=("Helvetica", 11, "bold"),
            fill=title,
        )

        samples = list(self._bandwidth_samples)
        if not self._baseline_enabled:
            canvas.create_text(
                width / 2,
                (top + bottom) / 2,
                text="Enable baseline comparison to graph bandwidth consumption.",
                fill=label,
                font=("Helvetica", 10),
            )
            return

        canvas.create_line(left, top, left, bottom, fill=axis)
        canvas.create_line(left, bottom, right, bottom, fill=axis)

        if not samples:
            for idx, tick_label in enumerate(("0", "kbps")):
                canvas.create_text(
                    left - 8,
                    bottom if idx == 0 else top,
                    text=tick_label,
                    anchor="e",
                    font=("Helvetica", 8),
                    fill="#94a3b8",
                )
            canvas.create_text(
                width / 2,
                (top + bottom) / 2,
                text="Waiting for paired adaptive/baseline bandwidth samples...",
                fill="#cbd5e1",
                font=("Helvetica", 10),
            )
            return

        max_kbps = max(
            max(
                float(sample.get("adaptive_kbps", 0.0)),
                float(sample.get("baseline_kbps", 0.0)),
            )
            for sample in samples
        )
        max_kbps = max(max_kbps, 1.0)
        y_max = max_kbps * 1.15

        def _fmt_kbps(value: float) -> str:
            if value >= 1000:
                return f"{value / 1000:.1f}M"
            return f"{value:.0f}"

        for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
            y = bottom - frac * (bottom - top)
            kbps = y_max * frac
            canvas.create_line(left, y, right, y, fill=grid)
            canvas.create_text(
                left - 8,
                y,
                text=_fmt_kbps(kbps),
                anchor="e",
                font=("Helvetica", 8),
                fill=label,
            )

        def _x(idx: int) -> float:
            if len(samples) == 1:
                return left
            return left + idx * (right - left) / float(len(samples) - 1)

        def _y(value: float) -> float:
            value = max(0.0, min(y_max, float(value)))
            return bottom - (value / y_max) * (bottom - top)

        def _draw_series(key: str, color: str):
            points = []
            for idx, sample in enumerate(samples):
                points.extend([_x(idx), _y(float(sample.get(key, 0.0)))])
            if len(points) >= 4:
                canvas.create_line(*points, fill=color, width=3, smooth=True)
            elif points:
                canvas.create_oval(
                    points[0] - 3, points[1] - 3, points[0] + 3, points[1] + 3,
                    fill=color, outline=color,
                )

        _draw_series("baseline_kbps", baseline_color)
        _draw_series("adaptive_kbps", adaptive_color)

        latest = samples[-1]
        savings = float(latest.get("savings_pct", 0.0))
        adaptive_kbps = float(latest.get("adaptive_kbps", 0.0))
        baseline_kbps = float(latest.get("baseline_kbps", 0.0))
        segment_id = int(latest.get("segment_id", 0))
        legend_x = left + 6
        legend_y = height - 14
        canvas.create_line(
            legend_x, legend_y, legend_x + 20, legend_y,
            fill=baseline_color, width=3,
        )
        canvas.create_text(
            legend_x + 26,
            legend_y,
            text="baseline",
            anchor="w",
            font=("Helvetica", 9),
            fill="#e2e8f0",
        )
        canvas.create_line(
            legend_x + 96, legend_y, legend_x + 116, legend_y,
            fill=adaptive_color, width=3,
        )
        canvas.create_text(
            legend_x + 122,
            legend_y,
            text="adaptive",
            anchor="w",
            font=("Helvetica", 9),
            fill="#e2e8f0",
        )
        canvas.create_text(
            right,
            16,
            text=(
                f"baseline {baseline_kbps:.0f} / adaptive {adaptive_kbps:.0f} kbps   "
                f"saved {savings:.1f}%   "
                f"seg {segment_id}"
            ),
            anchor="e",
            font=("Helvetica", 10),
            fill="#cbd5e1",
        )

    def _build_stats_panel(self, row: int):
        panel = tk.Frame(self, bg="#111827", padx=10, pady=10)
        panel.grid(row=row, column=0, columnspan=2, sticky="ew", pady=(8, 4))
        panel.grid_columnconfigure(0, weight=1)
        panel.grid_columnconfigure(1, weight=1)
        panel.grid_columnconfigure(2, weight=1)
        panel.grid_columnconfigure(3, weight=1)

        metrics = [
            ("bandwidth_saved_pct", "Bandwidth Saved", "#f59e0b"),
            ("crf", "Adaptive CRF", "#38bdf8"),
            ("avg_conf", "Avg. Confidence", "#34d399"),
            ("processing_fps", "Processing FPS", "#f472b6"),
        ]
        for col, (key, label, accent) in enumerate(metrics):
            value_var = tk.StringVar(value="--")
            self._metric_vars[key] = value_var
            card = tk.Frame(
                panel,
                bg="#111827",
                padx=10,
                pady=8,
            )
            card.grid(row=0, column=col, sticky="nsew", padx=6)
            panel.grid_columnconfigure(col, weight=1)
            tk.Label(
                card,
                text=label.upper(),
                bg="#111827",
                fg="#cbd5e1",
                font=("Helvetica", 9, "bold"),
                justify="center",
            ).pack(anchor="center")
            tk.Label(
                card,
                textvariable=value_var,
                bg="#111827",
                fg=accent,
                font=("Helvetica", 26, "bold"),
                justify="center",
            ).pack(anchor="center", pady=(6, 0))

    def _accumulate_live_metrics(self, sample: dict):
        self._bandwidth_totals["adaptive_bytes"] += float(
            sample.get("adaptive_bytes", 0.0)
        )
        self._bandwidth_totals["baseline_bytes"] += float(
            sample.get("baseline_bytes", 0.0)
        )
        conf = sample.get("adaptive_conf")
        if conf is not None:
            self._bandwidth_totals["conf_sum"] += float(conf)
            self._bandwidth_totals["conf_count"] += 1

    def _update_stats_panel(self):
        if not self._metric_vars:
            return
        samples = list(self._bandwidth_samples)
        if not samples:
            for value_var in self._metric_vars.values():
                value_var.set("--")
            return

        total_adaptive = self._bandwidth_totals["adaptive_bytes"]
        total_baseline = self._bandwidth_totals["baseline_bytes"]
        bandwidth_saved_pct = (
            (total_baseline - total_adaptive) / total_baseline * 100.0
            if total_baseline > 0 else 0.0
        )
        latest = samples[-1]
        conf_count = int(self._bandwidth_totals["conf_count"])
        avg_conf = (
            self._bandwidth_totals["conf_sum"] / conf_count
            if conf_count > 0 else 0.0
        )

        self._metric_vars["bandwidth_saved_pct"].set(f"{bandwidth_saved_pct:.1f}%")
        adaptive_crf = int(latest.get("adaptive_crf", 0))
        self._metric_vars["crf"].set(str(adaptive_crf))
        self._metric_vars["avg_conf"].set(f"{avg_conf:.3f}")
        self._metric_vars["processing_fps"].set(
            f"{float(latest.get('adaptive_processing_fps', 0.0)):.1f}"
        )
        # Do not modify the title above the video player with CRF

    def _update_save_status(self):
        if not self._save_artifacts:
            self._status_var.set("Session stopped. Artifacts saving was not enabled.")
            return
        out = Path(self._output_dir) if self._output_dir else None
        if out and out.exists():
            self._status_var.set(f"✓ Results saved to: {out}")
        else:
            self._status_var.set(
                "Session stopped. Output folder not found yet — "
                "it may still be writing."
            )


# ---------------------------------------------------------------------------
# Main app — orchestrates screens and subprocess lifecycle
# ---------------------------------------------------------------------------

class App:

    def __init__(self):
        _setup_app_logging()
        self.root = tk.Tk()
        self.root.title("StreamSense")
        self.root.resizable(True, True)
        self.root.report_callback_exception = self._report_callback_exception

        self._procs: list[subprocess.Popen] = []
        self._log_files: list[object] = []
        self._runtime_config: Path | None = None
        self._current_screen: tk.Frame | None = None

        log.info("app started")
        self._show_config_screen()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def run(self):
        self.root.mainloop()

    # --- screen transitions ---

    def _show_config_screen(self):
        if self._current_screen:
            self._current_screen.destroy()
        screen = ConfigScreen(self.root, on_start=self._on_start)
        screen.pack(fill="both", expand=True)
        self._current_screen = screen
        self.root.update_idletasks()
        width = min(self.root.winfo_reqwidth(), self.root.winfo_screenwidth() - 80)
        height = min(self.root.winfo_reqheight(), self.root.winfo_screenheight() - 120)
        self.root.geometry(f"{width}x{height}+60+60")
        self.root.minsize(min(width, 900), min(height, 560))
        self.root.deiconify()
        self.root.lift()
        self.root.focus_force()
        log.info("running screen shown size=%dx%d", width, height)

    def _show_running_screen(self, baseline_enabled: bool,
                             save_artifacts: bool, output_dir: str | None):
        if self._current_screen:
            self._current_screen.destroy()
        try:
            screen = RunningScreen(
                self.root,
                baseline_enabled=baseline_enabled,
                save_artifacts=save_artifacts,
                output_dir=output_dir,
                on_stop=self._on_stop,
            )
        except Exception:
            log.exception("failed to build running screen")
            messagebox.showerror(
                "UI error",
                "The live display failed to open. See logs/app.log for details.",
            )
            self._show_config_screen()
            return
        screen.pack(fill="both", expand=True)
        self._current_screen = screen

    def _report_callback_exception(self, exc, val, tb):
        log.exception("Tk callback failed", exc_info=(exc, val, tb))

    # --- session lifecycle ---

    def _on_start(self, input_source: str, loop_video: bool, classes: str,
                  baseline_enabled: bool, mask_blur_level: str,
                  save_artifacts: bool,
                  output_dir: str | None):
        """Write temp config, launch servers + camera, transition to Screen 2."""
        log.info(
            "start requested input=%s baseline=%s save_artifacts=%s",
            input_source,
            baseline_enabled,
            save_artifacts,
        )
        try:
            config_path = write_runtime_config(
                default_config_path=DEFAULT_CONFIG,
                input_source=input_source,
                loop_video=loop_video,
                classes=classes,
                baseline_enabled=baseline_enabled,
                mask_blur_level=mask_blur_level,
                save_artifacts=save_artifacts,
                output_dir=output_dir,
            )
            self._runtime_config = config_path
        except Exception as e:
            messagebox.showerror("Config error", str(e))
            return

        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)

        # Launch adaptive server (9999 → relay 9997)
        srv1 = launch_server(
            config_path=str(config_path),
            port=SERVER_PORT_ADAPTIVE,
            relay_port=RELAY_PORT_ADAPTIVE,
            log_path=log_dir / "server_9999.log",
            save_artifacts=save_artifacts,
            output_dir=output_dir,
        )
        self._procs.append(srv1)

        # Launch baseline server (9998 → relay 9996) if enabled
        srv2 = None
        if baseline_enabled:
            srv2 = launch_server(
                config_path=str(config_path),
                port=SERVER_PORT_BASELINE,
                relay_port=RELAY_PORT_BASELINE,
                log_path=log_dir / "server_9998.log",
                save_artifacts=save_artifacts,
                output_dir=output_dir,
            )
            self._procs.append(srv2)

        # Wait for servers to be ready before launching camera (mirrors run.sh)
        def _launch_in_background():
            ready1 = _wait_for_ready(
                log_dir / "server_9999.log", srv1.pid, SERVER_PORT_ADAPTIVE
            )
            if not ready1:
                log.error("server 9999 failed to become ready")
                self.root.after(0, lambda: messagebox.showerror(
                    "Launch error", "Server 9999 failed to start. Check logs/server_9999.log."
                ))
                return

            if baseline_enabled and srv2 is not None:
                ready2 = _wait_for_ready(
                    log_dir / "server_9998.log", srv2.pid, SERVER_PORT_BASELINE
                )
                if not ready2:
                    log.error("server 9998 failed to become ready")
                    self.root.after(0, lambda: messagebox.showerror(
                        "Launch error", "Server 9998 failed to start. Check logs/server_9998.log."
                    ))
                    return

            log.info("servers ready; launching camera")
            cam = launch_camera(
                config_path=str(config_path),
                log_path=log_dir / "camera.log",
                loop_video=loop_video,
                bandwidth_relay_port=(
                    BANDWIDTH_RELAY_PORT if baseline_enabled else None
                ),
            )
            self._procs.append(cam)

        threading.Thread(target=_launch_in_background, daemon=True).start()
        self._show_running_screen(baseline_enabled, save_artifacts, output_dir)

    def _on_stop(self):
        """Terminate all subprocesses and return to config screen."""
        log.info("stop requested; terminating %d subprocesses", len(self._procs))
        for proc in self._procs:
            try:
                proc.terminate()
            except OSError:
                pass
        # Give them a moment to exit cleanly
        for proc in self._procs:
            try:
                proc.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                proc.kill()
        self._procs.clear()

        if self._runtime_config and self._runtime_config.exists():
            try:
                self._runtime_config.unlink()
            except OSError:
                pass
            self._runtime_config = None

        self._show_config_screen()

    def _on_close(self):
        log.info("window close requested")
        self._on_stop()
        self.root.destroy()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    App().run()
