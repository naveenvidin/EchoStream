"""Minimal one-way mirror for run artifacts.

The camera writes `metrics.csv`, `baseline_metrics.csv`, and
`session_config.json` into a local run directory. When the server-hosted
Streamlit lives on a different machine, it can't see those files. This
module ships them over a dedicated TCP channel so they appear in real
time inside a matching run directory on the server.

The mirror is one-way and purely additive:

* It does NOT touch the inference protocol, masking, encoding/decoding,
  detection, or any pipeline logic.
* If the mirror server is unreachable, the main pipeline keeps running;
  the sender just logs and retries.
* The receiver is a standalone process started alongside `server_h264.py`
  but never integrated into it.

Wire format — newline-delimited JSON, one frame per line:

    {"type": "open",   "run_id": "compare_..."}
    {"type": "config", "run_id": "...", "file": "session_config.json", "content": "..."}
    {"type": "chunk",  "run_id": "...", "file": "metrics.csv",         "data": "..."}
    {"type": "chunk",  "run_id": "...", "file": "baseline_metrics.csv","data": "..."}
    {"type": "close",  "run_id": "..."}
"""
from __future__ import annotations

import argparse
import json
import logging
import socket
import threading
import time
from pathlib import Path
from typing import Dict, Optional


log = logging.getLogger("echostream.mirror")


_MIRRORED_FILES = ("metrics.csv", "baseline_metrics.csv")
_CONFIG_FILES = ("session_config.json",)


# ── camera side ──────────────────────────────────────────────────────────────


class MetricsMirrorSender:
    """Tail run-dir files and forward new bytes to the mirror server.

    Best-effort: connection failures are logged but never propagate into
    the main capture loop. Call `.start()` once after the run dir exists
    and `.stop()` before shutdown so any trailing bytes get flushed.
    """

    def __init__(
        self,
        run_dir: Path,
        host: str,
        port: int,
        poll_interval_sec: float = 0.25,
    ):
        self.run_dir = Path(run_dir)
        self.host = host
        self.port = int(port)
        self.poll_interval_sec = float(poll_interval_sec)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._sock: Optional[socket.socket] = None
        self._offsets: Dict[str, int] = {}
        self._sent_config: bool = False

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._run, name="metrics-mirror-sender", daemon=True,
        )
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        if self._sock is not None:
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None

    def _connect(self) -> bool:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(3.0)
            s.connect((self.host, self.port))
            s.settimeout(None)
            self._sock = s
            ok = self._send({"type": "open", "run_id": self.run_dir.name})
            if ok:
                log.info(
                    "mirror connected to %s:%d run_id=%s",
                    self.host, self.port, self.run_dir.name,
                )
            return ok
        except Exception as e:
            log.warning("mirror connect failed: %s", e)
            self._sock = None
            return False

    def _send(self, obj: dict) -> bool:
        if self._sock is None:
            return False
        try:
            payload = (json.dumps(obj) + "\n").encode("utf-8")
            self._sock.sendall(payload)
            return True
        except Exception as e:
            log.warning("mirror send failed: %s", e)
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None
            return False

    def _tail_file(self, name: str) -> None:
        path = self.run_dir / name
        if not path.exists():
            return
        prev = self._offsets.get(name, 0)
        try:
            size = path.stat().st_size
            if size <= prev:
                return
            with open(path, "rb") as f:
                f.seek(prev)
                chunk = f.read(size - prev)
            if not chunk:
                return
            ok = self._send({
                "type": "chunk",
                "run_id": self.run_dir.name,
                "file": name,
                "data": chunk.decode("utf-8", errors="replace"),
            })
            if ok:
                self._offsets[name] = size
        except Exception as e:
            log.debug("tail %s failed: %s", name, e)

    def _maybe_send_config(self) -> None:
        if self._sent_config:
            return
        for cfg_name in _CONFIG_FILES:
            cfg_path = self.run_dir / cfg_name
            if not cfg_path.exists():
                return
        for cfg_name in _CONFIG_FILES:
            try:
                content = (self.run_dir / cfg_name).read_text(encoding="utf-8")
                self._send({
                    "type": "config",
                    "run_id": self.run_dir.name,
                    "file": cfg_name,
                    "content": content,
                })
            except Exception as e:
                log.debug("mirror config %s failed: %s", cfg_name, e)
        self._sent_config = True

    def _drain_once(self) -> None:
        self._maybe_send_config()
        for name in _MIRRORED_FILES:
            self._tail_file(name)

    def _run(self) -> None:
        while not self._stop.is_set():
            if self._sock is None:
                if not self._connect():
                    # Wait before retrying so a missing server doesn't
                    # burn CPU; main pipeline is unaffected.
                    if self._stop.wait(2.0):
                        break
                    continue
            self._drain_once()
            if self._stop.wait(self.poll_interval_sec):
                break

        # Final flush before close so trailing rows aren't lost.
        try:
            if self._sock is not None:
                self._drain_once()
                self._send({"type": "close", "run_id": self.run_dir.name})
        except Exception:
            pass


# ── server side ──────────────────────────────────────────────────────────────


def _serve_one(conn: socket.socket, addr, output_dir: Path) -> None:
    """Read newline-delimited JSON frames from one connection and write
    received bytes to mirrored files under `output_dir/<run_id>/`.
    """
    log.info("mirror client %s connected", addr)
    buf = b""
    open_files: Dict[tuple, "object"] = {}
    try:
        while True:
            chunk = conn.recv(65536)
            if not chunk:
                break
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                if not line.strip():
                    continue
                try:
                    msg = json.loads(line.decode("utf-8"))
                except Exception as e:
                    log.warning("mirror bad json: %s", e)
                    continue

                t = msg.get("type")
                run_id = str(msg.get("run_id") or "unknown_run")
                # Reject path traversal; only basenames allowed.
                if "/" in run_id or "\\" in run_id or run_id in ("", ".", ".."):
                    log.warning("mirror reject run_id=%r", run_id)
                    continue
                run_path = output_dir / run_id
                run_path.mkdir(parents=True, exist_ok=True)

                if t == "open":
                    log.info("mirror run %s open from %s", run_id, addr)

                elif t == "config":
                    fname = msg.get("file") or "session_config.json"
                    if "/" in fname or "\\" in fname:
                        continue
                    try:
                        (run_path / fname).write_text(
                            msg.get("content", ""), encoding="utf-8",
                        )
                    except Exception as e:
                        log.warning("mirror config write failed: %s", e)

                elif t == "chunk":
                    fname = msg.get("file")
                    if not fname or "/" in fname or "\\" in fname:
                        continue
                    key = (run_id, fname)
                    f = open_files.get(key)
                    if f is None:
                        f = open(run_path / fname, "a", encoding="utf-8",
                                 buffering=1)
                        open_files[key] = f
                    try:
                        f.write(msg.get("data", ""))
                    except Exception as e:
                        log.warning("mirror chunk write failed: %s", e)

                elif t == "close":
                    log.info("mirror run %s closed by client", run_id)

    finally:
        for f in open_files.values():
            try:
                f.close()
            except Exception:
                pass
        try:
            conn.close()
        except Exception:
            pass
        log.info("mirror client %s disconnected", addr)


class MetricsMirrorServer:
    """Standalone TCP server that mirrors incoming run artifacts to disk."""

    def __init__(self, host: str, port: int, output_dir: Path):
        self.host = host
        self.port = int(port)
        self.output_dir = Path(output_dir)

    def serve_forever(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((self.host, self.port))
        s.listen(5)
        log.info(
            "mirror server listening on %s:%d -> %s",
            self.host, self.port, self.output_dir,
        )
        try:
            while True:
                conn, addr = s.accept()
                t = threading.Thread(
                    target=_serve_one,
                    args=(conn, addr, self.output_dir),
                    daemon=True,
                )
                t.start()
        finally:
            try:
                s.close()
            except Exception:
                pass


def main():
    p = argparse.ArgumentParser(
        description="EchoStream artifact mirror server.",
    )
    p.add_argument("--host", default="0.0.0.0",
                   help="Bind host (default 0.0.0.0).")
    p.add_argument("--port", type=int, default=9997,
                   help="Listen port (default 9997).")
    p.add_argument("--output-dir", default="runs",
                   help="Server-side root for mirrored run dirs.")
    args = p.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    MetricsMirrorServer(args.host, args.port, Path(args.output_dir)).serve_forever()


if __name__ == "__main__":
    main()
