"""Lightweight live bandwidth relay for the Tkinter demo UI.

Protocol:
    [uint32 length][utf-8 JSON payload] repeated per sample

The camera connects as a client and sends one sample after both the adaptive
and baseline encoders have completed the same segment. The UI listens locally
and drops stale samples if rendering falls behind.
"""
from __future__ import annotations

import json
import logging
import queue
import socket
import struct
import threading
from typing import Any

log = logging.getLogger("echostream.bandwidth_relay")

_QUEUE_MAXSIZE = 256


class BandwidthRelayReceiver:
    """Listen for live bandwidth samples from the camera process."""

    def __init__(self, port: int):
        self.port = int(port)
        self._q: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=_QUEUE_MAXSIZE)
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._accept_loop,
            daemon=True,
            name=f"bandwidth-relay-recv-{self.port}",
        )

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def drain(self) -> list[dict[str, Any]]:
        samples = []
        while True:
            try:
                samples.append(self._q.get_nowait())
            except queue.Empty:
                break
        return samples

    def _accept_loop(self) -> None:
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("127.0.0.1", self.port))
        srv.listen(1)
        srv.settimeout(1.0)
        log.info("BandwidthRelayReceiver listening on :%d", self.port)
        while not self._stop.is_set():
            try:
                conn, addr = srv.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            log.info("bandwidth relay UI accepted camera from %s", addr)
            self._recv_loop(conn)
        srv.close()

    def _recv_loop(self, conn: socket.socket) -> None:
        conn.settimeout(1.0)
        try:
            while not self._stop.is_set():
                header = self._recv_exact(conn, 4)
                if header is None:
                    break
                (length,) = struct.unpack("!I", header)
                if length <= 0 or length > 65536:
                    break
                payload = self._recv_exact(conn, int(length))
                if payload is None:
                    break
                try:
                    sample = json.loads(payload.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    continue
                self._put_latest(sample)
        except (OSError, struct.error):
            pass
        finally:
            try:
                conn.close()
            except OSError:
                pass

    def _recv_exact(self, conn: socket.socket, n: int) -> bytes | None:
        buf = b""
        while len(buf) < n and not self._stop.is_set():
            try:
                chunk = conn.recv(n - len(buf))
            except socket.timeout:
                continue
            except OSError:
                return None
            if not chunk:
                return None
            buf += chunk
        return buf if len(buf) == n else None

    def _put_latest(self, sample: dict[str, Any]) -> None:
        try:
            self._q.put_nowait(sample)
        except queue.Full:
            try:
                self._q.get_nowait()
            except queue.Empty:
                pass
            try:
                self._q.put_nowait(sample)
            except queue.Full:
                pass


class BandwidthRelaySender:
    """Best-effort sender used by the camera process."""

    def __init__(self, port: int, host: str = "127.0.0.1"):
        self.host = host
        self.port = int(port)
        self._q: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=_QUEUE_MAXSIZE)
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name=f"bandwidth-relay-send-{self.port}",
        )

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def send(self, sample: dict[str, Any]) -> None:
        if self._stop.is_set():
            return
        try:
            self._q.put_nowait(sample)
        except queue.Full:
            try:
                self._q.get_nowait()
            except queue.Empty:
                pass
            try:
                self._q.put_nowait(sample)
            except queue.Full:
                pass

    def _run(self) -> None:
        sock = None
        while not self._stop.is_set():
            if sock is None:
                sock = self._connect()
                if sock is None:
                    self._stop.wait(0.5)
                    continue
            try:
                sample = self._q.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                payload = json.dumps(sample, separators=(",", ":")).encode("utf-8")
                sock.sendall(struct.pack("!I", len(payload)) + payload)
            except OSError:
                try:
                    sock.close()
                except OSError:
                    pass
                sock = None
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass

    def _connect(self) -> socket.socket | None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1.0)
        try:
            sock.connect((self.host, self.port))
            sock.settimeout(None)
            log.info("bandwidth relay connected to %s:%d", self.host, self.port)
            return sock
        except OSError:
            try:
                sock.close()
            except OSError:
                pass
            return None
