"""FrameRelayServer — pipes annotated BGR frames from a server process to the UI.

Protocol (server → UI):
    [uint32 length][raw BGR bytes]   repeated per frame

The server calls relay.send(frame) after drawing its HUD. The relay is
non-blocking: if no UI is connected, or the send queue is full, the frame
is silently dropped so inference is never stalled.
"""
from __future__ import annotations

import logging
import queue
import socket
import struct
import threading

import cv2
import numpy as np

log = logging.getLogger("echostream.frame_relay")

_QUEUE_MAXSIZE = 4  # drop-oldest if UI can't keep up


class FrameRelayServer:
    """Listens on a local TCP port and streams annotated frames to one UI client.

    Usage inside server_h264.py:
        relay = FrameRelayServer(port=9997)
        relay.start()
        ...
        relay.send(annotated_frame)   # after draw_hud, non-blocking
        ...
        relay.stop()
    """

    def __init__(self, port: int, width: int = 640, height: int = 480):
        self.port = port
        self.width = width
        self.height = height
        self._q: queue.Queue[np.ndarray] = queue.Queue(maxsize=_QUEUE_MAXSIZE)
        self._stop = threading.Event()
        self._accept_thread = threading.Thread(
            target=self._accept_loop, daemon=True, name=f"relay-accept-{port}"
        )

    def start(self) -> None:
        self._accept_thread.start()
        log.info("FrameRelayServer listening on :%d", self.port)

    def stop(self) -> None:
        self._stop.set()

    def send(self, frame: np.ndarray) -> None:
        """Non-blocking. Drops oldest frame if queue is full."""
        if self._stop.is_set():
            return
        try:
            self._q.put_nowait(frame)
        except queue.Full:
            try:
                self._q.get_nowait()  # drop oldest
            except queue.Empty:
                pass
            try:
                self._q.put_nowait(frame)
            except queue.Full:
                pass

    # ------------------------------------------------------------------
    # Internal threads
    # ------------------------------------------------------------------

    def _accept_loop(self) -> None:
        """Accept one UI client at a time; restart on disconnect."""
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("127.0.0.1", self.port))
        srv.listen(1)
        srv.settimeout(1.0)
        while not self._stop.is_set():
            try:
                conn, addr = srv.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            log.info("relay:%d UI connected from %s", self.port, addr)
            self._send_loop(conn)
            log.info("relay:%d UI disconnected", self.port)
        srv.close()

    def _send_loop(self, conn: socket.socket) -> None:
        """Drain the queue and write frames to the connected UI client."""
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        try:
            while not self._stop.is_set():
                try:
                    frame = self._q.get(timeout=0.1)
                except queue.Empty:
                    continue
                # Resize to declared dimensions if needed (safety guard)
                if frame.shape[1] != self.width or frame.shape[0] != self.height:
                    frame = cv2.resize(frame, (self.width, self.height))
                raw = frame.tobytes()
                header = struct.pack("!I", len(raw))
                conn.sendall(header + raw)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            try:
                conn.close()
            except OSError:
                pass