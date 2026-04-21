"""Camera ↔ server wire protocol for prompted-class inference (v3).

Three message kinds travel on the TCP socket:

  1. **Handshake** (camera → server, once, right after TCP connect).
     The camera is the source of truth for the class prompts AND the
     reply heatmap resolution, so neither requires a server restart.

        !HHH  protocol_version, heat_w, heat_h   (6 bytes)
        !I    num_classes                         (4 bytes)
        for each class:
            !H   name_len                        (2 bytes)
            utf-8 bytes                           (name_len bytes)

     heat_w/heat_h are the resolution at which the server should return
     its detection heatmap. 80×60 is a sensible default for a 640×480
     pipeline (≈64× smaller than full-res; still a usable ROI hint once
     upsampled back on the camera side). Values of 0 mean "same as the
     camera frame" — only use for debugging.

  2. **Encoded video packet** (camera → server, per packet). v3 adds a
     ``sequence_id`` field to the wire framing — see ``src.codec.wire``.

  3. **Detection response** (server → camera, one per received packet).
     v3 adds a trailing ``sequence_id`` so the camera can correlate
     replies with sent packets.

        !fIIIIII   metric, heat_w, heat_h, num_boxes,
                   decode_us, infer_us, sequence_id
                   (28 bytes; all I fields are uint32)
        heatmap bytes                              (heat_w * heat_h uint8)
        num_boxes * !fffffI                        (24 bytes each:
                                                    x1,y1,x2,y2,conf,cls_idx)

Notes
-----
- `cls_idx` indexes the class list that was sent in the handshake.
- `decode_us`/`infer_us` are wall-clock microseconds on the server. They
  are uint32 — anything ≥ ~71 minutes saturates, which we never hit.
- ``sequence_id`` is echoed from the incoming packet header. The camera
  matches sent vs received IDs to detect stale or mismatched responses.
- Callers on both sides negotiate once, then trust the heatmap size
  embedded in every response (so a server can still override, e.g. if
  GPU memory forces a fallback).

**Breaking change in v3:** both wire framing and response header grew.
A v2 camera cannot talk to a v3 server and vice-versa — update both.
"""
from __future__ import annotations

import struct
from typing import List, Tuple

import numpy as np


PROTOCOL_VERSION = 3

HANDSHAKE_HEADER = struct.Struct("!HHH")   # version, heat_w, heat_h
HANDSHAKE_HEADER_SIZE = HANDSHAKE_HEADER.size  # 6

BOX_STRUCT = struct.Struct("!fffffI")
BOX_SIZE = BOX_STRUCT.size  # 24

# v3: trailing sequence_id so responses correlate with requests.
RESPONSE_HEADER = struct.Struct("!fIIIIII")
RESPONSE_HEADER_SIZE = RESPONSE_HEADER.size  # 28

DEFAULT_HEATMAP_WH = (80, 60)


Detection = Tuple[float, float, float, float, float, int]


# ── Shared socket helper ─────────────────────────────────────────────────────

def _recv_exact(sock, size: int) -> bytes:
    if size <= 0:
        return b""
    chunks, received = [], 0
    while received < size:
        chunk = sock.recv(size - received)
        if not chunk:
            raise ConnectionError("Socket closed during transfer.")
        chunks.append(chunk)
        received += len(chunk)
    return b"".join(chunks)


# ── Handshake ────────────────────────────────────────────────────────────────

def pack_handshake(class_names: List[str],
                   heat_wh: Tuple[int, int] = DEFAULT_HEATMAP_WH) -> bytes:
    heat_w, heat_h = heat_wh
    out = HANDSHAKE_HEADER.pack(PROTOCOL_VERSION, int(heat_w), int(heat_h))
    out += struct.pack("!I", len(class_names))
    for name in class_names:
        encoded = name.encode("utf-8")
        if len(encoded) > 0xFFFF:
            raise ValueError(f"class name too long: {name!r}")
        out += struct.pack("!H", len(encoded)) + encoded
    return out


def read_handshake(sock) -> Tuple[List[str], Tuple[int, int]]:
    """Return (class_names, (heat_w, heat_h)). Raises on unknown version."""
    version, heat_w, heat_h = HANDSHAKE_HEADER.unpack(
        _recv_exact(sock, HANDSHAKE_HEADER_SIZE)
    )
    if version != PROTOCOL_VERSION:
        raise ConnectionError(
            f"protocol mismatch: got v{version}, expected v{PROTOCOL_VERSION}"
        )
    (n,) = struct.unpack("!I", _recv_exact(sock, 4))
    names: List[str] = []
    for _ in range(n):
        (name_len,) = struct.unpack("!H", _recv_exact(sock, 2))
        names.append(_recv_exact(sock, name_len).decode("utf-8"))
    return names, (int(heat_w), int(heat_h))


# ── Detection response ───────────────────────────────────────────────────────

def encode_response(metric: float,
                    heatmap: np.ndarray,
                    detections: List[Detection],
                    decode_us: int = 0,
                    infer_us: int = 0,
                    sequence_id: int = 0) -> bytes:
    h, w = heatmap.shape[:2]
    hm = heatmap if heatmap.dtype == np.uint8 else heatmap.astype(np.uint8)
    payload = RESPONSE_HEADER.pack(
        float(metric), int(w), int(h), int(len(detections)),
        max(0, int(decode_us)), max(0, int(infer_us)),
        int(sequence_id) & 0xFFFFFFFF,
    )
    payload += hm.tobytes()
    if detections:
        # Single bulk pack is faster than N Python-level pack calls.
        box_buf = bytearray(BOX_SIZE * len(detections))
        for i, det in enumerate(detections):
            BOX_STRUCT.pack_into(box_buf, i * BOX_SIZE, *det)
        payload += bytes(box_buf)
    return payload


def read_response(sock) -> Tuple[float, np.ndarray, List[Detection], int, int, int]:
    """Return (metric, heatmap, detections, decode_us, infer_us, sequence_id)."""
    header = _recv_exact(sock, RESPONSE_HEADER_SIZE)
    metric, heat_w, heat_h, num_boxes, decode_us, infer_us, seq_id = \
        RESPONSE_HEADER.unpack(header)
    metric = max(0.0, min(1.0, float(metric)))
    heat_bytes = _recv_exact(sock, int(heat_w) * int(heat_h))
    heatmap = np.frombuffer(heat_bytes, dtype=np.uint8).reshape(
        (int(heat_h), int(heat_w))
    )
    detections: List[Detection] = []
    if num_boxes:
        box_bytes = _recv_exact(sock, int(num_boxes) * BOX_SIZE)
        for i in range(int(num_boxes)):
            chunk = box_bytes[i * BOX_SIZE:(i + 1) * BOX_SIZE]
            x1, y1, x2, y2, conf, cls_idx = BOX_STRUCT.unpack(chunk)
            detections.append((x1, y1, x2, y2, conf, int(cls_idx)))
    return metric, heatmap, detections, int(decode_us), int(infer_us), int(seq_id)
