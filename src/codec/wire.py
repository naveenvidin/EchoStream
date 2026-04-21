"""Wire framing for H.264 packets.

Every packet on the wire is (protocol v3):

    !QI  payload_len, sequence_id   (12 bytes)
    ...  payload                     (payload_len bytes)

``sequence_id`` is a monotonically-increasing uint32 set by the camera
for each packet. The server echoes it in the response header so the
camera can correlate replies, detect stale responses, and measure
request→response latency per frame even if the pipeline were to be
parallelised in the future.

Sequence IDs wrap at 2**32. At 30 fps that's ~4.5 years, well past any
session length we care about.

The previous v2 framing was ``!Q`` only; see ``PROTOCOL_VERSION`` in
``src.inference.protocol`` — both sides must be updated together.
"""
from __future__ import annotations

import struct

from .base import EncodedPacket

_HEADER = struct.Struct("!QI")
HEADER_SIZE = _HEADER.size  # 12


def pack_packet(pkt: EncodedPacket, sequence_id: int = 0) -> bytes:
    return _HEADER.pack(len(pkt.data), int(sequence_id) & 0xFFFFFFFF) + pkt.data


def _recv_exact(sock, size: int) -> bytes:
    chunks, received = [], 0
    while received < size:
        chunk = sock.recv(size - received)
        if not chunk:
            raise ConnectionError("Socket closed while receiving.")
        chunks.append(chunk)
        received += len(chunk)
    return b"".join(chunks)


def read_packet(sock, frame_index: int = 0):
    """Blocking read of one packet from a TCP socket.

    Returns (EncodedPacket, sequence_id). ``frame_index`` is kept for
    backwards compatibility on the EncodedPacket dataclass; the
    authoritative correlation key is ``sequence_id``.
    """
    header = _recv_exact(sock, HEADER_SIZE)
    payload_len, sequence_id = _HEADER.unpack(header)
    payload = _recv_exact(sock, payload_len)
    return (
        EncodedPacket(data=payload, frame_index=frame_index, is_keyframe=False),
        int(sequence_id),
    )
