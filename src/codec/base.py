"""Codec backend contract for EchoStream.

The pipeline was previously multi-codec (H.264 / neural / VVC / DCVC) with a
factory + a codec-id byte on every wire packet. It has been reduced to a
single H.264 path. The Protocols are kept so the backend still has a stable
shape, but there is only one implementation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Protocol

import numpy as np


@dataclass
class EncodedPacket:
    """A single transmission unit. `data` is opaque H.264 NAL bytes."""
    data: bytes
    frame_index: int
    is_keyframe: bool = False


class EncoderBackend(Protocol):
    def encode(self, frame_bgr: np.ndarray) -> List[EncodedPacket]:
        """Encode one input frame. May return 0..N packets (ffmpeg buffers)."""

    def set_quality(self, conf_score: float, force_keyframe: bool = False) -> None:
        """Adapt quality from a [0,1] confidence score."""

    @property
    def display_quality(self) -> int:
        """CRF value for the HUD."""

    def prewarm(self) -> None:
        """No-op for H.264 (ffmpeg already spawned in __init__)."""

    def close(self) -> None: ...


class DecoderBackend(Protocol):
    def push(self, packet: EncodedPacket) -> None:
        """Feed a packet into the decoder. Non-blocking."""

    def get_frame(self) -> Optional[np.ndarray]:
        """Return the next decoded BGR frame, or None if not ready."""

    def prewarm(self) -> None: ...

    def close(self) -> None: ...


def prewarm(backend) -> None:
    fn = getattr(backend, "prewarm", None)
    if callable(fn):
        fn()
