"""H.264 codec backend.

Previously a multi-codec factory (h264 / neural / vvc / dcvc). Now reduced
to H.264 + ffmpeg only — import the backend classes directly.

Usage:
    from src.codec.h264_backend import H264EncoderBackend, H264DecoderBackend
    from src.codec import wire
"""
from __future__ import annotations

from .base import EncodedPacket
from . import wire

__all__ = ["EncodedPacket", "wire"]
