"""H.264 backend — wraps the ffmpeg-based encoder/decoder.

Thin adapter over the persistent-ffmpeg `H264Encoder` / `H264Decoder`
classes defined in `src.streaming.camera_h264`. Adds:
- A reactive quality path that debounces CRF changes.
- An optional proactive planner that picks one CRF per GOP from an EMA of
  confidence + max ROI, enabled with `ECHOSTREAM_PROACTIVE=1`.
"""
from __future__ import annotations

import os
from typing import List, Optional

import numpy as np

from src.streaming.camera_h264 import H264Decoder, H264Encoder, conf_to_crf
from .base import EncodedPacket


def _conf_roi_to_crf(conf: float, roi_max: float) -> int:
    """Same shape as conf_to_crf, but lifts quality when ROI was large."""
    base = conf_to_crf(conf)
    # ROI in (0.1, 0.5] pulls CRF down by up to 4 (better quality).
    bump = int(round(min(max(roi_max - 0.1, 0.0), 0.4) / 0.1))
    return max(18, min(51, base - bump))


class _SegmentPlanner:
    """Per-GOP proactive CRF planner.

    Accumulates (conf, roi) during a GOP. At the GOP boundary, picks the
    next GOP's CRF from a smoothed view of those observations. No
    mid-GOP changes; a small dead-band suppresses micro-thrash.
    """

    def __init__(self, gop: int, initial_crf: int, ema_alpha: float = 0.4,
                 dead_band: int = 2):
        self.gop = gop
        self.crf = initial_crf
        self.ema_alpha = ema_alpha
        self.dead_band = dead_band
        self.ema_conf = 0.5
        self._roi_max = 0.0
        self._n = 0

    def observe(self, conf: float, roi: float) -> None:
        self.ema_conf = self.ema_alpha * conf + (1.0 - self.ema_alpha) * self.ema_conf
        if roi > self._roi_max:
            self._roi_max = roi
        self._n += 1

    def plan_next(self) -> int:
        if self._n == 0:
            return self.crf
        proposed = _conf_roi_to_crf(self.ema_conf, self._roi_max)
        self._roi_max = 0.0
        self._n = 0
        if abs(proposed - self.crf) < self.dead_band:
            return self.crf
        self.crf = proposed
        return proposed


class H264EncoderBackend:
    def __init__(self, width=640, height=480, fps=30, gop=30, initial_crf=28):
        self._enc = H264Encoder(
            width=width, height=height, crf=initial_crf, fps=fps, gop=gop
        )
        self._gop = gop
        self._frame_index = 0
        self._proactive = os.environ.get("ECHOSTREAM_PROACTIVE", "0") == "1"
        self._planner = _SegmentPlanner(gop=gop, initial_crf=initial_crf)
        self._restart_count = 0
        if self._proactive:
            print(f"[H264] proactive segment-planning mode (gop={gop})")

    @property
    def proactive(self) -> bool:
        return self._proactive

    @property
    def restart_count(self) -> int:
        return self._restart_count

    def prewarm(self) -> None:
        return None

    def encode(self, frame_bgr: np.ndarray) -> List[EncodedPacket]:
        # Proactive: at every GOP boundary, ask the planner for the next
        # CRF. If it differs, apply via the existing restart path — but
        # this restart is now aligned with an IDR, so it is a clean cut
        # rather than a mid-GOP stutter.
        if (self._proactive and self._frame_index > 0
                and self._frame_index % self._gop == 0):
            next_crf = self._planner.plan_next()
            if next_crf != self._enc._crf:
                self._enc.set_crf(next_crf, force_keyframe=True)
                self._restart_count += 1

        data = self._enc.encode(frame_bgr)
        self._frame_index += 1
        if not data:
            return []
        return [
            EncodedPacket(
                data=data,
                frame_index=self._frame_index,
                is_keyframe=False,
            )
        ]

    def set_quality(self, conf_score: float, force_keyframe: bool = False) -> None:
        # Proactive mode forbids mid-GOP restarts by construction.
        # The planner drives CRF; force_keyframe is intentionally dropped
        # (severe scene cuts get picked up at the next GOP boundary,
        # which is ≤1s at gop=30/fps=30).
        if self._proactive:
            return
        crf = conf_to_crf(float(conf_score))
        before = self._enc._crf
        self._enc.set_crf(crf, force_keyframe=force_keyframe)
        if self._enc._crf != before or force_keyframe:
            self._restart_count += 1

    def observe(self, conf_score: float, roi_ratio: float) -> None:
        """Per-frame signals for the proactive planner. No-op when reactive."""
        if self._proactive:
            self._planner.observe(float(conf_score), float(roi_ratio))

    @property
    def display_quality(self) -> int:
        return int(self._enc._crf)

    def close(self) -> None:
        self._enc.close()


class H264DecoderBackend:
    def __init__(self, width=640, height=480):
        self._dec = H264Decoder(width=width, height=height)

    def prewarm(self) -> None:
        return None

    def push(self, packet: EncodedPacket) -> None:
        self._dec.push(packet.data)

    def get_frame(self) -> Optional[np.ndarray]:
        return self._dec.get_frame()

    @property
    def drops(self) -> int:
        return self._dec.drops

    def close(self) -> None:
        self._dec.close()
