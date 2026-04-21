"""Adaptive CRF controller — the stable, single-source-of-truth quality loop.

The old control path lived inline in `camera_h264.main()`: it called
`encoder.set_quality(conf_score, force_keyframe=big_change)` on every
frame with `big_change = |conf - prev_conf| > 0.15`. Two problems:

- **Twitchy**: raw confidence bounces ±0.2 even on a stable scene, so
  CRF jumped around → encoder restarts → IDR floods → wasted bitrate.
- **Silent on lost detections**: when YOLO-World returned 0 boxes we
  fed metric=0.5 which masqueraded as "medium confidence" and kept the
  encoder at whatever CRF it happened to have.

`AdaptiveCRFController` centralizes three stabilizers:

- **EMA smoothing** on the confidence signal (same spirit as the
  proactive planner, but per-frame).
- **Dead-band hysteresis**: ignore |Δcrf| < `dead_band_crf`.
- **Rate limiting**: never restart the encoder more than once per
  `min_frames_between_restarts` (default = gop-length, so restarts
  coincide with IDRs for free).
- **No-detection drift**: if we go `zero_det_tolerance` frames without
  any detection, the controller nudges CRF back toward `fallback_crf`
  (a safe mid-quality setting). This keeps us from getting stuck at
  CRF 42 (high compression) when the scene has just gone empty.

All state is in this class. The camera loop just does:

    controller = AdaptiveCRFController(gop=30, ...)
    ...
    decision = controller.update(conf_score, num_detections)
    if decision.should_apply:
        encoder.set_quality(decision.crf, force_keyframe=decision.force_idr)

The reactive encoder's existing `set_quality` path is preserved; we just
feed it decisions from a controller that actually thinks before firing.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ControlDecision:
    crf: int
    should_apply: bool
    force_idr: bool
    reason: str


def _conf_to_crf(conf: float,
                 crf_min: int = 18,
                 crf_max: int = 51,
                 pivot_conf: float = 0.5,
                 pivot_crf: int = 32) -> int:
    """Piecewise-linear map from a [0,1] confidence to a CRF.

    Low confidence → low CRF (high quality) to help the detector.
    High confidence → high CRF (compress) since the detector is happy.

    Kept monotone and differentiable-ish so EMA smoothing is well-behaved.
    """
    conf = max(0.0, min(1.0, float(conf)))
    if conf <= pivot_conf:
        # Map [0, pivot_conf] → [crf_min, pivot_crf]
        t = conf / max(pivot_conf, 1e-6)
        return int(round(crf_min + t * (pivot_crf - crf_min)))
    # Map (pivot_conf, 1] → (pivot_crf, crf_max]
    t = (conf - pivot_conf) / max(1.0 - pivot_conf, 1e-6)
    return int(round(pivot_crf + t * (crf_max - pivot_crf)))


class AdaptiveCRFController:
    """Smoothed, rate-limited CRF control. See module docstring."""

    def __init__(self, *,
                 initial_crf: int = 28,
                 crf_min: int = 18,
                 crf_max: int = 51,
                 ema_alpha: float = 0.35,
                 dead_band_crf: int = 3,
                 min_frames_between_restarts: int = 30,
                 fallback_crf: int = 28,
                 zero_det_tolerance: int = 15,
                 big_jump_crf: int = 10):
        self.crf_min = int(crf_min)
        self.crf_max = int(crf_max)
        self.ema_alpha = float(ema_alpha)
        self.dead_band_crf = int(dead_band_crf)
        self.min_frames_between_restarts = int(min_frames_between_restarts)
        self.fallback_crf = int(fallback_crf)
        self.zero_det_tolerance = int(zero_det_tolerance)
        self.big_jump_crf = int(big_jump_crf)

        self._current_crf = int(initial_crf)
        self._ema_conf: Optional[float] = None
        self._frames_since_restart = self.min_frames_between_restarts  # eligible immediately
        self._zero_det_streak = 0
        self._transition_count = 0
        self._frame_counter = 0

    # ── Public state ─────────────────────────────────────────────────────────

    @property
    def current_crf(self) -> int:
        return self._current_crf

    @property
    def transition_count(self) -> int:
        return self._transition_count

    @property
    def ema_conf(self) -> float:
        return self._ema_conf if self._ema_conf is not None else 0.5

    # ── Main tick ────────────────────────────────────────────────────────────

    def update(self, conf_score: float, num_detections: int) -> ControlDecision:
        self._frame_counter += 1
        self._frames_since_restart += 1

        if num_detections <= 0:
            self._zero_det_streak += 1
        else:
            self._zero_det_streak = 0

        # EMA update. On first sample we seed to avoid a "warmup ramp".
        if self._ema_conf is None:
            self._ema_conf = float(conf_score)
        else:
            self._ema_conf = (self.ema_alpha * float(conf_score)
                              + (1.0 - self.ema_alpha) * self._ema_conf)

        target = _conf_to_crf(self._ema_conf,
                              crf_min=self.crf_min, crf_max=self.crf_max)

        # If the scene has been empty for a while, drift back toward a
        # medium-quality fallback so we don't stay pinned at high CRF.
        if self._zero_det_streak >= self.zero_det_tolerance:
            target = int(round(0.5 * target + 0.5 * self.fallback_crf))

        delta = target - self._current_crf
        abs_delta = abs(delta)

        # Rate-limit: normal change must wait N frames between restarts.
        # A "big jump" (abs_delta ≥ big_jump_crf) bypasses the rate limit
        # because it indicates a scene change we want to track.
        can_restart = (
            self._frames_since_restart >= self.min_frames_between_restarts
            or abs_delta >= self.big_jump_crf
        )

        if abs_delta < self.dead_band_crf:
            return ControlDecision(
                crf=self._current_crf, should_apply=False, force_idr=False,
                reason="dead_band",
            )
        if not can_restart:
            return ControlDecision(
                crf=self._current_crf, should_apply=False, force_idr=False,
                reason="rate_limited",
            )

        # Commit.
        force_idr = abs_delta >= self.big_jump_crf
        self._current_crf = int(max(self.crf_min, min(self.crf_max, target)))
        self._frames_since_restart = 0
        self._transition_count += 1
        return ControlDecision(
            crf=self._current_crf, should_apply=True, force_idr=force_idr,
            reason="big_jump" if force_idr else "gradual",
        )
