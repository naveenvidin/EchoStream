"""Isolated PI(D) controller for the adaptive CRF knob.

Ported from the PID/ reference implementation. The integral term carries
slow drift; the derivative gain is exposed but defaults to zero to match
the reference behavior. Kept deliberately small so the rest of the
camera/streaming code only sees a single `update(conf) -> int` call.

Also exposes `DiscreteStepCrfController` — the pre-PID 5-level lookup
restored as a separate controller class. Both classes expose the same
`update(conf) -> int` duck-typed interface so `ConfidenceListener` can
hold either one transparently.
"""

from __future__ import annotations


class CrfPIDController:
    def __init__(
        self,
        *,
        target: float,
        kp: float,
        ki: float,
        kd: float = 0.0,
        max_step: float,
        deadband: float,
        crf_min: int,
        crf_max: int,
        initial_crf: int,
        integral_clip: float = 8.0,
    ):
        self.target = float(target)
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.max_step = float(max_step)
        self.deadband = float(deadband)
        self.crf_min = int(crf_min)
        self.crf_max = int(crf_max)
        self.integral_clip = float(integral_clip)

        self._crf_value = float(initial_crf)
        self._integral = 0.0
        self._last_error = 0.0

    def update(self, conf: float) -> int:
        error = float(conf) - self.target
        if abs(error) < self.deadband:
            error = 0.0
        self._integral = max(
            -self.integral_clip,
            min(self.integral_clip, self._integral + error),
        )
        derivative = error - self._last_error
        self._last_error = error
        delta = self.kp * error + self.ki * self._integral + self.kd * derivative
        delta = max(-self.max_step, min(self.max_step, delta))
        self._crf_value = max(
            float(self.crf_min),
            min(float(self.crf_max), self._crf_value + delta),
        )
        return int(round(self._crf_value))


class DiscreteStepCrfController:
    """Five-level discrete CRF lookup with step size 5 between levels.

    Restores the pre-PID EchoStream control behavior so it can be A/B'd
    against the PID controller without touching pipeline logic. Direction
    matches the PID controller: low confidence selects a low CRF (higher
    quality), high confidence selects a high CRF (more compression).
    """

    LEVELS = [23, 28, 33, 38, 43]

    def __init__(self, *, initial_crf: int = 23, **_unused):
        self._last = int(initial_crf)

    def update(self, conf: float) -> int:
        c = max(0.0, min(1.0, float(conf)))
        idx = min(int(c * len(self.LEVELS)), len(self.LEVELS) - 1)
        self._last = self.LEVELS[idx]
        return self._last
