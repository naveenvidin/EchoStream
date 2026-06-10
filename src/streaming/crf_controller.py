"""Small CRF controllers for the adaptive stream.

The camera code only needs a controller with `update(conf) -> int`.
`EmaProbeCrfController` is the default: it smooths segment confidence,
backs off quality loss quickly when confidence falls, and probes CRF
upward every few stable intervals to search for more bandwidth savings.
"""

from __future__ import annotations


class EmaProbeCrfController:
    def __init__(
        self,
        *,
        target: float,
        margin: float,
        ema_alpha: float,
        up_step: int,
        down_step: int,
        probe_interval: int,
        crf_min: int,
        crf_max: int,
        initial_crf: int,
        crf_reset: int
    ):
        self.target = float(target)
        self.margin = float(margin)
        self.ema_alpha = max(0.0, min(1.0, float(ema_alpha)))
        self.up_step = max(1, int(up_step))
        self.down_step = max(1, int(down_step))
        self.probe_interval = max(1, int(probe_interval))
        self.crf_min = int(crf_min)
        self.crf_max = int(crf_max)
        self.crf_reset = int(crf_reset)

        self._crf_value = float(initial_crf)
        self._smoothed_conf = None
        self._probe_wait = 0

    def update(self, conf: float, has_detection: bool = True) -> int:
        if not has_detection:
            self._crf_value = float(self.crf_reset)
            self._smoothed_conf = 1.0
            self._probe_wait = 0
            return int(round(self._crf_value))

        conf = max(0.0, min(1.0, float(conf)))
        if self._smoothed_conf is None:
            self._smoothed_conf = conf
        else:
            self._smoothed_conf = (
                self.ema_alpha * conf
                + (1.0 - self.ema_alpha) * self._smoothed_conf
            )

        if self._smoothed_conf < self.target - self.margin:
            self._crf_value -= self.down_step
            self._probe_wait = 0
        elif self._smoothed_conf > self.target + self.margin:
            self._crf_value += self.up_step
            self._probe_wait = 0
        else:
            self._probe_wait += 1
            if self._probe_wait >= self.probe_interval:
                self._crf_value += self.up_step
                self._probe_wait = 0

        self._crf_value = max(float(self.crf_min), min(float(self.crf_max), self._crf_value))
        return int(round(self._crf_value))


class DiscreteStepCrfController:
    """Five-level discrete CRF lookup with step size 5 between levels.

    Restores the original EchoStream control behavior so it can be A/B'd
    against the EMA-probe controller without touching pipeline logic.
    Direction matches the adaptive controller: low confidence selects a low CRF (higher
    quality), high confidence selects a high CRF (more compression).
    """

    LEVELS = [23, 28, 33, 38, 43]

    def __init__(self, *, initial_crf: int = 23, **_unused):
        self._last = int(initial_crf)

    def update(self, conf: float, has_detection: bool = True) -> int:
        if not has_detection:
            self._last = self.LEVELS[-1]
            return self._last

        c = max(0.0, min(1.0, float(conf)))
        idx = min(int(c * len(self.LEVELS)), len(self.LEVELS) - 1)
        self._last = self.LEVELS[idx]
        return self._last
