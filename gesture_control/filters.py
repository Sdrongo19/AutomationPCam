"""Filtro One Euro: smorza il tremolio a mano ferma, resta reattivo nei movimenti veloci."""
from __future__ import annotations

import math


class _LowPass:
    def __init__(self):
        self.value = None

    def __call__(self, x: float, alpha: float) -> float:
        self.value = x if self.value is None else alpha * x + (1 - alpha) * self.value
        return self.value


def _alpha(cutoff: float, dt: float) -> float:
    tau = 1.0 / (2 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class OneEuro:
    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.5, d_cutoff: float = 1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.reset()

    def reset(self) -> None:
        self._x = _LowPass()
        self._dx = _LowPass()
        self._last_t = None

    def __call__(self, x: float, t: float) -> float:
        if self._last_t is None:
            self._last_t = t
            self._dx(0.0, 1.0)
            return self._x(x, 1.0)
        dt = max(t - self._last_t, 1e-3)
        self._last_t = t
        prev = self._x.value
        dx = self._dx((x - prev) / dt, _alpha(self.d_cutoff, dt))
        cutoff = self.min_cutoff + self.beta * abs(dx)
        return self._x(x, _alpha(cutoff, dt))


class OneEuro2D:
    def __init__(self, min_cutoff: float, beta: float):
        self.fx = OneEuro(min_cutoff, beta)
        self.fy = OneEuro(min_cutoff, beta)

    def reset(self) -> None:
        self.fx.reset()
        self.fy.reset()

    def __call__(self, x: float, y: float, t: float):
        return self.fx(x, t), self.fy(y, t)
