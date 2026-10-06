"""Low-latency smoothing filters."""

from __future__ import annotations

import math


def _alpha(cutoff: float, dt: float) -> float:
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class OneEuro2D:
    """One Euro filter (Casiez et al. 2012): heavy smoothing when still, low lag when fast."""

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.0, d_cutoff: float = 1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.reset()

    def reset(self) -> None:
        self.x = self.y = None
        self.dx = self.dy = 0.0
        self.t = None

    def __call__(self, x: float, y: float, t: float) -> tuple[float, float]:
        if self.t is None or t <= self.t:
            self.x, self.y, self.t = x, y, t
            return x, y
        dt = t - self.t
        self.t = t
        a_d = _alpha(self.d_cutoff, dt)
        self.dx += a_d * ((x - self.x) / dt - self.dx)
        self.dy += a_d * ((y - self.y) / dt - self.dy)
        a = _alpha(self.min_cutoff + self.beta * math.hypot(self.dx, self.dy), dt)
        self.x += a * (x - self.x)
        self.y += a * (y - self.y)
        return self.x, self.y

    @property
    def speed(self) -> float:
        return math.hypot(self.dx, self.dy)


class Ema:
    def __init__(self, k: float):
        self.k, self.v = k, None

    def reset(self) -> None:
        self.v = None

    def __call__(self, x: float) -> float:
        self.v = x if self.v is None else self.v + self.k * (x - self.v)
        return self.v


class OneEuroArray:
    """One Euro filter over an (N, D) array of points, e.g. 21 hand landmarks.

    The adaptive cutoff is computed per point from its own speed, so a fingertip flicking
    toward the thumb is not slowed down by a palm that is holding still.
    """

    def __init__(self, min_cutoff: float = 2.0, beta: float = 20.0, d_cutoff: float = 1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.reset()

    def reset(self) -> None:
        self.x = None
        self.dx = None
        self.t = None

    def __call__(self, x, t: float):
        import numpy as np

        x = np.asarray(x, dtype=np.float64)
        if self.x is None or self.t is None or t <= self.t or x.shape != self.x.shape:
            self.x, self.dx, self.t = x.copy(), np.zeros_like(x), t
            return x
        dt = t - self.t
        self.t = t
        self.dx += _alpha(self.d_cutoff, dt) * ((x - self.x) / dt - self.dx)
        cutoff = self.min_cutoff + self.beta * np.linalg.norm(self.dx, axis=-1, keepdims=True)
        tau = 1.0 / (2.0 * math.pi * cutoff)
        a = 1.0 / (1.0 + tau / dt)
        self.x += a * (x - self.x)
        return self.x.copy()
