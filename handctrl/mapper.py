"""Hybrid hand -> cursor mapping: relative motion with speed-dependent gain, plus a weak
pull toward the absolute box->screen position so hand and cursor never drift apart."""

from __future__ import annotations

from .config import Pointer
from .filters import OneEuro2D


def _clamp(v: float) -> float:
    return 0.0 if v < 0.0 else 1.0 if v > 1.0 else v


class Mapper:
    def __init__(self, cfg: Pointer):
        self.cfg = cfg
        self.filter = OneEuro2D(cfg.min_cutoff, cfg.beta)
        self.cx: float | None = None   # cursor, screen fraction 0..1
        self.cy: float | None = None
        self.prev: tuple[float, float] | None = None
        self.u = self.v = 0.0          # latest filtered anchor in box units

    def to_box(self, ax: float, ay: float) -> tuple[float, float]:
        x0, y0, x1, y1 = self.cfg.box
        return (ax - x0) / (x1 - x0), (ay - y0) / (y1 - y0)

    def lost(self) -> None:
        """Hand disappeared: forget motion history but keep the cursor where it is."""
        self.filter.reset()
        self.prev = None

    def gain(self, speed: float) -> float:
        c = self.cfg
        s2 = speed * speed
        return c.gain_slow + (c.gain_fast - c.gain_slow) * s2 / (s2 + c.speed_mid ** 2)

    def update(self, ax: float, ay: float, t: float, frozen: bool = False):
        """Feed the palm anchor; returns the new cursor (x, y) or None if unchanged."""
        u, v = self.filter(*self.to_box(ax, ay), t)
        self.u, self.v = u, v
        prev, self.prev = self.prev, (u, v)
        if self.cx is None:
            self.cx, self.cy = _clamp(u), _clamp(v)
            return self.cx, self.cy
        if prev is None or frozen:
            return None  # re-acquired (no jump) or deliberately held still
        speed = self.filter.speed
        g = self.gain(speed)
        cx = self.cx + (u - prev[0]) * g
        cy = self.cy + (v - prev[1]) * g
        pull = self.cfg.pull * min(1.0, speed / (2.0 * self.cfg.speed_mid))
        cx += pull * (_clamp(u) - cx)
        cy += pull * (_clamp(v) - cy)
        self.cx, self.cy = _clamp(cx), _clamp(cy)
        return self.cx, self.cy
