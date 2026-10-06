"""Gesture state machine: hand features in, pointer/button/wheel/key actions out.

Gesture map (the forearm rests on the desk; only the two most independent fingers pinch,
ring/pinky pinches strain the hand and get confused with the middle finger):
  index  pinch tap / hold+move   -> left click / drag
  middle pinch tap / hold+move   -> right click / scroll (with fling)
  middle/ring/pinky curled, index relaxed, thumb on its side (Quest-style microgestures):
    thumb tap                    -> middle click
    thumb swipe left / right     -> switch virtual desktop
    thumb swipe up               -> overview
  closed fist held                -> pause / resume
"""

from __future__ import annotations

import math
from typing import Callable, Protocol

from .config import Config
from .features import HandFeatures
from .filters import Ema
from .mapper import Mapper


PINCH_FINGERS = ("index", "middle")


class Output(Protocol):
    def move(self, x: float, y: float) -> None: ...
    def button(self, name: str, down: bool) -> None: ...
    def wheel(self, v: float, h: float) -> None: ...
    def combo(self, keys: list[str]) -> None: ...


class Engine:
    def __init__(self, cfg: Config, out: Output, on_toggle: Callable[[], None] | None = None):
        self.cfg = cfg
        self.g = cfg.gestures
        self.out = out
        self.on_toggle = on_toggle
        self.mapper = Mapper(cfg.pointer)
        self.dist = {f: Ema(0.6) for f in PINCH_FINGERS}
        self.paused = False
        self.finger: str | None = None  # finger currently pinched to the thumb
        self.mode = "hover"              # hover|press|drag|pending|scroll|micro
        self.t0 = 0.0
        self.u0 = self.v0 = 0.0
        self.prev_min_d: float | None = None
        self.hold_until = 0.0  # cursor stays put briefly after a release (double-click aim)
        self.fist_since: float | None = None
        self.fist_fired = False
        self.fling_v = self.fling_h = 0.0  # notches/s
        self.last_t: float | None = None
        # microgestures: the thumb's touch/lift "episodes" against the side of the index finger
        self.micro_since: float | None = None
        self.touching: bool | None = None
        self.ep_t0 = 0.0
        self.ep_start = (0.0, 0.0)
        self.ep_fired = False
        self.ep_bounded = False

    # ---- helpers -----------------------------------------------------------------
    def _moved(self) -> float:
        return math.hypot(self.mapper.u - self.u0, self.mapper.v - self.v0)

    def _end_pinch(self, cancel: bool = False) -> None:
        f, mode, dur = self.finger, self.mode, self.last_t - self.t0
        self.finger, self.mode = None, "hover"
        if not cancel and mode != "drag":
            self.hold_until = self.last_t + self.cfg.pointer.click_freeze
        if f == "index":
            self.out.button("left", False)
        elif cancel:
            return
        elif f == "middle" and mode == "pending":
            self.out.button("right", True)
            self.out.button("right", False)

    def release_all(self) -> None:
        if self.finger is not None:
            self._end_pinch(cancel=True)
        self._micro_end()
        self.fling_v = self.fling_h = 0.0

    # ---- main entry ----------------------------------------------------------------
    def update(self, feat: HandFeatures | None, t: float) -> None:
        dt = 0.0 if self.last_t is None else min(t - self.last_t, 0.2)
        self.last_t = t
        self._fling(dt)

        if feat is None:
            self.release_all()
            self.mapper.lost()
            for e in self.dist.values():
                e.reset()
            self.prev_min_d = None
            self.fist_since = None
            return

        if self._fist(feat, t) or self.paused:
            return

        if feat.micro and self.finger is None:
            self._micro(feat, t)
            self.mapper.update(feat.ax, feat.ay, t, frozen=True)  # cursor holds still
            return
        self._micro_end()

        d = {f: self.dist[f](feat.pinch[f]) for f in PINCH_FINGERS}
        frozen = self._pinch(d, t, feat.curled) or t < self.hold_until or self._approaching(d, dt)
        pos = self.mapper.update(feat.ax, feat.ay, t, frozen=frozen)
        if self.mode == "scroll":
            self._scroll(dt)
        elif pos is not None:
            self.out.move(*pos)

    # ---- pieces --------------------------------------------------------------------
    def _fist(self, feat: HandFeatures, t: float) -> bool:
        if not feat.fist:
            self.fist_since, self.fist_fired = None, False
            return False
        if self.finger is not None:
            self._end_pinch(cancel=True)
        if self.fist_since is None:
            self.fist_since = t
        elif not self.fist_fired and t - self.fist_since >= self.g.fist_hold:
            self.fist_fired = True
            self.release_all()
            if self.on_toggle:
                self.on_toggle()
            else:
                self.paused = not self.paused
        return True

    def _pinch(self, d: dict[str, float], t: float, curled: frozenset = frozenset()) -> bool:
        """Advance the pinch FSM; returns True if the cursor should hold still."""
        g = self.g
        if self.finger is None:
            f = min(d, key=d.get)
            # A thumb resting on a folded finger (pointing / peace sign) is not a pinch.
            if d[f] >= g.pinch_on or f in curled:
                return False
            self.finger, self.t0 = f, t
            self.u0, self.v0 = self.mapper.u, self.mapper.v
            self.fling_v = self.fling_h = 0.0
            if f == "index":
                self.out.button("left", True)
                self.mode = "press"
            else:
                self.mode = "pending"
            return True

        if d[self.finger] > g.pinch_off:
            self._end_pinch()
            return True  # hold still on the release frame too

        held = t - self.t0
        moved = self._moved()
        if self.finger == "index":
            if self.mode == "press" and (
                held >= self.cfg.pointer.click_freeze and moved >= self.cfg.pointer.drag_deadzone
            ):
                self.mode = "drag"
            return self.mode == "press"
        # middle
        if self.mode == "pending" and (held > g.tap_max or moved >= g.hold_move):
            self.mode = "scroll"
            self._su, self._sv = self.mapper.u, self.mapper.v
        return True

    def _micro(self, feat: HandFeatures, t: float) -> None:
        """Thumb against the side of the index finger. Every touch or lift starts an
        "episode"; a short episode is a tap, sliding far enough while touching is a swipe."""
        g = self.g
        if self.micro_since is None:
            self.micro_since, self.mode = t, "micro"
            self.touching = feat.thumb_gap < g.micro_touch
            self.ep_t0, self.ep_start, self.ep_fired, self.ep_bounded = t, feat.thumb_rel, False, False
            return
        settled = t - self.micro_since >= g.micro_settle
        lim = g.micro_lift if self.touching else g.micro_touch
        touching = feat.thumb_gap < lim
        dx = feat.thumb_rel[0] - self.ep_start[0]
        dy = feat.thumb_rel[1] - self.ep_start[1]
        travel = math.hypot(dx, dy)
        if touching != self.touching:
            if (self.ep_bounded and not self.ep_fired and t - self.ep_t0 <= g.micro_tap_max
                    and travel < g.micro_tap_move):
                self.out.button("middle", True)
                self.out.button("middle", False)
            self.touching = touching
            self.ep_t0, self.ep_start, self.ep_fired, self.ep_bounded = t, feat.thumb_rel, False, settled
            return
        if touching and settled and not self.ep_fired and travel >= g.micro_swipe:
            keys = self.cfg.keys
            if abs(dx) >= abs(dy):  # like dragging the desktop: thumb left -> desktop to the right
                self.out.combo(keys.desktop_right if dx < 0 else keys.desktop_left)
                self.ep_fired = True
            elif dy < 0:
                self.out.combo(keys.overview)
                self.ep_fired = True

    def _micro_end(self) -> None:
        if self.micro_since is not None:
            self.micro_since, self.touching = None, None
            if self.mode == "micro":
                self.mode = "hover"

    def _approaching(self, d: dict[str, float], dt: float) -> bool:
        """A finger closing fast toward the thumb: hold the cursor so the click lands where
        the user was aiming, not where the hand drifted while forming the pinch."""
        m = min(d.values())
        prev, self.prev_min_d = self.prev_min_d, m
        if prev is None or dt <= 0 or self.finger is not None:
            return False
        return m < self.g.approach_zone and (prev - m) / dt > self.g.approach_speed

    def _scroll(self, dt: float) -> None:
        u, v = self.mapper.u, self.mapper.v
        du, dv = u - self._su, v - self._sv
        self._su, self._sv = u, v
        sign = 1.0 if self.g.natural_scroll else -1.0
        wv = sign * dv * self.g.scroll_speed
        wh = -sign * du * self.g.scroll_speed
        if wv or wh:
            self.out.wheel(wv, wh)
        if dt > 0:
            k = 0.5
            self.fling_v += k * (wv / dt - self.fling_v)
            self.fling_h += k * (wh / dt - self.fling_h)

    def _fling(self, dt: float) -> None:
        if self.mode == "scroll" or dt <= 0 or (not self.fling_v and not self.fling_h):
            return
        decay = self.g.scroll_inertia ** (dt * 30.0)
        self.fling_v *= decay
        self.fling_h *= decay
        if abs(self.fling_v) < 0.5 and abs(self.fling_h) < 0.5:
            self.fling_v = self.fling_h = 0.0
            return
        self.out.wheel(self.fling_v * dt, self.fling_h * dt)
