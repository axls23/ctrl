"""Virtual input devices via the kernel uinput module (works on Wayland and X11)."""

from __future__ import annotations

from evdev import AbsInfo, UInput
from evdev import ecodes as e

ABS_MAX = 65535
HI_RES = 120  # REL_WHEEL_HI_RES units per notch
_BUTTONS = {"left": e.BTN_LEFT, "right": e.BTN_RIGHT, "middle": e.BTN_MIDDLE}


def _key(name: str) -> int:
    return getattr(e, "KEY_" + name.upper())


class UInputOutput:
    """Absolute pointer (like a QEMU USB tablet: libinput maps it across all monitors)
    plus a tiny keyboard for shortcut gestures."""

    def __init__(self, key_names: list[str]):
        abs_info = AbsInfo(value=0, min=0, max=ABS_MAX, fuzz=0, flat=0, resolution=0)
        self.ptr = UInput(
            {
                e.EV_KEY: list(_BUTTONS.values()),
                e.EV_ABS: [(e.ABS_X, abs_info), (e.ABS_Y, abs_info)],
                e.EV_REL: [e.REL_WHEEL, e.REL_HWHEEL, e.REL_WHEEL_HI_RES, e.REL_HWHEEL_HI_RES],
            },
            name="handctrl pointer",
            vendor=0x1209,
            product=0x4843,
            version=1,
        )
        self.kbd = UInput({e.EV_KEY: sorted({_key(k) for k in key_names})}, name="handctrl keys")
        self._last = (-1, -1)
        self._down: set[int] = set()
        self._acc = [0.0, 0.0]   # hi-res remainder (v, h)
        self._notch = [0, 0]     # hi-res units since last full notch

    def move(self, x: float, y: float) -> None:
        p = (round(x * ABS_MAX), round(y * ABS_MAX))
        if p == self._last:
            return
        self._last = p
        self.ptr.write(e.EV_ABS, e.ABS_X, p[0])
        self.ptr.write(e.EV_ABS, e.ABS_Y, p[1])
        self.ptr.syn()

    def button(self, name: str, down: bool) -> None:
        code = _BUTTONS[name]
        if down == (code in self._down):
            return
        (self._down.add if down else self._down.discard)(code)
        self.ptr.write(e.EV_KEY, code, int(down))
        self.ptr.syn()

    def wheel(self, v: float, h: float) -> None:
        wrote = False
        for i, (amount, hires, lowres) in enumerate(
            ((v, e.REL_WHEEL_HI_RES, e.REL_WHEEL), (h, e.REL_HWHEEL_HI_RES, e.REL_HWHEEL))
        ):
            self._acc[i] += amount * HI_RES
            units = int(self._acc[i])
            if not units:
                continue
            self._acc[i] -= units
            self.ptr.write(e.EV_REL, hires, units)
            self._notch[i] += units
            notches = int(self._notch[i] / HI_RES)
            if notches:
                self._notch[i] -= notches * HI_RES
                self.ptr.write(e.EV_REL, lowres, notches)
            wrote = True
        if wrote:
            self.ptr.syn()

    def combo(self, keys: list[str]) -> None:
        codes = [_key(k) for k in keys]
        for c in codes:
            self.kbd.write(e.EV_KEY, c, 1)
        self.kbd.syn()
        for c in reversed(codes):
            self.kbd.write(e.EV_KEY, c, 0)
        self.kbd.syn()

    def release(self) -> None:
        for code in list(self._down):
            self.ptr.write(e.EV_KEY, code, 0)
        self._down.clear()
        self.ptr.syn()

    def close(self) -> None:
        self.release()
        self.ptr.close()
        self.kbd.close()
