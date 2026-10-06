"""Main loop: camera -> tracker -> gesture engine -> uinput, with power states."""

from __future__ import annotations

import threading
import time

import cv2

from . import config as config_mod
from .camera import Camera
from .config import Config
from .control import ControlServer
from .features import extract
from .gestures import Engine
from .preview_link import PreviewLink
from .tracker import make_tracker

STATES = ("active", "paused", "off")


def features_of(hand, frame, cfg: Config):
    world = hand.world if cfg.gestures.pinch_3d else None
    return extract(hand.lm, frame.shape[1] / frame.shape[0], world=world, anchor_lm=hand.raw)
CALIBRATE_SECONDS = 6.0


class App:
    def __init__(self, cfg: Config, out, verbose: bool = False, tray: bool = True):
        self.cfg = cfg
        self.out = out
        self.verbose = verbose
        self.state = cfg.start_state if cfg.start_state in STATES else "active"
        self._wake = threading.Event()
        self._quit = False
        self._calib_until = 0.0
        self._calib_pts: list[tuple[float, float]] = []
        self.camera = Camera(cfg.camera)
        self.tracker = make_tracker(cfg)
        self._small = None  # previous downscaled frame for the idle motion gate
        self._probe_t = 0.0
        self.engine = Engine(cfg, out, on_toggle=lambda: self.command("toggle", source="gesture"))
        self.tray = None
        if tray:
            from .tray import Tray

            self.tray = Tray(self.command, self.state)
        self.preview = PreviewLink(cfg.preview_python)
        self.fps = 0.0
        self.control = ControlServer(self.command)

    # ---- commands (any thread) ------------------------------------------------------
    def command(self, cmd: str, source: str = "cli") -> str:
        cmd = {"on": "active", "resume": "active", "pause": "paused"}.get(cmd, cmd)
        if cmd == "status":
            return self.state
        if cmd == "toggle":
            cmd = "paused" if self.state == "active" else "active"
        if cmd in STATES:
            if cmd != self.state:
                self.state = cmd
                self._wake.set()
                if self.tray:
                    self.tray.set_state(cmd)
                    if source == "gesture":
                        self.tray.notify("Hand Control", "Resumed" if cmd == "active" else "Paused",
                                         timeout=1500)
                self._log(f"state -> {cmd} ({source})")
            return self.state
        if cmd == "preview":
            return "preview on" if self.preview.toggle() else "preview off"
        if cmd == "calibrate":
            if self.state != "active":
                self.command("active", source)
            self._calib_pts.clear()
            self._calib_until = time.monotonic() + CALIBRATE_SECONDS
            self._notify("Calibrating pointer area",
                         f"For {CALIBRATE_SECONDS:.0f} s, sweep your open hand over the area you "
                         "want to use (all four corners, comfortably).", 6000)
            return "calibrating"
        if cmd == "quit":
            self._quit = True
            self._wake.set()
            return "bye"
        return f"unknown command: {cmd}"

    def _notify(self, title: str, body: str, timeout: int = 4000) -> None:
        if self.tray:
            self.tray.notify(title, body, timeout)
        else:
            print(f"handctrl: {title}: {body}", flush=True)

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"handctrl: {msg}", flush=True)

    # ---- main loop -------------------------------------------------------------------
    def run(self) -> None:
        cfg, cam, engine = self.cfg, self.camera, self.engine
        seq, next_t, last_hand = 0, 0.0, 0.0
        prev_state = None
        if cfg.preview_on_start:
            self.preview.start()
        self._log(f"tracking on {self.tracker.name}")
        n, busy, stat_t = 0, 0.0, time.monotonic()
        try:
            while not self._quit:
                state = self.state
                if state != prev_state:
                    engine.release_all()
                    prev_state = state
                if state == "off":
                    if cam.is_open:
                        cam.close()
                        engine.update(None, time.monotonic())
                    self._wake.wait()
                    self._wake.clear()
                    continue
                if not cam.is_open:
                    try:
                        cam.open()
                        seq = 0
                    except RuntimeError as exc:
                        print(f"handctrl: {exc}; retrying in 3 s", flush=True)
                        self._wake.wait(3.0)
                        self._wake.clear()
                        continue

                now = time.monotonic()
                calibrating = now < self._calib_until
                if state == "paused":
                    fps = cfg.rates.paused_fps
                elif calibrating or self.preview.alive or now - last_hand < cfg.rates.idle_after:
                    fps = cfg.rates.active_fps
                else:
                    fps = cfg.rates.idle_fps
                cam.hot = fps >= cfg.rates.active_fps
                if now < next_t:
                    if self._wake.wait(next_t - now):
                        self._wake.clear()
                    continue
                next_t = now + 1.0 / fps - 0.004

                seq, frame = cam.latest(seq)
                if frame is None:
                    if not cam.is_open:  # camera vanished (unplugged / grabbed by another app)
                        cam.close()
                    continue
                t0 = time.monotonic()
                if t0 - last_hand > cfg.rates.idle_after and not calibrating \
                        and not self._motion(frame, t0):
                    self._show(frame, state, None, None, t0)
                    continue  # nothing moving and no hand: skip the network entirely
                hand = self.tracker.process(frame, t0)
                feat = features_of(hand, frame, cfg) if hand is not None else None
                if feat is not None:
                    last_hand = t0

                if self._calib_until:  # collecting, or just finished and needs saving
                    self._calibrate_step(feat, calibrating)
                else:
                    engine.paused = state == "paused"
                    engine.update(feat, t0)

                self._show(frame, state, hand, feat, t0)
                n += 1
                busy += time.monotonic() - t0
                if t0 - stat_t >= 2.0:
                    self.fps = n / (t0 - stat_t)
                    if self.verbose:
                        print(f"handctrl: {state} {n / (t0 - stat_t):4.1f} fps, "
                              f"{1000 * busy / max(n, 1):4.1f} ms/frame, "
                              f"hand={'yes' if feat else 'no '} pinch={engine.finger or '-'} "
                              f"mode={engine.mode}", flush=True)
                    n, busy, stat_t = 0, 0.0, t0
        finally:
            self.shutdown()

    def _show(self, frame, state, hand, feat, t) -> None:
        if not self.preview.alive:
            return
        e, g = self.engine, self.cfg.gestures
        fist = 0.0
        if e.fist_since is not None and not e.fist_fired:
            fist = min(1.0, (t - e.fist_since) / g.fist_hold)
        m = e.mapper
        info = {
            "state": state, "mode": e.mode, "finger": e.finger, "fps": self.fps,
            "delegate": self.tracker.name, "box": self.cfg.pointer.box,
            "pinch_on": g.pinch_on, "pinch_off": g.pinch_off, "fist": fist,
            "cursor": [m.cx, m.cy] if m.cx is not None else None,
            "calib": max(0.0, self._calib_until - t) if self._calib_until else 0,
        }
        if hand is not None:
            info["lm"] = [[round(p.x, 4), round(p.y, 4)] for p in hand.lm]
            info["dist"] = {k: v.v for k, v in e.dist.items() if v.v is not None}
            if feat is not None:
                info["anchor"] = [feat.ax, feat.ay]
        self.preview.push(frame, info)

    def _motion(self, frame, t: float) -> bool:
        small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (64, 48),
                           interpolation=cv2.INTER_AREA)
        prev, self._small = self._small, small
        if prev is None or t - self._probe_t >= self.cfg.rates.idle_probe \
                or cv2.absdiff(small, prev).mean() >= self.cfg.rates.motion_gate:
            self._probe_t = t
            return True
        return False

    def _calibrate_step(self, feat, still_running: bool) -> None:
        if still_running:
            if feat is not None:
                self._calib_pts.append((feat.ax, feat.ay))
            return
        self._calib_until = 0.0
        pts, self._calib_pts = self._calib_pts, []
        if len(pts) < 20:
            self._notify("Calibration failed", "Hand was not visible long enough. Try again.")
            return
        xs, ys = sorted(p[0] for p in pts), sorted(p[1] for p in pts)
        lo, hi = int(len(pts) * 0.03), int(len(pts) * 0.97) - 1
        # Shrink slightly so the screen edges are reachable without stretching.
        mx, my = (xs[hi] - xs[lo]) * 0.06, (ys[hi] - ys[lo]) * 0.06
        box = [xs[lo] + mx, ys[lo] + my, xs[hi] - mx, ys[hi] - my]
        if box[2] - box[0] < 0.12 or box[3] - box[1] < 0.12:
            self._notify("Calibration failed", "Movement range was too small. Try larger sweeps.")
            return
        self.cfg.pointer.box = box
        config_mod.save_box(box)
        self._notify("Calibration saved", "Pointer area: " + ", ".join(f"{v:.2f}" for v in box))

    def shutdown(self) -> None:
        self.engine.release_all()
        self.preview.stop()
        self.camera.close()
        self.control.close()
        try:
            self.tracker.close()
        except Exception:
            pass
        close = getattr(self.out, "close", None)
        if close:
            close()
