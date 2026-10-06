"""Offline tracking quality: record a webcam clip once, then score backends/settings on it.

Metrics (lower is better except detection):
  detected   frames with a hand
  drops      hand -> no hand transitions (track losses)
  jitter     median frame-to-frame landmark acceleration |p[t+1] - 2p[t] + p[t-1]|, as % of
             palm length. Real hand motion is smooth at 30 fps, so this is mostly noise.
  blips      pinch episodes shorter than 0.1 s (flicker), out of all pinch episodes
  ms/frame   tracker time per frame (mean / p95)
"""

from __future__ import annotations

import copy
import math
import statistics
import time
from dataclasses import dataclass

from .config import Config
from .features import MIDDLE_MCP, WRIST, extract
from .filters import Ema
from .gestures import PINCH_FINGERS as FINGERS

STEPS = (
    "hold your open hand still",
    "tap each finger to the thumb, a few times each",
    "move the hand around, pinching now and then",
)


def record(cfg: Config, path: str, secs: float) -> int:
    import cv2

    from .camera import Camera

    cam = Camera(cfg.camera)
    cam.open()
    seq, frame = 0, None
    while frame is None:
        seq, frame = cam.latest(seq)
    h, w = frame.shape[:2]
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), cfg.camera.fps, (w, h))
    print(f"recording {secs:.0f} s to {path} ({w}x{h})")
    n, t_start, step = 0, time.monotonic(), -1
    try:
        while (el := time.monotonic() - t_start) < secs:
            s = min(int(el / secs * len(STEPS)), len(STEPS) - 1)
            if s != step:
                step = s
                print(f"  {el:4.0f} s: {STEPS[s]}", flush=True)
            seq, frame = cam.latest(seq)
            if frame is not None:
                out.write(frame)
                n += 1
    finally:
        out.release()
        cam.close()
    fps = n / secs
    print(f"saved {n} frames ({fps:.1f} fps)")
    if abs(fps - cfg.camera.fps) > 0.1 * cfg.camera.fps:
        print(f"warning: camera delivered {fps:.1f} fps, not {cfg.camera.fps}; filter timing "
              "in `eval` assumes the configured rate")
    return 0


@dataclass
class Variant:
    label: str
    backend: str
    width: int
    smooth: bool
    pinch_3d: bool


def default_variants(cfg: Config) -> list[Variant]:
    vs = [
        Variant("mediapipe 320px raw 2D (old)", "mediapipe", 320, False, False),
        Variant("mediapipe 640px smooth 3D", "mediapipe", 640, True, True),
    ]
    try:
        import rtmlib  # noqa: F401

        vs.append(Variant(f"rtmpose/{cfg.tracking.rtmpose_device} 640px smooth",
                          "rtmpose", 640, True, False))
    except ImportError:
        print("(rtmpose not installed: uv sync --extra rtmpose to include it)")
    return vs


def _accel(prev2, prev, cur, aspect: float) -> float | None:
    if prev2 is None or prev is None:
        return None
    sx = aspect  # isotropic units, like features.extract
    s = math.hypot((prev[MIDDLE_MCP].x - prev[WRIST].x) * sx, prev[MIDDLE_MCP].y - prev[WRIST].y)
    if s < 1e-4:
        return None
    acc = sum(
        math.hypot((c.x - 2 * b.x + a.x) * sx, c.y - 2 * b.y + a.y)
        for a, b, c in zip(prev2, prev, cur)
    ) / len(cur)
    return 100.0 * acc / s


def score(cfg: Config, path: str, v: Variant) -> dict:
    import cv2

    from .tracker import make_tracker

    import os

    if not os.path.exists(path):
        raise SystemExit(f"handctrl: {path} not found — record one first: handctrl record")
    cfg = copy.deepcopy(cfg)
    cfg.tracking.smooth = v.smooth
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot read {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or cfg.camera.fps
    tracker = make_tracker(cfg, v.backend, v.width)
    g = cfg.gestures
    ema = {f: Ema(0.6) for f in FINGERS}  # same smoothing as gestures.Engine
    pinched = {f: False for f in FINGERS}
    run_len = {f: 0 for f in FINGERS}
    episodes = blips = 0
    blip_frames = max(1, round(0.1 * fps))
    n = hands = drops = 0
    times: list[float] = []
    jit, jit_raw = [], []
    hist: list = [None, None]
    hist_raw: list = [None, None]
    had = False
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = n / fps
            n += 1
            t0 = time.perf_counter()
            hand = tracker.process(frame, t)
            times.append(1000 * (time.perf_counter() - t0))
            if hand is None:
                drops += had
                had = False
                hist, hist_raw = [None, None], [None, None]
                for f in FINGERS:
                    ema[f].reset()
                    pinched[f], run_len[f] = False, 0
                continue
            had = True
            hands += 1
            aspect = frame.shape[1] / frame.shape[0]
            for buf, pts, out in ((hist, hand.lm, jit), (hist_raw, hand.raw, jit_raw)):
                a = _accel(buf[0], buf[1], pts, aspect)
                if a is not None:
                    out.append(a)
                buf[0], buf[1] = buf[1], pts
            feat = extract(hand.lm, aspect, world=hand.world if v.pinch_3d else None,
                           anchor_lm=hand.raw)
            for f in FINGERS:
                d = ema[f](feat.pinch[f])
                if not pinched[f] and d < g.pinch_on:
                    pinched[f], run_len[f] = True, 0
                    episodes += 1
                elif pinched[f] and d > g.pinch_off:
                    pinched[f] = False
                    blips += run_len[f] < blip_frames
                if pinched[f]:
                    run_len[f] += 1
    finally:
        cap.release()
        tracker.close()
    times.sort()
    return {
        "label": v.label,
        "frames": n,
        "detected": 100.0 * hands / max(n, 1),
        "drops": drops,
        "jitter": statistics.median(jit) if jit else float("nan"),
        "jitter_raw": statistics.median(jit_raw) if jit_raw else float("nan"),
        "episodes": episodes,
        "blips": blips,
        "ms": statistics.fmean(times) if times else 0.0,
        "ms95": times[int(0.95 * (len(times) - 1))] if times else 0.0,
    }


def evaluate(cfg: Config, path: str, variants: list[Variant]) -> int:
    rows = []
    for v in variants:
        print(f"scoring {v.label} …", flush=True)
        rows.append(score(cfg, path, v))
    print()
    head = (f"{'variant':34} {'detected':>8} {'drops':>5} {'jitter%':>7} {'raw%':>6} "
            f"{'pinch blips':>11} {'ms/frame':>9} {'p95':>5}")
    print(head)
    print("-" * len(head))
    for r in rows:
        print(f"{r['label']:34} {r['detected']:7.1f}% {r['drops']:5d} {r['jitter']:7.2f} "
              f"{r['jitter_raw']:6.2f} {r['blips']:4d} / {r['episodes']:<4d} "
              f"{r['ms']:9.1f} {r['ms95']:5.1f}")
    print("\njitter% = median landmark acceleration, % of palm length (smoothed; raw% before "
          "the landmark filter). blips = pinches shorter than 0.1 s / all pinches.")
    return 0
