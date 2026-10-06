"""Settings with sane defaults, optionally overridden by ~/.config/handctrl/config.toml."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path

CONFIG_DIR = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "handctrl"
CONFIG_PATH = CONFIG_DIR / "config.toml"
CACHE_DIR = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "handctrl"
MODEL_PATH = CACHE_DIR / "hand_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)
_RTM = "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
RTM_DET_URL = _RTM + "rtmdet_nano_8xb32-300e_hand-267f9c8f.zip"
RTM_POSE_URL = _RTM + "rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-74fb594_20230320.zip"


@dataclass
class Camera:
    device: int = 0
    width: int = 640
    height: int = 480
    fps: int = 30
    # Frames wider than this are downscaled before tracking. Keep it at the capture width: at
    # 320 px a desk-resting hand is only ~70 px tall and fingertip landmarks get noisy.
    process_width: int = 640


@dataclass
class Rates:
    active_fps: float = 30.0
    idle_fps: float = 8.0      # no hand seen for `idle_after` seconds
    paused_fps: float = 5.0    # only watching for the resume gesture
    idle_after: float = 3.0
    motion_gate: float = 2.0   # mean pixel change (0-255) that wakes the tracker when idle
    idle_probe: float = 1.0    # ...or run it anyway this often (seconds)


@dataclass
class Pointer:
    # Region of the (mirrored) camera image that maps onto the whole desktop.
    box: list[float] = field(default_factory=lambda: [0.2, 0.15, 0.8, 0.7])  # x0 y0 x1 y1
    gain_slow: float = 0.45    # precision gain when the hand moves slowly
    gain_fast: float = 2.6     # gain for fast sweeps
    speed_mid: float = 0.35    # box-widths/s where gain is halfway between slow and fast
    pull: float = 0.08         # max per-frame pull toward the absolute mapping
    click_freeze: float = 0.12  # seconds the cursor holds still when a pinch begins
    drag_deadzone: float = 0.012  # fraction of screen before a held pinch moves the cursor
    min_cutoff: float = 1.2    # One Euro filter params (anchor, normalized units)
    beta: float = 8.0


@dataclass
class Gestures:
    pinch_on: float = 0.22     # thumb-tip distance / hand scale to start a pinch
    pinch_off: float = 0.35    # ...and to release it (hysteresis)
    pinch_3d: bool = True      # measure pinches on 3D world landmarks when the backend has them
    approach_zone: float = 0.6   # pinch distance below which a closing finger freezes the cursor
    approach_speed: float = 2.0  # closing speed (hand-scales/s) that counts as "about to pinch"
    tap_max: float = 0.30      # seconds; longer pinches become holds
    hold_move: float = 0.02    # anchor travel (box units) that turns a pinch into a hold-drag
    fist_hold: float = 0.8     # seconds of closed fist to toggle pause
    scroll_speed: float = 14.0  # wheel notches per box-height of hand travel
    scroll_inertia: float = 0.92  # per-frame decay of the fling after release
    natural_scroll: bool = True
    # Thumb microgestures (fingers curled, thumb on the side of the index finger).
    # Distances are in hand scales (wrist -> middle knuckle, ~9 cm).
    micro_touch: float = 0.30  # thumb-to-index-side gap that counts as touching
    micro_lift: float = 0.42   # ...and as lifted off (hysteresis)
    micro_settle: float = 0.15  # seconds to hold the pose before taps count
    micro_tap_max: float = 0.35  # a touch or lift shorter than this is a tap
    micro_tap_move: float = 0.15  # max thumb travel during a tap
    micro_swipe: float = 0.35  # thumb travel along the finger that fires a swipe


@dataclass
class Tracking:
    backend: str = "mediapipe"  # mediapipe | rtmpose (pip extra: handctrl[rtmpose])
    rtmpose_device: str = "cpu"  # cpu | cuda (cuda needs onnxruntime-gpu instead of onnxruntime)
    smooth: bool = True         # One Euro filter on every landmark (stable skeleton & pinches)
    smooth_min_cutoff: float = 2.0   # Hz when still: lower = steadier, laggier
    smooth_beta: float = 20.0        # cutoff increase per unit of landmark speed
    rtm_min_score: float = 0.4  # mean keypoint score below which the RTMPose track is dropped


@dataclass
class Keys:
    # evdev key names (without KEY_ prefix), pressed together in order.
    overview: list[str] = field(default_factory=lambda: ["LEFTMETA", "W"])
    desktop_left: list[str] = field(default_factory=lambda: ["LEFTCTRL", "LEFTMETA", "LEFT"])
    desktop_right: list[str] = field(default_factory=lambda: ["LEFTCTRL", "LEFTMETA", "RIGHT"])


@dataclass
class Config:
    hand: str = "any"          # "right", "left" or "any"
    start_state: str = "active"  # active | paused | off
    tray: bool = True
    preview_on_start: bool = False
    preview_python: str = "/usr/bin/python3"  # needs PySide6 (Fedora: python3-pyside6)
    delegate: str = "auto"     # auto (GPU via OpenGL ES, falls back to CPU) | gpu | cpu
    threads: int = 2
    camera: Camera = field(default_factory=Camera)
    tracking: Tracking = field(default_factory=Tracking)
    rates: Rates = field(default_factory=Rates)
    pointer: Pointer = field(default_factory=Pointer)
    gestures: Gestures = field(default_factory=Gestures)
    keys: Keys = field(default_factory=Keys)


def _merge(obj, data: dict) -> None:
    for f in fields(obj):
        if f.name not in data:
            continue
        cur = getattr(obj, f.name)
        if is_dataclass(cur):
            _merge(cur, data[f.name])
        else:
            setattr(obj, f.name, data[f.name])


def load(path: Path = CONFIG_PATH) -> Config:
    cfg = Config()
    if path.exists():
        with open(path, "rb") as fh:
            _merge(cfg, tomllib.load(fh))
    return cfg


def save_box(box: list[float], path: Path = CONFIG_PATH) -> None:
    """Persist a calibrated pointer box, keeping the rest of the file intact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = path.read_text() if path.exists() else ""
    line = "box = [" + ", ".join(f"{v:.3f}" for v in box) + "]"
    lines = text.splitlines()
    try:
        i = lines.index("[pointer]")
    except ValueError:
        lines += ["", "[pointer]", line]
    else:
        j = i + 1
        while j < len(lines) and not lines[j].startswith("["):
            if lines[j].replace(" ", "").startswith("box="):
                lines[j] = line
                break
            j += 1
        else:
            lines.insert(i + 1, line)
    path.write_text("\n".join(lines).strip() + "\n")
