# handctrl

Control the Linux cursor with bare-hand gestures through your webcam — pinch-to-click in the
style of Apple Vision Pro / Meta Quest hand tracking. Runs as a light background service with a
tray icon, works on Wayland and X11 (uses a kernel `uinput` virtual device).

## Install

```sh
./install.sh          # uv venv + model, one sudo step for /dev/uinput access, autostart
```

## Gestures

| Gesture | Action |
|---|---|
| Open, relaxed hand | Move cursor (palm-anchored: pinching never moves the pointer) |
| Index + thumb pinch | Click · hold and move to drag / select |
| Middle + thumb pinch | Right click · hold and move to scroll (natural, with fling) |
| Middle, ring, pinky curled, index relaxed, thumb on its side: | *thumb microgestures* |
| · thumb tap | Middle click |
| · thumb swipe ← → along the finger | Switch desktops |
| · thumb swipe ↑ | Overview (Meta+W) |
| Closed fist, ~1 s | Pause / resume |

Only the index and middle fingers pinch: they are the most comfortable and independent fingers,
while ring/pinky pinches strain the hand and get misread as each other. The thumb-on-index
microgestures (as on Meta Quest) need almost no muscle effort.

Rest your forearm on the desk; small hand movements cover the screen (slow = precise, fast = far).

## Commands

```
handctrl toggle | pause | resume | off | on | status
handctrl preview      # pop-out live camera view with hand skeleton & gesture HUD
handctrl calibrate    # sweep your hand for 6 s to set the comfortable pointer area
handctrl bench        # measure tracking cost
handctrl record       # record a 30 s test clip (handctrl-clip.avi)
handctrl eval handctrl-clip.avi   # score tracking quality: old vs new settings vs rtmpose
handctrl run --dry-run --preview --verbose   # try it without installing
```

Tray icon: click = pause/resume, middle-click = camera off, menu = preview, calibrate, quit.
Bind `handctrl toggle` to a KDE shortcut for a hotkey.

## Resources

Tracking runs on the integrated GPU (OpenGL ES) when available, at 640 px. Active ≈ 15 % of
one core; idle (no hand) only runs the network when the image changes; **Off** releases the
camera entirely (~0 %).

## Configuration

`~/.config/handctrl/config.toml` — every field in `handctrl/config.py` can be overridden, e.g.

```toml
hand = "right"          # right | left | any
delegate = "auto"       # auto | gpu | cpu
[tracking]
backend = "mediapipe"   # mediapipe | rtmpose
smooth = true           # One Euro filter on every landmark
[pointer]
gain_slow = 0.45
gain_fast = 2.6
[gestures]
natural_scroll = true
[keys]
overview = ["LEFTMETA", "W"]
```

## Tracking backends

- **mediapipe** (default): light, runs on the iGPU, gives 3D landmarks. Pinches are measured
  in 3D (`gestures.pinch_3d`), so a thumb that only looks like it touches a finger from the
  camera's view doesn't click.
- **rtmpose**: RTMDet + RTMPose-m hand models (Apache-2.0, ONNX). Install with
  `uv sync --extra rtmpose`. It's 2D only, and it's heavy on CPU, so set `rtmpose_device = "cuda"`
  and replace `onnxruntime` with `onnxruntime-gpu` to run it on an NVIDIA GPU. It has no
  handedness output: with `hand = "left"/"right"`, the hand on that side is chosen only when
  both hands are visible.

Use `handctrl record` + `handctrl eval` to compare them on your own camera and lighting.
