"""handctrl — control the cursor with VR-style hand gestures through your webcam.

usage:
  handctrl [run] [--verbose] [--no-tray] [--preview] [--dry-run]   start the daemon
  handctrl toggle|pause|resume|off|on|status|preview|calibrate|quit
  handctrl bench [SECONDS] [--backend B]   measure tracking cost (no input is injected)
  handctrl record [SECONDS] [CLIP]         record a test clip (default 30 s, handctrl-clip.avi)
  handctrl eval CLIP [--backend B] [--width N] [--raw] [--pinch2d]
                                           score tracking quality on a clip; with no options,
                                           compares old settings, new settings and rtmpose
"""

from __future__ import annotations

import signal
import sys
import time

COMMANDS = {"toggle", "pause", "resume", "off", "on", "status", "preview", "calibrate", "quit"}


class DryRunOutput:
    """Prints actions instead of injecting them (no /dev/uinput access needed)."""

    def move(self, x, y): pass
    def button(self, name, down): print(f"[dry-run] {name} {'down' if down else 'up'}", flush=True)
    def wheel(self, v, h): pass
    def combo(self, keys): print(f"[dry-run] keys {'+'.join(keys)}", flush=True)


def _run(args: list[str]) -> int:
    from . import config
    from .control import is_running

    if is_running():
        print("handctrl is already running (use `handctrl toggle|quit`)", file=sys.stderr)
        return 1
    cfg = config.load()
    if "--preview" in args:
        cfg.preview_on_start = True
    try:
        if "--dry-run" in args:
            raise LookupError
        from .output import UInputOutput

        k = cfg.keys
        out = UInputOutput(k.overview + k.desktop_left + k.desktop_right)
    except LookupError:
        out = DryRunOutput()
    except (PermissionError, OSError) as exc:
        print(f"handctrl: cannot open /dev/uinput ({exc}).\n"
              "Run ./install.sh (installs a udev rule), then log out and back in.",
              file=sys.stderr)
        return 1

    from .app import App

    app = App(cfg, out, verbose="--verbose" in args or "-v" in args,
              tray=cfg.tray and "--no-tray" not in args)
    signal.signal(signal.SIGTERM, lambda *_: app.command("quit"))
    signal.signal(signal.SIGINT, lambda *_: app.command("quit"))
    signal.signal(signal.SIGUSR1, lambda *_: app.command("toggle"))
    app.run()
    return 0


def _bench(args: list[str]) -> int:
    from . import config
    from .camera import Camera
    from .features import extract
    from .tracker import make_tracker

    pos, opts = _split(args)
    secs = float(pos[0]) if pos else 10.0
    cfg = config.load()
    cam, tracker = Camera(cfg.camera), make_tracker(cfg, opts.get("--backend"))
    cam.open()
    print(f"benchmarking {secs:.0f} s on {tracker.name} — show your hand to the camera…")
    seq, n, hands, busy = 0, 0, 0, 0.0
    cpu0, t_start = time.process_time(), time.monotonic()
    while time.monotonic() - t_start < secs:
        seq, frame = cam.latest(seq)
        if frame is None:
            continue
        t0 = time.monotonic()
        hand = tracker.process(frame, t0)
        if hand is not None:
            extract(hand.lm, world=hand.world)
            hands += 1
        busy += time.monotonic() - t0
        n += 1
    wall = time.monotonic() - t_start
    cpu = time.process_time() - cpu0
    cam.close()
    tracker.close()
    try:
        rss = int(open("/proc/self/status").read().split("VmRSS:")[1].split()[0]) // 1024
    except Exception:
        rss = -1
    print(f"frames {n} ({n / wall:.1f} fps), hand in {hands}, "
          f"{1000 * busy / max(n, 1):.1f} ms/frame inference, "
          f"CPU {100 * cpu / wall:.0f}% of one core, RSS {rss} MB")
    return 0


def _split(args: list[str]) -> tuple[list[str], dict[str, str | bool]]:
    """Positional args, and --flag / --opt VALUE options."""
    pos, opts, i = [], {}, 0
    while i < len(args):
        a = args[i]
        if a in ("--backend", "--width"):
            if i + 1 >= len(args):
                raise SystemExit(f"handctrl: {a} needs a value")
            opts[a] = args[i + 1]
            i += 2
            continue
        if a.startswith("--"):
            opts[a] = True
        else:
            pos.append(a)
        i += 1
    return pos, opts


def _record(args: list[str]) -> int:
    from . import config
    from .evaluate import record

    pos, _ = _split(args)
    secs = float(pos[0]) if pos else 30.0
    path = pos[1] if len(pos) > 1 else "handctrl-clip.avi"
    return record(config.load(), path, secs)


def _eval(args: list[str]) -> int:
    from . import config
    from .evaluate import Variant, default_variants, evaluate

    pos, opts = _split(args)
    if not pos:
        print("usage: handctrl eval CLIP [--backend B] [--width N] [--raw] [--pinch2d]",
              file=sys.stderr)
        return 2
    cfg = config.load()
    if {"--backend", "--width", "--raw", "--pinch2d"} & opts.keys():
        backend = opts.get("--backend", cfg.tracking.backend)
        width = int(opts.get("--width", cfg.camera.process_width))
        smooth, p3d = not opts.get("--raw"), not opts.get("--pinch2d")
        label = f"{backend} {width}px {'smooth' if smooth else 'raw'} {'3D' if p3d else '2D'}"
        variants = [Variant(label, backend, width, smooth, p3d)]
    else:
        variants = default_variants(cfg)
    return evaluate(cfg, pos[0], variants)


def main() -> int:
    args = sys.argv[1:]
    cmd = args[0] if args and not args[0].startswith("-") else "run"
    rest = args[1:] if args and cmd == args[0] else args
    if cmd == "run":
        return _run(rest)
    if cmd == "bench":
        return _bench(rest)
    if cmd == "record":
        return _record(rest)
    if cmd == "eval":
        return _eval(rest)
    if cmd in COMMANDS:
        from .control import send

        try:
            print(send(cmd))
        except OSError:
            print("handctrl is not running", file=sys.stderr)
            return 1
        return 0
    print(__doc__)
    return 0 if cmd in ("-h", "--help", "help") else 2


if __name__ == "__main__":
    sys.exit(main())
