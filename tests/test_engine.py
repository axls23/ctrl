from types import SimpleNamespace

import pytest

from handctrl.config import Config
from handctrl.features import HandFeatures, extract
from handctrl.filters import OneEuro2D, OneEuroArray
from handctrl.gestures import Engine
from handctrl.mapper import Mapper
from handctrl.tracker import box_from_keypoints, normalize_keypoints, pick_hand

DT = 1 / 30


class FakeOut:
    def __init__(self):
        self.events = []
        self.pos = None

    def move(self, x, y):
        self.pos = (x, y)
        self.events.append(("move", x, y))

    def button(self, name, down):
        self.events.append(("btn", name, down))

    def wheel(self, v, h):
        self.events.append(("wheel", v, h))

    def combo(self, keys):
        self.events.append(("combo", tuple(keys)))

    def of(self, kind):
        return [e for e in self.events if e[0] == kind]


def feat(ax=0.5, ay=0.45, pinch=None, fist=False):
    d = {"index": 1.0, "middle": 1.1, "ring": 1.2, "pinky": 1.3}
    if pinch:
        d[pinch] = 0.05
    return HandFeatures(ax, ay, 0.2, d, fist)


class Sim:
    def __init__(self, cfg=None):
        self.cfg = cfg or Config()
        self.out = FakeOut()
        self.toggles = 0
        self.eng = Engine(self.cfg, self.out)
        self.t = 0.0

    def run(self, n, **kw):
        for _ in range(n):
            self.t += DT
            f = kw.get("f")
            self.eng.update(f(self.t) if callable(f) else f, self.t)


def test_hover_moves_cursor():
    s = Sim()
    s.run(5, f=feat())
    s.run(20, f=lambda t: feat(ax=0.5 + (t - 0.17) * 0.3))
    xs = [e[1] for e in s.out.of("move")]
    assert xs[-1] > xs[0]
    assert not s.out.of("btn")


def test_index_tap_is_click_without_cursor_motion():
    s = Sim()
    s.run(5, f=feat())
    pos = s.out.pos
    # palm drifts slightly while pinching, as real hands do
    s.run(3, f=lambda t: feat(ax=0.5 + (t - 0.17) * 0.02, pinch="index"))
    s.run(1, f=feat(ax=0.5 + 0.004))
    assert s.out.of("btn") == [("btn", "left", True), ("btn", "left", False)]
    assert s.out.pos == pos  # cursor held still through the whole tap


def test_index_hold_and_move_drags():
    s = Sim()
    s.run(5, f=feat())
    s.run(30, f=lambda t: feat(ax=0.5 + max(0, t - 0.3) * 0.5, pinch="index"))
    assert ("btn", "left", True) in s.out.events
    assert ("btn", "left", False) not in s.out.events
    assert s.eng.mode == "drag"
    s.run(3, f=feat(ax=0.8))
    assert s.out.events[-1] != ("btn", "left", True)
    assert ("btn", "left", False) in s.out.events


def test_middle_tap_is_right_click():
    s = Sim()
    s.run(5, f=feat())
    s.run(3, f=feat(pinch="middle"))
    s.run(3, f=feat())
    assert s.out.of("btn") == [("btn", "right", True), ("btn", "right", False)]


def test_middle_hold_move_scrolls_with_natural_direction_and_fling():
    s = Sim()
    s.run(5, f=feat())
    s.run(30, f=lambda t: feat(ay=0.45 - max(0, t - 0.2) * 0.4, pinch="middle"))  # hand up
    wheels = s.out.of("wheel")
    assert wheels and sum(w[1] for w in wheels) < 0  # natural: hand up -> scroll down
    assert not s.out.of("btn")  # no right click after a scroll
    before = len(s.out.of("wheel"))
    s.run(10, f=feat(ay=0.3))
    assert len(s.out.of("wheel")) > before  # inertial fling after release


def micro(gap, rel=(0.0, 0.0), ax=0.5):
    f = feat(ax=ax)
    f.micro, f.thumb_gap, f.thumb_rel = True, gap, rel
    f.curled = frozenset({"middle", "ring", "pinky"})
    return f


def test_micro_thumb_tap_is_middle_click_and_cursor_holds():
    s = Sim()
    s.run(5, f=feat())
    pos = s.out.pos
    s.run(10, f=micro(0.6))   # pose formed, thumb lifted; settle
    s.run(3, f=micro(0.1))    # brief touch on the index side
    s.run(5, f=micro(0.6, ax=0.6))
    assert s.out.of("btn") == [("btn", "middle", True), ("btn", "middle", False)]
    assert s.out.pos == pos
    assert s.eng.mode == "micro"


def test_micro_tap_from_resting_thumb_fires_once():
    s = Sim()
    s.run(10, f=micro(0.1))   # thumb resting on the finger
    s.run(3, f=micro(0.6))    # quick lift...
    s.run(10, f=micro(0.1))   # ...and back
    assert len(s.out.of("btn")) == 2


def test_micro_no_tap_while_forming_pose_or_on_long_rest():
    s = Sim()
    s.run(1, f=micro(0.6))
    s.run(2, f=micro(0.1))    # touch inside the settle window
    s.run(20, f=micro(0.6))
    s.run(30, f=micro(0.1))   # long rest, then lift: not a tap
    s.run(5, f=micro(0.6))
    assert not s.out.of("btn")


def test_micro_swipes_switch_desktop_and_open_overview():
    k = Config().keys
    s = Sim()
    s.run(10, f=micro(0.1))
    s.run(10, f=lambda t: micro(0.1, rel=(-min(0.6, (t - 0.33) * 3), 0.0)))  # thumb slides left
    s.run(3, f=micro(0.6))
    assert s.out.of("combo") == [("combo", tuple(k.desktop_right))]
    assert not s.out.of("btn")

    s = Sim()
    s.run(10, f=micro(0.1))
    s.run(10, f=lambda t: micro(0.1, rel=(0.0, -min(0.6, (t - 0.33) * 3))))  # slides up
    assert s.out.of("combo") == [("combo", tuple(k.overview))]


def test_ring_and_pinky_pinches_do_nothing():
    s = Sim()
    s.run(5, f=feat())
    s.run(3, f=feat(pinch="ring"))
    s.run(3, f=feat(pinch="pinky"))
    s.run(3, f=feat())
    assert not s.out.of("btn") and not s.out.of("combo")


def test_fist_toggles_pause_and_blocks_output():
    s = Sim()
    s.run(5, f=feat())
    s.run(30, f=feat(fist=True))
    assert s.eng.paused
    n = len(s.out.events)
    s.run(10, f=feat(ax=0.9))
    s.run(3, f=feat(pinch="index"))
    assert len(s.out.events) == n
    s.run(5, f=feat())
    s.run(30, f=feat(fist=True))
    assert not s.eng.paused


def test_hand_loss_releases_held_button_and_no_jump_on_return():
    s = Sim()
    s.run(5, f=feat())
    s.run(10, f=feat(pinch="index"))
    s.run(2, f=None)
    assert s.out.of("btn")[-1] == ("btn", "left", False)
    pos = s.out.pos
    s.run(1, f=feat(ax=0.9, ay=0.2))  # reappears elsewhere
    assert s.out.pos == pos


def test_gain_curve_is_monotonic():
    m = Mapper(Config().pointer)
    gains = [m.gain(v) for v in (0.0, 0.1, 0.35, 1.0, 5.0)]
    assert gains == sorted(gains)
    assert gains[0] == pytest.approx(Config().pointer.gain_slow)


def test_one_euro_smooths_jitter():
    f = OneEuro2D(1.2, 8.0)
    out = [f(0.5 + (0.003 if i % 2 else -0.003), 0.5, i * DT)[0] for i in range(60)]
    assert max(out[30:]) - min(out[30:]) < 0.003


def _hand(curled: bool, thumb=(0.35, 0.6)):
    """Crude synthetic right hand pointing up; wrist at bottom. Default thumb tip: away from
    the fingers, on the index side."""
    pts = [(0.5, 0.8)] + [(0.45, 0.7)] * 4  # wrist + thumb chain
    pts[4] = thumb
    for i, x in enumerate((0.45, 0.5, 0.55, 0.6)):
        mcp, pip = (x, 0.6), (x, 0.5)
        tip = (x, 0.58) if curled else (x, 0.35)
        dip = (x, 0.52) if curled else (x, 0.42)
        pts += [mcp, pip, dip, tip]
    return [SimpleNamespace(x=x, y=y) for x, y in pts]


def test_extract_fist_and_open():
    assert extract(_hand(True)).fist
    o = extract(_hand(False))
    assert not o.fist
    assert all(v > 0.35 for v in o.pinch.values())


def test_thumb_resting_on_curled_finger_is_not_a_pinch():
    s = Sim()
    s.run(5, f=feat())
    f = feat(pinch="middle")
    f.curled = frozenset({"middle", "ring", "pinky"})
    s.run(10, f=f)
    s.run(3, f=feat())
    assert not s.out.of("btn")


def _world(hand2d, thumb_z=0.0):
    """Lift a 2D synthetic hand into metres; optionally push the thumb tip toward the camera."""
    w = [SimpleNamespace(x=p.x * 0.4, y=p.y * 0.4, z=0.0) for p in hand2d]
    w[4].z = thumb_z
    return w


def test_pinch_3d_ignores_thumb_in_front_of_fingertip():
    lm = _hand(False)
    lm[4] = SimpleNamespace(x=lm[8].x, y=lm[8].y)  # thumb tip over the index tip in 2D
    assert extract(lm).pinch["index"] < 0.05  # 2D alone: looks like a pinch
    world = _world(lm, thumb_z=-0.05)          # ...but it is 5 cm in front of it
    assert extract(lm, world=world).pinch["index"] > 0.35
    assert extract(lm, world=_world(lm)).pinch["index"] < 0.05  # truly touching


def test_extract_fist_vs_micro_pose():
    lm = _hand(True, (0.43, 0.6))
    lm[7], lm[8] = SimpleNamespace(x=0.45, y=0.42), SimpleNamespace(x=0.45, y=0.35)  # index out
    side = extract(lm)
    assert side.micro and not side.fist and side.thumb_gap < 0.3
    fist = extract(_hand(True, (0.43, 0.6)))
    assert fist.fist and not fist.micro       # every finger curled: the pause gesture
    assert not extract(_hand(False)).micro    # open hand


def test_extract_3d_fist_and_anchor_from_raw():
    lm = _hand(True)
    assert extract(lm, world=_world(lm)).fist
    raw = [SimpleNamespace(x=p.x + 0.1, y=p.y) for p in lm]
    assert extract(lm, anchor_lm=raw).ax == pytest.approx(extract(lm).ax - 0.1)


def test_landmark_filter_smooths_still_jitter_but_follows_motion():
    import numpy as np

    f = OneEuroArray(2.0, 20.0)
    base = np.full((21, 2), 0.5)
    out = [f(base + (0.003 if i % 2 else -0.003), i * DT) for i in range(60)]
    assert np.ptp([o[0, 0] for o in out[30:]]) < 0.003
    f.reset()
    moving = [f(base + 0.02 * i, i * DT) for i in range(30)]  # 0.6 units/s sweep
    assert abs(moving[-1][0, 0] - (0.5 + 0.02 * 29)) < 0.03  # little lag


def test_rtmpose_adapters():
    import numpy as np

    k = np.array([[64.0, 48.0], [320.0, 240.0]] + [[100.0, 100.0]] * 19)
    n = normalize_keypoints(k, 640, 480)
    assert n[0].tolist() == pytest.approx([0.1, 0.1]) and n[1].tolist() == pytest.approx([0.5, 0.5])
    x0, y0, x1, y1 = box_from_keypoints(k, 640, 480, 1.1)
    assert x0 <= 64 and y0 <= 48 and x1 >= 320 and y1 >= 240 and x1 <= 639 and y1 <= 479

    left = np.full((21, 2), 100.0)   # raw-image left = user's right hand
    right = np.full((21, 2), 500.0)
    kp, good, bad = np.stack([right, left]), np.full(21, 0.9), np.full(21, 0.1)
    sc = np.stack([good, good])
    assert pick_hand(kp, sc, "right", 640, 0.4) == 1
    assert pick_hand(kp, sc, "left", 640, 0.4) == 0
    assert pick_hand(kp, np.stack([bad, good]), "any", 640, 0.4) == 1
    assert pick_hand(kp, np.stack([bad, bad]), "any", 640, 0.4) is None
