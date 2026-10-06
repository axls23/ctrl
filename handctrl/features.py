"""Turn 21 hand landmarks (MediaPipe order) into the few numbers the gesture engine needs."""

from __future__ import annotations

import math
from dataclasses import dataclass

WRIST = 0
THUMB_TIP = 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP = 9, 10, 12
RING_MCP, RING_PIP, RING_TIP = 13, 14, 16
PINKY_MCP, PINKY_PIP, PINKY_TIP = 17, 18, 20
PALM = (WRIST, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP)
FINGERS = ("index", "middle", "ring", "pinky")
_TIPS = (INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP)
_PIPS = (INDEX_PIP, MIDDLE_PIP, RING_PIP, PINKY_PIP)


@dataclass(slots=True)
class HandFeatures:
    ax: float              # palm anchor, mirrored image coords (0..1)
    ay: float
    scale: float           # wrist -> middle MCP length (image units)
    pinch: dict[str, float]  # thumb-tip to fingertip distance / scale, per finger
    fist: bool
    curled: frozenset = frozenset()  # fingers folded into the palm (can't start a pinch)
    # Microgesture pose: middle, ring and pinky curled, index relaxed (not curled, so it is not
    # a fist). The thumb then taps / swipes along the side of the index finger.
    micro: bool = False
    thumb_gap: float = 9.0  # thumb tip to the index finger's side (MCP-PIP-DIP) / scale
    thumb_rel: tuple[float, float] = (0.0, 0.0)  # thumb tip - index MCP, mirrored image / scale


def _d(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _d3(a, b) -> float:
    return math.dist((a.x, a.y, a.z), (b.x, b.y, b.z))


def _xyz(p) -> tuple:
    return (p.x, p.y, p.z) if hasattr(p, "z") else tuple(p)


def _seg_dist(p, a, b) -> float:
    """Distance from point p to segment ab (2D tuples or 3D points)."""
    p, a, b = _xyz(p), _xyz(a), _xyz(b)
    ab = [y - x for x, y in zip(a, b)]
    ap = [y - x for x, y in zip(a, p)]
    den = sum(v * v for v in ab) or 1e-12
    t = max(0.0, min(1.0, sum(u * v for u, v in zip(ab, ap)) / den))
    return math.dist(p, [x + t * v for x, v in zip(a, ab)])


def extract(lm, aspect: float = 4 / 3, world=None, anchor_lm=None) -> HandFeatures:
    """`lm` is a sequence of 21 objects/tuples with .x/.y (normalized image coords).

    x is scaled by the image aspect so distances are isotropic, and mirrored so that
    moving the hand right moves the cursor right (selfie view).

    With `world` (21 points .x/.y/.z in metres), pinch and curl are measured in 3D, so a thumb
    that is merely in front of a fingertip from the camera's view does not count as a pinch.
    The palm anchor comes from `anchor_lm` (default `lm`): the cursor has its own filter and
    should get the raw positions.
    """
    pts = [((1.0 - p.x) * aspect, p.y) for p in lm]
    apts = pts if anchor_lm is None else [((1.0 - p.x) * aspect, p.y) for p in anchor_lm]
    ax = sum(apts[i][0] for i in PALM) / len(PALM) / aspect
    ay = sum(apts[i][1] for i in PALM) / len(PALM)
    scale = max(_d(pts[WRIST], pts[MIDDLE_MCP]), 1e-4)
    if world is not None:
        d = _d3
        ref_scale = max(_d3(world[WRIST], world[MIDDLE_MCP]), 1e-4)
        ref = world
    else:
        d = _d
        ref_scale = scale
        ref = pts
    pinch = {n: d(ref[THUMB_TIP], ref[t]) / ref_scale for n, t in zip(FINGERS, _TIPS)}
    # Curled finger: tip is closer to the wrist than its PIP joint.
    curled = frozenset(
        name for name, t, p in zip(FINGERS, _TIPS, _PIPS)
        if d(ref[t], ref[WRIST]) < d(ref[p], ref[WRIST])
    )
    thumb = ref[THUMB_TIP]
    gap = min(_seg_dist(thumb, ref[INDEX_MCP], ref[INDEX_PIP]),
              _seg_dist(thumb, ref[INDEX_PIP], ref[INDEX_DIP])) / ref_scale
    rel = ((pts[THUMB_TIP][0] - pts[INDEX_MCP][0]) / scale,
           (pts[THUMB_TIP][1] - pts[INDEX_MCP][1]) / scale)
    micro = "index" not in curled and len(curled) == 3
    return HandFeatures(ax, ay, scale, pinch, len(curled) == 4, curled, micro, gap, rel)
