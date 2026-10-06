"""Hand landmark tracking backends (single hand, video/tracking mode).

Every backend returns a `Hand` with the 21 landmarks in MediaPipe order:
  mediapipe  BlazePalm + landmark net (TFLite, CPU or iGPU via OpenGL ES), 2D + metric 3D
  rtmpose    RTMDet-nano hand detector + RTMPose-m hand (ONNX, CPU or CUDA), 2D only
"""

from __future__ import annotations

import os
import urllib.request
from dataclasses import dataclass

from .config import MODEL_PATH, MODEL_URL, RTM_DET_URL, RTM_POSE_URL, Config
from .filters import OneEuroArray

NUM_LANDMARKS = 21


def ensure_model() -> str:
    if not MODEL_PATH.exists():
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = MODEL_PATH.with_suffix(".part")
        urllib.request.urlretrieve(MODEL_URL, tmp)
        tmp.rename(MODEL_PATH)
    return str(MODEL_PATH)


class Pt:
    __slots__ = ("x", "y", "z")

    def __init__(self, x: float, y: float, z: float = 0.0):
        self.x, self.y, self.z = x, y, z


def _pts(arr) -> list[Pt]:
    if arr.shape[1] == 2:
        return [Pt(float(x), float(y)) for x, y in arr]
    return [Pt(float(x), float(y), float(z)) for x, y, z in arr]


@dataclass(slots=True)
class Hand:
    lm: list            # 21 points .x/.y, normalized image coords (smoothed if enabled)
    raw: list           # same, unsmoothed: the cursor anchor has its own filter downstream
    world: list | None  # 21 points .x/.y/.z in metres (smoothed), or None for 2D-only backends


class _Smoother:
    def __init__(self, cfg: Config):
        t = cfg.tracking
        self.on = t.smooth
        self.img = OneEuroArray(t.smooth_min_cutoff, t.smooth_beta)
        # World landmarks are hand-centred metres: ~5x smaller motions than image units.
        self.world = OneEuroArray(t.smooth_min_cutoff, t.smooth_beta * 5.0)

    def reset(self) -> None:
        self.img.reset()
        self.world.reset()

    def __call__(self, img, world, t: float) -> Hand:
        raw = _pts(img)
        if not self.on:
            return Hand(raw, raw, _pts(world) if world is not None else None)
        lm = _pts(self.img(img, t))
        return Hand(lm, raw, _pts(self.world(world, t)) if world is not None else None)


def _downscale(frame_bgr, width: int):
    import cv2

    h, w = frame_bgr.shape[:2]
    if width and w > width:
        size = (width, round(h * width / w))
        frame_bgr = cv2.resize(frame_bgr, size, interpolation=cv2.INTER_AREA)
    return frame_bgr


class MediaPipeTracker:
    def __init__(self, cfg: Config, process_width: int | None = None):
        # Keep the inference runtime from spinning up a thread per core.
        os.environ.setdefault("OMP_NUM_THREADS", str(cfg.threads))
        os.environ.setdefault("GLOG_minloglevel", "2")
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        self._mp = mp
        self.hand = cfg.hand
        self.process_width = cfg.camera.process_width if process_width is None else process_width
        self._smooth = _Smoother(cfg)
        model = ensure_model()
        order = {"auto": ("gpu", "cpu"), "gpu": ("gpu",), "cpu": ("cpu",)}[cfg.delegate]
        for i, name in enumerate(order):
            opts = vision.HandLandmarkerOptions(
                base_options=BaseOptions(
                    model_asset_path=model,
                    delegate=BaseOptions.Delegate.GPU if name == "gpu" else BaseOptions.Delegate.CPU,
                ),
                running_mode=vision.RunningMode.VIDEO,
                num_hands=1 if cfg.hand == "any" else 2,
                min_hand_detection_confidence=0.6,
                min_hand_presence_confidence=0.5,
                # Lower = keep tracking longer before falling back to the palm detector,
                # whose fresh crop is what makes the skeleton "pop".
                min_tracking_confidence=0.5,
            )
            try:
                self._lm = vision.HandLandmarker.create_from_options(opts)
                self.delegate = name
                break
            except Exception:
                if i == len(order) - 1:
                    raise
        self.name = f"mediapipe/{self.delegate}"
        self._last_ms = -1

    def process(self, frame_bgr, t: float) -> Hand | None:
        """Returns the tracked hand, or None."""
        import cv2
        import numpy as np

        frame_bgr = _downscale(frame_bgr, self.process_width)
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        ms = max(int(t * 1000), self._last_ms + 1)  # timestamps must strictly increase
        self._last_ms = ms
        res = self._lm.detect_for_video(image, ms)
        idx = self._pick(res)
        if idx is None:
            self._smooth.reset()
            return None
        img = np.array([(p.x, p.y) for p in res.hand_landmarks[idx]])
        world = None
        if res.hand_world_landmarks and len(res.hand_world_landmarks) > idx:
            world = np.array([(p.x, p.y, p.z) for p in res.hand_world_landmarks[idx]])
        return self._smooth(img, world, t)

    def _pick(self, res) -> int | None:
        if not res.hand_landmarks:
            return None
        if self.hand == "any":
            return 0
        # MediaPipe labels handedness as seen in the (unmirrored) image, which is the
        # opposite of the user's real hand for a front-facing camera.
        want = "Left" if self.hand == "right" else "Right"
        for i, hd in enumerate(res.handedness):
            if hd and hd[0].category_name == want:
                return i
        return None

    def close(self) -> None:
        self._lm.close()


class RtmPoseTracker:
    """RTMDet finds the hand only when there is no track; afterwards the box follows the
    previous frame's keypoints, so the pose net sees a steady crop."""

    PAD = 1.1  # on top of the 1.25 padding RTMPose applies to every box

    def __init__(self, cfg: Config, process_width: int | None = None):
        try:
            from rtmlib import RTMDet, RTMPose
        except ImportError as exc:
            raise RuntimeError("backend 'rtmpose' needs the extra: uv sync --extra rtmpose") from exc
        import onnxruntime as ort

        t = cfg.tracking
        if t.rtmpose_device.startswith("cuda") and hasattr(ort, "preload_dlls"):
            ort.preload_dlls()  # CUDA/cuDNN from the nvidia-* pip wheels
        self.hand = cfg.hand
        self.process_width = cfg.camera.process_width if process_width is None else process_width
        self.min_score = t.rtm_min_score
        self.delegate = t.rtmpose_device
        self.name = f"rtmpose/{self.delegate}"
        self._smooth = _Smoother(cfg)
        self._det = RTMDet(RTM_DET_URL, model_input_size=(320, 320), backend="onnxruntime",
                           device=t.rtmpose_device)
        self._pose = RTMPose(RTM_POSE_URL, model_input_size=(256, 256), backend="onnxruntime",
                             device=t.rtmpose_device)
        for tool in (self._det, self._pose):
            self._limit_threads(tool, 1 if t.rtmpose_device.startswith("cuda") else cfg.threads)
        if t.rtmpose_device.startswith("cuda") and \
                "CUDAExecutionProvider" not in self._pose.session.get_providers():
            print("handctrl: CUDA unavailable, rtmpose runs on CPU "
                  "(needs onnxruntime-gpu: uv sync --extra rtmpose)", flush=True)
            self.delegate = "cpu"
            self.name = "rtmpose/cpu"
        self._box = None  # xyxy in processed-frame pixels

    @staticmethod
    def _limit_threads(tool, threads: int) -> None:
        # rtmlib builds its sessions with onnxruntime's default of one thread per core.
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        # Idle worker threads would otherwise busy-wait between frames, burning CPU.
        so.add_session_config_entry("session.intra_op.allow_spinning", "0")
        so.add_session_config_entry("session.inter_op.allow_spinning", "0")
        tool.session = ort.InferenceSession(tool.onnx_model, sess_options=so,
                                            providers=tool.session.get_providers())

    def process(self, frame_bgr, t: float) -> Hand | None:
        frame_bgr = _downscale(frame_bgr, self.process_width)
        h, w = frame_bgr.shape[:2]
        if self._box is not None:
            boxes = [self._box]
        else:
            boxes = list(self._det(frame_bgr)[:2])
            if not boxes:
                self._smooth.reset()
                return None
        kpts, scores = self._pose(frame_bgr, bboxes=boxes)
        idx = pick_hand(kpts, scores, self.hand, w, self.min_score)
        if idx is None:
            self._box = None
            self._smooth.reset()
            return None
        self._box = box_from_keypoints(kpts[idx], w, h, self.PAD)
        return self._smooth(normalize_keypoints(kpts[idx], w, h), None, t)

    def close(self) -> None:
        self._box = None


def normalize_keypoints(kpts, w: int, h: int):
    """Pixel keypoints (21, 2) -> normalized image coords like MediaPipe's."""
    import numpy as np

    return np.asarray(kpts, dtype=np.float64) / np.array([w, h], dtype=np.float64)


def box_from_keypoints(kpts, w: int, h: int, pad: float):
    import numpy as np

    lo, hi = kpts.min(axis=0), kpts.max(axis=0)
    c, half = (lo + hi) / 2, (hi - lo).max() / 2 * pad
    x0, y0 = np.maximum(c - half, 0)
    x1, y1 = np.minimum(c + half, (w - 1, h - 1))
    return [float(x0), float(y0), float(x1), float(y1)]


def pick_hand(kpts, scores, hand: str, w: int, min_score: float) -> int | None:
    """Best-scoring hand above `min_score`. With hand=left/right and two hands visible, take
    the one on that side of the user (RTMPose has no handedness output); a lone hand is
    accepted either way."""
    good = [i for i in range(len(kpts)) if float(scores[i].mean()) >= min_score]
    if not good:
        return None
    if hand != "any" and len(good) > 1:
        # Unmirrored front camera: the user's right hand appears on the image's left.
        side = sorted(good, key=lambda i: float(kpts[i][0][0]))
        return side[0] if hand == "right" else side[-1]
    return max(good, key=lambda i: float(scores[i].mean()))


def make_tracker(cfg: Config, backend: str | None = None, process_width: int | None = None):
    backend = backend or cfg.tracking.backend
    if backend == "mediapipe":
        return MediaPipeTracker(cfg, process_width)
    if backend == "rtmpose":
        return RtmPoseTracker(cfg, process_width)
    raise ValueError(f"unknown tracking backend: {backend!r} (mediapipe | rtmpose)")
