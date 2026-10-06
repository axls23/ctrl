"""Webcam capture on a background thread that only ever keeps the newest frame."""

from __future__ import annotations

import threading
import time

import cv2

from .config import Camera as CameraCfg

cv2.setNumThreads(1)


class Camera:
    def __init__(self, cfg: CameraCfg):
        self.cfg = cfg
        self._cap = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._frame = None
        self._seq = 0
        self._want = False
        self.hot = True  # decode every frame (active), or only on demand (idle/paused)
        self._running = False

    @property
    def is_open(self) -> bool:
        return self._running

    def open(self) -> None:
        if self._running:
            return
        cap = cv2.VideoCapture(self.cfg.device, cv2.CAP_V4L2)
        if not cap.isOpened():
            raise RuntimeError(f"cannot open camera /dev/video{self.cfg.device}")
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.height)
        cap.set(cv2.CAP_PROP_FPS, self.cfg.fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._cap, self._running = cap, True
        self._thread = threading.Thread(target=self._loop, name="camera", daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
            self._thread = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        with self._cond:
            self._frame = None
            self._cond.notify_all()

    def _loop(self) -> None:
        cap = self._cap
        fails = 0
        while self._running:
            # grab() only dequeues the buffer; JPEG decoding (retrieve) happens only when the
            # main loop is waiting, so low-rate states don't pay for decoding every frame.
            ok = cap.grab()
            if ok:
                fails = 0
                if not (self._want or self.hot):
                    continue
                ok, frame = cap.retrieve()
            if not ok:
                fails += 1
                if fails > 50:
                    self._running = False
                    break
                time.sleep(0.05)
                continue
            fails = 0
            with self._cond:
                self._frame = frame
                self._seq += 1
                self._want = False
                self._cond.notify_all()

    def latest(self, after_seq: int, timeout: float = 1.0):
        """Wait for the next decoded frame; returns (seq, frame) or (after_seq, None)."""
        with self._cond:
            if self._seq > after_seq and self._frame is not None:
                return self._seq, self._frame  # already have a newer frame: no waiting
            self._want = True
            if self._seq <= after_seq and self._running:
                self._cond.wait(timeout)
            if self._seq <= after_seq or self._frame is None:
                return after_seq, None
            return self._seq, self._frame
